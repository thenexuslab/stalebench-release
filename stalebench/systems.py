"""Production agent-memory systems as StaleBench policies.

The keying ladder in :mod:`stalebench.policies` is deliberately idealized: each
rung isolates one same-fact decision so it can be priced against an oracle. That
leaves the obvious question unanswered --- *where do the systems people actually
deploy sit on that ladder?* This module answers it by wrapping them in the same
``Policy`` interface, so a shipped system and a two-line heuristic are scored by
identical code on identical streams.

Wrapped here:

* :class:`Mem0Policy` --- Mem0's ADD/UPDATE/DELETE/NOOP loop. An LLM is shown
  the incoming statement plus the nearest stored memories and decides whether
  this is a new fact or an update to an existing one. That is exactly the
  same-fact decision the ladder prices, implemented by a production system, so
  Mem0 lands on the ladder as a *measured* rung rather than a cited one.

  **The version matters and is not provenance trivia.** In the 0.1.x line
  ``add()`` calls ``get_update_memory_messages`` and can emit UPDATE or DELETE.
  In the 2.0.x line ``add()`` runs a phased batch pipeline whose extraction
  prompt states that its sole operation is ADD, and the update prompts have no
  call site under ``mem0/memory/`` at all --- so an UPDATE count of zero there
  is fixed before any data is streamed and says nothing about the data. The two
  lines also disagree on the row-cap keyword (``limit`` vs ``top_k``), and
  passing the wrong one is ignored rather than raised. Both are handled by
  inspection, and every result records ``lib_version`` and
  ``adjudicates_on_add`` so a number from this wrapper cannot be read without
  the version that produced it.

* :class:`GraphitiPolicy` --- Graphiti's extract/dedupe/invalidate loop, the
  engine under Zep. Each statement becomes an episode dated by its year; the
  library extracts entities and facts, resolves them against the graph, and
  marks a contradicted fact ``invalid_at`` rather than deleting it. That is the
  temporal-KG mechanism the paper covers in idealized form, measured as
  shipped. Its analogue of ``adjudicates_on_add`` is ``invalidates_on_add``,
  derived the same way: by inspecting whether the installed ``add_episode``
  still reaches the contradiction and date-extraction prompts.

The wrappers are deliberately thin and configure each system with the same
encoder the dense baselines use (all-MiniLM-L6-v2), so a difference in score is
a difference in retention logic rather than in embedding quality. Every
statement carries its benchmark id in the system's own metadata, and rankings
are read back from that metadata: whatever the system chooses to surface is
what gets scored, including when it has merged or rewritten the text.
"""

from __future__ import annotations

import asyncio
import math
import os
import tempfile
import time
import uuid
import warnings
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

import numpy as np

from stalebench.policies import Policy
from stalebench.templama import _YEAR0

_MINILM = "sentence-transformers/all-MiniLM-L6-v2"
# chromadb validates collection names as 3-512 chars of [a-zA-Z0-9._-],
# so the obvious short name is rejected there but accepted elsewhere
_COLLECTION = "stalebench"


class Mem0Policy(Policy):
    """Mem0 (``mem0ai``) driven through the benchmark's Policy interface.

    ``observe`` hands the statement to ``Memory.add``, which runs Mem0's own
    extraction and ADD/UPDATE/NOOP decision; ``rank`` asks ``Memory.search``
    and maps the returned memories back to benchmark statement ids through the
    metadata Mem0 stores alongside each memory.

    Memories Mem0 has merged or rewritten keep the id of the statement that
    last wrote them, which is the honest reading: the benchmark scores the
    version the system would actually surface. That id comes from our own
    record of the latest writer rather than from the memory's payload metadata,
    because the update path does not rewrite metadata -- an updated memory
    still carries its *original* writer's id.
    """

    def __init__(self, llm_model: str = "qwen2.5:14b",
                 ollama_url: str = "http://localhost:11434",
                 embed_model: str = _MINILM, top_k: int = 20,
                 vector_store: Optional[str] = None,
                 extraction_prompt: Optional[str] = None):
        self.llm_model = llm_model
        self.ollama_url = ollama_url
        self.embed_model = embed_model
        self.top_k = top_k
        # The default fact-extraction prompt is user-centric ("Personal
        # Information Organizer"). A model that reads it literally extracts
        # nothing from a third-person statement, so the adjudication step is
        # never reached; the library exposes a replacement prompt for exactly
        # this, and which one ran is recorded in every result.
        self.extraction_prompt = extraction_prompt
        self.extraction_prompt_label = ("default" if extraction_prompt is None
                                        else "third-person")
        # The backend is part of the result, not an implementation detail: on
        # one release line the FAISS backend returned the two *least* similar
        # memories per query (unnormalised distances ranked as similarities),
        # which looks like a terrible system and is a broken retrieval layer.
        # Comparing two releases therefore requires holding it fixed, and
        # recording which one was held.
        self.vector_store = (vector_store
                             or os.environ.get("STALEBENCH_MEM0_STORE")
                             or "faiss")
        self.n_add = 0
        self.events: Dict[str, int] = {}
        super().__init__(budget=None)
        self.name = (f"mem0({llm_model})" if extraction_prompt is None
                     else f"mem0({llm_model}|third-person)")

    @property
    def lib_version(self) -> str:
        """The installed mem0 version -- load-bearing, not provenance trivia.

        The ADD/UPDATE/DELETE adjudication this benchmark prices is on the
        ``add()`` path in the 0.1.x line and not in the 2.0.x line, whose
        default extraction prompt states that its sole operation is ADD. A
        result from this wrapper is uninterpretable without the version.
        """
        import mem0
        return getattr(mem0, "__version__", "unknown")

    @property
    def adjudicates_on_add(self) -> bool:
        """Whether this version's ``add()`` can emit anything but ADD."""
        import inspect
        from mem0.memory import main as _m
        try:
            return "get_update_memory_messages" in inspect.getsource(_m)
        except Exception:
            return False

    def use_openai_compatible(self, base_url: str, api_key: str) -> None:
        """Drive the system's decisions with a hosted OpenAI-compatible model.

        Separates "this system does not supersede" from "the local model we gave
        it does not supersede" --- different findings that a single LLM cannot
        distinguish.
        """
        self._openai = {"base_url": base_url, "api_key": api_key}
        # keep the extraction-prompt suffix the constructor put in the name
        self._mem = None

    # -- lifecycle --------------------------------------------------------- #

    def reset(self) -> None:
        super().reset()
        self.n_add = 0
        self.n_silent_add = 0
        self.events = {}
        self.last_hits: List[Dict] = []
        self._sid_of: Dict[str, str] = {}       # mem0 memory id -> statement id
        self._user = f"sb{uuid.uuid4().hex[:8]}"
        self._mem = None                         # built lazily; see _memory()
        self._openai = getattr(self, "_openai", None)

    def _memory(self):
        if self._mem is None:
            warnings.filterwarnings("ignore")
            from mem0 import Memory
            store = tempfile.mkdtemp(prefix="sb_mem0_")
            if self._openai:
                llm = {"provider": "openai",
                       "config": {"model": self.llm_model, "temperature": 0.0,
                                  "openai_base_url": self._openai["base_url"],
                                  "api_key": self._openai["api_key"]}}
            else:
                llm = {"provider": "ollama",
                       "config": {"model": self.llm_model, "temperature": 0.0,
                                  "ollama_base_url": self.ollama_url}}
            if self.vector_store == "qdrant":
                vs = {"provider": "qdrant",
                      "config": {"collection_name": _COLLECTION,
                                 "embedding_model_dims": 384,
                                 "on_disk": False, "path": store}}
            elif self.vector_store == "chroma":
                vs = {"provider": "chroma",
                      "config": {"collection_name": _COLLECTION, "path": store}}
            else:
                vs = {"provider": "faiss",
                      "config": {"collection_name": _COLLECTION,
                                 "embedding_model_dims": 384, "path": store}}
            cfg = {
                "llm": llm,
                "embedder": {"provider": "huggingface",
                             "config": {"model": self.embed_model}},
                "vector_store": vs,
                "history_db_path": os.path.join(store, "history.db"),
            }
            if self.extraction_prompt is not None:
                cfg["custom_fact_extraction_prompt"] = self.extraction_prompt
            self._mem = Memory.from_config(cfg)
        return self._mem

    # -- Policy interface -------------------------------------------------- #

    def observe(self, emb: np.ndarray, item_id: str, t: float,
                text: Optional[str] = None) -> None:
        res = self._memory().add(text or "", user_id=self._user,
                                 metadata={"sid": item_id, "t": t})
        self.n_add += 1
        # Some (version, backend) pairs create and merge memories correctly but
        # return an empty results list, so the event histogram is unavailable
        # even though retention is observable. Record that explicitly: the
        # store's final size against the number of ingestion calls is then the
        # evidence of merging, and it does not depend on the report.
        if not ((res or {}).get("results") or []):
            self.n_silent_add = getattr(self, "n_silent_add", 0) + 1
        for r in (res or {}).get("results", []) or []:
            ev = (r.get("event") or "NONE").upper()
            self.events[ev] = self.events.get(ev, 0) + 1
            if r.get("id"):
                # a memory Mem0 updated keeps the id of its latest writer
                self._sid_of[r["id"]] = item_id

    # -- version portability ----------------------------------------------- #
    # The 0.1.x and 2.0.x lines differ in more than behaviour: the row cap is
    # ``limit`` in 0.1.x and ``top_k`` in 2.0.x, and passing the wrong one is
    # silently ignored rather than raising -- which reports a 20-slot store no
    # matter how much the system retained. Bind by inspection, once.

    def _cap_kwarg(self, fn) -> str:
        import inspect
        params = inspect.signature(fn).parameters
        return "top_k" if "top_k" in params else "limit"

    def _scope(self, fn) -> Dict:
        """How this version wants the entity scope passed.

        0.1.x takes ``user_id`` as a top-level keyword and *rejects* it inside
        ``filters``; 2.0.x requires it inside ``filters`` and rejects it at top
        level. Third incompatibility between the lines in the same two methods
        (after the ``limit``/``top_k`` cap and the ``threshold`` default), so it
        is resolved the same way: by inspection rather than by assumption.
        """
        import inspect
        params = inspect.signature(fn).parameters
        if "user_id" in params:
            return {"user_id": self._user}
        return {"filters": {"user_id": self._user}}

    @property
    def slots(self) -> int:
        try:
            mem = self._memory()
            kw = {self._cap_kwarg(mem.get_all): 1_000_000}
            got = mem.get_all(**self._scope(mem.get_all), **kw)
            n = len((got or {}).get("results", []) or [])
            if n:
                return n
        except Exception:
            pass
        return len(self._sid_of)

    def hits(self, query_text: Optional[str] = None) -> List[Dict]:
        """The memories the system surfaces, in its own rank order.

        This is what a reader would actually be handed, and it is strictly more
        than :meth:`rank` returns: a memory the wrapper cannot attribute to a
        benchmark statement is dropped from the id-level ranking but is still
        *in the context*. Keeping the raw hits lets the same run be scored three
        ways --- by id attribution, by the text the system wrote, and by what a
        reader answers from it --- without a second hour of ingestion.

        Each hit carries the system's own timestamps because a merged memory's
        payload metadata still names its original writer; ``updated_at`` is the
        only recency signal that survives an UPDATE.
        """
        mem = self._memory()
        kw = {self._cap_kwarg(mem.search): self.top_k}
        # threshold defaults to 0.1 in the 2.0.x line: an undisclosed relevance
        # cutoff would truncate the ranking and so change current@5 and
        # contamination@5. Ask for everything and let the metric decide.
        try:
            res = mem.search(query_text or "", threshold=0.0,
                             **self._scope(mem.search), **kw)
        except TypeError:
            res = mem.search(query_text or "",
                             **self._scope(mem.search), **kw)
        out: List[Dict] = []
        for r in (res or {}).get("results", []) or []:
            # a memory the system UPDATED keeps its original payload metadata,
            # so the id we recorded for the latest writer is the authoritative
            # one; metadata is only the fallback for memories never updated
            meta_sid = (r.get("metadata") or {}).get("sid")
            sid = self._sid_of.get(r.get("id")) or meta_sid
            # both attributions are recorded because they disagree, and the
            # disagreement is the finding: the update path does not rewrite
            # payload metadata, so meta_sid names the memory's *original*
            # writer while sid names its latest. Where a release relabels a
            # memory without rewriting its text, only the latter calls it
            # current -- and neither is the memory's actual content, which is
            # why the text is carried too
            out.append({"id": r.get("id"), "sid": sid, "meta_sid": meta_sid,
                        "text": r.get("memory") or r.get("text") or "",
                        "score": r.get("score"),
                        "created_at": r.get("created_at"),
                        "updated_at": r.get("updated_at")})
        self.last_hits = out
        return out

    def rank(self, query: np.ndarray,
             query_text: Optional[str] = None) -> List[str]:
        out: List[str] = []
        for h in self.hits(query_text):
            if h["sid"] and h["sid"] not in out:
                out.append(h["sid"])
        return out


# --------------------------------------------------------------------------- #
# Graphiti
# --------------------------------------------------------------------------- #

def statement_time(t: float) -> datetime:
    """The datetime Graphiti is told a statement was made.

    ``build_instance`` stamps each statement with ``t = (year - 2010) + 1e-4 *
    position``: the year is the integer part and the stream position the
    fraction. Graphiti dates its episodes, uses the date to pick the preceding
    episodes it shows the extractor, and copies it into a fact's ``valid_at``,
    so the year has to survive the round trip exactly. The position is carried
    as minutes: same-year statements keep their order and never leave January.
    """
    year = math.floor(t + 1e-9)
    seq = int(round((t - year) * 1e4))
    return datetime(_YEAR0 + year, 1, 1, tzinfo=timezone.utc) + timedelta(minutes=seq)


def _iso(dt: Optional[datetime]) -> Optional[str]:
    return dt.isoformat() if dt else None


def _minilm_embedder(model_name: str):
    """A graphiti ``EmbedderClient`` over the encoder the dense baselines use.

    Graphiti ships OpenAI-shaped embedders only. Wrapping sentence-transformers
    keeps the encoder identical to the ladder's, so a difference in score is a
    difference in the graph's retention logic rather than in embedding quality.
    Vectors are L2-normalised as in :mod:`stalebench.embed`, and the dimension
    is reported from the model rather than assumed.
    """
    from graphiti_core.embedder.client import EmbedderClient, EmbedderConfig
    from sentence_transformers import SentenceTransformer

    class _MiniLM(EmbedderClient):
        def __init__(self):
            self.model = SentenceTransformer(model_name)
            self.config = EmbedderConfig(
                embedding_dim=self.model.get_sentence_embedding_dimension())

        def _encode(self, texts) -> List[List[float]]:
            emb = self.model.encode(list(texts), convert_to_numpy=True,
                                    normalize_embeddings=True,
                                    show_progress_bar=False)
            return emb.astype(np.float32).tolist()

        async def create(self, input_data):
            texts = [input_data] if isinstance(input_data, str) else list(input_data)
            return self._encode(texts)[0]

        async def create_batch(self, input_data_list):
            return self._encode(input_data_list)

    return _MiniLM()


def _no_reranker():
    """The reranker slot, filled with nothing.

    The default search recipe never calls it, but leaving the slot empty makes
    Graphiti construct an OpenAI reranker, which demands an API key at import.
    """
    from graphiti_core.cross_encoder.client import CrossEncoderClient

    class _NoRerank(CrossEncoderClient):
        async def rank(self, query, passages):
            return [(p, 0.0) for p in passages]

    return _NoRerank()


def _ollama_llm(policy: "GraphitiPolicy"):
    """Graphiti's generic OpenAI-compatible client pointed at Ollama.

    It has to be this client and not the default ``OpenAIClient``: the default
    uses OpenAI's structured-output parse, which Ollama ignores, and the result
    is an extraction that silently returns nothing. Every attempt is counted on
    the policy so the cost of a statement is measured rather than estimated.
    """
    from graphiti_core.llm_client import LLMConfig
    from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient

    class _Counting(OpenAIGenericClient):
        async def _generate_response(self, *args, **kwargs):
            policy.n_llm_calls += 1
            return await super()._generate_response(*args, **kwargs)

    return _Counting(config=LLMConfig(
        api_key="ollama", base_url=policy.ollama_url.rstrip("/") + "/v1",
        model=policy.llm_model, small_model=policy.llm_model, temperature=0.0))


def _hosted_llm(policy: "GraphitiPolicy"):
    """The same generic client pointed at a hosted OpenAI-compatible endpoint.

    Hosted endpoints generally accept ``json_object`` and not the
    ``json_schema`` response format the client defaults to, so the schema is
    injected into the prompt instead; the library does that itself in this
    mode. Same counting as the local client, so the two are priced alike.
    """
    from graphiti_core.llm_client import LLMConfig
    from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient

    class _Counting(OpenAIGenericClient):
        async def _generate_response(self, *args, **kwargs):
            policy.n_llm_calls += 1
            return await super()._generate_response(*args, **kwargs)

    return _Counting(config=LLMConfig(
        api_key=policy.hosted_api_key, base_url=policy.hosted_base_url,
        model=policy.llm_model, small_model=policy.llm_model, temperature=0.0),
        structured_output_mode="json_object")


# Mem0's default extraction prompt asks for facts about *the user*; a model that
# reads it literally returns no facts for "Peter Sagan plays for X." and the
# adjudication is never reached. This replacement keeps the output contract
# (a JSON object with a "facts" list) and widens the subject to any named
# entity, so the adjudication step can be compared across models.
THIRD_PERSON_EXTRACTION_PROMPT = (
    "You are a Fact Organizer. Extract the factual assertions in the input "
    "about any named person, organization, team, party, or place --- not only "
    "the user --- and return them as a JSON object of the form "
    '{"facts": ["...", "..."]}: one short, self-contained sentence per fact, '
    "keeping every name, role, and value exactly as written. Return "
    '{"facts": []} only when the input contains no factual assertion.')


class GraphitiPolicy(Policy):
    """Graphiti (``graphiti-core``, the engine under Zep) as a benchmark policy.

    ``observe`` hands the statement to ``Graphiti.add_episode`` as a plain-text
    episode dated by the statement's year, which runs the library's own entity
    and fact extraction, deduplication against the graph, and edge
    invalidation. ``rank`` asks the library's default hybrid edge search and
    maps each returned fact back to benchmark statement ids through the
    episodes Graphiti itself records as that fact's sources.

    Supersession here is *invalidation*, not deletion: a contradicted fact
    stays in the graph with ``invalid_at`` set and is still returned by search.
    So the store has two sizes, ``slots`` (every fact) and ``active_slots``
    (facts not yet invalidated), and every hit carries its validity window so
    the same contexts can be scored as returned or with expired facts removed.

    A fact re-asserted by a later statement is resolved to the existing edge
    and the new episode is appended to that edge's sources. The latest source
    therefore names the statement that last wrote the fact -- the convention
    Mem0's UPDATE is scored under -- while the first source names the writer
    whose text is actually stored; both are recorded, as for Mem0.
    """

    def __init__(self, llm_model: str = "qwen2.5:14b",
                 ollama_url: str = "http://localhost:11434",
                 embed_model: str = _MINILM, top_k: int = 20,
                 falkor_host: str = "localhost", falkor_port: int = 6379,
                 llm_backend: str = "ollama",
                 hosted_base_url: Optional[str] = None,
                 hosted_api_key: Optional[str] = None):
        self.llm_model = llm_model
        self.ollama_url = ollama_url
        # "ollama" (local, json_schema enforced by the server) or a hosted
        # OpenAI-compatible endpoint; recorded with the model in every result
        self.llm_backend = llm_backend
        self.hosted_base_url = hosted_base_url
        self.hosted_api_key = hosted_api_key
        self.embed_model = embed_model
        self.top_k = top_k
        self.falkor_host = falkor_host
        self.falkor_port = falkor_port
        # recorded in every result for the same reason Mem0's vector_store is
        self.graph_backend = "falkordb"
        self.n_add = 0
        self.n_llm_calls = 0
        self.events: Dict[str, int] = {}
        super().__init__(budget=None)
        self.name = f"graphiti({llm_model})"

    @property
    def lib_version(self) -> str:
        from importlib.metadata import version
        try:
            return version("graphiti-core")
        except Exception:
            return "unknown"

    @property
    def invalidates_on_add(self) -> bool:
        """Whether this version's ``add_episode`` can invalidate an existing fact.

        Followed through the installed source one hop at a time rather than
        asserted: ``add_episode`` must reach the edge-resolution step, that
        step must call the resolver, and the resolver must still ask the
        contradiction prompt, extract dates, and mark ``invalid_at``. A release
        that removes any hop answers False before any data is streamed.
        """
        import inspect
        try:
            from graphiti_core import graphiti as _g
            from graphiti_core.utils.maintenance import edge_operations as _e
            add = inspect.getsource(_g.Graphiti.add_episode)
            hop = inspect.getsource(_g.Graphiti._extract_and_resolve_edges)
            res = inspect.getsource(_e.resolve_extracted_edge)
            con = inspect.getsource(_e.resolve_edge_contradictions)
            return ("_extract_and_resolve_edges" in add
                    and "resolve_extracted_edges" in hop
                    and "contradicted_facts" in res
                    and "_extract_edge_timestamps" in res
                    and "resolve_edge_contradictions" in res
                    and "invalid_at" in con)
        except Exception:
            return False

    # -- lifecycle --------------------------------------------------------- #

    def reset(self) -> None:
        super().reset()
        self._drop_graph()
        self.n_add = 0
        self.n_silent_add = 0
        self.n_llm_calls = 0
        self.events = {}
        self.last_hits: List[Dict] = []
        self._sid_of: Dict[str, str] = {}       # episode uuid -> statement id
        self._seq_of: Dict[str, int] = {}       # episode uuid -> stream position
        self._edges_seen: set = set()
        self._invalid_seen: set = set()
        # the group id is also the FalkorDB graph name, so a fresh id is a
        # fresh graph and nothing from the previous config can be searched
        self._group = f"sb{uuid.uuid4().hex[:8]}"
        self._g = None                          # built lazily; see _graph()
        self._loop = None
        self._t0 = None

    def _drop_graph(self) -> None:
        g = getattr(self, "_g", None)
        if g is None:
            return
        try:
            self._run(g.driver.client.select_graph(self._group).delete())
            self._run(g.close())
        except Exception:
            pass
        try:
            self._loop.close()
        except Exception:
            pass
        self._g = None
        self._loop = None

    def close(self) -> None:
        self._drop_graph()

    def _run(self, coro):
        return self._loop.run_until_complete(coro)

    def _graph(self):
        if self._g is None:
            warnings.filterwarnings("ignore")
            # the constructor phones home unless told not to
            os.environ.setdefault("GRAPHITI_TELEMETRY_ENABLED", "false")
            from graphiti_core import Graphiti
            from graphiti_core.driver.falkordb_driver import FalkorDriver
            self._loop = asyncio.new_event_loop()
            # bound to this run's graph outright: add_episode clones the
            # driver whenever the group id differs from the database name, and
            # a clone made inside the event loop schedules its own index build
            # to race the first search
            driver = FalkorDriver(host=self.falkor_host, port=self.falkor_port,
                                  database=self._group)
            self._g = Graphiti(graph_driver=driver,
                               llm_client=(_hosted_llm(self) if self.llm_backend != "ollama"
                                           else _ollama_llm(self)),
                               embedder=_minilm_embedder(self.embed_model),
                               cross_encoder=_no_reranker())
            self._run(self._g.build_indices_and_constraints())
        return self._g

    def _bump(self, ev: str, n: int = 1) -> None:
        if n:
            self.events[ev] = self.events.get(ev, 0) + n

    def _counts(self) -> Dict[str, int]:
        """Facts, invalidated facts and entities in this run's graph, from the graph."""
        g = self._graph()
        recs, _, _ = self._run(g.driver.execute_query(
            "MATCH ()-[e:RELATES_TO]->() WHERE e.group_id = $g "
            "RETURN count(e) AS edges, count(e.invalid_at) AS invalid",
            g=self._group))
        nrec, _, _ = self._run(g.driver.execute_query(
            "MATCH (n:Entity) WHERE n.group_id = $g RETURN count(n) AS nodes",
            g=self._group))
        e = recs[0] if recs else {}
        n = nrec[0] if nrec else {}
        return {"edges": int(e.get("edges") or 0),
                "invalid": int(e.get("invalid") or 0),
                "nodes": int(n.get("nodes") or 0)}

    # -- Policy interface -------------------------------------------------- #

    def observe(self, emb: np.ndarray, item_id: str, t: float,
                text: Optional[str] = None) -> None:
        from graphiti_core.nodes import EpisodeType
        g = self._graph()
        if self._t0 is None:
            self._t0 = time.time()
        before = self._counts()
        seq = self.n_add
        self.n_add += 1
        res = None
        # a local model occasionally answers an extraction prompt with JSON the
        # library's schema rejects; the library already retries the parse, and
        # this retries the statement, because one bad answer must not decide a
        # 329-statement run. A statement that fails three times is recorded
        # the way Mem0's unreported adds are: counted, never invented.
        for attempt in range(3):
            try:
                res = self._run(g.add_episode(
                    name=item_id, episode_body=text or "",
                    source=EpisodeType.text,
                    source_description="statement",
                    reference_time=statement_time(t), group_id=self._group))
                break
            except Exception as e:
                print(f"  [graphiti] add_episode failed ({attempt + 1}/3) for "
                      f"{item_id}: {str(e)[:200]!r}", flush=True)
        if res is None:
            self.n_silent_add += 1
            self._bump("ADD_FAILED")
            return
        self._sid_of[res.episode.uuid] = item_id
        self._seq_of[res.episode.uuid] = seq
        after = self._counts()
        # the graph's own accounting, diffed around the call, is the event
        # histogram: what the library returns is resolved plus invalidated
        # edges in one list, and does not say which existed before
        self._bump("EDGE_ADD", after["edges"] - before["edges"])
        self._bump("EDGE_INVALIDATE", after["invalid"] - before["invalid"])
        self._bump("NODE_ADD", after["nodes"] - before["nodes"])
        for e in res.edges:
            already = e.uuid in self._edges_seen
            if already and (e.invalid_at is None or e.uuid in self._invalid_seen):
                # an existing fact re-asserted: resolved to its edge unchanged
                self._bump("EDGE_DUPLICATE")
            self._edges_seen.add(e.uuid)
            if e.invalid_at is not None:
                self._invalid_seen.add(e.uuid)
        if not res.edges:
            # stored as an episode, but no fact came out of it: nothing an
            # edge search can ever return for this statement
            self._bump("EPISODE_NO_EDGES")
        if self.n_add % 25 == 0:
            el = time.time() - self._t0
            print(f"  [graphiti] {self.n_add} statements  {el / self.n_add:.1f}s"
                  f"/stmt  {self.n_llm_calls} llm calls  slots {after['edges']}"
                  f" active {after['edges'] - after['invalid']}  {self.events}",
                  flush=True)

    @property
    def slots(self) -> int:
        try:
            return self._counts()["edges"]
        except Exception:
            return len(self._edges_seen)

    @property
    def active_slots(self) -> int:
        try:
            c = self._counts()
            return c["edges"] - c["invalid"]
        except Exception:
            return len(self._edges_seen - self._invalid_seen)

    def written(self) -> List[str]:
        """Statement ids that back at least one fact in the graph, in stream order.

        The write-side half of the gate: a statement the library accepted but
        extracted nothing from is stored as an episode and unreachable by an
        edge search, which no retrieval metric distinguishes from a statement
        that was never given to it.
        """
        g = self._graph()
        recs, _, _ = self._run(g.driver.execute_query(
            "MATCH ()-[e:RELATES_TO]->() WHERE e.group_id = $g RETURN e.episodes AS eps",
            g=self._group))
        backed = {u for r in recs for u in (r.get("eps") or [])}
        sids = {self._sid_of[u] for u in backed if u in self._sid_of}
        order = sorted(self._sid_of.items(), key=lambda kv: self._seq_of[kv[0]])
        return [sid for _, sid in order if sid in sids]

    def hits(self, query_text: Optional[str] = None) -> List[Dict]:
        """The facts the system surfaces, in its own rank order.

        Search returns invalidated facts too; whether a deployment shows them
        is its own choice, so each hit carries the validity window and the
        scorer decides. ``sid`` follows the fact's latest source episode and
        ``meta_sid`` its first, for the same reason both are kept for Mem0.
        """
        g = self._graph()
        edges = self._run(g.search(query_text or "", group_ids=[self._group],
                                   num_results=self.top_k))
        out: List[Dict] = []
        for e in edges:
            known = sorted((self._seq_of[u], self._sid_of[u])
                           for u in (e.episodes or []) if u in self._sid_of)
            out.append({"id": e.uuid,
                        "sid": known[-1][1] if known else None,
                        "meta_sid": known[0][1] if known else None,
                        "text": e.fact or "",
                        "score": None,
                        "valid_at": _iso(e.valid_at),
                        "invalid_at": _iso(e.invalid_at),
                        "expired_at": _iso(e.expired_at),
                        "created_at": _iso(e.created_at)})
        self.last_hits = out
        return out

    def rank(self, query: np.ndarray,
             query_text: Optional[str] = None) -> List[str]:
        out: List[str] = []
        for h in self.hits(query_text):
            if h["sid"] and h["sid"] not in out:
                out.append(h["sid"])
        return out
