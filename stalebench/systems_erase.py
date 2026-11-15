"""ERASE (Li et al., 2025) as a StaleBench policy: rewrite-on-arrival.

The fourth deployed mechanism, and the one the paper names as not run.
Mem0 and LangMem adjudicate the *incoming* memory (ADD / UPDATE / DELETE);
Graphiti invalidates a contradicted edge and keeps it. ERASE edits the
*existing* entries: when a document arrives, the entries it is related to
are retrieved and a model decides, for each, whether it is still accurate
--- keep it, rewrite it to stay consistent, or delete it --- and the new
content is then added. That is correction applied to the store rather than
to the reader.

No public implementation is pinned to a release, so this is a
reimplementation of the loop from the paper, kept minimal on purpose: the
benchmark's MiniLM encoder retrieves the ``retrieve_k`` nearest stored facts
to the arriving statement, one greedy, cached model call returns a JSON
decision per fact, and the statement itself is inserted verbatim (each
statement already is one fact). A rewritten fact keeps its memory id and
takes the arriving statement as its latest writer, the convention every
other system in this track is scored under; its text is re-embedded so the
next arrival retrieves what is actually stored.

Parse failures are counted and treated as KEEP: a malformed decision must
not silently delete a fact, and must not pass for retention either.
"""

from __future__ import annotations

import json
import os
import re
from typing import Dict, List, Optional

import numpy as np

from stalebench.llm import make_client
from stalebench.policies import Policy

_MINILM = "sentence-transformers/all-MiniLM-L6-v2"

_PROMPT = """A memory store holds facts. A new statement has just arrived. For each stored fact below, decide whether it is still accurate now that the new statement is known.
- KEEP: the fact is unaffected by the new statement.
- DELETE: the new statement supersedes it entirely.
- REWRITE: part of it is worth keeping but it must change to stay consistent with the new statement; give the rewritten fact.
Only the new statement counts as evidence; do not use outside knowledge.

New statement: {new}

Stored facts:
{facts}

Answer with JSON only, one decision per stored fact, in this form:
{{"decisions": [{{"id": 1, "action": "KEEP"}}, {{"id": 2, "action": "REWRITE", "text": "..."}}, {{"id": 3, "action": "DELETE"}}]}}"""


def _strip_fences(s: str) -> str:
    s = s.strip()
    s = re.sub(r"^```(?:json)?\s*", "", s)
    s = re.sub(r"\s*```$", "", s)
    return s.strip()


class ErasePolicy(Policy):
    def __init__(self, llm_model: str = "qwen2.5:14b", llm_backend: str = "ollama",
                 embed_model: str = _MINILM, top_k: int = 20, retrieve_k: int = 5,
                 max_tokens: int = 512):
        self.llm_model = llm_model
        self.llm_backend = llm_backend
        self.embed_model = embed_model
        self.top_k = top_k
        # how many stored facts an arrival is checked against; ERASE
        # retrieves the entries related to the new document
        self.retrieve_k = retrieve_k
        self.max_tokens = max_tokens
        self.n_add = 0
        self.n_llm_calls = 0
        self.events: Dict[str, int] = {}
        self._encoder = None
        self._client = None
        super().__init__(budget=None)
        self.name = f"erase({llm_model})"

    # -- provenance ---------------------------------------------------------- #

    @property
    def lib_version(self) -> str:
        return "reimplemented from Li et al. (2025)"

    @property
    def rewrites_on_add(self) -> bool:
        return True

    # -- lifecycle ----------------------------------------------------------- #

    def reset(self) -> None:
        super().reset()
        self.n_add = 0
        self.n_silent_add = 0
        self.n_llm_calls = 0
        self.events = {}
        self.last_hits: List[Dict] = []
        self._mids: List[str] = []
        self._texts: List[str] = []
        self._embs: List[np.ndarray] = []
        self._sid: List[str] = []
        self._first: List[str] = []
        self._ts: List[float] = []
        self._alive: List[bool] = []
        self._next = 0

    def _embed(self, text: str) -> np.ndarray:
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer
            self._encoder = SentenceTransformer(self.embed_model)
        v = self._encoder.encode([text], normalize_embeddings=True)[0]
        return np.asarray(v, dtype=np.float32)

    def _llm(self):
        if self._client is None:
            self._client = make_client(f"{self.llm_backend}:{self.llm_model}")
        return self._client

    def _alive_idx(self) -> List[int]:
        return [i for i, a in enumerate(self._alive) if a]

    def _nearest(self, emb: np.ndarray, k: int) -> List[int]:
        idx = self._alive_idx()
        if not idx:
            return []
        m = np.stack([self._embs[i] for i in idx])
        sims = m @ emb
        order = np.argsort(-sims, kind="stable")[:k]
        return [idx[j] for j in order]

    # -- Policy interface ---------------------------------------------------- #

    def observe(self, emb: np.ndarray, item_id: str, t: float,
                text: Optional[str] = None) -> None:
        self.n_add += 1
        text = text or ""
        emb = np.asarray(emb, dtype=np.float32)
        cands = self._nearest(emb, self.retrieve_k)
        if cands:
            facts = "\n".join(f"{n + 1}. {self._texts[i]}" for n, i in enumerate(cands))
            prompt = _PROMPT.format(new=text, facts=facts)
            try:
                raw = self._llm().generate(prompt, self.max_tokens)
                self.n_llm_calls += 1
                decisions = json.loads(_strip_fences(raw)).get("decisions", [])
                by_id = {int(d.get("id", 0)): d for d in decisions if isinstance(d, dict)}
            except Exception:                            # noqa: BLE001
                self.events["PARSE_FAIL"] = self.events.get("PARSE_FAIL", 0) + 1
                by_id = {}
            for n, i in enumerate(cands):
                d = by_id.get(n + 1, {"action": "KEEP"})
                action = str(d.get("action", "KEEP")).upper()
                if action == "DELETE":
                    self._alive[i] = False
                elif action == "REWRITE" and d.get("text"):
                    self._texts[i] = str(d["text"])
                    self._embs[i] = self._embed(self._texts[i])
                    self._sid[i] = item_id
                    self._ts[i] = t
                else:
                    action = "KEEP"
                self.events[action] = self.events.get(action, 0) + 1
        self._mids.append(f"m{self._next}"); self._next += 1
        self._texts.append(text); self._embs.append(emb)
        self._sid.append(item_id); self._first.append(item_id)
        self._ts.append(t); self._alive.append(True)
        self.events["INSERT"] = self.events.get("INSERT", 0) + 1

    @property
    def slots(self) -> int:
        return len(self._alive_idx())

    def hits(self, query_text: Optional[str] = None) -> List[Dict]:
        q = self._embed(query_text or "")
        out: List[Dict] = []
        for i in self._nearest(q, self.top_k):
            out.append({"id": self._mids[i], "sid": self._sid[i], "meta_sid": self._first[i],
                        "text": self._texts[i], "score": float(self._embs[i] @ q),
                        "t": self._ts[i]})
        self.last_hits = out
        return out

    def rank(self, query: np.ndarray, query_text: Optional[str] = None) -> List[str]:
        out: List[str] = []
        for h in self.hits(query_text):
            if h["sid"] and h["sid"] not in out:
                out.append(h["sid"])
        return out
