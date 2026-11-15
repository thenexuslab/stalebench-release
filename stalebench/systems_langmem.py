"""LangMem (LangChain's long-term memory) as a StaleBench policy.

Third deployed system, same family of mechanism as Mem0: a model reads the
incoming statement beside the memories the store already holds that resemble
it, and returns the memories to insert, patch or remove. LangMem does it in
one structured-output call (``trustcall``: JSON patches against existing
memories), so a statement costs one model call, and the library ships with
inserts and updates on and deletes off --- the "as installed" configuration
this wrapper runs by default.

Attribution is the same convention as for Mem0. The manager returns the
memories it touched, and the store is diffed after every write: a key that
appears is an INSERT by this statement, a key whose content changes is an
UPDATE by it, a key that disappears a DELETE. A memory keeps the id of the
statement that last wrote it (``sid``) and of the one that created it
(``meta_sid``), and hits carry their text, because the library rewrites what
it stores and adds memories of its own ("the user is likely interested in
cycling") that no source statement wrote.

The store is LangGraph's in-memory store with the benchmark's MiniLM encoder
as its index, so retrieval differs from the dense baselines only in what the
system chose to keep.
"""

from __future__ import annotations

import os
import uuid
from typing import Dict, List, Optional

import numpy as np

from stalebench.policies import Policy

_MINILM = "sentence-transformers/all-MiniLM-L6-v2"
_ZAI_URL = "https://api.z.ai/api/paas/v4"


def _content_text(value) -> str:
    """The memory's text, whatever shape the library stored it in."""
    v = value
    if isinstance(v, dict):
        v = v.get("content", v)
    if isinstance(v, dict):
        v = v.get("content", v)
    return v if isinstance(v, str) else str(v)


class LangMemPolicy(Policy):
    def __init__(self, llm_model: str = "qwen2.5:14b", llm_backend: str = "ollama",
                 embed_model: str = _MINILM, top_k: int = 20,
                 enable_deletes: bool = False,
                 hosted_base_url: Optional[str] = None,
                 hosted_api_key: Optional[str] = None):
        self.llm_model = llm_model
        self.llm_backend = llm_backend
        self.embed_model = embed_model
        self.top_k = top_k
        # the library's default; switched on only as a labelled variant
        self.enable_deletes = enable_deletes
        self.hosted_base_url = hosted_base_url or _ZAI_URL
        self.hosted_api_key = hosted_api_key or os.environ.get("ZAI_API_KEY", "")
        self.n_add = 0
        self.n_llm_calls = 0
        self.events: Dict[str, int] = {}
        self._encoder = None
        super().__init__(budget=None)
        self.name = f"langmem({llm_model})" + ("|deletes" if enable_deletes else "")

    # -- provenance ---------------------------------------------------------- #

    @property
    def lib_version(self) -> str:
        from importlib.metadata import version
        try:
            return version("langmem")
        except Exception:
            return "unknown"

    @property
    def updates_on_add(self) -> bool:
        """Whether the manager this wrapper builds can patch an existing memory."""
        return True

    @property
    def deletes_on_add(self) -> bool:
        return self.enable_deletes

    # -- lifecycle ----------------------------------------------------------- #

    def reset(self) -> None:
        super().reset()
        self.n_add = 0
        self.n_silent_add = 0
        self.n_llm_calls = 0
        self.events = {}
        self.last_hits: List[Dict] = []
        self._sid_of: Dict[str, str] = {}
        self._first_sid: Dict[str, str] = {}
        self._prev: Dict[str, str] = {}
        self._user = f"sb{uuid.uuid4().hex[:8]}"
        self._ns = ("memories", self._user)
        self._store = None
        self._manager = None

    def _embed(self, texts):
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer
            self._encoder = SentenceTransformer(self.embed_model)
        return self._encoder.encode(list(texts), normalize_embeddings=True).tolist()

    def _chat_model(self):
        if self.llm_backend == "ollama":
            return f"ollama:{self.llm_model}"
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=self.llm_model, base_url=self.hosted_base_url,
                          api_key=self.hosted_api_key, temperature=0.0)

    def _build(self):
        if self._manager is None:
            from langgraph.store.memory import InMemoryStore
            from langmem import create_memory_store_manager
            self._store = InMemoryStore(index={"dims": 384, "embed": self._embed})
            self._manager = create_memory_store_manager(
                self._chat_model(), store=self._store,
                namespace=("memories", "{langgraph_user_id}"),
                enable_deletes=self.enable_deletes)
            from langchain_core.callbacks import BaseCallbackHandler
            policy = self

            class _Count(BaseCallbackHandler):
                def on_chat_model_start(self, *a, **k):
                    policy.n_llm_calls += 1
            self._config = {"configurable": {"langgraph_user_id": self._user},
                            "callbacks": [_Count()]}
        return self._manager

    def _snapshot(self) -> Dict[str, str]:
        items = self._store.search(self._ns, limit=1_000_000)
        return {it.key: _content_text(it.value) for it in items}

    # -- Policy interface ---------------------------------------------------- #

    def observe(self, emb: np.ndarray, item_id: str, t: float,
                text: Optional[str] = None) -> None:
        manager = self._build()
        self.n_add += 1
        err = None
        for attempt in range(3):
            try:
                manager.invoke({"messages": [{"role": "user", "content": text or ""}]},
                               config=self._config)
                err = None
                break
            except Exception as e:                       # noqa: BLE001
                err = e
        if err is not None:
            self.events["ADD_FAILED"] = self.events.get("ADD_FAILED", 0) + 1
            self.n_silent_add += 1
            # a run that fails every statement must say why, not read as an
            # empty store; the first message is kept and printed once
            if not getattr(self, "last_error", None):
                self.last_error = f"{type(err).__name__}: {str(err)[:300]}"
                print(f"  [langmem] add failed for {item_id}: {self.last_error}",
                      flush=True)
            return
        now = self._snapshot()
        changed = False
        for key, content in now.items():
            if key not in self._prev:
                self.events["INSERT"] = self.events.get("INSERT", 0) + 1
                self._sid_of[key] = item_id
                self._first_sid[key] = item_id
                changed = True
            elif content != self._prev[key]:
                self.events["UPDATE"] = self.events.get("UPDATE", 0) + 1
                self._sid_of[key] = item_id
                changed = True
        for key in self._prev:
            if key not in now:
                self.events["DELETE"] = self.events.get("DELETE", 0) + 1
                changed = True
        if not changed:
            # the model looked and wrote nothing: the analogue of Mem0's
            # empty result list, counted so it cannot pass for retention
            self.events["NOOP"] = self.events.get("NOOP", 0) + 1
            self.n_silent_add += 1
        self._prev = now

    @property
    def slots(self) -> int:
        if self._store is None:
            return 0
        return len(self._store.search(self._ns, limit=1_000_000))

    def hits(self, query_text: Optional[str] = None) -> List[Dict]:
        self._build()
        out: List[Dict] = []
        for it in self._store.search(self._ns, query=query_text or "", limit=self.top_k):
            out.append({"id": it.key, "sid": self._sid_of.get(it.key),
                        "meta_sid": self._first_sid.get(it.key),
                        "text": _content_text(it.value), "score": it.score,
                        "created_at": str(it.created_at), "updated_at": str(it.updated_at)})
        self.last_hits = out
        return out

    def rank(self, query: np.ndarray,
             query_text: Optional[str] = None) -> List[str]:
        out: List[str] = []
        for h in self.hits(query_text):
            if h["sid"] and h["sid"] not in out:
                out.append(h["sid"])
        return out
