"""Language-model clients for the reader and LLM-keyer tracks.

Three backends behind one interface:

* :class:`OllamaClient` --- a locally served open-weights model (no key, no
  cost, reproducible given the pinned tag).
* :class:`ZaiClient` --- a hosted large model (GLM) via Z.ai (``ZAI_API_KEY``).
* :class:`OpenRouterClient` --- any hosted model via the OpenRouter
  OpenAI-compatible endpoint (``OPENROUTER_API_KEY``).

All decode greedily (``temperature=0``) and cache every completion on disk,
keyed by a hash of (model, prompt, max_tokens). Caching is what makes the
reader track *reproducible* rather than merely repeatable: a rerun of the
paper's numbers replays the cache and issues no calls, and the cache ships
with the release so a reviewer can regenerate every reader number offline.

Usage cost is accumulated in ``self.usd`` from the provider's own accounting.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import threading
import time
import urllib.error
import urllib.request
from typing import Dict, List, Optional

_CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "llm_cache")

# Same-fact prompt for the LLM keyer rung. Deliberately austere: the model gets
# statement text only -- never a fact key, an id, or a timestamp.
_SAME_FACT = """You maintain an agent's memory store. A new statement has arrived.

New statement:
{new}

Existing memories:
{cands}

Which existing memory, if any, describes THE SAME FACT as the new statement --
same subject and same relation, so the new statement supersedes it? Two
statements about different subjects are never the same fact, even if they use
the same wording. A statement giving a different value for the same subject and
relation IS the same fact (it is an update).

Reply with the number of that memory, or 0 if none of them is the same fact.
Reply with the number only."""


class _BaseClient:
    name = "llm"
    max_workers = 8          # provider-appropriate request concurrency

    def __init__(self, cache_dir: str = _CACHE_DIR):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.n_calls = 0
        self.n_cached = 0
        self.usd = 0.0
        self._lock = threading.Lock()

    # -- caching ----------------------------------------------------------- #

    def _path(self, prompt: str, max_tokens: int) -> str:
        h = hashlib.md5(f"{self.model}\x1f{max_tokens}\x1f{prompt}"
                        .encode()).hexdigest()
        return os.path.join(self.cache_dir, f"{h}.json")

    def generate(self, prompt: str, max_tokens: int = 32) -> str:
        path = self._path(prompt, max_tokens)
        if not os.path.exists(path) and os.environ.get("STALEBENCH_LLM_CACHE_ONLY"):
            # a cache-only run must fail loudly rather than answer differently:
            # silently degrading an uncached judgment would change a published
            # number, and stalling on a busy server would be worse still
            raise RuntimeError(
                f"{self.model}: cache miss under STALEBENCH_LLM_CACHE_ONLY "
                f"({os.path.basename(path)})")
        if os.path.exists(path):
            with self._lock:
                self.n_cached += 1
            try:
                return json.load(open(path))["text"]
            except (json.JSONDecodeError, KeyError):
                os.remove(path)                       # corrupt entry, refetch
        text, usd = self._complete(prompt, max_tokens)
        tmp = f"{path}.{os.getpid()}.{threading.get_ident()}.tmp"
        with open(tmp, "w") as fh:
            json.dump({"model": self.model, "prompt": prompt, "text": text}, fh)
        os.replace(tmp, path)                         # atomic under concurrency
        with self._lock:
            self.n_calls += 1
            self.usd += usd
        return text

    def _complete(self, prompt: str, max_tokens: int):
        raise NotImplementedError

    def _post(self, url: str, body: Dict, headers: Dict,
              tries: Optional[int] = None, timeout: Optional[int] = None) -> Dict:
        # a hosted endpoint can hold a socket open without sending anything,
        # which outlives a generous timeout and stalls the whole run; keep it
        # tunable so a wedged provider costs one request, not an afternoon
        timeout = timeout or int(os.environ.get("STALEBENCH_LLM_TIMEOUT", "120"))
        # a local server that has wedged will not recover within one batch, and
        # 12 retries x 180s silently costs half an hour per bad request; the
        # caller records the failure instead
        tries = tries or int(os.environ.get("STALEBENCH_LLM_TRIES", "12"))
        last = None
        for i in range(tries):
            try:
                req = urllib.request.Request(
                    url, data=json.dumps(body).encode(),
                    headers={"Content-Type": "application/json", **headers})
                with urllib.request.urlopen(req, timeout=timeout) as r:
                    return json.load(r)
            except urllib.error.HTTPError as e:
                last = e
                if e.code in (400, 401, 402, 403, 404, 422):
                    raise RuntimeError(f"{self.model}: HTTP {e.code} "
                                       f"({e.reason}) -- not retryable: "
                                       f"{e.read()[:200]!r}")
                hinted = float(e.headers.get("Retry-After") or 0) if e.headers else 0
                if e.code == 429 and not hinted:
                    # these endpoints rate-limit a short burst window, not a
                    # cooldown: exponential backoff here costs far more
                    # throughput than it recovers. Jitter de-synchronizes the
                    # worker pool, which is what actually clears the burst.
                    time.sleep(min(5.0, 0.3 * 2 ** i) * (0.5 + random.random()))
                else:
                    time.sleep(max(hinted, min(60.0, 2.0 ** i)))
            except (urllib.error.URLError, TimeoutError, OSError,
                    json.JSONDecodeError) as e:
                last = e
                time.sleep(min(30.0, 2.0 ** i))
        raise RuntimeError(f"{self.model}: {tries} attempts failed: {last}")

    # -- tasks ------------------------------------------------------------- #

    def same_fact(self, new: str, candidates: List[str]) -> Optional[int]:
        """1-based index of the candidate describing the same fact, or 0/None."""
        if not candidates:
            return 0
        cands = "\n".join(f"{i + 1}. {c}" for i, c in enumerate(candidates))
        out = self.generate(_SAME_FACT.format(new=new, cands=cands), max_tokens=8)
        digits = "".join(c for c in out.strip()[:6] if c.isdigit() or c == " ")
        tok = digits.split()
        try:
            return int(tok[0]) if tok else None
        except ValueError:
            return None


class OllamaClient(_BaseClient):
    """Local open-weights reader served by ollama (free, offline, pinned tag)."""

    def __init__(self, model: str = "qwen2.5:14b",
                 host: str = "http://localhost:11434", **kw):
        self.model = model
        self.host = host
        self.name = model
        super().__init__(**kw)

    def _complete(self, prompt: str, max_tokens: int):
        r = self._post(f"{self.host}/api/generate",
                       {"model": self.model, "prompt": prompt, "stream": False,
                        "options": {"temperature": 0.0, "seed": 0,
                                    "num_predict": max_tokens}},
                       headers={}, timeout=180)
        return r["response"].strip(), 0.0


class OpenRouterClient(_BaseClient):
    """Hosted frontier reader through OpenRouter's OpenAI-compatible endpoint."""

    def __init__(self, model: str = "anthropic/claude-sonnet-4.6",
                 api_key: Optional[str] = None, **kw):
        self.model = model
        self.name = model.split("/")[-1]
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        if not self.api_key:
            raise RuntimeError("OPENROUTER_API_KEY is not set")
        super().__init__(**kw)

    def _complete(self, prompt: str, max_tokens: int):
        r = self._post("https://openrouter.ai/api/v1/chat/completions",
                       {"model": self.model,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": max_tokens, "temperature": 0.0},
                       headers={"Authorization": f"Bearer {self.api_key}"})
        if "choices" not in r:
            raise RuntimeError(f"openrouter error: {str(r)[:300]}")
        usd = float((r.get("usage") or {}).get("cost") or 0.0)
        return (r["choices"][0]["message"]["content"] or "").strip(), usd


class ZaiClient(_BaseClient):
    """Z.ai GLM endpoint --- the hosted large-model reader.

    GLM-4.x emits reasoning tokens by default and would spend the whole answer
    budget thinking about a one-phrase reply, so thinking is disabled: the
    reader task is deliberately a short-answer task, and every model in the
    comparison must answer under the same budget.
    """

    max_workers = 1          # this account admits ~1 in-flight request;
                             # more workers merely trade throughput for 429s

    def __init__(self, model: str = "glm-4.6", api_key: Optional[str] = None, **kw):
        self.model = model
        self.name = model
        self.api_key = api_key or os.environ.get("ZAI_API_KEY", "")
        if not self.api_key:
            raise RuntimeError("ZAI_API_KEY is not set")
        super().__init__(**kw)

    def _complete(self, prompt: str, max_tokens: int):
        r = self._post("https://api.z.ai/api/paas/v4/chat/completions",
                       {"model": self.model,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": max_tokens, "temperature": 0.0,
                        "thinking": {"type": "disabled"}},
                       headers={"Authorization": f"Bearer {self.api_key}"})
        if "choices" not in r:
            raise RuntimeError(f"z.ai error: {str(r)[:300]}")
        return (r["choices"][0]["message"].get("content") or "").strip(), 0.0


def make_client(spec: str) -> _BaseClient:
    """``ollama:qwen2.5:14b``, ``zai:glm-4.6``, ``openrouter:<vendor>/<model>``."""
    backend, _, model = spec.partition(":")
    if backend == "ollama":
        return OllamaClient(model=model)
    if backend == "openrouter":
        return OpenRouterClient(model=model)
    if backend == "zai":
        return ZaiClient(model=model)
    raise ValueError(f"unknown backend {backend!r}")
