"""Controlled stale-fact environment with value-agnostic queries.

Each entity has a value that updates over time. An event's embedding mixes a
stable **key** direction (the entity) with a **value** direction (the current
value); ``value_gap`` controls how far the value moves the embedding. Queries are
**value-agnostic**: they point at the entity's key only, so a similarity
retriever cannot distinguish the current value from stale ones by the query ---
which is exactly the real failure mode.

Ground truth is generator-defined and independent of any retrieval score: the
current version of each entity at query time. A retrieved item is *stale* if it
is the right entity but an older version.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class StaleStream:
    events: List[Tuple[np.ndarray, int, int, int]]   # (emb, entity, version, t)
    queries: List[Tuple[np.ndarray, int, int]]        # (query_emb, entity, current_v)
    n_entities: int
    dim: int


def make_stream(n_entities: int = 12, n_steps: int = 300, dim: int = 64,
                update_p: float = 0.15, value_gap: float = 0.4,
                query_noise: float = 0.05, key_jitter: float = 0.0,
                seed: int = 0) -> StaleStream:
    """Generate one stale-fact stream.

    Args:
        n_entities: number of distinct entities (keys).
        n_steps: length of the event stream.
        dim: embedding dimension.
        update_p: per-step probability the sampled entity's value updates.
        value_gap: how far a value change moves the embedding (0 = invisible
            update, large = a new value looks like a different entity).
        query_noise: noise on the value-agnostic query embedding.
        key_jitter: per-event perturbation of the key direction — the
            controlled analog of *paraphrase*: the same fact re-verbalized no
            longer has an identical key component, so similarity-based same-key
            detection must survive surface variation. 0 = every mention of an
            entity shares an exact key direction (the easy regime).
        seed: RNG seed.
    """
    rng = np.random.default_rng(seed)
    keys = rng.standard_normal((n_entities, dim))
    keys /= np.linalg.norm(keys, axis=1, keepdims=True)
    ver = {e: 0 for e in range(n_entities)}
    valvec: Dict[Tuple[int, int], np.ndarray] = {}

    def val(e, v):
        if (e, v) not in valvec:
            x = rng.standard_normal(dim); valvec[(e, v)] = x / np.linalg.norm(x)
        return valvec[(e, v)]

    events = []
    for t in range(n_steps):
        e = int(rng.integers(0, n_entities))
        if rng.random() < update_p:
            ver[e] += 1
        k = keys[e]
        if key_jitter > 0:
            k = k + key_jitter * rng.standard_normal(dim)
            k = k / np.linalg.norm(k)
        x = k + value_gap * val(e, ver[e])
        x = (x / np.linalg.norm(x)).astype(np.float32)
        events.append((x, e, ver[e], t))

    queries = []
    for e in range(n_entities):
        q = keys[e] + query_noise * rng.standard_normal(dim)   # names the KEY only
        q = (q / np.linalg.norm(q)).astype(np.float32)
        queries.append((q, e, ver[e]))
    return StaleStream(events, queries, n_entities, dim)
