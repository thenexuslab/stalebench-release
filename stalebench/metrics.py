"""StaleBench metrics: how fresh is the retrieved answer?

Items are identified by ``"key:version"``. Ground truth is a mapping
``current[key] -> set of versions`` whose stored *value* equals the value valid
at query time (a set, so a value that recurs — A, B, A — is scored by value,
not by emission index). Each query's ranking is scored as:

* **current@1**  — top-1 is the right key AND a current version.
* **stale_rate** — top-1 is the right key but a non-current version (the failure).
* **miss_rate**  — top-1 is the wrong key (or nothing retrieved).
* **current@k**  — any current version of the key in the top-k.
* **contamination@k** — top-k contains BOTH a current and a stale version of the
  queried key: the context handed to a reader argues with itself. Time-aware
  reranking cannot reduce this; retention policies drive it to ~0.
* **current_set_recall@k** — share of the key's *distinct* current values that
  appear in the top-k (needs an id → value lookup). Equals current@k for a
  single-valued key; for a key with several simultaneously-valid values it is
  the metric that charges a one-slot-per-fact policy for the still-valid
  values it discarded, which current@k, satisfied by any one of them, cannot.
* **slots** — store size after the stream (memory footprint).

``score_texts`` re-scores the same context by the *text* the store surfaced,
for systems that rewrite or merge what they were given and so cannot be scored
by id alone.

Statistics: ``bootstrap_ci`` (nonparametric CI over queries or seeds) and
``paired_permutation`` (sign-flip test on per-query paired differences), plus
Holm correction for the headline family of comparisons.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np


def _decode(item_id: str) -> Tuple[str, int]:
    k, v = item_id.rsplit(":", 1)
    return k, int(v)


# --------------------------------------------------------------------------- #
# Ranking-level scoring
# --------------------------------------------------------------------------- #


def score_ranking(ranked: Sequence[str], key: str, current_vers: Set[int],
                  ks: Iterable[int] = (5,),
                  value_of: Optional[Mapping[str, str]] = None) -> Dict[str, float]:
    """Score one query's ranking against the current-version set of its key.

    ``value_of`` (item id -> stored value) switches on ``current_set_recall@k``;
    without it the output is exactly the id-level decomposition and nothing
    else, so callers that never pass it are unaffected.
    """
    rec: Dict[str, float] = {}
    if not ranked:
        rec.update({"current@1": 0.0, "stale_rate": 0.0, "miss_rate": 1.0})
    else:
        k0, v0 = _decode(ranked[0])
        if k0 != key:
            rec.update({"current@1": 0.0, "stale_rate": 0.0, "miss_rate": 1.0})
        elif v0 in current_vers:
            rec.update({"current@1": 1.0, "stale_rate": 0.0, "miss_rate": 0.0})
        else:
            rec.update({"current@1": 0.0, "stale_rate": 1.0, "miss_rate": 0.0})
    # distinct by value, not by version: an A-B-A key has one value to recall,
    # and a store that surfaces both of its A statements is not credited twice
    want = ({value_of[f"{key}:{v}"] for v in current_vers}
            if value_of is not None else None)
    for k in ks:
        top = ranked[:k]
        has_cur = has_stale = False
        seen: Set[str] = set()
        for iid in top:
            ik, iv = _decode(iid)
            if ik != key:
                continue
            if iv in current_vers:
                has_cur = True
                if want is not None:
                    seen.add(value_of[iid])
            else:
                has_stale = True
        rec[f"current@{k}"] = float(has_cur)
        rec[f"contamination@{k}"] = float(has_cur and has_stale)
        if want is not None:
            rec[f"current_set_recall@{k}"] = len(seen) / len(want)
    return rec


def score_texts(texts: Sequence[str], gold: str,
                stale_values: Sequence[str],
                ks: Iterable[int] = (5,)) -> Dict[str, float]:
    """Score a retrieved context by *what it says*, not by which id it carries.

    Every other metric here identifies a retrieved item by ``key:version``,
    which presumes the store returns the statement it was given. A system that
    rewrites or merges memories breaks that presumption: an UPDATE that fuses
    "Sagan rides for Bora" into "Sagan rides for TotalEnergies (formerly Bora)"
    is attributed to its latest writer and therefore scores as current and
    uncontaminated, while the text a reader sees names both values. Scoring the
    surfaced text against the source-defined current and superseded values
    bounds that gap instead of disclosing it.

    Metric names mirror the id-level ones with a ``_text`` suffix, plus
    ``hedged@1_text`` --- top-1 names the current value *and* a superseded one
    --- which the id-level decomposition has no way to express.

    Uses the reader's answer matcher, so "the context names the stale value"
    and "the reader answered the stale value" are the same judgement applied at
    two points in the pipeline.
    """
    from stalebench.reader import _matches

    stale = [v for v in stale_values if v != gold]

    def says(t: str) -> Tuple[bool, bool]:
        return _matches(t, gold), any(_matches(t, v) for v in stale)

    rec: Dict[str, float] = {}
    if not texts:
        rec.update({"current@1_text": 0.0, "stale_rate_text": 0.0,
                    "hedged@1_text": 0.0, "miss_rate_text": 1.0})
    else:
        g0, s0 = says(texts[0])
        rec.update({"current@1_text": float(g0 and not s0),
                    "stale_rate_text": float(s0 and not g0),
                    "hedged@1_text": float(g0 and s0),
                    "miss_rate_text": float(not g0 and not s0)})
    for k in ks:
        flags = [says(t) for t in texts[:k]]
        has_cur = any(g for g, _ in flags)
        has_stale = any(s for _, s in flags)
        rec[f"current@{k}_text"] = float(has_cur)
        rec[f"contamination@{k}_text"] = float(has_cur and has_stale)
    return rec


def score_stream(policy, stream, ks: Iterable[int] = (5,)) -> Dict[str, float]:
    """Run a policy over a controlled StaleStream and return freshness metrics.

    Returns a flat dict of means over queries, plus ``slots`` and a ``records``
    list (one dict per query) for paired statistics.
    """
    policy.reset()
    for emb, e, v, t in stream.events:
        policy.observe(emb, f"{e}:{v}", t)
    records = [score_ranking(policy.rank(q), str(e), {cur_v}, ks)
               for q, e, cur_v in stream.queries]
    out = {m: float(np.mean([r[m] for r in records])) for m in records[0]}
    out["slots"] = float(policy.slots)
    out["records"] = records
    return out


# --------------------------------------------------------------------------- #
# Aggregation & statistics
# --------------------------------------------------------------------------- #


def bootstrap_ci(x: Sequence[float], n_boot: int = 10000, seed: int = 0,
                 alpha: float = 0.05) -> Dict[str, float]:
    """Nonparametric percentile bootstrap CI for the mean of ``x``."""
    x = np.asarray(x, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    means = x[idx].mean(axis=1)
    return {"mean": float(x.mean()),
            "lo": float(np.quantile(means, alpha / 2)),
            "hi": float(np.quantile(means, 1 - alpha / 2))}


def paired_permutation(a: Sequence[float], b: Sequence[float],
                       n_perm: int = 10000, seed: int = 0) -> float:
    """Two-sided sign-flip permutation test on paired differences a - b."""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    if np.allclose(d, 0):
        return 1.0
    rng = np.random.default_rng(seed)
    obs = abs(d.mean())
    signs = rng.choice([-1.0, 1.0], size=(n_perm, len(d)))
    null = np.abs((signs * d).mean(axis=1))
    return float((np.sum(null >= obs) + 1) / (n_perm + 1))


def holm(pvals: Dict[str, float]) -> Dict[str, float]:
    """Holm–Bonferroni step-down correction over a family of p-values."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    out: Dict[str, float] = {}
    running = 0.0
    for i, (name, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        out[name] = running
    return out


def aggregate(rows: List[Dict[str, float]], seed: int = 0) -> Dict[str, Dict[str, float]]:
    """Mean + bootstrap 95% CI per metric over rows (e.g. per-seed results)."""
    keys = [k for k in rows[0] if k != "records"]
    return {k: bootstrap_ci([r[k] for r in rows], seed=seed) for k in keys}
