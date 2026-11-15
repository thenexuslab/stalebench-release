"""Run the StaleBench controlled diagnostic and print the regime map.

The controlled generator is a *diagnostic*, not the headline (that is the
TempLAMA track, ``run_real.py``): it maps exactly when staleness bites and when
each fix holds, on knobs real data cannot turn independently.

Sweeps:
  (1) value_gap   — how far an update moves the embedding
  (2) key_jitter  — surface variation of the same fact (paraphrase analog)
  (3) tau         — the false-merge / false-split trade-off of reconsolidation
  (4) budget      — memory footprint under eviction pressure

Run:  cd stalebench && python -m stalebench.run
"""

from __future__ import annotations

import json
import os
from typing import Callable, Dict, List, Tuple

import numpy as np

from stalebench.env import make_stream
from stalebench.metrics import aggregate, score_stream
from stalebench.policies import (CosineRAG, KeepLatestOracle, Recency,
                                 Reconsolidation, RecencyWeighted, TimeRerank)

_HERE = os.path.dirname(__file__)
_OUT = os.path.join(_HERE, "..", "results", "stalebench_results.json")

N_SEEDS = 20
N_TUNE = 10
_VAL_SEEDS = range(1000, 1000 + N_TUNE)     # disjoint from the reported seeds

_K_GRID = [2, 3, 5, 10, 20]
_LAM_GRID = [0.1, 0.3, 0.6, 1.0]
_DECAY_GRID = [0.8, 0.9, 0.99]
_TAU_GRID = [0.1, 0.2, 0.3, 0.45, 0.6, 0.75, 0.9]


def _val_c1(fac: Callable, **stream_kw) -> float:
    return float(np.mean([score_stream(fac(), make_stream(seed=s, **stream_kw))
                          ["current@1"] for s in _VAL_SEEDS]))


def tune(budget=None, **stream_kw) -> Dict:
    """Grid-search each tunable policy on held-out generator seeds.

    The paper's guardrail is that every policy is re-tuned in every regime, so
    that a store-all fix is never made to look bad by a setting chosen
    elsewhere. An earlier revision hard-coded k=5, lam=0.5, tau=0.6 across the
    whole controlled track, which understated time-reranking badly at low
    ``value_gap`` (identical repeated mentions fill its top-k) and understated
    threshold keying under jitter (the useful thresholds are below 0.3).
    """
    def c1(fac):
        return _val_c1(fac, **stream_kw)
    k = max(_K_GRID, key=lambda k: c1(lambda: TimeRerank(k=k, budget=budget)))
    lam, decay = max(((l, d) for l in _LAM_GRID for d in _DECAY_GRID),
                     key=lambda ld: c1(lambda: RecencyWeighted(
                         lam=ld[0], decay=ld[1], budget=budget)))
    tau = max(_TAU_GRID,
              key=lambda t: c1(lambda: Reconsolidation(tau=t, budget=budget)))
    return {"k": k, "lam": lam, "decay": decay, "tau": tau}


def _core(tuned: Dict, budget=None) -> List[Tuple[str, Callable]]:
    return [
        ("cosine-rag", lambda: CosineRAG()),
        ("time-rerank", lambda: TimeRerank(k=tuned["k"], budget=budget)),
        ("recency-weighted", lambda: RecencyWeighted(
            lam=tuned["lam"], decay=tuned["decay"], budget=budget)),
        ("reconsolidation", lambda: Reconsolidation(tau=tuned["tau"],
                                                    budget=budget)),
        ("keep-latest-oracle", lambda: KeepLatestOracle()),
    ]


def _cell(policies, n_seeds: int = N_SEEDS, **stream_kw) -> Dict:
    out = {}
    for name, fac in policies:
        rows = [score_stream(fac(), make_stream(seed=s, **stream_kw))
                for s in range(n_seeds)]
        out[name] = aggregate(rows)
    return out


def run(n_seeds: int = N_SEEDS) -> dict:
    out: Dict = {"n_seeds": n_seeds, "n_tune_seeds": N_TUNE, "tuned": {}}

    out["value_gap_sweep"] = {}
    for g in [0.3, 0.5, 0.8]:
        tuned = tune(value_gap=g)
        out["tuned"][f"gap={g}"] = tuned
        out["value_gap_sweep"][f"gap={g}"] = _cell(_core(tuned), n_seeds,
                                                   value_gap=g)

    # the v0.2 operating point, kept as a comparison arm: it is what makes the
    # difference between "the knob breaks threshold keying" and "the knob moves
    # its operating point" visible rather than asserted
    out["value_gap_sweep_fixed_tau"] = {
        f"gap={g}": _cell([("reconsolidation", lambda: Reconsolidation(tau=0.6))],
                          n_seeds, value_gap=g)["reconsolidation"]
        for g in [0.3, 0.5, 0.8]}

    out["key_jitter_sweep"] = {}
    for j in [0.0, 0.2, 0.4, 0.8]:
        tuned = tune(value_gap=0.4, key_jitter=j)
        out["tuned"][f"jitter={j}"] = tuned
        out["key_jitter_sweep"][f"jitter={j}"] = _cell(
            _core(tuned), n_seeds, value_gap=0.4, key_jitter=j)

    # tau sweep: two regimes (clean, paraphrased), grid extended below 0.3
    # because the tuned optimum under jitter lives there
    out["tau_sweep"] = {}
    for regime, kw in [("clean", dict(value_gap=0.4)),
                       ("jitter=0.4", dict(value_gap=0.4, key_jitter=0.4))]:
        out["tau_sweep"][regime] = {
            f"tau={t}": _cell([("reconsolidation",
                                lambda t=t: Reconsolidation(tau=t))],
                              n_seeds, **kw)["reconsolidation"]
            for t in _TAU_GRID}

    out["budget_sweep"] = {}
    for B in [12, 20, 40]:
        tuned = tune(budget=B, value_gap=0.4)
        out["tuned"][f"B={B}"] = tuned
        out["budget_sweep"][f"B={B}"] = _cell([
            ("recency", lambda B=B: Recency(budget=B)),
            ("time-rerank", lambda B=B, t=tuned: TimeRerank(k=t["k"], budget=B)),
            ("reconsolidation", lambda B=B, t=tuned: Reconsolidation(
                tau=t["tau"], budget=B)),
            ("keep-latest-oracle", lambda: KeepLatestOracle()),
        ], n_seeds, value_gap=0.4)
    return out


def _f(c: Dict) -> str:
    return f"{c['current@1']['mean']:.2f}/{c['stale_rate']['mean']:.2f}"


def main():
    res = run()
    os.makedirs(os.path.dirname(_OUT), exist_ok=True)
    json.dump(res, open(_OUT, "w"), indent=1)

    print("=" * 72)
    print(f"StaleBench (controlled diagnostic): current@1 / stale-rate, "
          f"{res['n_seeds']} seeds")
    print("=" * 72)

    for title, key, cols in [
        ("value-gap sweep (full store)", "value_gap_sweep",
         ["gap=0.3", "gap=0.5", "gap=0.8"]),
        ("key-jitter sweep (gap=0.4) — paraphrase analog", "key_jitter_sweep",
         ["jitter=0.0", "jitter=0.2", "jitter=0.4", "jitter=0.8"]),
    ]:
        print(f"\n[{title}]")
        print(f"{'policy':20s}  " + "  ".join(f"{c:11s}" for c in cols))
        for name in ["cosine-rag", "time-rerank", "recency-weighted",
                     "reconsolidation", "keep-latest-oracle"]:
            cells = [_f(res[key][c][name]) for c in cols]
            print(f"{name:20s}  " + "  ".join(f"{c:11s}" for c in cells))

    print("\n[tau sweep, reconsolidation]  current@1 (clean | jitter=0.4)")
    taus = [f"tau={t}" for t in _TAU_GRID]
    for t in taus:
        a = res["tau_sweep"]["clean"][t]["current@1"]["mean"]
        b = res["tau_sweep"]["jitter=0.4"][t]["current@1"]["mean"]
        print(f"{t:9s}  {a:.3f} | {b:.3f}")

    print("\n[budget sweep, gap=0.4]  current@1")
    print(f"{'policy':20s}  " + "  ".join(f"B={b}" for b in [12, 20, 40]))
    for name in ["recency", "time-rerank", "reconsolidation", "keep-latest-oracle"]:
        cells = [f"{res['budget_sweep'][f'B={b}'][name]['current@1']['mean']:.3f}"
                 for b in [12, 20, 40]]
        print(f"{name:20s}  " + "  ".join(cells))
    print("\n[tuned per regime, on held-out seeds]")
    for regime, t in res["tuned"].items():
        print(f"  {regime:12s} k={t['k']:<3d} lam={t['lam']} decay={t['decay']} "
              f"tau={t['tau']}")
    print(f"\nWrote {os.path.abspath(_OUT)}")


if __name__ == "__main__":
    main()
