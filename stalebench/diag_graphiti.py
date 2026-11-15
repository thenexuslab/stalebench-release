"""Score an invalidating system's contexts by three read paths.

Graphiti marks a superseded edge ``invalid_at`` on the write path and then, on
the read path its library ships, returns invalidated edges beside valid ones
and breaks ties among versions of one fact by insertion order --- oldest
first. So the number a deployment gets ("as returned") and the number the
mechanism would earn if the read path honoured its own flags are different
numbers, and the difference is a property of the system rather than of our
bookkeeping. This module computes both from the contexts sidecar
:mod:`stalebench.run_systems` writes, with no second ingestion:

* ``as_returned`` --- the system's own ranking, what a deployment gets;
* ``valid_first`` --- the same hits with non-invalidated edges ranked first,
  which is the read path of the paper's temporal-KG rung;
* ``valid_only`` --- invalidated edges dropped, the retention reading.

Each arm is scored by id and by text exactly as the systems track scores the
system, and the arms are paired by query for the sign-flip test.

Run:  python -m stalebench.diag_graphiti results/<contexts.json>
"""

from __future__ import annotations

import json
import os
import sys
from typing import Dict, List, Sequence

import numpy as np

from stalebench.metrics import (bootstrap_ci, holm, paired_permutation,
                                score_ranking, score_texts)
from stalebench.reader import stale_values_of
from stalebench.templama import build_instance, load_facts, split_facts

_HERE = os.path.dirname(__file__)
_OUT = os.environ.get("STALEBENCH_GRAPHITI_OUT") or os.path.join(
    _HERE, "..", "results", "stalebench_graphiti_readpath.json")
_KS = (5,)


def _is_valid(hit: Dict) -> bool:
    return hit.get("invalid_at") is None


def arms_of(hits: Sequence[Dict]) -> Dict[str, List[Dict]]:
    return {
        "as_returned": list(hits),
        # stable: within each tier the system's own order is kept
        "valid_first": sorted(hits, key=lambda h: 0 if _is_valid(h) else 1),
        "valid_only": [h for h in hits if _is_valid(h)],
    }


def _score(hits: Sequence[Dict], key: str, cur, gold: str, stale) -> Dict:
    ranked: List[str] = []
    for h in hits:
        if h.get("sid") and h["sid"] not in ranked:
            ranked.append(h["sid"])
    rec = score_ranking(ranked, key, cur, _KS)
    rec.update(score_texts([h.get("text") or "" for h in hits], gold, stale, _KS))
    return rec


def run(ctx_path: str) -> Dict:
    ctx = json.load(open(ctx_path))
    cfg = ctx["config"]
    facts, _ = load_facts()
    _, test = split_facts(facts)
    test = test[:cfg["n_test_facts"]] if cfg.get("subsampled") else test
    stale_all = [stale_values_of(f.runs, f.current_value) for f in test]
    out: Dict = {"config": {"contexts": os.path.basename(ctx_path),
                            "source_config": cfg}, "by_config": {}}
    raw: Dict[str, float] = {}
    for name, cell in ctx["by_config"].items():
        inst = build_instance(test, paraphrase_p=cfg["paraphrase_p"], seed=0,
                              alias_p=1.0 if name == "anchor_free" else 0.0)
        cell_out: Dict = {}
        for pol, hits_per_query in cell.items():
            if not pol.startswith("graphiti"):
                continue
            recs: Dict[str, List[Dict]] = {}
            for hits, (_, key, cur), g, sv in zip(hits_per_query, inst.queries,
                                                  inst.gold, stale_all):
                for arm, ah in arms_of(hits).items():
                    recs.setdefault(arm, []).append(_score(ah, key, cur, g, sv))
            pol_out: Dict = {}
            for arm, rs in recs.items():
                pol_out[arm] = {
                    "summary": {m: float(np.mean([r[m] for r in rs])) for m in rs[0]},
                    "ci": {m: bootstrap_ci([r[m] for r in rs], seed=1)
                           for m in ("current@1", "current@1_text",
                                     "contamination@5", "contamination@5_text")},
                    "n": len(rs)}
                if arm != "as_returned":
                    for m in ("current@1", "current@1_text",
                              "contamination@5_text"):
                        raw[f"{name}:{pol}:{arm}-vs-as_returned:{m}"] = \
                            paired_permutation([r[m] for r in rs],
                                               [r[m] for r in recs["as_returned"]],
                                               seed=7)
            # how many queries the system's own order puts an invalidated edge
            # above a valid one for the queried fact: the read path ignoring
            # the write path, counted directly
            n_inv_first = sum(
                1 for hits in hits_per_query
                if hits and not _is_valid(hits[0])
                and any(_is_valid(h) for h in hits))
            pol_out["n_invalid_ranked_first"] = n_inv_first
            pol_out["n_queries"] = len(hits_per_query)
            cell_out[pol] = pol_out
        out["by_config"][name] = cell_out
    out["tests"] = {"raw": raw, "holm": holm(raw) if raw else {},
                    "family_size": len(raw)}
    return out


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    res = run(sys.argv[1])
    os.makedirs(os.path.dirname(_OUT), exist_ok=True)
    json.dump(res, open(_OUT, "w"), indent=1)
    for name, cell in res["by_config"].items():
        for pol, pol_out in cell.items():
            print(f"\n[{name}] {pol}  ({pol_out['n_invalid_ranked_first']} of "
                  f"{pol_out['n_queries']} queries rank an invalidated edge first)")
            for arm in ("as_returned", "valid_first", "valid_only"):
                s = pol_out[arm]["summary"]
                print(f"  {arm:12s} cur@1 {s['current@1']:.3f} / text "
                      f"{s['current@1_text']:.3f}   contam@5 "
                      f"{s['contamination@5']:.3f} / text "
                      f"{s['contamination@5_text']:.3f}   miss {s['miss_rate']:.3f}")
    print(f"\nWrote {os.path.abspath(_OUT)}")


if __name__ == "__main__":
    main()
