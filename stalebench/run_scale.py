"""v1.1 scale track: do the findings hold on 6x the facts?

v1.0 queries the 892 TempLAMA keys that are single-valued in every year, because
set-valued gold complicates crisp top-1 scoring. That restriction invites the
most common objection a benchmark gets --- *your n is small and your slice is
unrepresentative* --- and it is answerable rather than arguable: keep every
changing key, score a top-1 as current when its value is valid in the final
year, and let a fact with several simultaneously-valid values have several
current runs. The queried set goes from 892 to ~5.6k.

Nothing else changes. Same policies, same metrics, same val/test hash split,
same tuning protocol. If the ladder's ordering survives a sixfold increase in
facts and the arrival of genuinely multi-valued gold, the v1.0 numbers were not
an artifact of the restriction.

That scoring is the easier reading, and it favours retention: a top-1 is
current when *any* still-valid value is, so a one-slot-per-fact policy can
discard a still-valid value unseen. ``current_set_recall@5`` --- the share of
a fact's distinct currently-valid values in the top-5 --- is the metric that
sees it, and is reported for the track and for the multi-valued subset, the
only place it can differ from current@5. The tuning protocol still selects on
current@1, so the numbers the paper already quotes are untouched.

Run:  python -m stalebench.run_scale
"""

from __future__ import annotations

import json
import os
from typing import Callable, Dict, List

import numpy as np

from stalebench.alias import build_alias_map
from stalebench.embed import MINILM, Embedder
from stalebench.metrics import bootstrap_ci, holm, paired_permutation, score_ranking
from stalebench.policies import (AliasKey, BM25RAG, CosineRAG, FuzzyKey,
                                 KeepLatestOracle, LexicalKey, ReadWriteFuzzyKey,
                                 Reconsolidation, RecencyWeighted, TimeRerank)
from stalebench.templama import (build_instance, load_all_changing_facts,
                                 split_facts, subjects_with_class)

_HERE = os.path.dirname(__file__)
_OUT = os.path.join(_HERE, "..", "results", "stalebench_scale_results.json")
_PARA = 0.5
_KS = (5,)
_CI = ("current@1", "contamination@5", "current_set_recall@5")


def _score(policy, inst, ev, qe) -> Dict:
    policy.reset()
    for (text, iid, t), e in zip(inst.events, ev):
        policy.observe(e, iid, t, text)
    recs = [score_ranking(policy.rank(q, qt), key, cur, _KS, value_of=inst.value_of)
            for (qt, key, cur), q in zip(inst.queries, qe)]
    summ = {m: float(np.mean([r[m] for r in recs])) for m in recs[0]}
    summ["slots"] = float(policy.slots)
    return {"summary": summ, "records": recs}


def _subset(recs: List[Dict], keep: List[bool], slots: float) -> Dict:
    """Summary + CI over the queries ``keep`` selects (one store, so one slots)."""
    sub = [r for r, m in zip(recs, keep) if m]
    summ = {m: float(np.mean([r[m] for r in sub])) for m in sub[0]}
    summ["slots"] = slots
    return {"summary": summ,
            "ci": {m: bootstrap_ci([r[m] for r in sub], seed=1) for m in _CI}}


def _factories(tuned: Dict, gaz: List[str], amap: Dict) -> Dict[str, Callable]:
    tau = tuned["reconsolidation"]["tau"]
    return {
        "bm25": lambda: BM25RAG(),
        "cosine-rag": lambda: CosineRAG(),
        "time-rerank": lambda: TimeRerank(k=tuned["time-rerank"]["k"]),
        "recency-weighted": lambda: RecencyWeighted(
            lam=tuned["recency-weighted"]["lam"],
            decay=tuned["recency-weighted"]["decay"]),
        "reconsolidation": lambda: Reconsolidation(tau=tau),
        "lexical-key": lambda: LexicalKey(gaz),
        "fuzzy-key": lambda: FuzzyKey(gaz, tuned["fuzzy-key"]["tau"]),
        "fuzzy-rw-key": lambda: ReadWriteFuzzyKey(gaz, tuned["fuzzy-key"]["tau"]),
        "alias-key(recent)": lambda: AliasKey(gaz, amap, "recent"),
        "keep-latest-oracle": lambda: KeepLatestOracle(),
    }


def _tune(inst, ev, qv, gaz) -> Dict:
    def c1(p):
        return _score(p, inst, ev, qv)["summary"]["current@1"]
    ks = [2, 3, 5, 10, 20]
    lam, decay = max(((l, d) for l in (0.1, 0.3, 0.6, 1.0) for d in (0.8, 0.9, 0.99)),
                     key=lambda ld: c1(RecencyWeighted(lam=ld[0], decay=ld[1])))
    taus = [round(float(x), 2) for x in np.arange(0.50, 0.91, 0.05)]
    ftaus = [round(float(x), 2) for x in np.arange(0.20, 0.66, 0.05)]
    return {"time-rerank": {"k": max(ks, key=lambda k: c1(TimeRerank(k=k)))},
            "recency-weighted": {"lam": lam, "decay": decay},
            "reconsolidation": {"tau": max(taus,
                                           key=lambda t: c1(Reconsolidation(tau=t)))},
            "fuzzy-key": {"tau": max(ftaus, key=lambda t: c1(FuzzyKey(gaz, t)))}}


def run() -> Dict:
    facts, stats = load_all_changing_facts()
    val, test = split_facts(facts)
    gaz_val = [f.subject for f in val]
    gaz_test = [f.subject for f in test]
    amap_test = build_alias_map(subjects_with_class(test))
    multi = [f.is_multi_valued for f in test]        # one query per fact, in order
    emb = Embedder(MINILM)
    out: Dict = {"stats": dict(
        stats, val_facts=len(val), test_facts=len(test), paraphrase_p=_PARA,
        multi_valued_test=sum(1 for f in test if f.is_multi_valued),
        mean_current_values=float(np.mean([len(f.current_values) for f in test])),
    ), "by_alias": {}}

    for a in (0.0, 1.0):
        inst_v = build_instance(val, paraphrase_p=_PARA, seed=0, alias_p=a)
        inst_t = build_instance(test, paraphrase_p=_PARA, seed=0, alias_p=a)
        tag = f"scale_a{a}"
        ev_v = emb.encode([e[0] for e in inst_v.events], f"val_{tag}")
        qv = emb.encode([q[0] for q in inst_v.queries], f"valq_{tag}")
        ev_t = emb.encode([e[0] for e in inst_t.events], f"test_{tag}")
        qt = emb.encode([q[0] for q in inst_t.queries], f"testq_{tag}")
        tuned = _tune(inst_v, ev_v, qv, gaz_val)
        res = {n: _score(f(), inst_t, ev_t, qt)
               for n, f in _factories(tuned, gaz_test, amap_test).items()}
        assert len(multi) == len(inst_t.queries)
        orc = res["keep-latest-oracle"]["summary"]["current@1"]

        def pp(x, y, m="current@1"):
            return paired_permutation([r[m] for r in res[x]["records"]],
                                      [r[m] for r in res[y]["records"]], seed=2)
        raw = {"rerank_vs_cosine": pp("time-rerank", "cosine-rag"),
               "lexical_vs_oracle": pp("lexical-key", "keep-latest-oracle"),
               "fuzzy_vs_oracle": pp("fuzzy-key", "keep-latest-oracle"),
               "recon_vs_fuzzy": pp("reconsolidation", "fuzzy-key"),
               # symmetric ordinary resolution vs asymmetric perfect resolution
               "rw_vs_fuzzy": pp("fuzzy-rw-key", "fuzzy-key"),
               "rw_vs_alias": pp("fuzzy-rw-key", "alias-key(recent)"),
               "rw_vs_oracle": pp("fuzzy-rw-key", "keep-latest-oracle")}
        out["by_alias"][f"alias_p={a}"] = {
            "tuned": tuned,
            "n_statements": len(inst_t.events),
            "test": {n: {"summary": r["summary"],
                         "ci": {m: bootstrap_ci([x[m] for x in r["records"]], seed=1)
                                for m in _CI}}
                     for n, r in res.items()},
            "keying_deficit": {n: orc - r["summary"]["current@1"]
                               for n, r in res.items()},
            "paired_p": {"raw": raw, "holm": holm(raw)},
            # the facts on which set recall can fall below current@5
            "test_multi_valued": {n: _subset(r["records"], multi, r["summary"]["slots"])
                                  for n, r in res.items()},
        }
    return out


def main():
    res = run()
    os.makedirs(os.path.dirname(_OUT), exist_ok=True)
    json.dump(res, open(_OUT, "w"), indent=1)
    st = res["stats"]
    print("=" * 78)
    print(f"StaleBench v1.1 scale track: {st['test_facts']} test facts "
          f"({st['multi_valued_test']} multi-valued), "
          f"mean {st['mean_current_values']:.2f} current values/fact")
    print("=" * 78)
    for a, cell in res["by_alias"].items():
        print(f"\n[{a}]  {cell['n_statements']} statements  tuned={cell['tuned']}")
        for n, r in cell["test"].items():
            s = r["summary"]
            ms = cell["test_multi_valued"][n]["summary"]
            print(f"  {n:20s} cur@1 {s['current@1']:.3f}  stale {s['stale_rate']:.3f}"
                  f"  miss {s['miss_rate']:.3f}  contam@5 {s['contamination@5']:.3f}"
                  f"  setrec@5 {s['current_set_recall@5']:.3f}"
                  f" (multi-valued: cur@5 {ms['current@5']:.3f}"
                  f" setrec@5 {ms['current_set_recall@5']:.3f})"
                  f"  slots {s['slots']:.0f}  deficit "
                  f"{100 * cell['keying_deficit'][n]:+.1f}")
        print(f"  paired p (Holm): {cell['paired_p']['holm']}")
    print(f"\nWrote {os.path.abspath(_OUT)}")


if __name__ == "__main__":
    main()
