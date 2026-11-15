"""Verbalization robustness: is the ladder an artifact of how we phrase facts?

Every result in this paper is reported at ``paraphrase_p=0.5`` --- half the
statements use the relation's base template, half use a hand-written
alternative ("X plays for Y" vs "Y is the club X turns out for"). That single
setting invites a fair objection: the keying ladder separates policies by how
well they detect that two statements describe one fact, and how hard *that* is
depends on how much the two statements look alike. A benchmark that fixes
verbalization diversity is measuring keying at one point of an axis it never
varies.

So vary it. ``p=0.0`` is the worst case for the objection and the best case for
a naive keyer: every statement about a fact is the same sentence with the
object swapped, so surface overlap is maximal. ``p=1.0`` is the opposite. If
the ladder's ordering is stable across that range while the *absolute* numbers
move, the ordering is a property of the keying mechanisms rather than of our
sentence templates.

This is retrieval-only and CPU-only: no LLM keyer, no reader, so it costs
embedding time and nothing else.

Run:  python -m stalebench.run_paraphrase
"""

from __future__ import annotations

import json
import os
from typing import Dict, List

import numpy as np

from stalebench.alias import build_alias_map
from stalebench.embed import MINILM, Embedder
from stalebench.metrics import holm, paired_permutation, score_ranking
from stalebench.policies import (AliasKey, CosineRAG, FuzzyKey, KeepLatestOracle,
                                 LexicalKey, ReadWriteFuzzyKey, RecencyWeighted,
                                 Reconsolidation, TimeRerank)
from stalebench.templama import (build_instance, load_facts, split_facts,
                                 subjects_with_class)

_HERE = os.path.dirname(__file__)
_OUT = os.path.join(_HERE, "..", "results", "stalebench_paraphrase_results.json")
_PS = (0.0, 0.5, 1.0)
_KS = (5,)


def _score(policy, inst, ev, qe) -> Dict:
    policy.reset()
    for (text, iid, t), e in zip(inst.events, ev):
        policy.observe(e, iid, t, text)
    recs = [score_ranking(policy.rank(q, qt), key, cur, _KS)
            for (qt, key, cur), q in zip(inst.queries, qe)]
    summ = {m: float(np.mean([r[m] for r in recs])) for m in recs[0]}
    summ["slots"] = float(policy.slots)
    return {"summary": summ, "records": recs}


def _factories(gaz: List[str], amap: Dict, tuned: Dict) -> Dict:
    tau, ftau = tuned["tau"], tuned["fuzzy_tau"]
    return {
        "cosine-rag": lambda: CosineRAG(),
        "time-rerank": lambda: TimeRerank(k=tuned["rerank_k"]),
        "recency-weighted": lambda: RecencyWeighted(
            lam=tuned["lam"], decay=tuned["decay"]),
        "reconsolidation": lambda: Reconsolidation(tau=tau),
        "lexical-key": lambda: LexicalKey(gaz),
        "fuzzy-key": lambda: FuzzyKey(gaz, ftau),
        "fuzzy-rw-key": lambda: ReadWriteFuzzyKey(gaz, ftau),
        "alias-key(recent)": lambda: AliasKey(gaz, amap, "recent"),
        "keep-latest-oracle": lambda: KeepLatestOracle(),
    }


def run() -> Dict:
    facts, _ = load_facts()
    val, test = split_facts(facts)
    gaz_val = [f.subject for f in val]
    gaz_test = [f.subject for f in test]
    amap = build_alias_map(subjects_with_class(test))
    emb = Embedder(MINILM)
    out: Dict = {"config": {"paraphrase_levels": list(_PS),
                            "n_test_facts": len(test)}, "cells": {}}

    for p in _PS:
        for a in (0.0, 1.0):
            tag = f"p{p}_a{a}"
            inst_v = build_instance(val, paraphrase_p=p, seed=0, alias_p=a)
            inst_t = build_instance(test, paraphrase_p=p, seed=0, alias_p=a)
            ev_v = emb.encode([e[0] for e in inst_v.events], f"pv_{tag}")
            qv = emb.encode([q[0] for q in inst_v.queries], f"pvq_{tag}")
            ev_t = emb.encode([e[0] for e in inst_t.events], f"pt_{tag}")
            qt = emb.encode([q[0] for q in inst_t.queries], f"ptq_{tag}")

            # tuned on val at every cell, as everywhere else in this paper
            def c1(pol):
                return _score(pol, inst_v, ev_v, qv)["summary"]["current@1"]
            taus = [round(float(x), 2) for x in np.arange(0.50, 0.91, 0.05)]
            ftaus = [round(float(x), 2) for x in np.arange(0.20, 0.66, 0.05)]
            # every tunable is re-tuned per cell, because the paper says so
            lam, decay = max(((l, dd) for l in (0.1, 0.3, 0.6, 1.0)
                              for dd in (0.8, 0.9, 0.99)),
                             key=lambda ld: c1(RecencyWeighted(lam=ld[0],
                                                               decay=ld[1])))
            tuned = {
                "tau": max(taus, key=lambda t: c1(Reconsolidation(tau=t))),
                "fuzzy_tau": max(ftaus, key=lambda t: c1(FuzzyKey(gaz_val, t))),
                "rerank_k": max([2, 3, 5, 10, 20],
                                key=lambda k: c1(TimeRerank(k=k))),
                "lam": lam, "decay": decay}

            res = {n: _score(f(), inst_t, ev_t, qt)
                   for n, f in _factories(gaz_test, amap, tuned).items()}
            orc = res["keep-latest-oracle"]["summary"]["current@1"]

            def pp(x, y):
                return paired_permutation(
                    [r["current@1"] for r in res[x]["records"]],
                    [r["current@1"] for r in res[y]["records"]], seed=4)
            raw = {"lexical_vs_oracle": pp("lexical-key", "keep-latest-oracle"),
                   "fuzzy_vs_lexical": pp("fuzzy-key", "lexical-key"),
                   "rw_vs_alias": pp("fuzzy-rw-key", "alias-key(recent)")}
            out["cells"][tag] = {
                "paraphrase_p": p, "alias_p": a,
                "tuned": tuned,
                "test": {n: r["summary"] for n, r in res.items()},
                "keying_deficit": {n: orc - r["summary"]["current@1"]
                                   for n, r in res.items()},
                "paired_p": {"raw": raw, "holm": holm(raw)},
            }
            print(f"[p={p} a={a}] " + "  ".join(
                f"{n.split('-')[0]}={r['summary']['current@1']:.3f}"
                for n, r in res.items()), flush=True)
    return out


def main():
    res = run()
    os.makedirs(os.path.dirname(_OUT), exist_ok=True)
    json.dump(res, open(_OUT, "w"), indent=1)
    print("\n" + "=" * 76)
    print("Verbalization robustness: current@1 (keying deficit) by paraphrase_p")
    print("=" * 76)
    for tag, cell in res["cells"].items():
        print(f"\n[paraphrase_p={cell['paraphrase_p']} alias_p={cell['alias_p']}]"
              f"  tuned={cell['tuned']}")
        for n, s in cell["test"].items():
            print(f"  {n:20s} cur@1 {s['current@1']:.3f}  stale "
                  f"{s['stale_rate']:.3f}  contam@5 {s['contamination@5']:.3f}"
                  f"  deficit {100 * cell['keying_deficit'][n]:+.1f}")
    print(f"\nWrote {os.path.abspath(_OUT)}")


if __name__ == "__main__":
    main()
