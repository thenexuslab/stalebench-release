"""Diagnostic: is the controlled pair actually controlled?

The reader track's headline compares two keyers that are supposed to be the
same policy on the read path, so that any difference in reader accuracy is
attributable to ranks 2..k. That argument is only as good as the premise, and
the premise is checkable per query rather than inferable from two aggregate
scores that happen to match. This measures it directly at both ends of the
anchor axis and writes the counts the paper quotes.

Anchor-free the pair is *not* exactly controlled: a query whose subject fails
to resolve falls back to similarity ranking, and the two policies rank over
differently sized stores (one retains every version, the other merges). The
paper therefore makes the controlled claim only in the anchored configuration,
where the reader track runs and where agreement is exact.

Run:  python -m stalebench.diag_pair
"""

from __future__ import annotations

import json
import os
from typing import Dict

from stalebench.embed import MINILM, Embedder
from stalebench.policies import ReadWriteFuzzyKey
from stalebench.templama import build_instance, load_facts, split_facts

_HERE = os.path.dirname(__file__)
_OUT = os.path.join(_HERE, "..", "results", "stalebench_pair_diag.json")
_PARA = 0.5
_TAU = 0.2


def run() -> Dict:
    facts, _ = load_facts()
    _, test = split_facts(facts)
    gaz = [f.subject for f in test]
    emb = Embedder(MINILM)
    out: Dict = {"config": {"paraphrase_p": _PARA, "fuzzy_tau": _TAU,
                            "n_test_facts": len(test)}, "by_alias": {}}

    for a in (0.0, 1.0):
        inst = build_instance(test, paraphrase_p=_PARA, seed=0, alias_p=a)
        tag = f"pt_p{_PARA}_a{a}"
        ev = emb.encode([e[0] for e in inst.events], tag)
        qe = emb.encode([q[0] for q in inst.queries], f"q{tag}")
        tops = {}
        for name, write_keys in (("read", False), ("rw", True)):
            pol = ReadWriteFuzzyKey(gaz, _TAU, write_keys=write_keys)
            pol.reset()
            for (text, iid, t), e in zip(inst.events, ev):
                pol.observe(e, iid, t, text)
            tops[name] = [pol.rank(q, qt)[0]
                          for (qt, _, _), q in zip(inst.queries, qe)]
        agree = sum(1 for x, y in zip(tops["read"], tops["rw"]) if x == y)
        n = len(tops["read"])
        out["by_alias"][f"alias_p={a}"] = {
            "n_queries": n, "rank1_agree": agree,
            "rank1_agree_frac": agree / n,
            "exactly_controlled": agree == n,
        }
        print(f"[alias_p={a}] rank-1 identical on {agree}/{n} queries "
              f"({100 * agree / n:.1f}%)", flush=True)
    return out


def main():
    res = run()
    os.makedirs(os.path.dirname(_OUT), exist_ok=True)
    json.dump(res, open(_OUT, "w"), indent=1)
    print(f"Wrote {os.path.abspath(_OUT)}")


if __name__ == "__main__":
    main()
