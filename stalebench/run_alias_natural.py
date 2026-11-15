"""StaleBench anchor-free track, natural alias source (v1.2).

The anchor-free ladder (:mod:`stalebench.run_alias`) withdraws the lexical
anchor with short forms derived *mechanically* from the subject's name --- the
head token, the leading token, an initialed form, a drop-last shortening. The
objection writes itself: the axis is synthetic. This runner reruns the ladder
with *natural* short forms --- Wikidata aliases and English Wikipedia
redirects (``v2/natural_aliases.py``) --- so the paper
can report the gap between the two instead of asserting it is small.

Two configurations, because the natural inventory reaches only 270 of the 467
test subjects:

* ``natural`` --- an uncovered subject stays anchored on every statement, so
  the stream is exactly as anchor-free as the public record allows;
* ``natural+fallback`` --- an uncovered subject uses the mechanical forms, so
  coverage matches the mechanical run and only the *forms* differ.

Everything else is the mechanical protocol: same rungs, same val grids (every
tunable re-tuned per configuration and level, on a val stream built from the
val half of the inventory), same test split, same encoder, same retrieval-free
keying audit. The entity-linker rung is handed the natural inventory --- the
analogue of holding the generator that inserted the mechanical aliases.

Run:  python -m stalebench.run_alias_natural
"""

from __future__ import annotations

import json
import os
import time
from typing import Dict

from stalebench.alias import alias_map_from_pools, pool_stats
from stalebench.embed import MINILM, Embedder
from stalebench.policies import (AliasKeyer, FuzzyKeyer, LexicalKeyer,
                                 OracleKeyer, ThresholdKeyer)
from stalebench.run_alias import (_PARA, _audit_keyer, _factories, _sweep_cell,
                                  _tune, score_real)
from stalebench.templama import (NATURAL_INVENTORY, alias_pools, build_instance,
                                 load_facts, load_natural_inventory, split_facts)

_HERE = os.path.dirname(__file__)
_OUT = os.path.join(_HERE, "..", "results", "stalebench_alias_natural_results.json")
_MECHANICAL = os.path.join(_HERE, "..", "results", "stalebench_alias_results.json")
# tuning happens on val, so val needs its own inventory (same builder, --split val)
_INVENTORY = {"test": NATURAL_INVENTORY,
              "val": os.path.join(os.path.dirname(NATURAL_INVENTORY),
                                  "alias_inventory_natural_val.json")}
_SOURCES = ("natural", "natural+fallback")
_ALIAS_LEVELS = (1.0, 0.5)
_GAP_METRICS = ("current@1", "contamination@5", "slots")


def run() -> Dict:
    t0 = time.time()
    facts, stats = load_facts()
    val, test = split_facts(facts)
    gaz_val = [f.subject for f in val]
    gaz_test = [f.subject for f in test]
    inv = {split: load_natural_inventory(p) for split, p in _INVENTORY.items()}
    emb = Embedder(MINILM)
    mech = json.load(open(_MECHANICAL)) if os.path.exists(_MECHANICAL) else {}
    out: Dict = {
        "stats": dict(stats, val_facts=len(val), test_facts=len(test),
                      paraphrase_p=_PARA),
        "inventory": {s: {"path": os.path.abspath(p), "n_subjects": len(inv[s])}
                      for s, p in _INVENTORY.items()},
        "mechanical_results": os.path.abspath(_MECHANICAL) if mech else None,
        "alias_inventory": {}, "alias_sweep": {}, "keying_audit": {},
        "gap_vs_mechanical": {},
    }
    for src in _SOURCES:
        slug = src.replace("+", "-")
        pools_t = alias_pools(test, src, inv["test"])
        amap_test = alias_map_from_pools(pools_t)
        out["alias_inventory"][src] = {
            "val": pool_stats(alias_pools(val, src, inv["val"])),
            "test": pool_stats(pools_t)}
        for key in ("alias_sweep", "keying_audit", "gap_vs_mechanical"):
            out[key][src] = {}
        for a in _ALIAS_LEVELS:
            inst_v = build_instance(val, paraphrase_p=_PARA, seed=0, alias_p=a,
                                    alias_source=src,
                                    alias_inventory=_INVENTORY["val"])
            inst_t = build_instance(test, paraphrase_p=_PARA, seed=0, alias_p=a,
                                    alias_source=src,
                                    alias_inventory=_INVENTORY["test"])
            ev_v = emb.encode([e[0] for e in inst_v.events],
                              f"val_ev_p{_PARA}_a{a}_{slug}")
            qv = emb.encode([q[0] for q in inst_v.queries],
                            f"val_q_p{_PARA}_a{a}_{slug}")
            ev_t = emb.encode([e[0] for e in inst_t.events],
                              f"test_ev_p{_PARA}_a{a}_{slug}")
            qt = emb.encode([q[0] for q in inst_t.queries],
                            f"test_q_p{_PARA}_a{a}_{slug}")
            tuned = _tune(inst_v, ev_v, qv, gaz_val)
            res = {n: score_real(f(), inst_t, ev_t, qt)
                   for n, f in _factories(tuned, gaz_test, amap_test).items()}
            cell = _sweep_cell(res, inst_t, tuned, paired=True)
            cell["coverage"] = {"test": inst_t.alias_coverage,
                                "val": inst_v.alias_coverage}
            out["alias_sweep"][src][f"alias_p={a}"] = cell
            out["keying_audit"][src][f"alias_p={a}"] = {
                "threshold": _audit_keyer(
                    ThresholdKeyer(tuned["reconsolidation"]["tau"]), inst_t, ev_t),
                "lexical": _audit_keyer(LexicalKeyer(gaz_test), inst_t, ev_t),
                "fuzzy": _audit_keyer(
                    FuzzyKeyer(gaz_test, tuned["fuzzy-key"]["tau"]), inst_t, ev_t),
                "alias(recent)": _audit_keyer(
                    AliasKeyer(gaz_test, amap_test, "recent"), inst_t, ev_t),
                "oracle": _audit_keyer(OracleKeyer(), inst_t, ev_t),
            }
            # the number the paper reports: natural minus mechanical, per rung,
            # at the same anchor level (both re-tuned on val)
            mcell = mech.get("alias_sweep", {}).get(f"alias_p={a}")
            if mcell:
                out["gap_vs_mechanical"][src][f"alias_p={a}"] = {
                    n: dict({m: r["summary"][m] - mcell["test"][n]["summary"][m]
                             for m in _GAP_METRICS},
                            keying_deficit=(cell["keying_deficit"][n]
                                            - mcell["keying_deficit"][n]))
                    for n, r in res.items() if n in mcell["test"]}
    out["wall_clock_s"] = time.time() - t0
    return out


def main():
    res = run()
    os.makedirs(os.path.dirname(_OUT), exist_ok=True)
    json.dump(res, open(_OUT, "w"), indent=1)

    print("=" * 78)
    print("StaleBench anchor-free track, natural alias source (TempLAMA), "
          "test split, paraphrase p=0.5")
    print("=" * 78)
    for src in _SOURCES:
        for split in ("val", "test"):
            inv = res["alias_inventory"][src][split]
            print(f"{src:17s} {split:4s} inventory: {inv['n_with_alias']}/"
                  f"{inv['n_subjects']} subjects ({inv['frac_with_alias']:.1%}) "
                  f"have >=1 form; {inv['n_ambiguous_forms']}/{inv['n_alias_forms']} "
                  f"forms ambiguous")
        for a in _ALIAS_LEVELS:
            cov = res["alias_sweep"][src][f"alias_p={a}"]["coverage"]["test"]
            print(f"{src:17s} a={a:<4} test: {cov['n_alias_events']}/"
                  f"{cov['n_alias_eligible']} eligible statements aliased; "
                  f"{cov['n_subjects_natural']} subjects natural, "
                  f"{cov['n_subjects_fallback']} fallback")

    mech = (json.load(open(_MECHANICAL)) if res["mechanical_results"] else {})
    cols = ["mechanical"] + list(_SOURCES)
    for a in _ALIAS_LEVELS:
        cells = {"mechanical": mech.get("alias_sweep", {}).get(f"alias_p={a}")}
        cells.update({s: res["alias_sweep"][s][f"alias_p={a}"] for s in _SOURCES})
        print(f"\n[a={a}]  current@1 / contamination@5 / slots / deficit vs oracle")
        print(f"{'policy':20s}" + "".join(f"  {c:>28s}" for c in cols))
        for n in res["alias_sweep"][_SOURCES[0]][f"alias_p={a}"]["test"]:
            row = ""
            for c in cols:
                cell = cells[c]
                if not cell or n not in cell["test"]:
                    row += f"  {'-':>28s}"
                    continue
                s = cell["test"][n]["summary"]
                row += (f"  {s['current@1']:.3f}/{s['contamination@5']:.3f}/"
                        f"{s['slots']:4.0f}/{cell['keying_deficit'][n]:+.3f}")
            print(f"{n:20s}{row}")
        for c in cols:
            if cells[c]:
                print(f"  tuned[{c}]: {cells[c]['tuned']}")

    print("\n[keying audit: cluster errors, independent of retrieval]")
    for src in _SOURCES:
        for a, cell in res["keying_audit"][src].items():
            print(f"  {src} {a}")
            for k, v in cell.items():
                print(f"    {k:16s} clusters {v['n_clusters']:4d}  impure "
                      f"{v['impure_clusters']:3d}  split facts {v['split_facts']:3d} "
                      f"({v['frac_split']:.1%})  unlinked {v['unlinked']:4d}")
    print(f"\nwall-clock {res['wall_clock_s']:.0f}s")
    print(f"Wrote {os.path.abspath(_OUT)}")


if __name__ == "__main__":
    main()
