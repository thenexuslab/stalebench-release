"""StaleBench anchor-free track (v1.1): same-fact detection without a verbatim anchor.

v0.2's keying ladder found that a gazetteer string keyer nearly matched the
true-key oracle, because every templated statement carried the subject's
surface form verbatim. That made the paper's stated open problem --- same-fact
detection --- look solved, and left the *actual* open problem (detection
without a lexical anchor) outside what the instrument could measure.

This track supplies the missing axis. ``alias_p`` is the probability that a
non-introducing statement refers to its subject by a short form
(:mod:`stalebench.alias`) instead of the full name. The first statement about
each fact always introduces the full name and the query always uses it, so
every fact stays resolvable in principle --- what degrades is the *cost* of
resolving it.

Four experiments:

* ``alias_sweep`` --- every rung of the keying ladder across
  ``alias_p in {0, .25, .5, .75, 1}``, re-tuned on val at each level.
* ``mechanism`` --- overwrite vs. temporal-KG invalidation under the same
  keyer: what tombstoning buys and what it costs.
* ``full_corpus`` --- the same comparison with all 4,939 non-queried TempLAMA
  fact keys streamed in as distractors (a ~7x larger, realistic retrieval pool).
* ``ambiguity`` --- alias inventory coverage, collision rate, and the linker's
  own accounting of unresolvable mentions.

Run:  python -m stalebench.run_alias
"""

from __future__ import annotations

import json
import os
from collections import Counter
from typing import Callable, Dict, List, Optional

import numpy as np

from stalebench.alias import alias_stats, build_alias_map
from stalebench.embed import MINILM, MPNET, Embedder
from stalebench.metrics import bootstrap_ci, holm, paired_permutation, score_ranking
from stalebench.policies import (AliasKey, AliasKeyer, BM25RAG, CosineRAG,
                                 FuzzyKey, FuzzyKeyer, KeepLatestOracle,
                                 LexicalKey, LexicalKeyer, OracleKeyer,
                                 ReadKeyedRetention, ReadWriteFuzzyKey,
                                 Reconsolidation,
                                 RecencyWeighted, SurfaceMatcher,
                                 TemporalKG, ThresholdKeyer, TimeRerank)
from stalebench.templama import (RealInstance, build_instance, load_distractor_facts,
                                 load_facts, split_facts, subjects_with_class)

_HERE = os.path.dirname(__file__)
_OUT = os.path.join(_HERE, "..", "results", "stalebench_alias_results.json")

_KS = (5,)
_ALIAS_LEVELS = (0.0, 0.25, 0.5, 0.75, 1.0)
_PARA = 0.5                      # headline paraphrase rate, as in the main track


def score_real(policy, inst: RealInstance, emb_events: np.ndarray,
               emb_queries: np.ndarray) -> Dict:
    policy.reset()
    for (text, iid, t), e in zip(inst.events, emb_events):
        policy.observe(e, iid, t, text)
    records = [score_ranking(policy.rank(qe, qtext), key, cur, _KS)
               for (qtext, key, cur), qe in zip(inst.queries, emb_queries)]
    summary = {m: float(np.mean([r[m] for r in records])) for m in records[0]}
    summary["slots"] = float(policy.slots)
    summary["active_slots"] = float(policy.active_slots)
    return {"summary": summary, "records": records}


def _factories(tuned: Dict[str, Dict], gaz: List[str],
               amap: Dict) -> Dict[str, Callable]:
    """Every rung of the keying ladder, plus the store-all reference points."""
    tau = tuned["reconsolidation"]["tau"]
    return {
        "bm25": lambda: BM25RAG(),
        "bm25+time-rerank": lambda: BM25RAG(rerank_k=tuned["time-rerank"]["k"]),
        "cosine-rag": lambda: CosineRAG(),
        "time-rerank": lambda: TimeRerank(k=tuned["time-rerank"]["k"]),
        "recency-weighted": lambda: RecencyWeighted(
            lam=tuned["recency-weighted"]["lam"],
            decay=tuned["recency-weighted"]["decay"]),
        "reconsolidation": lambda: Reconsolidation(tau=tau),
        "lexical-key": lambda: LexicalKey(gaz),
        "fuzzy-key": lambda: FuzzyKey(gaz, tuned["fuzzy-key"]["tau"]),
        "fuzzy-read-key": lambda: ReadWriteFuzzyKey(
            gaz, tuned["fuzzy-key"]["tau"], write_keys=False),
        "fuzzy-rw-key": lambda: ReadWriteFuzzyKey(
            gaz, tuned["fuzzy-key"]["tau"], write_keys=True),
        "alias-key(skip)": lambda: AliasKey(gaz, amap, "skip"),
        "alias-key(recent)": lambda: AliasKey(gaz, amap, "recent"),
        # the control that separates *path* from *resolver quality*: the same
        # complete alias table, applied on both paths. Without it, "symmetric
        # ordinary beats asymmetric perfect" confounds the two.
        "alias-rw-key": lambda: ReadKeyedRetention(
            AliasKeyer(gaz, amap, "recent"), name="alias-rw-key"),
        "keep-latest-oracle": lambda: KeepLatestOracle(),
    }


def _tune(inst: RealInstance, ev: np.ndarray, qv: np.ndarray,
          gaz_for_tune: Optional[List[str]] = None) -> Dict[str, Dict]:
    """Grid-search the tunable policies on val by current@1 (as in the main track)."""
    gaz_for_tune = gaz_for_tune or []

    def c1(p) -> float:
        return score_real(p, inst, ev, qv)["summary"]["current@1"]
    ks = [2, 3, 5, 10, 20]
    lam_grid, decay_grid = [0.1, 0.3, 0.6, 1.0], [0.8, 0.9, 0.99]
    lam, decay = max(((l, d) for l in lam_grid for d in decay_grid),
                     key=lambda ld: c1(RecencyWeighted(lam=ld[0], decay=ld[1])))
    taus = [round(float(x), 2) for x in np.arange(0.50, 0.91, 0.05)]
    ftaus = [round(float(x), 2) for x in np.arange(0.20, 0.66, 0.05)]
    return {"time-rerank": {"k": max(ks, key=lambda k: c1(TimeRerank(k=k)))},
            "recency-weighted": {"lam": lam, "decay": decay},
            "reconsolidation": {"tau": max(taus,
                                           key=lambda t: c1(Reconsolidation(tau=t)))},
            "fuzzy-key": {"tau": max(ftaus,
                                     key=lambda t: c1(FuzzyKey(gaz_for_tune, t)))}}


def _sweep_cell(res: Dict[str, Dict], inst_t: RealInstance,
                tuned: Dict[str, Dict], paired: bool) -> Dict:
    """One cell of the sweep: summaries with CIs, the deficit vs. the oracle,
    and (at the endpoints) the paired-comparison family. The natural-source
    runner builds its cells here too, so the two result files share a schema
    by construction rather than by convention."""
    cell = {
        "tuned": tuned,
        "n_alias_events": inst_t.n_alias_events,
        "n_events": len(inst_t.events),
        "test": {n: {"summary": r["summary"],
                     "ci": {m: bootstrap_ci([x[m] for x in r["records"]],
                                            seed=1)
                            for m in ("current@1", "current@5",
                                      "stale_rate", "miss_rate",
                                      "contamination@5")}}
                 for n, r in res.items()},
    }
    # the quantity the paper is about: how many points of current@1 does a
    # realistic keyer lose to the oracle, as the anchor disappears?
    orc = res["keep-latest-oracle"]["summary"]["current@1"]
    cell["keying_deficit"] = {n: orc - r["summary"]["current@1"]
                              for n, r in res.items()}
    if paired:
        def pp(x, y, m="current@1"):
            return paired_permutation([r[m] for r in res[x]["records"]],
                                      [r[m] for r in res[y]["records"]], seed=2)
        raw = {"lexical_vs_oracle": pp("lexical-key", "keep-latest-oracle"),
               "alias_vs_oracle": pp("alias-key(recent)", "keep-latest-oracle"),
               "alias_vs_lexical": pp("alias-key(recent)", "lexical-key"),
               "recon_vs_lexical": pp("reconsolidation", "lexical-key"),
               # the read-path rung: does resolving the query too actually
               # beat resolving only the write path, and the alias table?
               "rw_vs_fuzzy": pp("fuzzy-rw-key", "fuzzy-key"),
               "rw_vs_alias": pp("fuzzy-rw-key", "alias-key(recent)"),
               "rw_vs_oracle": pp("fuzzy-rw-key", "keep-latest-oracle"),
               # path vs resolver quality, separated
               "aliasrw_vs_alias": pp("alias-rw-key", "alias-key(recent)"),
               "aliasrw_vs_fuzzyrw": pp("alias-rw-key", "fuzzy-rw-key")}
        cell["paired_p"] = {"raw": raw, "holm": holm(raw)}
    return cell


def _audit_keyer(keyer, inst: RealInstance, emb: np.ndarray) -> Dict:
    """Replay a keyer over the stream and score its same-fact decisions directly.

    Keying is scored as a clustering of statements against the true fact
    partition, independent of retrieval: a *merge error* puts two different
    facts in one slot, a *split error* leaves one fact in several slots.
    """
    from stalebench.policies import KeyedRetention
    p = KeyedRetention(keyer)
    p.reset()
    members: Dict[str, List[str]] = {}      # cluster id -> true fact keys in it
    for (text, iid, t), e in zip(inst.events, emb):
        # key once, then mirror KeyedRetention.observe: re-calling observe here
        # would double every keyer decision (and every LLM call)
        k = p.keyer.key_of(p, e, iid, text)
        cid = k if k is not None else f"__unkeyed{len(members)}"
        j = p._slot_of.get(k) if k is not None else None
        if j is None:
            p._push(e, iid, t, text, k)
            if k is not None:
                p._slot_of[k] = p._n - 1
        else:
            p._overwrite(j, e, iid, t, text)
        members.setdefault(cid, []).append(iid.split(":")[0])
    impure = sum(1 for v in members.values() if len(set(v)) > 1)
    fact_clusters: Dict[str, set] = {}
    for cid, facts in members.items():
        for f in set(facts):
            fact_clusters.setdefault(f, set()).add(cid)
    split = sum(1 for v in fact_clusters.values() if len(v) > 1)
    return {"n_clusters": len(members),
            "n_facts": len(fact_clusters),
            "impure_clusters": impure,
            "split_facts": split,
            "frac_split": split / max(1, len(fact_clusters)),
            "frac_impure": impure / max(1, len(members)),
            "unlinked": int(getattr(keyer, "n_unlinked", 0)),
            "ambiguous_hits": int(getattr(keyer, "n_ambiguous_hits", 0))}


def run() -> Dict:
    facts, stats = load_facts()
    val, test = split_facts(facts)
    gaz_val = [f.subject for f in val]
    gaz_test = [f.subject for f in test]
    cls_val, cls_test = subjects_with_class(val), subjects_with_class(test)
    amap_val = build_alias_map(cls_val)
    amap_test = build_alias_map(cls_test)
    emb = Embedder(MINILM)
    out: Dict = {
        "stats": dict(stats, val_facts=len(val), test_facts=len(test),
                      paraphrase_p=_PARA),
        "alias_inventory": {"val": alias_stats(cls_val),
                            "test": alias_stats(cls_test)},
        "alias_sweep": {}, "mechanism": {}, "full_corpus": {}, "keying_audit": {},
    }

    # -- 1. the sweep: every rung vs. anchor availability ------------------- #
    for a in _ALIAS_LEVELS:
        inst_v = build_instance(val, paraphrase_p=_PARA, seed=0, alias_p=a)
        inst_t = build_instance(test, paraphrase_p=_PARA, seed=0, alias_p=a)
        ev_v = emb.encode([e[0] for e in inst_v.events], f"val_ev_p{_PARA}_a{a}")
        qv = emb.encode([q[0] for q in inst_v.queries], f"val_q_p{_PARA}_a{a}")
        ev_t = emb.encode([e[0] for e in inst_t.events], f"test_ev_p{_PARA}_a{a}")
        qt = emb.encode([q[0] for q in inst_t.queries], f"test_q_p{_PARA}_a{a}")
        tuned = _tune(inst_v, ev_v, qv, gaz_val)
        res = {n: score_real(f(), inst_t, ev_t, qt)
               for n, f in _factories(tuned, gaz_test, amap_test).items()}
        out["alias_sweep"][f"alias_p={a}"] = _sweep_cell(
            res, inst_t, tuned, paired=a in (0.0, 1.0))

    # -- 1b. how much does mention detection matter? ------------------------ #
    # A bare-substring matcher (v0.2's) links inventory forms occurring inside
    # other names and other words. Quantified on the distractor corpus, since
    # that is where it does the damage: every hit there is a cross-fact merge.
    dist_all = load_distractor_facts()
    _, dist_t0 = split_facts(dist_all)
    inst_fc = build_instance(test, paraphrase_p=_PARA, seed=0, alias_p=0.0,
                             distractors=dist_t0)
    forms = sorted(set(gaz_test) | set(amap_test), key=lambda x: (-len(x), x))
    matcher = SurfaceMatcher(forms)
    naive = strict = n_dist = 0
    for text, iid, _ in inst_fc.events:
        if not iid.startswith("d"):
            continue
        n_dist += 1
        if any(f in text for f in forms):
            naive += 1
        if matcher.match(text) is not None:
            strict += 1
    out["mention_detection"] = {
        "n_distractor_statements": n_dist,
        "substring_false_link_rate": naive / max(1, n_dist),
        "mention_false_link_rate": strict / max(1, n_dist),
    }

    # -- 2. keying audit: cluster quality, independent of retrieval --------- #
    for a in (0.0, 1.0):
        inst_t = build_instance(test, paraphrase_p=_PARA, seed=0, alias_p=a)
        ev_t = emb.encode([e[0] for e in inst_t.events], f"test_ev_p{_PARA}_a{a}")
        tau = out["alias_sweep"][f"alias_p={a}"]["tuned"]["reconsolidation"]["tau"]
        out["keying_audit"][f"alias_p={a}"] = {
            "threshold": _audit_keyer(ThresholdKeyer(tau), inst_t, ev_t),
            "lexical": _audit_keyer(LexicalKeyer(gaz_test), inst_t, ev_t),
            "fuzzy": _audit_keyer(
                FuzzyKeyer(gaz_test,
                           out["alias_sweep"][f"alias_p={a}"]["tuned"]
                           ["fuzzy-key"]["tau"]), inst_t, ev_t),
            "alias(recent)": _audit_keyer(
                AliasKeyer(gaz_test, amap_test, "recent"), inst_t, ev_t),
            "oracle": _audit_keyer(OracleKeyer(), inst_t, ev_t),
        }

    # -- 3. mechanism: overwrite vs. invalidation, same keyer --------------- #
    for a in (0.0, 1.0):
        inst_t = build_instance(test, paraphrase_p=_PARA, seed=0, alias_p=a)
        ev_t = emb.encode([e[0] for e in inst_t.events], f"test_ev_p{_PARA}_a{a}")
        qt = emb.encode([q[0] for q in inst_t.queries], f"test_q_p{_PARA}_a{a}")
        tau = out["alias_sweep"][f"alias_p={a}"]["tuned"]["reconsolidation"]["tau"]
        pairs = {
            "overwrite[tau]": lambda: Reconsolidation(tau=tau),
            "temporal-kg[tau]": lambda: TemporalKG(ThresholdKeyer(tau)),
            "overwrite[alias]": lambda: AliasKey(gaz_test, amap_test, "recent"),
            "temporal-kg[alias]": lambda: TemporalKG(
                AliasKeyer(gaz_test, amap_test, "recent")),
            "overwrite[oracle]": lambda: KeepLatestOracle(),
            "temporal-kg[oracle]": lambda: TemporalKG(OracleKeyer()),
        }
        out["mechanism"][f"alias_p={a}"] = {
            n: score_real(f(), inst_t, ev_t, qt)["summary"] for n, f in pairs.items()}

    # -- 4. full corpus: 4,939 non-queried fact keys as distractors --------- #
    dist = load_distractor_facts()
    dist_v, dist_t = split_facts(dist)
    # A keyer's inventory can be *open-world* (it knows only the facts it is
    # asked about, and every distractor entity is out-of-inventory) or
    # *closed-world* (it knows every subject in the stream). The contrast
    # separates two very different claims: entity resolution is hard, versus
    # your gazetteer is incomplete.
    gaz_closed = gaz_test + [f.subject for f in dist_t]
    amap_closed = build_alias_map(cls_test + subjects_with_class(dist_t))
    for a in (0.0, 1.0):
        inst_v = build_instance(val, paraphrase_p=_PARA, seed=0, alias_p=a,
                                distractors=dist_v)
        inst_t = build_instance(test, paraphrase_p=_PARA, seed=0, alias_p=a,
                                distractors=dist_t)
        ev_v = emb.encode([e[0] for e in inst_v.events], f"val_ev_full_a{a}")
        qv = emb.encode([q[0] for q in inst_v.queries], f"val_q_full_a{a}")
        ev_t = emb.encode([e[0] for e in inst_t.events], f"test_ev_full_a{a}")
        qt = emb.encode([q[0] for q in inst_t.queries], f"test_q_full_a{a}")
        tuned = _tune(inst_v, ev_v, qv, gaz_val)
        cell = {"tuned": tuned,
                "n_events": len(inst_t.events),
                "n_distractor_events": inst_t.n_distractor_events,
                "n_inventory": {"open_world": len(gaz_test),
                                "closed_world": len(gaz_closed)}}
        for world, (g, am) in {"open_world": (gaz_test, amap_test),
                               "closed_world": (gaz_closed, amap_closed)}.items():
            res = {n: score_real(f(), inst_t, ev_t, qt)
                   for n, f in _factories(tuned, g, am).items()}
            orc = res["keep-latest-oracle"]["summary"]["current@1"]
            cell[world] = {
                "test": {n: {"summary": r["summary"],
                             "ci": {m: bootstrap_ci([x[m] for x in r["records"]],
                                                    seed=1)
                                    for m in ("current@1", "contamination@5")}}
                         for n, r in res.items()},
                "keying_deficit": {n: orc - r["summary"]["current@1"]
                                   for n, r in res.items()}}
        out["full_corpus"][f"alias_p={a}"] = cell

    # -- 4b. inventory completeness as a second priced axis ----------------- #
    # The fuzzy and alias rungs are handed an inventory that exactly covers the
    # queried subjects with no out-of-vocabulary mass, which no deployed agent
    # has. Holding the corpus fixed and enlarging the inventory isolates how
    # much of the reported deficit is a property of that gift.
    out["inventory_scaling"] = {}
    dist_subjects = [f.subject for f in dist_t]
    for a in (0.0, 1.0):
        inst_t = build_instance(test, paraphrase_p=_PARA, seed=0, alias_p=a)
        ev_t = emb.encode([e[0] for e in inst_t.events], f"test_ev_p{_PARA}_a{a}")
        qt = emb.encode([q[0] for q in inst_t.queries], f"test_q_p{_PARA}_a{a}")
        tau = out["alias_sweep"][f"alias_p={a}"]["tuned"]["fuzzy-key"]["tau"]
        orc = out["alias_sweep"][f"alias_p={a}"]["test"]["keep-latest-oracle"][
            "summary"]["current@1"]
        cell = {}
        for n_extra in (0, 500, 1500, len(dist_subjects)):
            gaz = gaz_test + dist_subjects[:n_extra]
            r = score_real(FuzzyKey(gaz, tau), inst_t, ev_t, qt)
            # the paper's headline surface was measured on the write-path rung
            # while its headline fix is the symmetric one; measuring both on
            # the same axis is what makes the surface apply to the fix
            rw = score_real(ReadWriteFuzzyKey(gaz, tau), inst_t, ev_t, qt)
            cell[f"n={len(gaz)}"] = {
                "n_inventory": len(gaz),
                "current@1": r["summary"]["current@1"],
                "deficit": orc - r["summary"]["current@1"],
                "rw_current@1": rw["summary"]["current@1"],
                "rw_deficit": orc - rw["summary"]["current@1"],
                "rw_contamination@5": rw["summary"]["contamination@5"]}
        out["inventory_scaling"][f"alias_p={a}"] = cell

    # -- 5. encoder robustness: is the ladder a MiniLM artifact? ------------ #
    emb2 = Embedder(MPNET)
    out["mpnet"] = {}
    for a in (0.0, 1.0):
        inst_v = build_instance(val, paraphrase_p=_PARA, seed=0, alias_p=a)
        inst_t = build_instance(test, paraphrase_p=_PARA, seed=0, alias_p=a)
        ev_v = emb2.encode([e[0] for e in inst_v.events], f"val_ev_mpnet_a{a}")
        qv = emb2.encode([q[0] for q in inst_v.queries], f"val_q_mpnet_a{a}")
        ev_t = emb2.encode([e[0] for e in inst_t.events], f"test_ev_mpnet_a{a}")
        qt = emb2.encode([q[0] for q in inst_t.queries], f"test_q_mpnet_a{a}")
        tuned = _tune(inst_v, ev_v, qv, gaz_val)
        res = {n: score_real(f(), inst_t, ev_t, qt)
               for n, f in _factories(tuned, gaz_test, amap_test).items()}
        orc = res["keep-latest-oracle"]["summary"]["current@1"]
        out["mpnet"][f"alias_p={a}"] = {
            "tuned": tuned,
            "test": {n: r["summary"] for n, r in res.items()},
            "keying_deficit": {n: orc - r["summary"]["current@1"]
                               for n, r in res.items()}}
    return out


def main():
    res = run()
    os.makedirs(os.path.dirname(_OUT), exist_ok=True)
    json.dump(res, open(_OUT, "w"), indent=1)

    print("=" * 78)
    print("StaleBench anchor-free track (TempLAMA), test split, paraphrase p=0.5")
    print("=" * 78)
    inv = res["alias_inventory"]["test"]
    print(f"alias inventory: {inv['n_with_alias']}/{inv['n_subjects']} subjects "
          f"({inv['frac_with_alias']:.1%}) have >=1 alias; "
          f"{inv['n_ambiguous_forms']}/{inv['n_alias_forms']} forms ambiguous "
          f"({inv['frac_ambiguous_forms']:.1%}), touching "
          f"{inv['frac_subjects_touched_by_ambiguity']:.1%} of subjects")

    names = list(res["alias_sweep"]["alias_p=0.0"]["test"])
    print(f"\n[current@1 vs anchor availability]")
    print(f"{'policy':20s}" + "".join(f"  a={a:<6.2f}" for a in _ALIAS_LEVELS))
    for n in names:
        row = "".join(
            f"  {res['alias_sweep'][f'alias_p={a}']['test'][n]['summary']['current@1']:<8.3f}"
            for a in _ALIAS_LEVELS)
        print(f"{n:20s}{row}")
    print(f"\n[keying deficit vs oracle, current@1 points]")
    for n in names:
        if n in ("keep-latest-oracle", "cosine-rag"):
            continue
        row = "".join(
            f"  {res['alias_sweep'][f'alias_p={a}']['keying_deficit'][n]:<8.3f}"
            for a in _ALIAS_LEVELS)
        print(f"{n:20s}{row}")

    md = res.get("mention_detection")
    if md:
        print(f"\n[mention detection] of {md['n_distractor_statements']} distractor "
              f"statements, bare substring matching links "
              f"{md['substring_false_link_rate']:.1%} into a queried fact; "
              f"token-boundary mention matching links "
              f"{md['mention_false_link_rate']:.1%}")

    print("\n[keying audit: cluster errors, independent of retrieval]")
    for a, cell in res["keying_audit"].items():
        print(f"  {a}")
        for k, v in cell.items():
            print(f"    {k:16s} clusters {v['n_clusters']:4d}  impure "
                  f"{v['impure_clusters']:3d}  split facts {v['split_facts']:3d} "
                  f"({v['frac_split']:.1%})  unlinked {v['unlinked']:4d}")

    print("\n[mechanism: overwrite vs temporal-KG invalidation]")
    for a, cell in res["mechanism"].items():
        print(f"  {a}")
        for n, s in cell.items():
            print(f"    {n:22s} cur@1 {s['current@1']:.3f}  contam@5 "
                  f"{s['contamination@5']:.3f}  slots {s['slots']:.0f} "
                  f"(active {s['active_slots']:.0f})")

    print("\n[full corpus: +4,939 non-queried fact keys as distractors]")
    for a, cell in res["full_corpus"].items():
        print(f"  {a}  ({cell['n_events']} events, "
              f"{cell['n_distractor_events']} distractor; inventory "
              f"{cell['n_inventory']['open_world']} open / "
              f"{cell['n_inventory']['closed_world']} closed)")
        for world in ("open_world", "closed_world"):
            print(f"    [{world}]")
            for n, r in cell[world]["test"].items():
                s = r["summary"]
                print(f"      {n:20s} cur@1 {s['current@1']:.3f}  stale "
                      f"{s['stale_rate']:.3f}  miss {s['miss_rate']:.3f}  contam@5 "
                      f"{s['contamination@5']:.3f}  slots {s['slots']:.0f}")
    print("\n[inventory scaling: fuzzy rung, corpus fixed]")
    for a, cell in res.get("inventory_scaling", {}).items():
        row = "  ".join(f"n={v['n_inventory']}:{v['current@1']:.3f}"
                        f"(-{100 * v['deficit']:.1f})" for v in cell.values())
        print(f"  {a}  {row}")

    print("\n[encoder robustness: mpnet vs MiniLM]  current@1 (deficit vs oracle)")
    for a, cell in res.get("mpnet", {}).items():
        print(f"  {a}")
        for n, s2 in cell["test"].items():
            print(f"    {n:20s} {s2['current@1']:.3f} "
                  f"({100 * cell['keying_deficit'][n]:+.1f})")
    print(f"\nWrote {os.path.abspath(_OUT)}")


if __name__ == "__main__":
    main()
