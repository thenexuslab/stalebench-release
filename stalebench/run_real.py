"""StaleBench real-data track: TempLAMA updating facts.

Protocol:

* Facts are split val/test by key hash. **All hyperparameters (tau, k, lam,
  decay) are tuned on val only**, by current@1; test numbers are reported with
  bootstrap 95% CIs over queries and paired sign-flip permutation tests
  (Holm-corrected) against the strongest store-all baseline.
* ``paraphrase_p`` sweeps {0, 0.5, 1}: the surface-variation stress on
  similarity-based same-key detection. Headline table at p=0.5.
* Encoders: MiniLM everywhere; mpnet repeated at p=0.5 as an
  encoder-robustness check.

Review-board additions (all at p=0.5 on the test split unless noted):

* ``lexical-key`` — gazetteer/NER-idealized string keyer, the second tier of
  the keying ladder (embedding-threshold -> lexical -> true-key oracle).
* ``stratified`` — current@1 by per-fact version count (2 vs >=3).
* ``t_shuffle`` — timestamps randomly permuted across events (embeddings and
  arrival order unchanged): the regime where logged time metadata is
  unreliable. Time-aware fixes should collapse; retention never reads t for
  decisions and should be unchanged.
* ``recon_audit`` — reconsolidation's destructive-failure accounting:
  overwrites, cross-fact overwrites, residual duplicate slots, facts left
  with no slot.
* ``template_seed_var`` — current@1 across 3 template-assignment realizations
  (build_instance seeds 0/1/2) per p, hyperparameters fixed from the seed-0
  val tuning.
* ``query_paraphrase`` — queries sampled from all question templates instead
  of the canonical one.

Run:  cd stalebench && python -m stalebench.run_real
"""

from __future__ import annotations

import itertools
import json
import os
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np

from stalebench.embed import Embedder, MINILM, MPNET
from stalebench.metrics import (bootstrap_ci, holm, paired_permutation,
                                score_ranking)
from stalebench.policies import (BM25RAG, CosineRAG, FuzzyKey,
                                 KeepLatestOracle, LexicalKey, LexicalKeyer,
                                 NewerOnlyLexicalKey, NewerOnlyOracle,
                                 ReadWriteFuzzyKey, Reconsolidation,
                                 RecencyWeighted, TemporalKG, TimeRerank)
from stalebench.templama import (RealInstance, build_instance, load_facts,
                                 split_facts)

_HERE = os.path.dirname(__file__)
_OUT = os.path.join(_HERE, "..", "results", "stalebench_real_results.json")

_KS = (5,)


def score_real(policy, inst: RealInstance, emb_events: np.ndarray,
               emb_queries: np.ndarray) -> Dict:
    policy.reset()
    for (text, iid, t), e in zip(inst.events, emb_events):
        policy.observe(e, iid, t, text)
    records = []
    for (qtext, key, cur), qe in zip(inst.queries, emb_queries):
        records.append(score_ranking(policy.rank(qe, qtext), key, cur, _KS))
    summary = {m: float(np.mean([r[m] for r in records])) for m in records[0]}
    summary["slots"] = float(policy.slots)
    return {"summary": summary, "records": records}


def _factories(tuned: Dict[str, Dict], gazetteer: List[str]) -> Dict[str, callable]:
    """Policy factories given tuned hyperparameters and the split's gazetteer."""
    return {
        "bm25": lambda: BM25RAG(),
        "bm25+time-rerank": lambda: BM25RAG(rerank_k=tuned["bm25+time-rerank"]["k"]),
        "cosine-rag": lambda: CosineRAG(),
        "time-rerank": lambda: TimeRerank(k=tuned["time-rerank"]["k"]),
        "recency-weighted": lambda: RecencyWeighted(
            lam=tuned["recency-weighted"]["lam"],
            decay=tuned["recency-weighted"]["decay"]),
        "reconsolidation": lambda: Reconsolidation(tau=tuned["reconsolidation"]["tau"]),
        "lexical-key": lambda: LexicalKey(gazetteer),
        "keep-latest-oracle": lambda: KeepLatestOracle(),
    }


def tune(inst: RealInstance, emb_events: np.ndarray,
         emb_queries: np.ndarray,
         gazetteer: Optional[List[str]] = None) -> Dict[str, Dict]:
    """Grid-search each tunable policy on the val split by current@1."""
    def c1(policy) -> float:
        return score_real(policy, inst, emb_events, emb_queries)["summary"]["current@1"]

    tuned: Dict[str, Dict] = {}
    ks = [2, 3, 5, 10, 20]
    tuned["time-rerank"] = {"k": max(ks, key=lambda k: c1(TimeRerank(k=k)))}
    tuned["bm25+time-rerank"] = {"k": max(ks, key=lambda k: c1(BM25RAG(rerank_k=k)))}
    lam_grid = [0.1, 0.3, 0.6, 1.0]
    decay_grid = [0.8, 0.9, 0.99]
    lam, decay = max(itertools.product(lam_grid, decay_grid),
                     key=lambda ld: c1(RecencyWeighted(lam=ld[0], decay=ld[1])))
    tuned["recency-weighted"] = {"lam": lam, "decay": decay}
    taus = [round(float(x), 2) for x in np.arange(0.50, 0.91, 0.05)]
    tuned["reconsolidation"] = {"tau": max(taus, key=lambda t: c1(Reconsolidation(tau=t)))}
    if gazetteer is not None:
        # the corruption grid carries the fuzzy rungs; same grid as run_systems
        ftaus = [round(float(x), 2) for x in np.arange(0.20, 0.66, 0.05)]
        tuned["fuzzy-key"] = {"tau": max(ftaus,
                                         key=lambda t: c1(FuzzyKey(gazetteer, t)))}
    return tuned


def _shuffle_t(inst: RealInstance, seed: int = 7) -> RealInstance:
    """Permute timestamps across events; text, embeddings, arrival order fixed."""
    rng = np.random.default_rng(seed)
    ts = [t for _, _, t in inst.events]
    perm = rng.permutation(len(ts))
    events = [(text, iid, float(ts[perm[i]]))
              for i, (text, iid, _) in enumerate(inst.events)]
    return RealInstance(events, inst.queries, inst.n_facts)


# --------------------------------------------------------------------------- #
# The two temporal signals, corrupted separately and together
# --------------------------------------------------------------------------- #
# ``_shuffle_t`` corrupts the clock and keeps arrival order, which is the
# signal overwrite retention actually reads (last write wins; a keyer never
# looks at t). So that regime shows retention "immune" only because the signal
# it depends on was left intact. The regimes below corrupt arrival order with
# the clock intact, then both, and add a dose of realistic partial disorder.

_CORR_SEEDS = (7, 8, 9)
_LATE_Q = (0.1, 0.25, 0.5)


def _permute_order(inst: RealInstance, emb_events: np.ndarray,
                   seed: int) -> Tuple[RealInstance, np.ndarray]:
    """Permute arrival order; each statement keeps its own timestamp.

    The mirror image of :func:`_shuffle_t`: delayed or out-of-order
    ingestion. Embeddings travel with their statements.
    """
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(inst.events))
    inst2 = RealInstance([inst.events[i] for i in perm], inst.queries, inst.n_facts)
    return inst2, emb_events[perm]


def _late_report(inst: RealInstance, emb_events: np.ndarray, q: float,
                 seed: int) -> Tuple[RealInstance, np.ndarray, int, int]:
    """For a fraction ``q`` of facts the final version arrives *before* its
    predecessor: a stale report landing after the current one. Only the two
    arrival positions swap; timestamps travel with their statements."""
    rng = np.random.default_rng(seed)
    pos: Dict[str, List[int]] = {}
    for i, (_, iid, _) in enumerate(inst.events):
        pos.setdefault(iid.rsplit(":", 1)[0], []).append(i)
    keys = sorted(k for k, v in pos.items() if len(v) >= 2)
    n = int(round(q * len(keys)))
    order = list(range(len(inst.events)))
    for k in (rng.choice(keys, size=n, replace=False) if n else []):
        a, b = pos[k][-2], pos[k][-1]
        order[a], order[b] = order[b], order[a]
    perm = np.asarray(order)
    inst2 = RealInstance([inst.events[i] for i in perm], inst.queries, inst.n_facts)
    return inst2, emb_events[perm], n, len(keys)


def _corruption_factories(tuned: Dict[str, Dict],
                          gazetteer: List[str]) -> Dict[str, callable]:
    """The Table 1 policies plus the rungs whose temporal signal differs.

    Invalidation (temporal-KG) supersedes by arrival like overwrite does; the
    read-path rung retains everything and sorts the resolved key's slots by
    ``t``; the symmetric rung does both; the newer-only variants overwrite by
    ``t`` instead of by arrival. Between them the grid can attribute every
    collapse to one signal.
    """
    f = _factories(tuned, gazetteer)
    ft = tuned["fuzzy-key"]["tau"]
    f.update({
        "temporal-kg": lambda: TemporalKG(LexicalKeyer(gazetteer), name="temporal-kg"),
        "fuzzy-read-key": lambda: ReadWriteFuzzyKey(gazetteer, ft, write_keys=False),
        "fuzzy-rw-key": lambda: ReadWriteFuzzyKey(gazetteer, ft),
        "lexical-key-newer": lambda: NewerOnlyLexicalKey(gazetteer),
        "keep-newest-oracle": lambda: NewerOnlyOracle(),
    })
    return f


def _mean_over_seeds(runs: List[Dict[str, Dict]]) -> Dict[str, Dict]:
    return {pn: {m: float(np.mean([r[pn][m] for r in runs])) for m in runs[0][pn]}
            for pn in runs[0]}


def _corruption_grid(inst: RealInstance, emb_events: np.ndarray,
                     emb_queries: np.ndarray, tuned: Dict[str, Dict],
                     gazetteer: List[str]) -> Dict:
    facs = _corruption_factories(tuned, gazetteer)

    def run_all(inst2, ev2) -> Dict[str, Dict]:
        return {n: score_real(f(), inst2, ev2, emb_queries)["summary"]
                for n, f in facs.items()}

    def both(seed):
        inst2, ev2 = _permute_order(inst, emb_events, seed)
        return _shuffle_t(inst2, seed + 100), ev2

    regimes = {
        "clock": lambda s: (_shuffle_t(inst, s), emb_events),
        "arrival": lambda s: _permute_order(inst, emb_events, s),
        "both": both,
    }
    out: Dict = {"seeds": list(_CORR_SEEDS),
                 "regimes": {"clean": {"mean": run_all(inst, emb_events)}},
                 "late_report": {}}
    for name, build in regimes.items():
        runs = [run_all(*build(s)) for s in _CORR_SEEDS]
        out["regimes"][name] = {"mean": _mean_over_seeds(runs), "per_seed": runs}
    for q in _LATE_Q:
        runs = []
        for s in _CORR_SEEDS:
            inst2, ev2, n_swapped, n_eligible = _late_report(inst, emb_events, q, s)
            runs.append(run_all(inst2, ev2))
        out["late_report"][f"q={q}"] = {
            "mean": _mean_over_seeds(runs), "per_seed": runs,
            "n_facts_swapped": n_swapped, "n_facts_eligible": n_eligible}
    return out


def _audit_recon(inst: RealInstance, emb_events: np.ndarray, tau: float) -> Dict:
    """Replay reconsolidation, accounting for its destructive failures."""
    ids: List[str] = []
    embs: List[np.ndarray] = []
    overwrites = cross = 0
    for (text, iid, t), e in zip(inst.events, emb_events):
        if embs:
            sims = np.stack(embs) @ e
            j = int(np.argmax(sims))
            if sims[j] > tau:
                overwrites += 1
                if ids[j].split(":")[0] != iid.split(":")[0]:
                    cross += 1
                ids[j] = iid; embs[j] = e
                continue
        ids.append(iid); embs.append(e)
    per_fact = Counter(i.split(":")[0] for i in ids)
    all_facts = {iid.split(":")[0] for _, iid, _ in inst.events}
    return {"overwrites": overwrites,
            "cross_fact_overwrites": cross,
            "n_events": len(inst.events),
            "residual_duplicate_slots": sum(c - 1 for c in per_fact.values() if c > 1),
            "facts_zero_slot": len(all_facts - set(per_fact)),
            "slots": len(ids)}


def _stratify(results: Dict[str, Dict], facts, metric: str = "current@1") -> Dict:
    """current@1 by per-fact version count (2 vs >=3 runs)."""
    n_runs = [len(f.runs) for f in facts]
    out: Dict[str, Dict] = {}
    for name, r in results.items():
        two = [rec[metric] for rec, n in zip(r["records"], n_runs) if n == 2]
        more = [rec[metric] for rec, n in zip(r["records"], n_runs) if n >= 3]
        out[name] = {"2ver": {"n": len(two), "mean": float(np.mean(two))},
                     "3plus": {"n": len(more), "mean": float(np.mean(more))}}
    return out


def run() -> Dict:
    facts, stats = load_facts()
    val, test = split_facts(facts)
    stats.update({"val_facts": len(val), "test_facts": len(test)})
    out: Dict = {"stats": stats, "by_p": {}}
    gaz_val = [f.subject for f in val]
    gaz_test = [f.subject for f in test]

    emb = Embedder(MINILM)
    tuned_by_p: Dict[float, Dict] = {}
    for p in (0.0, 0.5, 1.0):
        cell: Dict = {}
        inst_v = build_instance(val, paraphrase_p=p, seed=0)
        inst_t = build_instance(test, paraphrase_p=p, seed=0)
        ev_v = emb.encode([e[0] for e in inst_v.events], f"val_ev_p{p}")
        qv = emb.encode([q[0] for q in inst_v.queries], f"val_q_p{p}")
        ev_t = emb.encode([e[0] for e in inst_t.events], f"test_ev_p{p}")
        qt = emb.encode([q[0] for q in inst_t.queries], f"test_q_p{p}")

        tuned = tune(inst_v, ev_v, qv, gaz_val)
        tuned_by_p[p] = tuned
        cell["tuned"] = tuned

        results = {name: score_real(fac(), inst_t, ev_t, qt)
                   for name, fac in _factories(tuned, gaz_test).items()}
        cell["test"] = {
            name: {"summary": r["summary"],
                   "ci": {m: bootstrap_ci([rec[m] for rec in r["records"]], seed=1)
                          for m in r["records"][0]}}
            for name, r in results.items()}

        # paired tests: reconsolidation / rerank vs the strongest store-all.
        # The comparator is chosen on VAL. Selecting it on test (as an earlier
        # revision did) makes the "pre-specified" claim false, even though the
        # choice happens to be conservative against our own policies.
        store_all = ["bm25", "bm25+time-rerank", "cosine-rag", "time-rerank",
                     "recency-weighted"]
        val_results = {name: score_real(fac(), inst_v, ev_v, qv)
                       for name, fac in _factories(tuned, gaz_val).items()
                       if name in store_all}
        best_sa = max(store_all,
                      key=lambda n: val_results[n]["summary"]["current@1"])
        cell["best_store_all"] = best_sa
        cell["best_store_all_selected_on"] = "val"

        def pp(a: str, b: str, metric: str = "current@1") -> float:
            return paired_permutation([r[metric] for r in results[a]["records"]],
                                      [r[metric] for r in results[b]["records"]],
                                      seed=2)
        raw = {
            f"reconsolidation_vs_{best_sa}": pp("reconsolidation", best_sa),
            "time-rerank_vs_cosine-rag": pp("time-rerank", "cosine-rag"),
            "oracle_vs_reconsolidation": pp("keep-latest-oracle", "reconsolidation"),
            f"recon_vs_{best_sa}_contam": pp("reconsolidation", best_sa,
                                            "contamination@5"),
        }
        cell["paired_p"] = {"raw": raw, "holm": holm(raw)}

        if p == 0.5:
            cell["stratified"] = _stratify(results, test)
            cell["recon_audit"] = _audit_recon(inst_t, ev_t,
                                              tuned["reconsolidation"]["tau"])
            # timestamps corrupted: same embeddings, permuted t
            inst_sh = _shuffle_t(inst_t)
            cell["t_shuffle"] = {
                name: score_real(fac(), inst_sh, ev_t, qt)["summary"]
                for name, fac in _factories(tuned, gaz_test).items()}
            # query-template robustness: sample all question paraphrases
            inst_qp = build_instance(test, paraphrase_p=p, seed=0,
                                     query_template_seed=1)
            qt_qp = emb.encode([q[0] for q in inst_qp.queries], "test_q_p0.5_qp1")
            cell["query_paraphrase"] = {
                name: score_real(fac(), inst_qp, ev_t, qt_qp)["summary"]
                for name, fac in _factories(tuned, gaz_test).items()
                if name in ("cosine-rag", "time-rerank", "reconsolidation",
                            "lexical-key", "keep-latest-oracle")}
            # clock x arrival corruption, and the late-stale-report dose
            cell["corruption"] = _corruption_grid(inst_t, ev_t, qt, tuned, gaz_test)
        out["by_p"][f"p={p}"] = cell

    # template-assignment variance: 3 realizations, hyperparameters fixed
    seed_var: Dict = {}
    for p in (0.0, 0.5, 1.0):
        per_pol: Dict[str, List[float]] = {}
        for s in (0, 1, 2):
            inst_s = build_instance(test, paraphrase_p=p, seed=s)
            ev_s = emb.encode([e[0] for e in inst_s.events], f"test_ev_p{p}_s{s}")
            q_s = emb.encode([q[0] for q in inst_s.queries], f"test_q_p{p}_s{s}")
            for name, fac in _factories(tuned_by_p[p], gaz_test).items():
                if name in ("cosine-rag", "time-rerank", "reconsolidation",
                            "lexical-key", "keep-latest-oracle"):
                    c1 = score_real(fac(), inst_s, ev_s, q_s)["summary"]["current@1"]
                    per_pol.setdefault(name, []).append(c1)
        seed_var[f"p={p}"] = per_pol
    out["template_seed_var"] = seed_var

    # encoder robustness: mpnet at p=0.5, tuned on val with mpnet embeddings
    emb2 = Embedder(MPNET)
    inst_v = build_instance(val, paraphrase_p=0.5, seed=0)
    inst_t = build_instance(test, paraphrase_p=0.5, seed=0)
    ev_v = emb2.encode([e[0] for e in inst_v.events], "val_ev_p0.5")
    qv = emb2.encode([q[0] for q in inst_v.queries], "val_q_p0.5")
    ev_t = emb2.encode([e[0] for e in inst_t.events], "test_ev_p0.5")
    qt = emb2.encode([q[0] for q in inst_t.queries], "test_q_p0.5")
    tuned2 = tune(inst_v, ev_v, qv)
    out["mpnet_p0.5"] = {
        "tuned": tuned2,
        "test": {name: score_real(fac(), inst_t, ev_t, qt)["summary"]
                 for name, fac in _factories(tuned2, gaz_test).items()
                 if name not in ("bm25", "bm25+time-rerank")}}
    return out


def _fmt(x: Dict) -> str:
    return f"{x['mean']:.3f} [{x['lo']:.3f},{x['hi']:.3f}]"


def main():
    res = run()
    os.makedirs(os.path.dirname(_OUT), exist_ok=True)
    json.dump(res, open(_OUT, "w"), indent=1)

    print("=" * 78)
    print("StaleBench real-data track (TempLAMA), test split")
    print("=" * 78)
    st = res["stats"]
    print(f"facts: {st['fact_keys']} keys -> {st['single_valued']} single-valued -> "
          f"{st['single_valued_changing']} changing; val/test = "
          f"{st['val_facts']}/{st['test_facts']}")
    for p_key, cell in res["by_p"].items():
        print(f"\n[{p_key}]  tuned: {cell['tuned']}")
        print(f"{'policy':20s}  cur@1              cur@5  stale  contam@5  slots")
        for name, r in cell["test"].items():
            s, ci = r["summary"], r["ci"]
            print(f"{name:20s}  {_fmt(ci['current@1']):18s} {s['current@5']:.3f}  "
                  f"{s['stale_rate']:.3f}  {s['contamination@5']:.3f}     "
                  f"{s['slots']:.0f}")
        print(f"best store-all: {cell['best_store_all']}; "
              f"paired p (Holm): {cell['paired_p']['holm']}")

    c5 = res["by_p"]["p=0.5"]
    print("\n[stratified by version count, p=0.5]  current@1 (2ver n=%d / 3+ n=%d)"
          % (c5["stratified"]["cosine-rag"]["2ver"]["n"],
             c5["stratified"]["cosine-rag"]["3plus"]["n"]))
    for name, s in c5["stratified"].items():
        print(f"{name:20s}  {s['2ver']['mean']:.3f} / {s['3plus']['mean']:.3f}")
    print("\n[t-shuffle, p=0.5]  current@1 clean -> corrupted-timestamps")
    for name, s in c5["t_shuffle"].items():
        clean = c5["test"][name]["summary"]["current@1"]
        print(f"{name:20s}  {clean:.3f} -> {s['current@1']:.3f}")
    corr = c5["corruption"]
    print(f"\n[corruption grid, p=0.5]  current@1: clean / clock / arrival / both"
          f"  (mean over seeds {corr['seeds']})")
    for name in corr["regimes"]["clean"]["mean"]:
        row = "  ".join(f"{corr['regimes'][r]['mean'][name]['current@1']:.3f}"
                        for r in ("clean", "clock", "arrival", "both"))
        print(f"{name:20s}  {row}")
    print("\n[late stale report, p=0.5]  current@1 by q")
    for name in corr["regimes"]["clean"]["mean"]:
        row = "  ".join(f"{corr['late_report'][k]['mean'][name]['current@1']:.3f}"
                        for k in corr["late_report"])
        print(f"{name:20s}  {row}")
    print("\n[recon audit, p=0.5]", c5["recon_audit"])
    print("\n[query-paraphrase, p=0.5]  current@1")
    for name, s in c5["query_paraphrase"].items():
        print(f"{name:20s}  {s['current@1']:.3f}")
    print("\n[template-seed variance]  current@1 across seeds 0/1/2")
    for p_key, pols in res["template_seed_var"].items():
        row = "  ".join(f"{n}:{min(v):.3f}-{max(v):.3f}" for n, v in pols.items())
        print(f"{p_key}: {row}")
    print("\n[mpnet, p=0.5]")
    for name, s in res["mpnet_p0.5"]["test"].items():
        print(f"{name:20s}  {s['current@1']:.3f}  stale {s['stale_rate']:.3f}  "
              f"contam@5 {s['contamination@5']:.3f}")
    print(f"\nWrote {os.path.abspath(_OUT)}")


if __name__ == "__main__":
    main()
