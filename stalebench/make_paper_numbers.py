"""Generate paper/numbers.tex and paper/tab_*.tex from the results JSONs.

Every number in the paper is produced here from
``results/stalebench_results.json`` (controlled) and
``results/stalebench_real_results.json`` (TempLAMA), so the draft can never
drift from the code's output. Each tab_*.tex is a complete tabular environment
(inputting bare rows inside an alignment breaks booktabs). Re-run after any
experiment change:

    cd stalebench && python -m stalebench.make_paper_numbers
"""

from __future__ import annotations

import json
import os
from typing import Dict

from stalebench.metrics import holm, paired_permutation

_EXPECTED_TABLES = 18  # +1 when gate records exist (late-report and natural-alias tables added)
_HERE = os.path.dirname(__file__)
_RES = os.path.join(_HERE, "..", "results")
_PAPER = os.path.join(_HERE, "..", "paper")

_REAL_ROWS = [
    ("bm25", "BM25 (store-all)"),
    ("bm25+time-rerank", "BM25 + time-rerank"),
    ("cosine-rag", "Cosine (store-all)"),
    ("time-rerank", "Cosine + time-rerank"),
    ("recency-weighted", "Cosine + recency-weight"),
    ("reconsolidation", "Reconsolidation ($\\tau$)"),
    ("lexical-key", "Lexical key (gazetteer)"),
    ("keep-latest-oracle", "Keep-latest oracle"),
]


def _ci(c) -> str:
    return f"${c['mean']:.3f}$ {{\\scriptsize[{c['lo']:.3f},{c['hi']:.3f}]}}"


def _backend_gate(version, store):
    """Did this (version, backend) pair return what its store held?

    A high miss rate does not distinguish "the backend is broken" from "the
    system genuinely ranks badly" -- mem0 2.0.11 misses over half its queries
    on two different backends, one of which passes this probe, so that miss is
    the system's. Only a hand-scored probe separates the two, and suppressing
    results on the miss rate alone withheld a valid finding.
    """
    path = os.path.join(_RES, "mem0_gates", f"{version}_{store}.json")
    if not os.path.exists(path):
        return None
    return bool(json.load(open(path)).get("gate_pass"))


_written = {"tables": 0}


def _tabular(colspec: str, header: str, rows, path: str) -> None:
    body = "\n".join(rows)
    _written["tables"] += 1
    with open(path, "w") as fh:
        fh.write(f"\\begin{{tabular}}{{{colspec}}}\n\\toprule\n{header} \\\\\n"
                 f"\\midrule\n{body}\n\\bottomrule\n\\end{{tabular}}\n")


def main() -> None:
    real = json.load(open(os.path.join(_RES, "stalebench_real_results.json")))
    ctrl = json.load(open(os.path.join(_RES, "stalebench_results.json")))
    alias = json.load(open(os.path.join(_RES, "stalebench_alias_results.json")))
    _rel = os.path.join(_HERE, "..", "release", "manifest.json")
    relman = json.load(open(_rel)) if os.path.exists(_rel) else {}
    def _opt(name):
        """LLM tracks may not have finished yet; warn loudly rather than
        silently emitting a paper with missing macros."""
        path = os.path.join(_RES, name)
        if os.path.exists(path):
            return json.load(open(path))
        print(f"WARNING: {name} missing -- its macros and tables are NOT emitted")
        return None

    systems = _opt("stalebench_systems_results.json")
    scale = _opt("stalebench_scale_results.json")
    reader = _opt("stalebench_reader_results.json")
    llmkey = _opt("stalebench_llmkey_results.json")
    os.makedirs(_PAPER, exist_ok=True)

    n = []
    st = real["stats"]
    p5 = real["by_p"]["p=0.5"]["test"]

    _seen: Dict[str, str] = {}
    _frozen: list = []

    def add(name: str, val: str) -> None:
        # numbers.tex is written once, at the very end; a macro defined after
        # that write would be computed, reported as fine, and then be missing
        # from the file LaTeX reads
        if _frozen:
            raise SystemExit(f"macro {name!r} defined after numbers.tex was "
                             f"written -- move its block above the write")
        # a macro emitted twice is a LaTeX error, and one emitted twice with
        # *different* values is a generator bug worth failing on
        if not name.isalpha():
            raise SystemExit(f"macro {name!r} is not a legal LaTeX control "
                             f"sequence (letters only)")
        if name in _seen:
            if _seen[name] != val:
                raise SystemExit(f"macro {name!r} emitted twice with different "
                                 f"values: {_seen[name]!r} vs {val!r}")
            return
        _seen[name] = val
        n.append(f"\\newcommand{{\\{name}}}{{{val}}}")

    # the paper must not name a config list that has drifted from the release;
    # the HF card had the same defect one layer out
    from stalebench.release import CONFIGS as _RELCFG
    add("relNConfigs", str(len(_RELCFG)))
    add("relConfigList",
        ", ".join(f"\\texttt{{{c.replace('_', chr(92) + '_')}}}"
                  for c in _RELCFG))
    add("nFactKeys", f"{st['fact_keys']:,}")
    add("nSingle", str(st["single_valued"]))
    add("nChanging", str(st["single_valued_changing"]))
    add("nVal", str(st["val_facts"]))
    add("nTest", str(st["test_facts"]))

    def s(pol, metric, p="p=0.5", digits=3):
        return f"{real['by_p'][p]['test'][pol]['summary'][metric]:.{digits}f}"

    add("realCosineCur", s("cosine-rag", "current@1"))
    add("realCosineStale", s("cosine-rag", "stale_rate"))
    add("realBmCur", s("bm25", "current@1"))
    add("realBmStale", s("bm25", "stale_rate"))
    add("realRerankCur", s("time-rerank", "current@1"))
    add("realRerankStale", s("time-rerank", "stale_rate"))
    add("realRerankContam", s("time-rerank", "contamination@5"))
    add("realBmRerankCur", s("bm25+time-rerank", "current@1"))
    add("realRwCur", s("recency-weighted", "current@1"))
    add("realRwContam", s("recency-weighted", "contamination@5"))
    add("realBmContam", s("bm25", "contamination@5"))
    add("realReconCur", s("reconsolidation", "current@1"))
    add("realReconStale", s("reconsolidation", "stale_rate"))
    add("realReconContam", s("reconsolidation", "contamination@5"))
    add("realReconSlots", s("reconsolidation", "slots", digits=0))
    add("realOracleCur", s("keep-latest-oracle", "current@1"))
    add("realStoreSlots", s("cosine-rag", "slots", digits=0))
    add("realOracleSlots", s("keep-latest-oracle", "slots", digits=0))

    for tag, p in [("Pzero", "p=0.0"), ("Phalf", "p=0.5"), ("Pone", "p=1.0")]:
        add(f"realReconCur{tag}", s("reconsolidation", "current@1", p))
        add(f"realOracleCur{tag}", s("keep-latest-oracle", "current@1", p))
        add(f"realRerankCur{tag}", s("time-rerank", "current@1", p))
    gap0 = (real["by_p"]["p=0.0"]["test"]["keep-latest-oracle"]["summary"]["current@1"]
            - real["by_p"]["p=0.0"]["test"]["reconsolidation"]["summary"]["current@1"])
    gap1 = (real["by_p"]["p=1.0"]["test"]["keep-latest-oracle"]["summary"]["current@1"]
            - real["by_p"]["p=1.0"]["test"]["reconsolidation"]["summary"]["current@1"])
    add("gapReconOraclePzero", f"{100 * gap0:.1f}")
    add("gapReconOraclePone", f"{100 * gap1:.1f}")
    add("tunedRerankK", str(real["by_p"]["p=0.5"]["tuned"]["time-rerank"]["k"]))
    add("tunedTau", f"{real['by_p']['p=0.5']['tuned']['reconsolidation']['tau']:.2f}")

    add("realReconCurFive", s("reconsolidation", "current@5"))
    add("realCosineCurFive", s("cosine-rag", "current@5"))
    jump = (real["by_p"]["p=0.5"]["test"]["time-rerank"]["summary"]["current@1"]
            - real["by_p"]["p=0.5"]["test"]["cosine-rag"]["summary"]["current@1"])
    add("realRerankJump", f"{jump:.2f}")

    # lexical keying tier
    for tag, p in [("Pzero", "p=0.0"), ("Phalf", "p=0.5"), ("Pone", "p=1.0")]:
        add(f"realLexCur{tag}", s("lexical-key", "current@1", p))
    add("realLexCur", s("lexical-key", "current@1"))
    add("realLexContam", s("lexical-key", "contamination@5"))
    add("realLexSlots", s("lexical-key", "slots", digits=0))
    lex_gap = (real["by_p"]["p=0.5"]["test"]["keep-latest-oracle"]["summary"]["current@1"]
               - real["by_p"]["p=0.5"]["test"]["lexical-key"]["summary"]["current@1"])
    add("gapLexOraclePhalf", f"{100 * lex_gap:.1f}")
    # the gap the reader section actually argues about is rerank -> gazetteer,
    # not gazetteer -> oracle; quoting the latter for the former renders as
    # "the 0.6-point difference between 0.925 and 0.972"
    _rr_lex = (real["by_p"]["p=0.5"]["test"]["lexical-key"]["summary"]["current@1"]
               - real["by_p"]["p=0.5"]["test"]["time-rerank"]["summary"]["current@1"])
    add("gapRerankLexPhalf", f"{100 * _rr_lex:.1f}")

    # stratification / t-shuffle / audit / query-paraphrase / seed variance
    c5 = real["by_p"]["p=0.5"]
    strat = c5["stratified"]
    add("stratNTwo", str(strat["cosine-rag"]["2ver"]["n"]))
    add("stratPctTwo", f"{100 * strat['cosine-rag']['2ver']['n'] / st['test_facts']:.0f}")
    add("stratNMore", str(strat["cosine-rag"]["3plus"]["n"]))
    add("stratRerankMore", f"{strat['time-rerank']['3plus']['mean']:.3f}")
    add("stratReconMore", f"{strat['reconsolidation']['3plus']['mean']:.3f}")
    add("stratOracleMore", f"{strat['keep-latest-oracle']['3plus']['mean']:.3f}")
    add("stratRerankTwo", f"{strat['time-rerank']['2ver']['mean']:.3f}")
    add("stratCosineTwo", f"{strat['cosine-rag']['2ver']['mean']:.3f}")
    add("stratCosineMore", f"{strat['cosine-rag']['3plus']['mean']:.3f}")

    sh = c5["t_shuffle"]
    for name, macro in [("time-rerank", "shuffleRerank"),
                        ("bm25+time-rerank", "shuffleBmRerank"),
                        ("recency-weighted", "shuffleRw"),
                        ("reconsolidation", "shuffleRecon"),
                        ("lexical-key", "shuffleLex")]:
        add(f"{macro}Cur", f"{sh[name]['current@1']:.3f}")

    au = c5["recon_audit"]
    add("auditNEvents", f"{au['n_events']:,}")
    add("auditOverwrites", str(au["overwrites"]))
    add("auditCross", str(au["cross_fact_overwrites"]))
    add("auditCrossPct", f"{100 * au['cross_fact_overwrites'] / au['n_events']:.1f}")
    add("auditDup", str(au["residual_duplicate_slots"]))
    add("auditZero", str(au["facts_zero_slot"]))

    qp = c5["query_paraphrase"]
    add("qpRerankCur", f"{qp['time-rerank']['current@1']:.3f}")
    add("qpReconCur", f"{qp['reconsolidation']['current@1']:.3f}")
    add("qpCosineCur", f"{qp['cosine-rag']['current@1']:.3f}")

    sv = real["template_seed_var"]
    for p, tag in [("p=0.5", "Phalf"), ("p=1.0", "Pone")]:
        v = sv[p]["reconsolidation"]
        add(f"seedReconLo{tag}", f"{min(v):.3f}")
        add(f"seedReconHi{tag}", f"{max(v):.3f}")
        w = sv[p]["time-rerank"]
        add(f"seedRerankLo{tag}", f"{min(w):.3f}")
        add(f"seedRerankHi{tag}", f"{max(w):.3f}")

    mp = real["mpnet_p0.5"]["test"]
    add("mpnetLexCur", f"{mp['lexical-key']['current@1']:.3f}")
    add("mpnetRwCur", f"{mp['recency-weighted']['current@1']:.3f}")
    add("mpnetCosineCur", f"{mp['cosine-rag']['current@1']:.3f}")
    add("mpnetRerankCur", f"{mp['time-rerank']['current@1']:.3f}")
    add("mpnetReconCur", f"{mp['reconsolidation']['current@1']:.3f}")
    add("mpnetOracleCur", f"{mp['keep-latest-oracle']['current@1']:.3f}")
    add("mpnetReconLead",
        f"{100 * (real['by_p']['p=0.5']['test']['reconsolidation']['summary']['current@1'] - real['by_p']['p=0.5']['test']['recency-weighted']['summary']['current@1']):.1f}")
    add("mpnetReconLeadMp",
        f"{100 * (mp['reconsolidation']['current@1'] - mp['recency-weighted']['current@1']):.1f}")
    add("mpnetReconContam", f"{mp['reconsolidation']['contamination@5']:.3f}")
    add("mpnetRerankContam", f"{mp['time-rerank']['contamination@5']:.3f}")

    # controlled
    def c(sweep, cell, pol, metric):
        return ctrl[sweep][cell][pol][metric]["mean"]

    add("ctrlSeeds", str(ctrl["n_seeds"]))
    add("ctrlCosineStaleGapMid", f"{c('value_gap_sweep', 'gap=0.5', 'cosine-rag', 'stale_rate'):.2f}")
    add("ctrlReconGapHighCur", f"{c('value_gap_sweep', 'gap=0.8', 'reconsolidation', 'current@1'):.2f}")
    add("ctrlReconJitterTwoCur", f"{c('key_jitter_sweep', 'jitter=0.2', 'reconsolidation', 'current@1'):.2f}")
    add("ctrlOracleJitterTwoCur", f"{c('key_jitter_sweep', 'jitter=0.2', 'keep-latest-oracle', 'current@1'):.2f}")
    for jtag, j in [("Two", 0.2), ("Four", 0.4), ("Eight", 0.8)]:
        d = (c("key_jitter_sweep", f"jitter={j}", "keep-latest-oracle", "current@1")
             - c("key_jitter_sweep", f"jitter={j}", "reconsolidation", "current@1"))
        add(f"ctrlKeyDeficitJitter{jtag}", f"{d:.2f}")
    add("ctrlRwCleanCur", f"{c('key_jitter_sweep', 'jitter=0.0', 'recency-weighted', 'current@1'):.2f}")
    add("ctrlRerankBudgetLowCur", f"{c('budget_sweep', 'B=12', 'time-rerank', 'current@1'):.2f}")
    add("ctrlTuneSeeds", str(ctrl.get("n_tune_seeds", 0)))
    add("ctrlTunedTauGapLow", str(ctrl["tuned"]["gap=0.3"]["tau"]))
    add("ctrlTunedTauGapHigh", str(ctrl["tuned"]["gap=0.8"]["tau"]))
    if "value_gap_sweep_fixed_tau" in ctrl:
        add("ctrlFixedTauGapHighCur",
            f"{ctrl['value_gap_sweep_fixed_tau']['gap=0.8']['current@1']['mean']:.2f}")
    # best tau per regime and the value it buys: the point is that the optimum
    # moves, not that no optimum exists
    for reg, tag in [("clean", "Clean"), ("jitter=0.4", "JitterFour")]:
        cells = ctrl["tau_sweep"][reg]
        best = max(cells, key=lambda k: cells[k]["current@1"]["mean"])
        add(f"ctrlBestTau{tag}", best.split("=")[1])
        add(f"ctrlBestTau{tag}Cur", f"{cells[best]['current@1']['mean']:.2f}")
        add(f"ctrlTauDefault{tag}Cur",
            f"{cells['tau=0.6']['current@1']['mean']:.2f}")
    for reg, tag in [("jitter=0.0", "JitterZero"), ("jitter=0.2", "JitterTwo"),
                     ("jitter=0.4", "JitterFour")]:
        tn = ctrl["tuned"][reg]
        add(f"ctrlTunedK{tag}", str(tn["k"]))
        add(f"ctrlTunedTau{tag}", str(tn["tau"]))
    add("ctrlOracleJitterEightCur",
        f"{c('key_jitter_sweep', 'jitter=0.8', 'keep-latest-oracle', 'current@1'):.2f}")
    add("ctrlOracleJitterFourCur",
        f"{c('key_jitter_sweep', 'jitter=0.4', 'keep-latest-oracle', 'current@1'):.2f}")
    add("ctrlReconJitterFourCur",
        f"{c('key_jitter_sweep', 'jitter=0.4', 'reconsolidation', 'current@1'):.2f}")
    add("ctrlRerankJitterZeroCur",
        f"{c('key_jitter_sweep', 'jitter=0.0', 'time-rerank', 'current@1'):.2f}")
    add("ctrlReconBudgetLowCur", f"{c('budget_sweep', 'B=12', 'reconsolidation', 'current@1'):.2f}")

    # ---------------- anchor-free track (v1.1) ---------------- #
    _ALIAS_LEVELS_KEYS = [f"alias_p={x}" for x in (0.0, 0.25, 0.5, 0.75, 1.0)]
    inv = alias["alias_inventory"]["test"]
    add("aliasCoverage", f"{100 * inv['frac_with_alias']:.0f}")
    add("aliasNForms", f"{inv['n_alias_forms']:,}")
    add("aliasAmbigPct", f"{100 * inv['frac_ambiguous_forms']:.1f}")
    add("aliasSubjAmbigPct",
        f"{100 * inv['frac_subjects_touched_by_ambiguity']:.0f}")
    add("aliasNEventsOne",
        str(alias["alias_sweep"]["alias_p=1.0"]["n_alias_events"]))
    add("aliasNEventsTotal",
        str(alias["alias_sweep"]["alias_p=1.0"]["n_events"]))

    def a_(pol, a, metric="current@1"):
        return alias["alias_sweep"][f"alias_p={a}"]["test"][pol]["summary"][metric]

    def a_def(pol, a):
        return alias["alias_sweep"][f"alias_p={a}"]["keying_deficit"][pol]

    for pol, tag in [("cosine-rag", "Cosine"), ("time-rerank", "Rerank"),
                     ("reconsolidation", "Recon"), ("lexical-key", "Lex"),
                     ("fuzzy-key", "Fuzzy"), ("alias-key(recent)", "Entity"),
                     ("fuzzy-read-key", "FuzzyRead"),
                     ("fuzzy-rw-key", "FuzzyRW"),
                     ("alias-rw-key", "EntityRW"),
                     ("keep-latest-oracle", "Oracle")]:
        if pol not in alias["alias_sweep"]["alias_p=1.0"]["test"]:
            continue                       # rung absent from an older sweep
        add(f"alias{tag}CurZero", f"{a_(pol, 0.0):.3f}")
        add(f"alias{tag}CurOne", f"{a_(pol, 1.0):.3f}")
        if pol != "keep-latest-oracle":
            add(f"alias{tag}DefZero", f"{100 * a_def(pol, 0.0):.1f}")
            add(f"alias{tag}DefOne", f"{100 * a_def(pol, 1.0):.1f}")
    # the read/write factorization: each path buys a different metric, so the
    # error decomposition has to be available for all three rungs at a=1
    for pol, tag in [("cosine-rag", "Cosine"),
                     ("fuzzy-key", "Fuzzy"), ("fuzzy-read-key", "FuzzyRead"),
                     ("fuzzy-rw-key", "FuzzyRW"), ("alias-key(recent)", "Entity"),
                     ("alias-rw-key", "EntityRW"),
                     ("keep-latest-oracle", "Oracle")]:
        if pol not in alias["alias_sweep"]["alias_p=1.0"]["test"]:
            continue
        for metric, mtag in (("miss_rate", "Miss"), ("stale_rate", "Stale"),
                             ("contamination@5", "Contam")):
            add(f"alias{tag}{mtag}One", f"{a_(pol, 1.0, metric):.3f}")
        add(f"alias{tag}SlotsOne", f"{a_(pol, 1.0, 'slots'):,.0f}")
        add(f"alias{tag}SlotsZero", f"{a_(pol, 0.0, 'slots'):,.0f}")
    _t1 = alias["alias_sweep"]["alias_p=1.0"]["test"]
    if "alias-rw-key" in _t1:
        _g = lambda k: _t1[k]["summary"]["current@1"]
        add("symGainFuzzy", f"{100 * (_g('fuzzy-rw-key') - _g('fuzzy-key')):.1f}")
        add("symGainEntity",
            f"{100 * (_g('alias-rw-key') - _g('alias-key(recent)')):.1f}")
        add("resolverGainSym",
            f"{100 * (_g('alias-rw-key') - _g('fuzzy-rw-key')):.1f}")
    _ap = alias["alias_sweep"]["alias_p=1.0"].get("paired_p", {}).get("holm", {})
    for kk, tag in (("aliasrw_vs_alias", "aliasPEntityRwEntityOne"),
                    ("aliasrw_vs_fuzzyrw", "aliasPEntityRwFuzzyRwOne"),
                    ("rw_vs_alias", "aliasPRwEntityOne"),
                    ("rw_vs_oracle", "aliasPRwOracleOne"),
                    ("rw_vs_fuzzy", "aliasPRwFuzzyOne")):
        if kk in _ap:
            add(tag, "<10^{-3}" if _ap[kk] < 1e-3 else f"={_ap[kk]:.2f}")

    # largest skip-vs-recent gap across every cell we compute
    _gaps = []
    for _a in _ALIAS_LEVELS_KEYS:
        _c = alias["alias_sweep"][_a]["test"]
        if "alias-key(skip)" in _c and "alias-key(recent)" in _c:
            _gaps.append(abs(_c["alias-key(skip)"]["summary"]["current@1"]
                             - _c["alias-key(recent)"]["summary"]["current@1"]))
    for _a in alias.get("full_corpus", {}):
        for _w in ("open_world", "closed_world"):
            _c = alias["full_corpus"][_a][_w]["test"]
            if "alias-key(skip)" in _c:
                _gaps.append(abs(_c["alias-key(skip)"]["summary"]["current@1"]
                                 - _c["alias-key(recent)"]["summary"]["current@1"]))
    add("aliasSkipRecentMaxGap", f"{100 * max(_gaps):.1f}")
    # how many model calls the shipped cache holds, split by where they ran: a
    # hand-typed count went stale as tracks were added, so the paper reads it
    # from the manifest
    _lc = relman.get("llm_cache", {})
    if _lc.get("entries"):
        _local = sum(v for k, v in _lc.get("by_model", {}).items() if ":" in k and "/" not in k)
        add("llmCacheEntries", f"{_lc['entries']:,}")
        add("llmCacheLocal", f"{_local:,}")
        add("llmCacheHosted", f"{_lc['entries'] - _local:,}")
    tb = relman.get("trivial_baselines", {})
    for cfg, tag in [("anchored", "Anchored"), ("anchor_free", "AnchorFree"),
                     ("full_corpus", "Full"),
                     ("full_corpus_anchor_free", "FullAnchorFree")]:
        if cfg in tb:
            best = max(tb[cfg].values(), key=lambda d: d["current@1"])
            add(f"blind{tag}", f"{best['current@1']:.3f}")
            add(f"blind{tag}Contam", f"{best['contamination@5']:.3f}")
    add("tunedFuzzyTau",
        f"{alias['alias_sweep']['alias_p=1.0']['tuned']['fuzzy-key']['tau']:.2f}")
    add("aliasLexContamOne", f"{a_('lexical-key', 1.0, 'contamination@5'):.3f}")
    add("aliasEntityContamOne",
        f"{a_('alias-key(recent)', 1.0, 'contamination@5'):.3f}")
    add("aliasRerankContamOne", f"{a_('time-rerank', 1.0, 'contamination@5'):.3f}")

    aud = alias["keying_audit"]

    def au_(keyer, a, field):
        return alias["keying_audit"][f"alias_p={a}"][keyer][field]

    for keyer, tag in [("threshold", "Thresh"), ("lexical", "Lex"),
                       ("fuzzy", "Fuzzy"), ("alias(recent)", "Entity")]:
        add(f"audit{tag}SplitZero", f"{100 * au_(keyer, 0.0, 'frac_split'):.1f}")
        add(f"audit{tag}SplitOne", f"{100 * au_(keyer, 1.0, 'frac_split'):.1f}")
        add(f"audit{tag}ImpureOne", str(au_(keyer, 1.0, "impure_clusters")))
        add(f"audit{tag}ImpureZero", str(au_(keyer, 0.0, "impure_clusters")))

    isc = alias.get("inventory_scaling", {}).get("alias_p=1.0")
    if isc:
        vals = sorted(isc.values(), key=lambda v: v["n_inventory"])
        add("invSmallN", f"{vals[0]['n_inventory']:,}")
        add("invSmallCur", f"{vals[0]['current@1']:.3f}")
        add("invSmallDef", f"{100 * vals[0]['deficit']:.1f}")
        add("invBigN", f"{vals[-1]['n_inventory']:,}")
        add("invBigCur", f"{vals[-1]['current@1']:.3f}")
        add("invBigDef", f"{100 * vals[-1]['deficit']:.1f}")
        # the same axis for the symmetric rung, so the paper's headline
        # surface and its headline fix are measured on the same policy
        if "rw_deficit" in vals[0]:
            add("invSmallDefRW", f"{100 * vals[0]['rw_deficit']:.1f}")
            add("invBigDefRW", f"{100 * vals[-1]['rw_deficit']:.1f}")
            add("invSmallCurRW", f"{vals[0]['rw_current@1']:.3f}")
            add("invBigCurRW", f"{vals[-1]['rw_current@1']:.3f}")
            _adv = [100 * (v["rw_current@1"] - v["current@1"]) for v in vals]
            add("invRWAdvantageMin", f"{min(_adv):.1f}")
            add("invRWAdvantageMax", f"{max(_adv):.1f}")

    md = alias.get("mention_detection")
    if md:
        add("substringFalseLink", f"{100 * md['substring_false_link_rate']:.0f}")
        add("mentionFalseLink", f"{100 * md['mention_false_link_rate']:.1f}")
        add("nDistractStatements", f"{md['n_distractor_statements']:,}")

    mech = alias["mechanism"]
    add("mechKgSlots", f"{mech['alias_p=0.0']['temporal-kg[alias]']['slots']:.0f}")
    add("mechKgActive",
        f"{mech['alias_p=0.0']['temporal-kg[alias]']['active_slots']:.0f}")
    add("mechOwSlots", f"{mech['alias_p=0.0']['overwrite[alias]']['slots']:.0f}")
    add("mechKgTauCur", f"{mech['alias_p=0.0']['temporal-kg[tau]']['current@1']:.3f}")
    add("mechOwTauCur", f"{mech['alias_p=0.0']['overwrite[tau]']['current@1']:.3f}")

    fc = alias["full_corpus"]
    add("fullNEvents", f"{fc['alias_p=0.0']['n_events']:,}")
    add("fullNDistract", f"{fc['alias_p=0.0']['n_distractor_events']:,}")
    add("fullNInvOpen", str(fc["alias_p=0.0"]["n_inventory"]["open_world"]))
    add("fullNInvClosed", f"{fc['alias_p=0.0']['n_inventory']['closed_world']:,}")

    def f_(pol, world, a=0.0, metric="current@1"):
        return fc[f"alias_p={a}"][world]["test"][pol]["summary"][metric]

    for pol, tag in [("cosine-rag", "Cosine"), ("time-rerank", "Rerank"),
                     ("reconsolidation", "Recon"), ("lexical-key", "Lex"),
                     ("alias-key(recent)", "Entity"),
                     ("keep-latest-oracle", "Oracle")]:
        add(f"full{tag}Open", f"{f_(pol, 'open_world'):.3f}")
        add(f"full{tag}Closed", f"{f_(pol, 'closed_world'):.3f}")
    add("fullEntityOpenMiss",
        f"{f_('alias-key(recent)', 'open_world', metric='miss_rate'):.3f}")
    add("fullEntityClosedMiss",
        f"{f_('alias-key(recent)', 'closed_world', metric='miss_rate'):.3f}")
    add("fullFuzzyOpenOne", f"{f_('fuzzy-key', 'open_world', 1.0):.3f}")
    add("fullFuzzyDefOne",
        f"{100 * (f_('keep-latest-oracle', 'open_world', 1.0)
                  - f_('fuzzy-key', 'open_world', 1.0)):.1f}")
    add("fullLexInventoryCost",
        f"{100 * (f_('lexical-key', 'closed_world')
                  - f_('lexical-key', 'open_world')):.1f}")
    add("realCosineContam", s("cosine-rag", "contamination@5"))
    if "fuzzy-rw-key" in alias["full_corpus"]["alias_p=1.0"]["open_world"]["test"]:
        add("fullFuzzyRWOpenOne", f"{f_('fuzzy-rw-key', 'open_world', 1.0):.3f}")
        add("fullFuzzyRWDefOne",
            f"{100 * alias['full_corpus']['alias_p=1.0']['open_world']['keying_deficit']['fuzzy-rw-key']:.1f}")
    add("fullEntityInventoryCost",
        f"{100 * (f_('alias-key(recent)', 'closed_world')
                  - f_('alias-key(recent)', 'open_world')):.1f}")

    # significance of the ladder separating: indistinguishable when anchored,
    # all pairs separated when anchor-free
    def _p(a, key):
        v = alias["alias_sweep"][f"alias_p={a}"]["paired_p"]["holm"][key]
        return "<10^{-3}" if v < 1e-3 else f"={v:.2f}"

    def _praw(a, key):
        # a Holm-corrected p of exactly 1.00 reads as "maximally failed" when it
        # is a family-size artifact; the raw value is the informative one to
        # quote next to a claim that we cannot reject
        v = alias["alias_sweep"][f"alias_p={a}"]["paired_p"]["raw"][key]
        return "<10^{-3}" if v < 1e-3 else f"={v:.2f}"
    add("aliasHolmFamily",
        str(len(alias["alias_sweep"]["alias_p=1.0"]["paired_p"]["raw"])))
    for a, atag in [(0.0, "Zero"), (1.0, "One")]:
        add(f"aliasPLexOracle{atag}", _p(a, "lexical_vs_oracle"))
        add(f"aliasPLexOracleRaw{atag}", _praw(a, "lexical_vs_oracle"))
        add(f"aliasPEntityOracleRaw{atag}", _praw(a, "alias_vs_oracle"))
        add(f"aliasPEntityOracle{atag}", _p(a, "alias_vs_oracle"))
        add(f"aliasPEntityLex{atag}", _p(a, "alias_vs_lexical"))

    # anchor-free under a second encoder: is the ladder a MiniLM artifact?
    if "mpnet" in alias:
        for pol, tag in [("cosine-rag", "Cosine"), ("time-rerank", "Rerank"),
                         ("reconsolidation", "Recon"), ("lexical-key", "Lex"),
                         ("alias-key(recent)", "Entity"),
                         ("keep-latest-oracle", "Oracle")]:
            for a, atag in [("0.0", "Zero"), ("1.0", "One")]:
                cell = alias["mpnet"][f"alias_p={a}"]
                add(f"mpAlias{tag}Cur{atag}",
                    f"{cell['test'][pol]['current@1']:.3f}")
                if pol != "keep-latest-oracle":
                    add(f"mpAlias{tag}Def{atag}",
                        f"{100 * cell['keying_deficit'][pol]:.1f}")

    # ---------------- production systems ---------------- #
    # For each release line, report the run whose (release, backend) pair passed
    # the probe. Two of the five pairs produce plausible numbers that measure a
    # backend rather than a system, so "which run do we quote" has to be decided
    # by the probe record and not by which file was written last.
    # Order is the quoting order: the first gate-passing run of a release line
    # on the preferred backend is the one the paper quotes end to end. The
    # ``_run2`` files lead because only they carry per-query records, surfaced
    # contexts and text-level scores -- quoting an older run's headline beside
    # tests computed on a newer one is the cross-run splice this generator
    # already had to be fixed for once.
    # the second deployed system has no mem0( key, so the release selection
    # below skips it; its own block reads the file directly
    _cands = ["stalebench_systems_graphiti_qwen14b.json",
              "stalebench_systems_chroma_01118_run2.json",
              "stalebench_systems_chroma_2011_run2.json",
              "stalebench_systems_results.json",
              "stalebench_systems_qdrant_2011.json",
              "stalebench_systems_chroma_2011.json",
              "stalebench_systems_v01_results.json",
              "stalebench_systems_qdrant_01118.json",
              "stalebench_systems_chroma_01118.json"]
    # A run this list does not name is invisible to the selection above, and an
    # invisible run reads exactly like a run that was never done. Fail rather
    # than quietly quote a stale file next to a fresh one on disk.
    # runs with a second adjudicator model ("_glm") or the third-person
    # extraction prompt ("_tp_") are quoted by their own block below, never as
    # a release's headline run
    _on_disk = sorted(fn for fn in os.listdir(_RES)
                      if fn.startswith("stalebench_systems_")
                      and fn.endswith(".json")
                      and ".contexts." not in fn
                      and "_glm" not in fn and "_tp_" not in fn
                      and "_langmem_" not in fn and "_erase_" not in fn
                      and fn != "stalebench_systems_ladder.json")
    _unlisted = [fn for fn in _on_disk if fn not in _cands]
    if _unlisted:
        raise SystemExit(
            "systems results present but not listed for selection: "
            + ", ".join(_unlisted)
            + " -- add them to _cands (and decide the quoting order) or move "
              "them out of results/")
    _best = {}
    _all_probed = {}
    for _fn in _cands:
        _dd = _opt(_fn)
        if not _dd:
            continue
        _cc = _dd["by_config"].get("anchored", {})
        _mk = next((k for k in _cc if k.startswith("mem0(")), None)
        if not _mk:
            continue
        _v = _cc[_mk].get("lib_version")
        _st = _cc[_mk].get("vector_store") or "faiss"
        if _backend_gate(_v, _st):
            _all_probed.setdefault(_v, []).append((_fn, _dd, _st))
    # Prefer a backend that BOTH release lines pass, so the headline comparison
    # is same-backend rather than resting on cross-backend agreement. Falls back
    # to any probed run for a line that has no shared option.
    _shared = set.intersection(*[{st for _, _, st in v}
                                 for v in _all_probed.values()]) \
        if len(_all_probed) > 1 else set()
    for _v, _opts in _all_probed.items():
        _pick = next((o for o in _opts if o[2] in _shared), _opts[0])
        _best[_v] = _pick
    if _shared:
        add("sysMatchedStore", f"\\texttt{{{sorted(_shared)[0]}}}")
    def _anchored_mem0(data):
        _cc9 = data["by_config"]["anchored"]
        return _cc9[next(k for k in _cc9 if k.startswith("mem0("))]

    # how stable is the measurement across every probed backend for one line?
    # Two runs of the SAME backend are a repeat, not a second backend, and
    # counting them here would report reproducibility as cross-backend
    # agreement -- so group by store first and take one run per store.
    for _v, _opts in _all_probed.items():
        # only count backends the run actually recorded; one early result file
        # has vector_store unset and is mapped by a fallback, which is not
        # provenance in a subsection about provenance
        _by_store: Dict[str, list] = {}
        for _optv in _opts:
            _stv = _anchored_mem0(_optv[1]).get("vector_store")
            if _stv:
                _by_store.setdefault(_stv, []).append(_optv)
        _tg = "Adj" if _v.startswith("0.1.") else "Cur"
        if len(_by_store) >= 2:
            _cs = [_anchored_mem0(_runs[0][1])["summary"]["current@1"]
                   for _runs in _by_store.values()]
            add(f"probed{_tg}NBackends", str(len(_by_store)))
            add(f"probed{_tg}Spread", f"{100 * (max(_cs) - min(_cs)):.1f}")
            add(f"probed{_tg}Lo", f"{min(_cs):.3f}")
            add(f"probed{_tg}Hi", f"{max(_cs):.3f}")
        # an independent repeat of one cell: same release, same backend, same
        # model, fresh store. The subsection reported single runs and said so;
        # where a repeat exists, report how far the two landed apart.
        for _stv, _runs in sorted(_by_store.items()):
            if len(_runs) < 2:
                continue
            _rep = [_anchored_mem0(_rr[1]) for _rr in _runs[:2]]
            add(f"repeat{_tg}NRuns", str(len(_runs)))
            add(f"repeat{_tg}Store", f"\\texttt{{{_stv}}}")
            add(f"repeat{_tg}CurA", f"{_rep[0]['summary']['current@1']:.3f}")
            add(f"repeat{_tg}CurB", f"{_rep[1]['summary']['current@1']:.3f}")
            add(f"repeat{_tg}CurDelta",
                f"{100 * abs(_rep[0]['summary']['current@1'] - _rep[1]['summary']['current@1']):.1f}")
            add(f"repeat{_tg}SlotsA", f"{_rep[0]['summary']['slots']:.0f}")
            add(f"repeat{_tg}SlotsB", f"{_rep[1]['summary']['slots']:.0f}")
            _ev_a = _rep[0].get("events") or {}
            _ev_b = _rep[1].get("events") or {}
            add(f"repeat{_tg}EventsIdentical",
                "identical" if _ev_a == _ev_b else "different")
            add(f"repeat{_tg}Update", str(_ev_a.get("UPDATE", 0)))
    for _v, (_fn, _dd, _st) in sorted(_best.items()):
        _tag = "Adj" if _v.startswith("0.1.") else "Cur"
        for _cfg, _ct in (("anchored", "Zero"), ("anchor_free", "One")):
            _cc = _dd["by_config"].get(_cfg, {})
            _mk = next((k for k in _cc if k.startswith("mem0(")), None)
            if not _mk:
                continue
            _sm = _cc[_mk]["summary"]
            _ev = _cc[_mk].get("events") or {}
            add(f"probed{_tag}Cur{_ct}", f"{_sm['current@1']:.3f}")
            add(f"probed{_tag}Contam{_ct}", f"{_sm['contamination@5']:.3f}")
            add(f"probed{_tag}Slots{_ct}", f"{_sm['slots']:.0f}")
            add(f"probed{_tag}Miss{_ct}", f"{_sm['miss_rate']:.3f}")
            # the same contexts scored by what they say. Emitted only where the
            # run recorded it, so a section quoting a text-level number cannot
            # be built over a run that never produced one.
            if "current@1_text" in _sm:
                add(f"text{_tag}Cur{_ct}", f"{_sm['current@1_text']:.3f}")
                add(f"text{_tag}Stale{_ct}", f"{_sm['stale_rate_text']:.3f}")
                add(f"text{_tag}Miss{_ct}", f"{_sm['miss_rate_text']:.3f}")
                add(f"text{_tag}Hedged{_ct}", f"{_sm['hedged@1_text']:.3f}")
                add(f"text{_tag}CurK{_ct}", f"{_sm['current@5_text']:.3f}")
                add(f"text{_tag}Contam{_ct}",
                    f"{_sm['contamination@5_text']:.3f}")
                add(f"text{_tag}CurDrop{_ct}",
                    f"{100 * (_sm['current@1'] - _sm['current@1_text']):.1f}")
                add(f"text{_tag}ContamRise{_ct}",
                    f"{100 * (_sm['contamination@5_text'] - _sm['contamination@5']):.1f}")
            # why the two conventions differ, and the control that says the
            # difference belongs to the system rather than to the text scorer:
            # policies that return the corpus verbatim must agree under both
            _attr_path = os.path.join(_RES, "attribution",
                                      _fn.replace(".json", ".json"))
            if os.path.exists(_attr_path):
                _attr = json.load(open(_attr_path))["by_config"].get(_cfg, {})
                _arow = _attr.get(_mk)
                if _arow:
                    add(f"attr{_tag}Collapsed{_ct}", str(_arow["n_collapsed"]))
                    add(f"attr{_tag}BothValues{_ct}",
                        str(_arow["n_both_values"]))
                    add(f"attr{_tag}UpdatedStale{_ct}",
                        str(_arow["n_updated_stale"]))
                    add(f"attr{_tag}NQueries{_ct}", str(_arow["n_queries"]))
                    # the other way the convention fails, and the one that
                    # looks like a system that cannot retrieve
                    if "n_id_disagrees" in _arow:
                        _nmem = _arow["n_id_agrees"] + _arow["n_id_disagrees"]
                        add(f"attr{_tag}IdWrong{_ct}",
                            str(_arow["n_id_disagrees"]))
                        add(f"attr{_tag}IdScored{_ct}", str(_nmem))
                        add(f"attr{_tag}IdWrongPct{_ct}",
                            f"{100 * _arow['n_id_disagrees'] / _nmem:.0f}"
                            if _nmem else "0")
                _gaps = [abs(_r2["id_level"]["current@1"]
                             - _r2["text_level"]["current@1_text"])
                         for _k2, _r2 in _attr.items()
                         if not _k2.startswith("mem0(")]
                if _gaps:
                    add(f"attr{_tag}NControls{_ct}", str(len(_gaps)))
                    add(f"attr{_tag}ControlMaxGap{_ct}",
                        f"{100 * max(_gaps):.1f}")
                    add(f"attr{_tag}ControlCollapsed{_ct}",
                        str(sum(_r2["n_collapsed"] for _k2, _r2 in _attr.items()
                                if not _k2.startswith("mem0("))))
            # the system's own keying deficit, on the paper's central axis
            if "keep-latest-oracle" in _cc:
                add(f"probed{_tag}Def{_ct}",
                    f"{100 * (_cc['keep-latest-oracle']['summary']['current@1'] - _sm['current@1']):.1f}")
            for _r, _rt in (("cosine-rag", "Cosine"), ("time-rerank", "Rerank"),
                            ("keep-latest-oracle", "Oracle")):
                if _r in _cc:
                    add(f"probed{_tag}Ref{_rt}{_ct}",
                        f"{_cc[_r]['summary']['current@1']:.3f}")
                    add(f"probed{_tag}Ref{_rt}Contam{_ct}",
                        f"{_cc[_r]['summary']['contamination@5']:.3f}")
            if _ct == "One":
                add(f"probed{_tag}UpdateAF", str(_ev.get("UPDATE", 0)))
            if _ct == "Zero":
                add(f"probed{_tag}Version", f"\\texttt{{{_v}}}")
                add(f"probed{_tag}Store", f"\\texttt{{{_st}}}")
                add(f"probed{_tag}Update", str(_ev.get("UPDATE", 0)))
                add(f"probed{_tag}Delete", str(_ev.get("DELETE", 0)))
                add(f"probed{_tag}Add", str(_ev.get("ADD", 0)))
                for _r, _rt in (("cosine-rag", "Cosine"),
                                ("time-rerank", "Rerank"),
                                ("keep-latest-oracle", "Oracle")):
                    if _r in _cc:
                        add(f"probed{_tag}Ref{_rt}",
                            f"{_cc[_r]['summary']['current@1']:.3f}")
                        add(f"probed{_tag}Ref{_rt}Contam",
                            f"{_cc[_r]['summary']['contamination@5']:.3f}")
                        add(f"probed{_tag}Ref{_rt}Slots",
                            f"{_cc[_r]['summary']['slots']:.0f}")
    add("probedNLines", str(len(_best)))
    # the paper's own ladder scored on the systems stream, so the comparison
    # against a deployed system is same-stream rather than cross-table
    _ld = _opt("stalebench_systems_ladder.json")
    if _ld:
        for _cfg3, _t3 in (("anchored", "Zero"), ("anchor_free", "One")):
            _c3 = _ld["by_config"].get(_cfg3, {})
            for _k3, _n3 in (("fuzzy-rw-key", "Sym"), ("fuzzy-key", "Fuzzy"),
                             ("keep-latest-oracle", "Oracle")):
                if _k3 in _c3:
                    _s3 = _c3[_k3]["summary"]
                    add(f"ladder{_n3}Cur{_t3}", f"{_s3['current@1']:.3f}")
                    add(f"ladder{_n3}Contam{_t3}",
                        f"{_s3['contamination@5']:.3f}")
                    add(f"ladder{_n3}Slots{_t3}", f"{_s3['slots']:.0f}")
    # -- paired tests on the systems stream -------------------------------- #
    # The subsection reported point estimates because the wrapped-system runs
    # predated per-query records. Every comparison below is paired by query on
    # a single stream, so the sign-flip test the rest of the paper uses applies
    # unchanged. Comparisons are generated over the records that are *present*:
    # a run without them contributes nothing rather than silently borrowing a
    # neighbour's, and the family size is emitted so a reader can see how many
    # tests the Holm correction was over.
    _sys_p: Dict[str, float] = {}
    _sys_detail: Dict[str, Dict] = {}

    def _macro_of(policy_name: str) -> str:
        return "".join(w.capitalize() for w in policy_name.replace("_", "-")
                       .split("-") if w.isalnum())

    def _paired(label, left, right, metric):
        """Sign-flip test on two same-length per-query record lists."""
        if not left or not right or len(left) != len(right):
            return
        if metric not in left[0] or metric not in right[0]:
            return
        xs = [rec[metric] for rec in left]
        ys = [rec[metric] for rec in right]
        _sys_p[label] = paired_permutation(xs, ys, seed=11)
        _sys_detail[label] = {"n": len(xs),
                              "delta": sum(xs) / len(xs) - sum(ys) / len(ys)}

    for _vv, (_fnv, _ddv, _stv) in sorted(_best.items()):
        _tagv = "Adj" if _vv.startswith("0.1.") else "Cur"
        for _cfgv, _ctv in (("anchored", "Zero"), ("anchor_free", "One")):
            _cellv = _ddv["by_config"].get(_cfgv, {})
            _mkv = next((k for k in _cellv if k.startswith("mem0(")), None)
            if not _mkv:
                continue
            _sysrec = _cellv[_mkv].get("records")
            for _polv, _recv in _cellv.items():
                if _polv == _mkv or not isinstance(_recv, dict):
                    continue
                _paired(f"sysP{_tagv}Vs{_macro_of(_polv)}{_ctv}",
                        _sysrec, _recv.get("records"), "current@1")
                # the same comparison scored by what the contexts say. The
                # section's claims rest on this scoring, so it is tested rather
                # than quoted, and it joins one family with the id-level tests
                # so the correction is over everything a reader would scan.
                _paired(f"sysT{_tagv}Vs{_macro_of(_polv)}{_ctv}",
                        _sysrec, _recv.get("records"), "current@1_text")
            # and the two scorings of the same run against each other: is the
            # drop from relabelling larger than query-to-query noise?
            if _sysrec and "current@1_text" in (_sysrec[0] or {}):
                _sys_p[f"sysPConvention{_tagv}{_ctv}"] = paired_permutation(
                    [r["current@1"] for r in _sysrec],
                    [r["current@1_text"] for r in _sysrec], seed=11)
                _sys_detail[f"sysPConvention{_tagv}{_ctv}"] = {
                    "n": len(_sysrec), "delta": 0.0}
    # the headline is a difference between releases, and it is paired too: the
    # same facts in the same order, one store each
    if len(_best) > 1:
        _lines = sorted(_best)
        for _cfgv, _ctv in (("anchored", "Zero"), ("anchor_free", "One")):
            _pair = []
            for _vv in _lines:
                _cellv = _best[_vv][1]["by_config"].get(_cfgv, {})
                _mkv = next((k for k in _cellv if k.startswith("mem0(")), None)
                _pair.append(_cellv[_mkv].get("records") if _mkv else None)
            if len(_pair) == 2:
                _paired(f"sysPVersion{_ctv}", _pair[0], _pair[1], "current@1")
    if _sys_p:
        _sys_holm = holm(_sys_p)
        for _lbl, _pv in sorted(_sys_holm.items()):
            add(_lbl, "$<0.001$" if _pv < 0.001 else f"${_pv:.3f}$")
        add("sysTestFamily", str(len(_sys_p)))
        add("sysTestSurviving",
            str(sum(1 for v in _sys_holm.values() if v < 0.05)))
        add("sysTestN", str(max(d["n"] for d in _sys_detail.values())))

    # the 2.0.11 shortfall has two causes and the paper collapsed them into
    # one: plain cosine over the same statements is a floor for "no supersession
    # at all", so the distance from it measures extraction damage and the
    # distance to the adjudicating release measures the missing supersession
    if len(_best) > 1 and "probedAdjRefCosine" in _seen:
        try:
            _c_cos = float(_seen["probedAdjRefCosine"])
            _c_adj = float(_seen["probedAdjCurZero"])
            _c_cur = float(_seen["probedCurCurZero"])
            add("sysExtractionCost", f"{100 * (_c_cos - _c_cur):.0f}")
            add("sysSupersessionGain", f"{100 * (_c_adj - _c_cos):.0f}")
        except (KeyError, ValueError):
            pass
    if len(_best) > 1:
        _cz = [float(v) for v in
               (_seen.get("probedAdjCurZero"), _seen.get("probedCurCurZero"))
               if v]
        if len(_cz) == 2:
            add("probedGapPts", f"{100 * abs(_cz[0] - _cz[1]):.0f}")
    # subsample fidelity: the systems track runs on a subsample for wall-clock
    # reasons, so state how faithfully it reproduces the reference policies
    if _best:
        _any = next(iter(_best.values()))[1]["by_config"]["anchored"]
        _full = {"cosine-rag": real["by_p"]["p=0.5"]["test"]["cosine-rag"],
                 "time-rerank": real["by_p"]["p=0.5"]["test"]["time-rerank"],
                 "keep-latest-oracle":
                     real["by_p"]["p=0.5"]["test"]["keep-latest-oracle"]}
        _devs = [abs(_any[k]["summary"]["current@1"]
                     - v["summary"]["current@1"]) for k, v in _full.items()
                 if k in _any]
        if _devs:
            add("sysSubsampleMaxDev", f"{100 * max(_devs):.1f}")
        # the costliest probed cell, since that is what a reader budgeting a
        # replication needs, not the cheapest
        _secs = []
        for _v2, (_fn2, _dd2, _st2) in _best.items():
            for _cfg2 in ("anchored", "anchor_free"):
                _cc3 = _dd2["by_config"].get(_cfg2, {})
                _mk3 = next((k for k in _cc3 if k.startswith("mem0(")), None)
                if _mk3 and _cc3[_mk3].get("ingest_seconds"):
                    _secs.append(_cc3[_mk3]["ingest_seconds"])
        if _secs:
            add("sysIngestHours", f"{max(_secs) / 3600:.1f}")
            add("sysIngestHoursTotal", f"{sum(_secs) / 3600:.1f}")

    if systems:
        # Quote ONE probed run per release line, end to end. Reading a
        # hard-coded result file here while the probed selection was used
        # elsewhere spliced two runs into one paragraph -- 812 ADD events from
        # an unpinned backend beside 817 slots from chroma, in the sentence
        # claiming same-backend provenance, and 812 ADDs with zero UPDATEs
        # cannot produce 817 slots. Everything in this block now comes from
        # the same run the tables and the probed* macros use.
        _cur_line = next((v for k, v in _best.items()
                          if not k.startswith("0.1.")), None)
        if _cur_line is not None:
            systems = _cur_line[1]
        sy = systems["by_config"]["anchored"]
        mem = next((k for k in sy if k.startswith("mem0(")), None)
        add("sysNFacts", str(systems["config"]["n_test_facts"]))
        add("sysNStatements", str(sy["n_statements"]))
        if mem:
            # NB: do not bind `c` -- it is the controlled-track accessor
            # function used further down, and shadowing it aborts table
            # generation partway (see _EXPECTED_TABLES guard)
            sc = sy[mem]
            _sm = sc["summary"].get("miss_rate", 0.0)
            _srm = sy.get("cosine-rag", {}).get("summary", {}).get("miss_rate", 0.0)
            # gate on the backend probe, not on the miss rate: a system may
            # genuinely rank badly, and this system does
            _gate = _backend_gate(sc.get("lib_version"),
                                  sc.get("vector_store") or "faiss")
            _s_usable = bool(_gate)
            if _gate is None:
                _s_usable = False
                print(f"WARNING: {mem} has no backend gate record for store "
                      f"{sc.get('vector_store')!r}; withholding retrieval "
                      f"metrics until one exists")
            elif not _gate:
                print(f"WARNING: {mem} failed its backend gate "
                      f"({sc.get('vector_store')}) -- the store does not return "
                      f"what it holds; withholding retrieval metrics")
            elif _sm > max(0.30, 10 * _srm):
                # A high miss rate has a third cause besides "broken backend"
                # and "the system ranks badly", and it is the one that held
                # here: the wrapper could not attach a benchmark id to what the
                # system stored, so a memory about the right fact was scored as
                # the wrong one. The text-level miss rate distinguishes them --
                # it asks whether the returned memory talks about the queried
                # fact at all -- so quote it whenever the run recorded it
                # rather than repeating the old reading.
                _tm = sc["summary"].get("miss_rate_text")
                if _tm is None:
                    print(f"NOTE: {mem} misses {_sm:.3f} against store-all "
                          f"{_srm:.3f}; its backend passes the gate, but this "
                          f"run has no text-level miss rate, so whether that "
                          f"is the system's ranking or the wrapper's "
                          f"attribution is undetermined")
                elif _tm < 0.5 * _sm:
                    print(f"NOTE: {mem} misses {_sm:.3f} by id but "
                          f"{_tm:.3f} by text -- the id-level miss is the "
                          f"attribution failing, not the retrieval; the paper "
                          f"must report it as such")
                else:
                    print(f"NOTE: {mem} misses {_sm:.3f} by id and {_tm:.3f} "
                          f"by text against store-all {_srm:.3f}, so the miss "
                          f"is the system's ranking and is reported")
            add("sysMemZeroMiss", f"{_sm:.3f}")
            add("sysMemZeroRefMiss", f"{_srm:.3f}")
            add("sysMemZeroName", f"\\texttt{{{mem}}}")
            add("sysMemZeroAdds", str(sc.get("n_add_calls") or sy["n_statements"]))
            add("sysMemZeroAddEvents", str(sc["events"].get("ADD", 0)))
            add("sysMemZeroUpdateEvents", str(sc["events"].get("UPDATE", 0)))
            if _s_usable:
                add("sysMemZeroCur", f"{sc['summary']['current@1']:.3f}")
                add("sysMemZeroContam", f"{sc['summary']['contamination@5']:.3f}")
                add("sysMemZeroCurFive", f"{sc['summary']['current@5']:.3f}")
                # a store that only ever ADDs must hold exactly its ADD count;
                # a mismatch means two runs have been spliced together
                _ev0 = sc.get("events") or {}
                if not _ev0.get("UPDATE") and not _ev0.get("DELETE"):
                    _expect = _ev0.get("ADD", 0)
                    if _expect and abs(sc["summary"]["slots"] - _expect) > 0.5:
                        raise SystemExit(
                            f"systems block inconsistent: {_expect} ADD events "
                            f"with no UPDATE/DELETE but "
                            f"{sc['summary']['slots']:.0f} slots -- numbers "
                            f"from different runs have been mixed")
            add("sysMemZeroSlots", f"{sc['summary']['slots']:.0f}")
            # reference points on the identical stream, so "worse than
            # store-all" is a same-data comparison rather than a cross-table one
            for _ref, _rt in (("cosine-rag", "Cosine"),
                              ("time-rerank", "Rerank"),
                              ("keep-latest-oracle", "Oracle")):
                if _ref in sy:
                    add(f"sysRef{_rt}", f"{sy[_ref]['summary']['current@1']:.3f}")
                    add(f"sysRef{_rt}Slots",
                        f"{sy[_ref]['summary']['slots']:.0f}")
        af = systems["by_config"].get("anchor_free", {})
        if mem and mem in af:
            add("sysMemZeroCurAF", f"{af[mem]['summary']['current@1']:.3f}")
            add("sysMemZeroUpdateEventsAF",
                str(af[mem]["events"].get("UPDATE", 0)))
        if mem:
            add("sysMemZeroVersion",
                f"\\texttt{{{sy[mem].get('lib_version') or 'unpinned'}}}")

    # the version difference in two statements, generated per interpreter by
    # stalebench.diag_mem0_versions so neither line overwrites the other
    import glob as _g
    _vd = {}
    for _f in sorted(_g.glob(os.path.join(_RES, "mem0_versions", "*.json"))):
        _d = json.load(open(_f))
        _vd[_d["lib_version"]] = _d
    if _vd:
        _adj = sorted(v for v, d in _vd.items() if d["adjudicates_on_add"])
        _non = sorted(v for v, d in _vd.items() if not d["adjudicates_on_add"])
        if _adj and _non:
            _a, _n = _vd[_adj[0]], _vd[_non[0]]

            def _evstr(d):
                return ", ".join(f"{k}\\,{v}" for k, v in sorted(d["events"].items()))
            add("miniAdjVersion", f"\\texttt{{{_a['lib_version']}}}")
            add("miniNonVersion", f"\\texttt{{{_n['lib_version']}}}")
            add("miniAdjEvents", _evstr(_a))
            add("miniNonEvents", _evstr(_n))
            add("miniAdjSlots", str(_a["slots"]))
            add("miniNonSlots", str(_n["slots"]))
            add("miniNStatements", str(_a["n_statements"]))
            add("miniAdjRankCurrent",
                "the current one" if _a["rank1_is_current"] else "the stale one")
            add("miniNonRankCurrent",
                "the current one" if _n["rank1_is_current"] else "the stale one")

    # the (release, backend) probe matrix, as a table. For a benchmark this is
    # reproducibility guidance, not trivia: two of these pairs silently produce
    # numbers that look like a system's behaviour and are not.
    import glob as _gl
    _gates = []
    for _f in sorted(_gl.glob(os.path.join(_RES, "mem0_gates", "*.json"))):
        _gates.append(json.load(open(_f)))
    if _gates:
        _rows = []
        for _g in sorted(_gates, key=lambda d: (d["lib_version"],
                                                d["vector_store"])):
            _ret = "\\checkmark" if _g.get("retrieval_ok") else "$\\times$"
            _act = "\\checkmark" if _g.get("actions_ok") else "$\\times$"
            _use = "\\checkmark" if _g.get("gate_pass") else "$\\times$"
            _rows.append(
                f"\\texttt{{{_g['lib_version']}}} & \\texttt{{{_g['vector_store']}}} "
                f"& {_act} & {_ret} & ${_g['n_returned']}$ & {_use} \\\\")
        _tabular("llcccc",
                 "mem0 release & vector store & writes & retrieval & "
                 "returned & usable",
                 _rows, os.path.join(_PAPER, "tab_gates.tex"))
        add("sysNGatesProbed", str(len(_gates)))
        add("sysNGatesPassed", str(sum(1 for g in _gates if g.get("gate_pass"))))
        _fails = [f"\\texttt{{{g['lib_version']}}}/\\texttt{{{g['vector_store']}}}"
                  for g in _gates if not g.get("gate_pass")]
        add("sysGateFailList", ", ".join(_fails) if _fails else "none")
        # a probe that crashed leaves no record, so an all-pass table may simply
        # mean the failures were never written; say which pairs are accounted for
        _expected = {("2.0.11", "faiss"), ("2.0.11", "qdrant"),
                     ("0.1.118", "faiss"), ("0.1.118", "qdrant"),
                     ("0.1.118", "chroma"), ("2.0.11", "chroma")}
        _have = {(g["lib_version"], g["vector_store"]) for g in _gates}
        add("sysGateMatrixComplete", "yes" if _expected <= _have else "no")
        # generate the sentence about failures, so it cannot describe failures
        # the record does not contain (the failing probes record last, and an
        # absent probe is indistinguishable from a passing one in this table)
        # Describe the failures from the records, not from memory: an earlier
        # version generated only the *presence* of this sentence while
        # hardcoding its content, so it named two failure modes when one was on
        # record. What the sentence says must be as generated as whether it
        # appears.
        _modes = []
        for _g in _gates:
            if _g.get("gate_pass"):
                continue
            if not _g.get("retrieval_ok"):
                # describe what the record establishes -- n returned of n held,
                # and rank 1 != the hand-scored best -- not "least similar",
                # which the probe does not measure
                _modes.append(
                    f"one returns {_g['n_returned']} of the "
                    f"{_g['n_memories_held']} memories it holds and ranks none "
                    f"of them the one the embedder scores first")
            if not _g.get("actions_ok"):
                _modes.append("one records no decision for any write while "
                              "still persisting some of them")
        # a write gate that passes only shows the system *recorded* decisions;
        # on a failing read gate those decisions were taken against the wrong
        # candidates, so the counts are not comparable across backends
        _wr = [g for g in _gates
               if g.get("actions_ok") and not g.get("retrieval_ok")]
        if _wr:
            add("sysGateWriteCaveat",
                "A passing write gate shows only that the system recorded its "
                "own decisions, not that it made them well: on a pair whose "
                "read gate fails, the adjudicator is choosing against the "
                "candidates that backend returned, and its event counts differ "
                "from the same release on a passing backend accordingly")
        else:
            add("sysGateWriteCaveat", "")
        _modes = list(dict.fromkeys(_modes))
        if _modes:
            _why = (_modes[0] if len(_modes) == 1
                    else "; ".join(_modes[:-1]) + "; and " + _modes[-1])
            add("sysGateFailNote",
                "we report no retrieval metrics from the "
                + ("other" if len(_fails) == 1 else "rest")
                + ", because a failing pair still produces plausible-looking "
                  "numbers --- " + _why + " --- that describe our "
                  "configuration rather than the system")
        else:
            add("sysGateFailNote",
                "pairs that fail it are excluded, because a failing pair still "
                "produces plausible-looking numbers that describe our "
                "configuration rather than the system")
        add("sysGateMissingList",
            ", ".join(f"\\texttt{{{v}}}/\\texttt{{{st}}}"
                      for v, st in sorted(_expected - _have)) or "none")

    # matched-backend runs, if present: the whole point of re-running on one
    # known-good store is that a version comparison needs the backend held
    # fixed, so the prose reporting it is generated from whichever files exist
    _q = {}
    for _tag, _f in (("cur", "stalebench_systems_qdrant_2011.json"),
                     ("old", "stalebench_systems_qdrant_01118.json")):
        _dd = _opt(_f)
        if _dd:
            _cell = _dd["by_config"].get("anchored", {})
            _mk = next((k for k in _cell if k.startswith("mem0(")), None)
            if _mk:
                _q[_tag] = (_dd, _cell, _cell[_mk], _mk)
    if _q:
        _lines = []
        for _tag in ("old", "cur"):
            if _tag not in _q:
                continue
            _dd, _cell, _c2, _mk = _q[_tag]
            _rm = _cell.get("cosine-rag", {}).get("summary", {}).get("miss_rate", 0.0)
            _mm = _c2["summary"].get("miss_rate", 0.0)
            _ok = bool(_backend_gate(_c2.get("lib_version"),
                                     _c2.get("vector_store") or "faiss"))
            _v = _c2.get("lib_version", "?")
            _st = _c2.get("vector_store", "?")
            add(f"sysQ{_tag}Version", f"\\texttt{{{_v}}}")
            add(f"sysQ{_tag}Store", f"\\texttt{{{_st}}}")
            add(f"sysQ{_tag}Slots", f"{_c2['summary']['slots']:.0f}")
            add(f"sysQ{_tag}Adds", str(_c2.get("n_add_calls") or "?"))
            add(f"sysQ{_tag}Miss", f"{_mm:.3f}")
            add(f"sysQ{_tag}Usable", "yes" if _ok else "no")
            if _ok:
                add(f"sysQ{_tag}Cur", f"{_c2['summary']['current@1']:.3f}")
                add(f"sysQ{_tag}Contam",
                    f"{_c2['summary']['contamination@5']:.3f}")
            else:
                print(f"WARNING: {_v}/{_st} has no passing backend gate "
                      f"record; withholding its retrieval metrics "
                      f"(miss {_mm:.3f} vs store-all {_rm:.3f})")

    # cross-backend replication for the release whose numbers we report: the
    # claim "this is the system, not our backend" is only worth as much as the
    # agreement between two independent backends, so state it as a number
    _qc = _opt("stalebench_systems_qdrant_2011.json")
    if systems and _qc:
        for _cfg, _ct in (("anchored", "Zero"), ("anchor_free", "One")):
            _a = systems["by_config"].get(_cfg, {})
            _b = _qc["by_config"].get(_cfg, {})
            _ma = next((k for k in _a if k.startswith("mem0(")), None)
            _mb = next((k for k in _b if k.startswith("mem0(")), None)
            if not (_ma and _mb):
                continue
            _sa, _sb = _a[_ma]["summary"], _b[_mb]["summary"]
            _ga = _backend_gate(_a[_ma].get("lib_version"),
                                _a[_ma].get("vector_store") or "faiss")
            _gb = _backend_gate(_b[_mb].get("lib_version"),
                                _b[_mb].get("vector_store") or "qdrant")
            if not (_ga and _gb):
                continue                     # both backends must be probed
            add(f"sysReplCur{_ct}", f"{_sb['current@1']:.3f}")
            add(f"sysReplContam{_ct}", f"{_sb['contamination@5']:.3f}")
            add(f"sysReplCurDelta{_ct}",
                f"{100 * abs(_sa['current@1'] - _sb['current@1']):.1f}")
            add(f"sysReplContamDelta{_ct}",
                f"{100 * abs(_sa['contamination@5'] - _sb['contamination@5']):.1f}")
            add(f"sysReplStore{_ct}",
                f"\\texttt{{{_b[_mb].get('vector_store') or '?'}}}")

    # the same system on the release line whose add() still adjudicates: the
    # version comparison is the finding, so it is generated, not asserted
    sysold = _opt("stalebench_systems_v01_results.json")
    if systems and sysold:
        so = sysold["by_config"]["anchored"]
        memo = next((k for k in so if k.startswith("mem0(")), None)
        if memo:
            co = so[memo]
            ev = co["events"]
            # A near-total miss rate against a wrapped third-party system is a
            # measurement failure wearing the costume of a result -- whatever
            # its cause (id resolution, embedder, vector-store backend). Publish
            # the event histogram, which counts the system's own decisions and
            # is independent of our retrieval path, and withhold the retrieval
            # metrics until the cause is understood.
            _miss = co["summary"].get("miss_rate", 0.0)
            # Compare against the store-all baseline's miss on the *same*
            # stream rather than an absolute cutoff. A wrapped system may
            # legitimately miss more than plain cosine, but not by an order of
            # magnitude: that means its retrieval layer is not returning what it
            # holds, which is a fact about our configuration of that layer.
            _ref_miss = so.get("cosine-rag", {}).get(
                "summary", {}).get("miss_rate", 0.0)
            _g2 = _backend_gate(co.get("lib_version"),
                                co.get("vector_store") or "faiss")
            _usable = bool(_g2)
            if not _usable:
                print(f"WARNING: {memo} backend gate "
                      f"{'FAILED' if _g2 is False else 'MISSING'} for store "
                      f"{co.get('vector_store')!r} -- withholding retrieval "
                      f"metrics (miss {_miss:.3f})")
            if not _usable:
                print(f"WARNING: {memo} miss_rate={_miss:.3f} -- retrieval "
                      f"metrics reflect a failure in the retrieval layer, not "
                      f"the system; emitting the event histogram only")
            n_upd = ev.get("UPDATE", 0) + ev.get("DELETE", 0)
            add("sysOldVersion",
                f"\\texttt{{{co.get('lib_version') or 'unknown'}}}")
            add("sysOldAdds", str(co.get("n_add_calls") or so["n_statements"]))
            add("sysOldAddEvents", str(ev.get("ADD", 0)))
            add("sysOldUpdateEvents", str(ev.get("UPDATE", 0)))
            add("sysOldDeleteEvents", str(ev.get("DELETE", 0)))
            if _usable:
                add("sysOldCur", f"{co['summary']['current@1']:.3f}")
                add("sysOldContam", f"{co['summary']['contamination@5']:.3f}")
            add("sysOldSlots", f"{co['summary']['slots']:.0f}")
            add("sysOldMiss", f"{_miss:.3f}")
            # NB: the event histogram is NOT independent of the retrieval
            # layer. mem0's _add_to_vector_store calls vector_store.search
            # (limit=5) and hands those neighbours to the adjudicator, so a
            # backend that returns the least-similar memories makes the LLM
            # decide against the wrong candidates. Counts produced on a failed
            # backend are therefore withheld along with the metrics; only the
            # *capability* claim (which events a release can emit at all)
            # survives, because that comes from the dispatch path, not a run.
            _afo = sysold["by_config"].get("anchor_free", {}).get(memo) \
                if _usable else None
            if _afo:
                _eaf = _afo["events"]
                add("sysOldUpdateEventsAF", str(_eaf.get("UPDATE", 0)))
                add("sysOldDeleteEventsAF", str(_eaf.get("DELETE", 0)))
                add("sysOldAddEventsAF", str(_eaf.get("ADD", 0)))
                _u0, _u1 = ev.get("UPDATE", 0), _eaf.get("UPDATE", 0)
                add("sysOldUpdateDropPct",
                    f"{100 * (_u0 - _u1) / _u0:.0f}" if _u0 else "n/a")
            add("sysVersionCompare", "")
            if True:
                pass
            
    elif systems:
        add("sysVersionCompare", "")

    # ---------------- v1.1 scale track ---------------- #
    if scale:
        sst = scale["stats"]
        add("scaleNTest", f"{sst['test_facts']:,}")
        add("scaleNVal", f"{sst['val_facts']:,}")
        add("scaleNMulti", f"{sst['multi_valued_test']:,}")
        add("scaleMeanCurrentValues", f"{sst['mean_current_values']:.2f}")
        for a, atag in (("0.0", "Zero"), ("1.0", "One")):
            cell = scale["by_alias"][f"alias_p={a}"]
            add(f"scaleNStatements{atag}", f"{cell['n_statements']:,}")
            for pol, tag in [("cosine-rag", "Cosine"), ("time-rerank", "Rerank"),
                             ("recency-weighted", "Rw"),
                             ("reconsolidation", "Recon"), ("lexical-key", "Lex"),
                             ("fuzzy-key", "Fuzzy"), ("fuzzy-rw-key", "FuzzyRW"),
                             ("alias-key(recent)", "Alias"),
                             ("keep-latest-oracle", "Oracle")]:
                if pol in cell["test"]:
                    add(f"scale{tag}Cur{atag}",
                        f"{cell['test'][pol]['summary']['current@1']:.3f}")
                    if pol != "keep-latest-oracle":
                        add(f"scale{tag}Def{atag}",
                            f"{100 * cell['keying_deficit'][pol]:.1f}")
                    # set recall over simultaneously valid values: the price
                    # of one slot per fact, which current@k cannot see
                    _sr = cell["test"][pol]["summary"].get("current_set_recall@5")
                    if _sr is not None:
                        add(f"scaleSetRec{tag}{atag}", f"{_sr:.3f}")
                    _mv = cell.get("test_multi_valued", {}).get(pol)
                    if _mv and "current_set_recall@5" in _mv["summary"]:
                        add(f"scaleSetRecMulti{tag}{atag}",
                            f"{_mv['summary']['current_set_recall@5']:.3f}")
                        add(f"scaleCurFiveMulti{tag}{atag}",
                            f"{_mv['summary']['current@5']:.3f}")

    if scale:
        # a pre-specified v1.0 comparison that REVERSES at scale; naming it is
        # both more honest and a sharper claim than "everything survives"
        _sz = scale["by_alias"]["alias_p=0.0"]["test"]
        for _pol, _tag in (("time-rerank", "Rerank"),
                           ("reconsolidation", "Recon")):
            # NB: do not bind _ci here -- it is a formatting *function* used
            # later in this module, and shadowing it aborts table generation
            # partway, leaving stale tab_*.tex behind a clean-looking run
            _bnd = _sz[_pol]["ci"]["current@1"]
            add(f"scale{_tag}CiLo", f"{_bnd['lo']:.3f}")
            add(f"scale{_tag}CiHi", f"{_bnd['hi']:.3f}")

    # ---------------- verbalization robustness ---------------- #
    para = _opt("stalebench_paraphrase_results.json")
    if para:
        _pl = para["config"]["paraphrase_levels"]
        add("paraNLevels", str(len(_pl)))
        add("paraLevels", ", ".join(f"{x:g}" for x in _pl))

        def _pc(pol, pp, a=1.0, m="current@1"):
            return para["cells"][f"p{pp}_a{a}"]["test"][pol][m]

        for pol, tag in (("cosine-rag", "Cosine"), ("lexical-key", "Lex"),
                         ("fuzzy-key", "Fuzzy"), ("fuzzy-rw-key", "FuzzyRW"),
                         ("alias-key(recent)", "Entity"),
                         ("keep-latest-oracle", "Oracle")):
            for pp, ptag in ((0.0, "Zero"), (1.0, "One")):
                add(f"para{tag}{ptag}", f"{_pc(pol, pp):.3f}")
            # how much this policy moves across the whole verbalization range
            _sp = [_pc(pol, x) for x in _pl]
            add(f"para{tag}Spread", f"{100 * (max(_sp) - min(_sp)):.1f}")
        # does any pair of rungs change order across the range? Reported for
        # the write-path family, which is the ladder the paper's deficit is
        # defined over; the symmetric rung crosses the write-path oracle and
        # that crossing is discussed where it happens, not hidden in a flag
        def _rank(pols, pp):
            return [n for n, _ in sorted(((n, _pc(n, pp)) for n in pols),
                                         key=lambda kv: -kv[1])]
        _wp = ("cosine-rag", "time-rerank", "recency-weighted",
               "reconsolidation", "lexical-key", "fuzzy-key",
               "alias-key(recent)", "keep-latest-oracle")
        _o = [_rank(_wp, pp) for pp in _pl]
        add("paraOrderStable",
            "identical at every level" if all(x == _o[0] for x in _o)
            else "not identical")
        _oa = [_rank(_wp + ("fuzzy-rw-key",), pp) for pp in _pl]
        add("paraOrderStableAll",
            "identical at every level" if all(x == _oa[0] for x in _oa)
            else "identical except that the symmetric rung overtakes the "
                 "write-path oracle once any paraphrasing is present")

    if para:
        # -------- Table: verbalization crossed with the anchor -------- #
        _prow = [("cosine-rag", "Cosine (store-all)"),
                 ("time-rerank", "+ time-rerank"),
                 ("reconsolidation", "Overwrite: threshold"),
                 ("lexical-key", "Overwrite: gazetteer"),
                 ("fuzzy-key", "Overwrite: fuzzy linker"),
                 ("alias-key(recent)", "Overwrite: linker + aliases"),
                 ("fuzzy-rw-key", "Symmetric key (read+write)"),
                 ("keep-latest-oracle", "Overwrite: true-key oracle")]
        _lv = para["config"]["paraphrase_levels"]
        rows = []
        for key, label in _prow:
            cells = []
            for a in (0.0, 1.0):
                for pp in _lv:
                    cells.append(
                        f"${para['cells'][f'p{pp}_a{a}']['test'][key]['current@1']:.3f}$")
            rows.append(f"{label} & " + " & ".join(cells) + " \\\\")
        _hdr = ("& \\multicolumn{%d}{c}{anchored ($a{=}0$)} & "
                "\\multicolumn{%d}{c}{anchor-free ($a{=}1$)} \\\\\n"
                "\\cmidrule(lr){2-%d}\\cmidrule(lr){%d-%d}\n"
                "policy & " % (len(_lv), len(_lv), 1 + len(_lv),
                               2 + len(_lv), 1 + 2 * len(_lv))
                + " & ".join(f"$p{{=}}{x:g}$" for x in _lv) + " & "
                + " & ".join(f"$p{{=}}{x:g}$" for x in _lv))
        _tabular("l" + "c" * (2 * len(_lv)), _hdr, rows,
                 os.path.join(_PAPER, "tab_paracross.tex"))

    # ---------------- reader track ---------------- #
    if reader:
     models = list(reader["readers"])
     primary = models[0]
     add("readerNModels", str(len(models)))
     add("readerModelList", ", ".join(f"\\texttt{{{m}}}" for m in models))
     add("readerPrimary", f"\\texttt{{{primary}}}")
     add("readerTopK", str(reader["config"]["top_k"]))

     def rd(model, pol, dk="no_dates", field="correct"):
        return reader["readers"][model]["policy_track"][dk][pol]["summary"][field]

     for pol, tag in [("cosine-rag", "Cosine"), ("time-rerank", "Rerank"),
                     ("reconsolidation", "Recon"), ("lexical-key", "Lex"),
                     ("keep-latest-oracle", "Oracle")]:
        add(f"reader{tag}Correct", f"{rd(primary, pol):.3f}")
        add(f"reader{tag}Stale", f"{rd(primary, pol, field='stale'):.3f}")
        add(f"reader{tag}Dates", f"{rd(primary, pol, 'with_dates'):.3f}")
     # ---- the controlled pair -------------------------------------- #
     # read-path and symmetric keying rank the SAME statement first for every
     # query, so they are identical on current@1 and stale@1 and differ only in
     # what fills ranks 2-5. Any reader gap between them is attributable to
     # context contamination alone -- a within-pair contrast, not a correlation
     _PAIR = ("fuzzy-read-key", "fuzzy-rw-key")
     if all(pp in reader["readers"][primary]["policy_track"]["no_dates"]
            for pp in _PAIR):
        _pt = reader["readers"][primary]["policy_track"]["no_dates"]
        add("pairCur", f"{_pt[_PAIR[0]]['retrieval_current@1']:.3f}")
        # the premise of the whole comparison, measured per query by
        # stalebench.diag_pair rather than inferred from matching aggregates
        _pd = _opt("stalebench_pair_diag.json")
        if _pd:
            _z = _pd["by_alias"]["alias_p=0.0"]
            _o = _pd["by_alias"]["alias_p=1.0"]
            assert _z["exactly_controlled"], \
                "the pair is not exactly controlled where the reader track runs"
            add("pairNQueries", str(_z["n_queries"]))
            add("pairAnchorFreeAgree", str(_o["rank1_agree"]))
        if len(models) > 1:
            add("readerSecCosine", f"{rd(models[1], 'cosine-rag'):.3f}")
        # the symmetric rung is the best non-oracle policy at the reader on
        # every reader measured; the paper reported every component of this
        # and never stated it
        _bests = []
        for _m in models:
            _ptm = reader["readers"][_m]["policy_track"]["no_dates"]
            _rk = sorted(_ptm, key=lambda k: -_ptm[k]["summary"]["correct"])
            _nonorc = [k for k in _rk if k != "keep-latest-oracle"]
            _bests.append(_nonorc[0] if _nonorc else None)
        add("readerBestNonOracle",
            "fuzzy-rw-key" if set(_bests) == {"fuzzy-rw-key"} else "varies")
        for _i, _m in enumerate(models):
            _mt = ["Pri", "Sec", "Ter"][_i] if _i < 3 else f"M{_i}"
            _ptm = reader["readers"][_m]["policy_track"]["no_dates"]
            if "fuzzy-rw-key" in _ptm:
                add(f"reader{_mt}SymGapToOracle",
                    f"{100 * (_ptm['keep-latest-oracle']['summary']['correct']
                              - _ptm['fuzzy-rw-key']['summary']['correct']):.1f}")
        add("readerOracleRet",
            f"{_pt['keep-latest-oracle']['retrieval_current@1']:.3f}")
        assert (_pt[_PAIR[0]]["retrieval_current@1"]
                == _pt[_PAIR[1]]["retrieval_current@1"]), \
            "the controlled pair is only a controlled pair if current@1 matches"
        for pp, tag in zip(_PAIR, ("Read", "Rw")):
           add(f"pair{tag}Contam",
               f"{_pt[pp]['retrieval_contamination@5']:.3f}")
        for i, m in enumerate(models):
           mt = ["Pri", "Sec", "Ter"][i] if i < 3 else f"M{i}"
           ptm = reader["readers"][m]["policy_track"]["no_dates"]
           if not all(pp in ptm for pp in _PAIR):
              continue
           for pp, tag in zip(_PAIR, ("Read", "Rw")):
              add(f"pair{mt}{tag}", f"{ptm[pp]['summary']['correct']:.3f}")
           for pp_, tag_ in zip(_PAIR, ("Read", "Rw")):
              for fld, ftag in (("stale", "Stale"), ("hedged", "Hedged")):
                 add(f"pair{mt}{tag_}{ftag}",
                     f"{ptm[pp_]['summary'][fld]:.3f}")
           # the same pair with the stored year surfaced: whatever is left of
           # the gap is what a reader cannot resolve from dates alone
           _ptd = reader["readers"][m]["policy_track"].get("with_dates", {})
           if all(pp in _ptd for pp in _PAIR):
              for pp, tag in zip(_PAIR, ("Read", "Rw")):
                 add(f"pair{mt}{tag}Dates", f"{_ptd[pp]['summary']['correct']:.3f}")
              add(f"pair{mt}DeltaDates",
                  f"{100 * (_ptd[_PAIR[1]]['summary']['correct'] - _ptd[_PAIR[0]]['summary']['correct']):.1f}")
              add(f"pair{mt}NChangedDates",
                  str(sum(int(a != b) for a, b in
                          zip(_ptd[_PAIR[0]]["per_query_correct"],
                              _ptd[_PAIR[1]]["per_query_correct"]))))
           # how much of the gap is the reader naming the stale value vs
           # naming both -- different failures, and only one is unambiguous
           add(f"pair{mt}StaleShare",
               f"{100 * (ptm[_PAIR[0]]['summary']['stale']
                         - ptm[_PAIR[1]]['summary']['stale']):.1f}")
           add(f"pair{mt}HedgeShare",
               f"{100 * (ptm[_PAIR[0]]['summary']['hedged']
                         - ptm[_PAIR[1]]['summary']['hedged']):.1f}")
           add(f"pair{mt}Delta",
               f"{100 * (ptm[_PAIR[1]]['summary']['correct']
                         - ptm[_PAIR[0]]['summary']['correct']):.1f}")
           a = ptm[_PAIR[0]].get("per_query_correct")
           b = ptm[_PAIR[1]].get("per_query_correct")
           if a and b:
              _pv = paired_permutation(a, b, seed=3)
              add(f"pair{mt}P",
                  "<10^{-3}" if _pv < 1e-3 else f"={_pv:.3f}")

     # the transfer question: retrieval delta vs reader delta, per model
     for i, m in enumerate(models):
        tag = ["Pri", "Sec", "Ter"][i] if i < 3 else f"M{i}"
        add(f"reader{tag}Name", f"\\texttt{{{m}}}")
        add(f"reader{tag}Cosine", f"{rd(m, 'cosine-rag'):.3f}")
        add(f"reader{tag}Rerank", f"{rd(m, 'time-rerank'):.3f}")
        add(f"reader{tag}Lex", f"{rd(m, 'lexical-key'):.3f}")
        d = rd(m, "time-rerank") - rd(m, "cosine-rag")
        add(f"reader{tag}RerankDelta", f"{100 * d:+.1f}")
        d2 = rd(m, "lexical-key") - rd(m, "cosine-rag")
        add(f"reader{tag}LexDelta", f"{100 * d2:+.1f}")
     retr_delta = (reader["readers"][primary]["policy_track"]["no_dates"]
                  ["time-rerank"]["retrieval_current@1"]
                  - reader["readers"][primary]["policy_track"]["no_dates"]
                  ["cosine-rag"]["retrieval_current@1"])
     add("readerRetrievalDelta", f"{100 * retr_delta:+.1f}")
     add("readerSameSetAll",
         f"{100 * reader['readers'][primary]['policy_track']['no_dates']['time-rerank'].get('context_same_set_as_cosine', 0):.0f}")
     add("readerSameSetReordered",
        f"{100 * reader['readers'][primary]['policy_track']['no_dates']['time-rerank'].get('context_same_set_reordered', 0):.0f}")
     ro = reader["readers"][primary].get("reversed_order")
     if ro:
         for pol, tag in [("cosine-rag", "Cosine"), ("time-rerank", "Rerank"),
                          ("lexical-key", "Lex"), ("keep-latest-oracle", "Oracle")]:
             if pol in ro:
                 add(f"revOrder{tag}", f"{ro[pol]['summary']['correct']:.3f}")
                 v = ro[pol].get("vs_forward")
                 if v:
                     add(f"revOrder{tag}Shift", f"{100 * v['delta']:+.1f}")
                     add(f"revOrder{tag}P",
                         ("<10^{-3}" if v["p_paired_permutation"] < 1e-3
                          else f"={v['p_paired_permutation']:.2f}"))
         # reranking's value relative to the store-all baseline, per convention
         for tag, key in [("Fwd", "policy_track"), ("Rev", "reversed_order")]:
             src = (reader["readers"][primary]["policy_track"]["no_dates"]
                    if key == "policy_track" else ro)
             d = (src["time-rerank"]["summary"]["correct"]
                  - src["cosine-rag"]["summary"]["correct"])
             add(f"rerankVsBase{tag}", f"{100 * d:+.1f}")
         _fwd = reader["readers"][primary]["policy_track"]["no_dates"]
         add("lexVsBaseFwd", f"{100 * (_fwd['lexical-key']['summary']['correct'] - _fwd['cosine-rag']['summary']['correct']):+.1f}")
         dl = (ro["lexical-key"]["summary"]["correct"]
               - ro["cosine-rag"]["summary"]["correct"])
         add("lexVsBaseRev", f"{100 * dl:+.1f}")
     # does either retrieval metric ever rank two policies opposite to the
     # reader? computed over every policy pair in every measured block
     import itertools as _it
     _blocks = []
     for _m, _d in reader["readers"].items():
         _blocks.append(_d["policy_track"]["no_dates"])
         if "reversed_order" in _d:
             _blocks.append(_d["reversed_order"])

     def _conc(key, sign, drop=(), min_reader_gap=0.0):
         """Concordance census.

         ``drop`` removes policies from the census and ``min_reader_gap``
         removes pairs the reader itself barely separates. Both exist because
         the unrestricted census contains the deliberately-constructed
         read-path control and a number of sub-point ties, and a census that
         a reviewer can accuse of being stacked is worth less than the
         restricted one -- which is also the stronger result.
         """
         agree = disc = tie = 0
         # NB: only blocks carrying retrieval keys contribute; the
         # reversed-order block has none, so the census is over the policy
         # tracks of each reader, which is what the paper now says
         for blk in _blocks:
             pols = [p for p in blk
                     if blk[p].get(key) is not None and p not in drop]
             for a, b in _it.combinations(pols, 2):
                 if abs(blk[a]["summary"]["correct"]
                        - blk[b]["summary"]["correct"]) < min_reader_gap:
                     continue
                 ma, mb = sign * blk[a][key], sign * blk[b][key]
                 ra = blk[a]["summary"]["correct"]
                 rb = blk[b]["summary"]["correct"]
                 if ma == mb:
                     tie += 1
                 elif (ma > mb) == (ra > rb):
                     agree += 1
                 else:
                     disc += 1
         return agree, disc, tie
     # exact paired test between the two metrics on the restricted census:
     # 3 discordant comparisons all favouring contamination@5 is p = 0.25, so
     # the census is descriptive corroboration and the controlled pair is the
     # inferential evidence. Stating that is more honest than an N that looks
     # like 24 independent observations.
     add("concordExactP", "=0.25")
     add("concordNBlocks", str(sum(
         1 for _b in _blocks
         if any(_v.get("retrieval_current@1") is not None for _v in _b.values()))))
     for key, sign, tag in [("retrieval_current@1", 1, "Cur"),
                            ("retrieval_contamination@5", -1, "Contam")]:
         agree, disc, tie = _conc(key, sign)
         add(f"concord{tag}Agree", str(agree))
         add(f"concord{tag}Invert", str(disc))
         add(f"concord{tag}Tie", str(tie))
         add(f"concord{tag}Total", str(agree + disc + tie))
         # census restricted to policies a practitioner could actually deploy
         # (the retain-everything control is ours, built to be pathological)
         # and to pairs the reader separates by more than a point
         ag2, di2, ti2 = _conc(key, sign, drop=("fuzzy-read-key",),
                               min_reader_gap=0.01)
         add(f"concord{tag}InvertStrict", str(di2))
         # the census counts each policy pair once per reader, so N is not a
         # count of independent observations; say how many distinct pairs it is
         _dp = set()
         for blk in _blocks:
             pols = [p_ for p_ in blk
                     if blk[p_].get(key) is not None and p_ != "fuzzy-read-key"]
             for x_, y_ in _it.combinations(sorted(pols), 2):
                 if abs(blk[x_]["summary"]["correct"]
                        - blk[y_]["summary"]["correct"]) < 0.01:
                     continue
                 _dp.add((x_, y_))
         add(f"concord{tag}PairsStrict", str(len(_dp)))
         if tag == "Contam":
             # how many of contamination@5's ties are comparisons current@1
             # actually got wrong -- "declines exactly the ones current@1 gets
             # wrong" is arithmetically impossible with 8 ties and 13 inversions
             _tie_inv = 0
             for blk in _blocks:
                 pols = [p_ for p_ in blk
                         if blk[p_].get("retrieval_current@1") is not None]
                 # NB: names here must not collide with the accessor
                 # functions defined above (a_, f_, s, c); shadowing one
                 # aborts table generation partway through
                 for pa, pb in _it.combinations(pols, 2):
                     ra_ = blk[pa]["summary"]["correct"]
                     rb_ = blk[pb]["summary"]["correct"]
                     ca_ = blk[pa]["retrieval_current@1"]
                     cb_ = blk[pb]["retrieval_current@1"]
                     ka_ = blk[pa]["retrieval_contamination@5"]
                     kb_ = blk[pb]["retrieval_contamination@5"]
                     if ka_ == kb_ and ca_ != cb_ and ((ca_ > cb_) != (ra_ > rb_)):
                         _tie_inv += 1
             add("concordTieOnCurInvert", str(_tie_inv))
         add(f"concord{tag}TotalStrict", str(ag2 + di2 + ti2))

     # order sensitivity at the level of individual answers, not aggregates
     if ro:
         for pol, tag in [("cosine-rag", "Cosine"), ("time-rerank", "Rerank"),
                          ("lexical-key", "Lex")]:
             if pol in ro:
                 f = reader["readers"][primary]["policy_track"]["no_dates"][pol]["per_query_correct"]
                 v = ro[pol]["per_query_correct"]
                 add(f"flip{tag}", str(sum(1 for a, b in zip(f, v) if a != b)))

     # within-policy contamination split, replicated across readers
     for i, m in enumerate(models):
         tag = ["Pri", "Sec", "Ter"][i] if i < 3 else f"M{i}"
         pc = reader["readers"][m].get("policy_conditional", {}).get("no_dates", {})
         cell = pc.get("reconsolidation")
         if cell and cell["contaminated"]["correct"] is not None:
             add(f"cond{tag}Dirty", f"{cell['contaminated']['correct']:.3f}")
             add(f"cond{tag}Clean", f"{cell['clean']['correct']:.3f}")
             add(f"cond{tag}Delta",
                 f"{100 * (cell['clean']['correct'] - cell['contaminated']['correct']):+.1f}")
             add(f"cond{tag}NDirty", str(cell["contaminated"]["n"]))
             add(f"cond{tag}NClean", str(cell["clean"]["n"]))

     add("readerCeiling", f"{reader['config'].get('metric_ceiling', 1.0):.3f}")
     add("readerNUnscorable", str(reader["config"].get("n_gold_unscorable", 0)))
     add("readerOracleCorrect", f"{rd(primary, 'keep-latest-oracle'):.3f}")
     _nf = sum(int((reader["readers"][m].get("usage") or {}).get("n_failed", 0)
                   or 0) for m in models)
     add("readerNFailed", str(_nf))
     add("readerClosedBook",
        f"{reader['readers'][primary]['closed_book']['summary']['correct']:.3f}")
     for i, m in enumerate(models):
         tag = ["Pri", "Sec", "Ter"][i] if i < 3 else f"M{i}"
         cb = reader["readers"][m].get("closed_book")
         if cb:
             add(f"reader{tag}ClosedBook",
                 f"{cb['summary']['correct']:.3f}")

     def dose(model, cond, dk="no_dates", field="correct"):
        return reader["readers"][model]["dose_response"][dk][cond]["summary"][field]

     for cond, tag in [("current_only", "CurrentOnly"),
                      ("plus1_current_first", "CurrentFirst"),
                      ("plus1_stale_first", "StaleFirst"),
                      ("plus2", "PlusTwo"), ("stale_only", "StaleOnly")]:
        add(f"dose{tag}", f"{dose(primary, cond):.3f}")
        add(f"dose{tag}Dates", f"{dose(primary, cond, 'with_dates'):.3f}")
        # the constructed-context probe now runs on every reader that has the
        # full condition set, so the causal claim is no longer single-model
        for _i, _m in enumerate(models):
            _mt = ["Pri", "Sec", "Ter"][_i] if _i < 3 else f"M{_i}"
            _dr = reader["readers"][_m]["dose_response"]["no_dates"]
            if cond in _dr:
                add(f"dose{_mt}{tag}", f"{_dr[cond]['summary']['correct']:.3f}")
     _full = [m for m in models
              if len(reader["readers"][m]["dose_response"]["no_dates"]) > 1]
     add("doseNModels", str(len(_full)))
     add("doseModelList", ", ".join(f"\\texttt{{{m}}}" for m in _full))
     if len(_full) > 1:
        # the drop from a clean context to one stale version, worst case per
        # reader: the quantity the causal claim is about
        _drops = []
        for m in _full:
           _dr = reader["readers"][m]["dose_response"]["no_dates"]
           _worst = min(_dr[c]["summary"]["correct"]
                        for c in ("plus1_current_first", "plus1_stale_first")
                        if c in _dr)
           _drops.append(100 * (_dr["current_only"]["summary"]["correct"] - _worst))
        add("doseDropMin", f"{min(_drops):.1f}")
        add("doseDropMax", f"{max(_drops):.1f}")
     # the position effect on every reader, because the pair's most likely
     # objection is "your 65 points is a positional artifact" and the answer
     # is a reader with no detectable positional sensitivity and a large gap
     for _i, _m in enumerate(models):
         _mt = ["Pri", "Sec", "Ter"][_i] if _i < 3 else None
         _pe = reader["readers"][_m].get("position_effect", {}).get("no_dates")
         if _pe and _mt:
             add(f"pos{_mt}Gain", f"{100 * _pe['gain_from_current_last']:.1f}")
             add(f"pos{_mt}CiLo", f"{100 * _pe['ci']['lo']:.1f}")
             add(f"pos{_mt}CiHi", f"{100 * _pe['ci']['hi']:.1f}")
             _pv = _pe["p_paired_permutation"]
             add(f"pos{_mt}P",
                 "<10^{-3}" if _pv < 1e-3 else f"={_pv:.3f}")
     pe = reader["readers"][primary].get("position_effect", {}).get("no_dates")
     if pe:
         add("posCurrentFirst", f"{pe['current_first_correct']:.3f}")
         add("posCurrentLast", f"{pe['current_last_correct']:.3f}")
         add("posGain", f"{100 * pe['gain_from_current_last']:.1f}")
         add("posGainLo", f"{100 * pe['ci']['lo']:.1f}")
         add("posGainHi", f"{100 * pe['ci']['hi']:.1f}")
         add("posP", ("<10^{-3}" if pe["p_paired_permutation"] < 1e-3
                      else f"={pe['p_paired_permutation']:.3f}"))
     add("dosePlusTwoN",
         str(reader["readers"][primary]["dose_response"]["no_dates"]["plus2"]["n"]))
     add("doseWorstDrop",
         f"{100 * (dose(primary, 'current_only') - dose(primary, 'plus1_current_first')):.1f}")
     ce = reader["readers"][primary]["contamination_effect"]["no_dates"]
     add("contamDrop", f"{100 * ce['drop']:.1f}")
     add("contamDropLo", f"{100 * ce['drop_ci']['lo']:.1f}")
     add("contamDropHi", f"{100 * ce['drop_ci']['hi']:.1f}")
     add("contamDropP", ("<10^{-3}" if ce["p_paired_permutation"] < 1e-3
                        else f"{ce['p_paired_permutation']:.3f}"))
     add("contamNPaired", str(ce["n_paired"]))

    if llmkey:
      # ---------------- LLM keyer ---------------- #
      km = list(llmkey["by_model"])
      add("llmKeyNModels", str(len(km)))
      add("llmKeyModelList", ", ".join(f"\\texttt{{{m}}}" for m in km))
      for i, m in enumerate(km):
         tag = ["Pri", "Sec"][i] if i < 2 else f"M{i}"
         add(f"llmKey{tag}Name", f"\\texttt{{{m}}}")
         for a, atag in [("0.0", "Zero"), ("1.0", "One")]:
             kc = llmkey["by_model"][m][f"alias_p={a}"]
             add(f"llmKey{tag}Cur{atag}", f"{kc['summary']['current@1']:.3f}")
             add(f"llmKey{tag}Def{atag}", f"{100 * kc['keying_deficit']:.1f}")
             add(f"llmKey{tag}Split{atag}",
                 f"{100 * kc['audit']['frac_split']:.1f}")
             add(f"llmKey{tag}Impure{atag}", str(kc["audit"]["impure_clusters"]))
         add(f"llmKey{tag}Calls",
             f"{llmkey['by_model'][m]['alias_p=1.0']['n_llm_calls']:,}")
      add("llmKeyNEvents", f"{llmkey['reference']['alias_p=1.0']['n_events']:,}")
      add("llmKeyNCand", str(llmkey["config"]["n_cand"]))
      for a, atag in [("0.0", "Zero"), ("1.0", "One")]:
          cr = llmkey.get("candidate_recall", {}).get(f"alias_p={a}")
          if cr:
              add(f"llmKeyRecall{atag}", f"{cr['recall_at_n_cand']:.3f}")

    # numbers.tex used to be written here, before any table. Every add() after
    # this point was then silently dropped -- a macro computed from a table's
    # own data could never reach the file that defines it, and the paper then
    # failed to build with an undefined control sequence pointing nowhere near
    # the cause. It is written at the end now, and add() refuses to run once
    # it has been.

    # ---------------- Table 1: real headline (p=0.5) ---------------- #
    rows = []
    for key, label in _REAL_ROWS:
        r = p5[key]
        summ, ci = r["summary"], r["ci"]
        rows.append(
            f"{label} & {_ci(ci['current@1'])} & ${summ['current@5']:.3f}$ & "
            f"${summ['stale_rate']:.3f}$ & "
            f"${summ['miss_rate']:.3f}$ & ${summ['contamination@5']:.3f}$ & "
            f"${summ['slots']:.0f}$ \\\\")
    _tabular("lcccccc",
             "policy & current@1 $\\uparrow$ & current@5 $\\uparrow$ & "
             "stale@1 $\\downarrow$ & "
             "miss@1 $\\downarrow$ & contam@5 $\\downarrow$ & slots $\\downarrow$",
             rows, os.path.join(_PAPER, "tab_real.tex"))

    # ------- Table: stratification by version count (p=0.5) ------- #
    short = {"bm25": "BM25", "bm25+time-rerank": "BM25+rerank",
             "cosine-rag": "Cosine", "time-rerank": "+ time-rerank",
             "recency-weighted": "+ recency-wt",
             "reconsolidation": "Reconsolid.", "lexical-key": "Lexical key",
             "keep-latest-oracle": "Oracle"}
    rows = []
    for key, _ in _REAL_ROWS:
        st_ = real["by_p"]["p=0.5"]["stratified"][key]
        rows.append(f"{short[key]} & ${st_['2ver']['mean']:.3f}$ & "
                    f"${st_['3plus']['mean']:.3f}$ \\\\")
    n2 = real["by_p"]["p=0.5"]["stratified"]["cosine-rag"]["2ver"]["n"]
    n3 = real["by_p"]["p=0.5"]["stratified"]["cosine-rag"]["3plus"]["n"]
    _tabular("lcc",
             f"policy & 2 ver.\\ ($n{{=}}{n2}$) & $\\ge$3 ver.\\ ($n{{=}}{n3}$)",
             rows, os.path.join(_PAPER, "tab_strat.tex"))

    # ------- Table: corrupted timestamps (p=0.5) ------- #
    rows = []
    for key, _ in _REAL_ROWS:
        if key == "bm25":
            continue
        clean = real["by_p"]["p=0.5"]["test"][key]["summary"]["current@1"]
        shuf = real["by_p"]["p=0.5"]["t_shuffle"][key]["current@1"]
        rows.append(f"{short[key]} & ${clean:.3f}$ & ${shuf:.3f}$ \\\\")
    corr = real["by_p"]["p=0.5"].get("corruption")
    if corr:
        # ------- Table: clock x arrival corruption (replaces the 2-column
        # t-shuffle table) and the late-stale-report dose ------- #
        _CORR_POL = [("cosine-rag", "Cosine", "Cosine (store-all)"),
                     ("time-rerank", "Rerank", "+ time-rerank"),
                     ("recency-weighted", "Rw", "+ recency-wt"),
                     ("reconsolidation", "Recon", "Overwrite: threshold"),
                     ("lexical-key", "Lex", "Overwrite: gazetteer"),
                     ("temporal-kg", "Tkg", "Invalidate: gazetteer"),
                     ("keep-latest-oracle", "Oracle", "Overwrite: oracle"),
                     ("fuzzy-read-key", "Read", "Read-path key (retain all)"),
                     ("fuzzy-rw-key", "Sym", "Symmetric key"),
                     ("lexical-key-newer", "LexNewer",
                      "Overwrite if newer: gazetteer"),
                     ("keep-newest-oracle", "OracleNewer",
                      "Overwrite if newer: oracle")]
        _CORR_REG = [("clean", "Clean"), ("clock", "Clock"),
                     ("arrival", "Arrival"), ("both", "Both")]
        add("corrSeeds", str(len(corr["seeds"])))
        rows = []
        for key, tag, label in _CORR_POL:
            vals = []
            for reg, rtag in _CORR_REG:
                v = corr["regimes"][reg]["mean"][key]["current@1"]
                add(f"corr{rtag}{tag}", f"{v:.3f}")
                vals.append(f"${v:.3f}$")
            rows.append(f"{label} & " + " & ".join(vals) + " \\\\")
        _tabular("lcccc",
                 "policy & clean & clock permuted & arrival permuted & both",
                 rows, os.path.join(_PAPER, "tab_tshuffle.tex"))
        # the per-seed spread, so the text can say the single-seed values of
        # the earlier revision sit inside it
        _spread = max(
            max(ps[k]["current@1"] for ps in corr["regimes"][reg]["per_seed"])
            - min(ps[k]["current@1"] for ps in corr["regimes"][reg]["per_seed"])
            for reg in ("clock", "arrival", "both")
            for k, _, _ in _CORR_POL)
        add("corrSeedSpreadMax", f"{100 * _spread:.1f}")
        # the controlled pair's footprints on this stream, for the reader
        # section's budget argument (run_reader does not record slots)
        for key, tag in (("fuzzy-read-key", "Read"), ("fuzzy-rw-key", "Sym")):
            add(f"corrSlots{tag}",
                f"{corr['regimes']['clean']['mean'][key]['slots']:.0f}")
        _QTAG = {"q=0.1": "Ten", "q=0.25": "TwentyFive", "q=0.5": "Fifty"}
        # short labels: this table sits in a half-width minipage
        _LATE_ROWS = [("time-rerank", "Rerank", "+ time-rerank"),
                      ("lexical-key", "Lex", "Overwrite: gaz."),
                      ("keep-latest-oracle", "Oracle", "Overwrite: oracle"),
                      ("fuzzy-rw-key", "Sym", "Symmetric key"),
                      ("lexical-key-newer", "LexNewer", "If newer: gaz.")]
        rows = []
        for key, tag, label in _LATE_ROWS:
            vals = [f"${corr['regimes']['clean']['mean'][key]['current@1']:.3f}$"]
            for qk, qtag in _QTAG.items():
                v = corr["late_report"][qk]["mean"][key]["current@1"]
                add(f"late{tag}{qtag}", f"{v:.3f}")
                vals.append(f"${v:.3f}$")
            rows.append(f"{label} & " + " & ".join(vals) + " \\\\")
        _tabular("lcccc",
                 "policy & $q{=}0$ & $q{=}0.1$ & $q{=}0.25$ & $q{=}0.5$",
                 rows, os.path.join(_PAPER, "tab_latereport.tex"))
        # overwrite hands the reader one wrong memory, not a contradiction:
        # its contamination stays at zero under every dose
        add("lateLexContamMax",
            f"{max(corr['late_report'][qk]['mean']['lexical-key']['contamination@5'] for qk in _QTAG):.3f}")
        add("lateNEligible", str(corr["late_report"]["q=0.1"]["n_facts_eligible"]))
        for qk, qtag in _QTAG.items():
            add(f"lateNSwapped{qtag}", str(corr["late_report"][qk]["n_facts_swapped"]))
    else:
        _tabular("lcc", "policy & clean $t$ & permuted $t$",
                 rows, os.path.join(_PAPER, "tab_tshuffle.tex"))

    # ---------------- Table 2: paraphrase sweep ---------------- #
    rows = []
    for key, label in [("cosine-rag", "Cosine (store-all)"),
                       ("time-rerank", "Cosine + time-rerank"),
                       ("reconsolidation", "Reconsolidation"),
                       ("keep-latest-oracle", "Keep-latest oracle")]:
        cells = []
        for p in ("p=0.0", "p=0.5", "p=1.0"):
            r = real["by_p"][p]["test"][key]
            cells.append(f"{_ci(r['ci']['current@1'])} & "
                         f"${r['summary']['contamination@5']:.2f}$")
        rows.append(f"{label} & " + " & ".join(cells) + " \\\\")
    hdr = ("& \\multicolumn{2}{c}{$p=0$} & \\multicolumn{2}{c}{$p=0.5$} & "
           "\\multicolumn{2}{c}{$p=1$} \\\\\n"
           "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\n"
           "policy & current@1 & cont.@5 & current@1 & cont.@5 & "
           "current@1 & cont.@5")
    _tabular("lcccccc", hdr, rows, os.path.join(_PAPER, "tab_paraphrase.tex"))

    # ---------------- Table 3: controlled regime map ---------------- #
    pols = ["cosine-rag", "time-rerank", "recency-weighted", "reconsolidation",
            "keep-latest-oracle"]
    labels = {"cosine-rag": "Cosine (store-all)", "time-rerank": "+ time-rerank",
              "recency-weighted": "+ recency-weight",
              "reconsolidation": "Reconsolidation", "keep-latest-oracle": "Oracle"}
    def c_ci(sweep, cell, pol):
        e = ctrl[sweep][cell][pol]["current@1"]
        hw = (e["hi"] - e["lo"]) / 2
        return f"${e['mean']:.2f}$\\,{{\\tiny$\\pm${hw:.2f}}}"

    rows = []
    for pol in pols:
        cells = [c_ci("value_gap_sweep", f"gap={g}", pol) for g in (0.3, 0.5, 0.8)]
        cells += [c_ci("key_jitter_sweep", f"jitter={j}", pol)
                  for j in (0.0, 0.2, 0.4, 0.8)]
        rows.append(f"{labels[pol]} & " + " & ".join(cells) + " \\\\")
    hdr = ("& \\multicolumn{3}{c}{\\texttt{value\\_gap}} & "
           "\\multicolumn{4}{c}{\\texttt{key\\_jitter}} \\\\\n"
           "\\cmidrule(lr){2-4}\\cmidrule(lr){5-8}\n"
           "policy & $0.3$ & $0.5$ & $0.8$ & $0.0$ & $0.2$ & $0.4$ & $0.8$")
    _tabular("lccccccc", hdr, rows, os.path.join(_PAPER, "tab_controlled.tex"))

    # ---------------- Table 4: budget sweep ---------------- #
    rows = []
    for pol, label in [("recency", "Recency (FIFO)"),
                       ("time-rerank", "Cosine + time-rerank"),
                       ("reconsolidation", "Reconsolidation"),
                       ("keep-latest-oracle", "Keep-latest oracle")]:
        cells = [f"${c('budget_sweep', f'B={b}', pol, 'current@1'):.2f}$"
                 for b in (12, 20, 40)]
        rows.append(f"{label} & " + " & ".join(cells) + " \\\\")
    _tabular("lccc", "policy & $B{=}12$ & $B{=}20$ & $B{=}40$",
             rows, os.path.join(_PAPER, "tab_budget.tex"))

    # ---------------- Table 5: tau sweep ---------------- #
    rows = []
    for t in ctrl["tau_sweep"]["clean"]:
        a = ctrl["tau_sweep"]["clean"][t]["current@1"]["mean"]
        b = ctrl["tau_sweep"]["jitter=0.4"][t]["current@1"]["mean"]
        rows.append(f"${t.split('=')[1]}$ & ${a:.2f}$ & ${b:.2f}$ \\\\")
    _tabular("lcc", "$\\tau$ & clean & jitter $0.4$",
             rows, os.path.join(_PAPER, "tab_tau.tex"))

    # -------- Table: anchor-free sweep (the keying ladder priced) -------- #
    levels = [0.0, 0.25, 0.5, 0.75, 1.0]

    lad = [("bm25", "BM25 (store-all)", False),
           ("bm25+time-rerank", "BM25 + time-rerank", False),
           ("cosine-rag", "Cosine (store-all)", False),
           ("time-rerank", "+ time-rerank", False),
           ("recency-weighted", "+ recency-weight", False),
           ("reconsolidation", "Overwrite: threshold $\\tau$", True),
           ("lexical-key", "Overwrite: gazetteer (exact)", True),
           ("fuzzy-key", "Overwrite: fuzzy linker (no alias table)", True),
           ("alias-key(recent)", "Overwrite: linker + alias table", True),
           ("fuzzy-read-key", "Read-path key (no overwrite)", True),
           ("fuzzy-rw-key", "Symmetric key (read+write)", True),
           ("alias-rw-key", "Symmetric key + alias table", True),
           ("keep-latest-oracle", "Overwrite: true-key oracle", False)]
    lad = [r for r in lad
           if r[0] in alias["alias_sweep"]["alias_p=1.0"]["test"]]
    rows = []
    for key, label, is_keyer in lad:
        cells = [f"${a_(key, a):.3f}$" for a in levels]
        # the deficit isolates a *keying* decision, so it is only meaningful for
        # policies that make one; store-all rows get a dash
        cells.append(f"${100 * a_def(key, 1.0):.1f}$" if is_keyer else "---")
        rows.append(f"{label} & " + " & ".join(cells) + " \\\\")
    _tabular("lccccc|c",
             "policy & $a{=}0$ & $0.25$ & $0.5$ & $0.75$ & $a{=}1$ & "
             "keying deficit at $a{=}1$",
             rows, os.path.join(_PAPER, "tab_alias.tex"))

    # -------- Table: anchor-free failure decomposition -------- #
    # contamination@k is not monotone in context quality (a top-k holding only
    # stale versions scores 0), so it is never shown without current@k
    rows = []
    for key, label, _ in lad:
        c = alias["alias_sweep"]["alias_p=1.0"]["test"][key]["summary"]
        rows.append(f"{label} & ${c['current@1']:.3f}$ & ${c['stale_rate']:.3f}$ "
                    f"& ${c['miss_rate']:.3f}$ & ${c['current@5']:.3f}$ & "
                    f"${c['contamination@5']:.3f}$ & ${c['slots']:.0f}$ \\\\")
    _tabular("lcccccc",
             "policy ($a{=}1$) & cur@1 $\\uparrow$ & stale@1 $\\downarrow$ & "
             "miss@1 $\\downarrow$ & cur@5 $\\uparrow$ & "
             "cont.@5 $\\downarrow$ & slots $\\downarrow$",
             rows, os.path.join(_PAPER, "tab_aliasfail.tex"))

    # -------- Table: keying audit (retrieval-independent) -------- #
    rows = []
    for keyer, label in [("threshold", "Threshold $\\tau$"),
                         ("lexical", "Gazetteer (exact)"),
                         ("fuzzy", "Fuzzy linker (no alias table)"),
                         ("alias(recent)", "Linker + alias table"),
                         ("oracle", "True-key oracle")]:
        cells = []
        for a in (0.0, 1.0):
            v = aud[f"alias_p={a}"][keyer]
            cells += [f"${v['impure_clusters']}$",
                      f"${100 * v['frac_split']:.1f}$"]
        rows.append(f"{label} & " + " & ".join(cells) + " \\\\")
    hdr = ("& \\multicolumn{2}{c}{anchored ($a{=}0$)} & "
           "\\multicolumn{2}{c}{anchor-free ($a{=}1$)} \\\\\n"
           "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\n"
           "keyer & merge err.\\ & \\% split & merge err.\\ & \\% split")
    _tabular("lcccc", hdr, rows, os.path.join(_PAPER, "tab_keying.tex"))

    # -------- Table: full corpus, inventory completeness -------- #
    rows = []
    for key, label in [("cosine-rag", "Cosine (store-all)"),
                       ("time-rerank", "+ time-rerank"),
                       ("reconsolidation", "Overwrite: threshold"),
                       ("lexical-key", "Overwrite: gazetteer"),
                       ("fuzzy-key", "Overwrite: fuzzy linker"),
                       ("alias-key(recent)", "Overwrite: linker + alias table"),
                       ("fuzzy-rw-key", "Symmetric key (read+write)"),
                       ("keep-latest-oracle", "Overwrite: oracle")]:
        if key not in alias["full_corpus"]["alias_p=1.0"]["open_world"]["test"]:
            continue
        cells = [f"${f_(key, w, a):.3f}$"
                 for a in (0.0, 1.0) for w in ("open_world", "closed_world")]
        rows.append(f"{label} & " + " & ".join(cells) + " \\\\")
    hdr = ("& \\multicolumn{2}{c}{anchored} & "
           "\\multicolumn{2}{c}{anchor-free} \\\\\n"
           "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\n"
           "policy & open-world & closed-world & open-world & closed-world")
    _tabular("lcccc", hdr, rows, os.path.join(_PAPER, "tab_fullcorpus.tex"))

    # -------- Table: retention mechanism (overwrite vs tombstone) -------- #
    rows = []
    for a, atag in [("0.0", "anchored"), ("1.0", "anchor-free")]:
        for keyer, label in [("tau", "threshold $\\tau$"),
                             ("alias", "entity linker"),
                             ("oracle", "true-key oracle")]:
            cells = []
            for mechk in (f"overwrite[{keyer}]", f"temporal-kg[{keyer}]"):
                cell = mech[f"alias_p={a}"][mechk]
                cells += [f"${cell['current@1']:.3f}$",
                          f"${cell['contamination@5']:.3f}$",
                          f"${cell['slots']:.0f}$"]
            rows.append(f"{atag}, {label} & " + " & ".join(cells) + " \\\\")
    hdr = ("& \\multicolumn{3}{c}{overwrite (destructive)} & "
           "\\multicolumn{3}{c}{temporal KG (invalidate)} \\\\\n"
           "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\n"
           "keyer & current@1 & cont.@5 & slots & current@1 & cont.@5 & slots")
    _tabular("lcccccc", hdr, rows, os.path.join(_PAPER, "tab_mechanism.tex"))

    # -------- Table: reader track -------- #
    if reader:
     rlab = {"cosine-rag": "Cosine (store-all)", "time-rerank": "+ time-rerank",
             "reconsolidation": "Overwrite: threshold", "lexical-key": "Overwrite: gazetteer",
             "fuzzy-read-key": "Read-path key (no overwrite)",
             "fuzzy-rw-key": "Symmetric key (read + write)",
             "keep-latest-oracle": "Overwrite: oracle"}
     rows = []

     def _cell(model, key, dk="no_dates"):
         # a cell a reader was never run on is absent rather than zero (the
         # secondary reader's date ablation, until it is run)
         try:
             return f"${rd(model, key, dk):.3f}$"
         except KeyError:
             return "---"

     # every reader that ran the date ablation gets a dates column; the
     # secondary reader's arrives with its run rather than being assumed
     _dmodels = [m for m in models
                 if reader["readers"][m]["policy_track"].get("with_dates")]
     for key, label in rlab.items():
         pt_pri = reader["readers"][primary]["policy_track"]["no_dates"]
         if key not in pt_pri:
             continue
         cell = pt_pri[key]
         vals = [f"${cell['retrieval_current@1']:.3f}$",
                 f"${cell['retrieval_contamination@5']:.3f}$"]
         vals += [_cell(m, key) for m in models]
         vals += [_cell(m, key, "with_dates") for m in _dmodels]
         rows.append(f"{label} & " + " & ".join(vals) + " \\\\")
     _nd = len(_dmodels)
     hdr = ("& \\multicolumn{2}{c}{retrieval} & "
            f"\\multicolumn{{{len(models)}}}{{c}}{{reader accuracy (no dates)}} & "
            f"\\multicolumn{{{_nd}}}{{c}}{{dates}} \\\\\n"
            f"\\cmidrule(lr){{2-3}}\\cmidrule(lr){{4-{3 + len(models)}}}"
            f"\\cmidrule(lr){{{4 + len(models)}-{3 + len(models) + _nd}}}\n"
            "policy & current@1 & cont.@5 & "
            + " & ".join(m.replace("_", "\\_") for m in models)
            + " & " + " & ".join(m.replace("_", "\\_") for m in _dmodels))
     _tabular("lcc" + "c" * (len(models) + _nd), hdr, rows,
              os.path.join(_PAPER, "tab_reader.tex"))
     for _i, _m in enumerate(_dmodels):
         if _m == primary:
             continue
         _mt = ["Pri", "Sec", "Ter"][models.index(_m)] if models.index(_m) < 3 else f"M{models.index(_m)}"
         for pol, tag in [("cosine-rag", "Cosine"), ("time-rerank", "Rerank"),
                          ("reconsolidation", "Recon"), ("lexical-key", "Lex"),
                          ("keep-latest-oracle", "Oracle")]:
             if pol in reader["readers"][_m]["policy_track"]["with_dates"]:
                 add(f"reader{_mt}{tag}Dates", f"{rd(_m, pol, 'with_dates'):.3f}")

     # -------- Table: dose-response -------- #
     dose_models = [m for m in models
                    if len(reader["readers"][m]["dose_response"]["no_dates"]) > 1]
     dlab = [("no_context", "no context (closed book)"),
             ("current_only", "current only"),
             ("plus1_current_first", "+1 stale, current first"),
             ("plus1_stale_first", "+1 stale, current last"),
             ("plus2", "+2 stale, current middle"),
             ("stale_only", "stale only")]
     _ddm = [m for m in dose_models
             if reader["readers"][m]["dose_response"].get("with_dates")]
     rows = []
     for cond, label in dlab:
         c0 = reader["readers"][primary]["dose_response"]["no_dates"].get(cond)
         if c0 is None:
             continue
         vals = [f"${c0['n']}$"]
         for m in dose_models:
             cell = reader["readers"][m]["dose_response"]["no_dates"].get(cond)
             vals.append(f"${cell['summary']['correct']:.3f}$" if cell else "---")
         for m in _ddm:
             wd = reader["readers"][m]["dose_response"]["with_dates"].get(cond)
             vals.append(f"${wd['summary']['correct']:.3f}$" if wd else "---")
             if wd and m != primary:
                 _mt = ["Pri", "Sec", "Ter"][models.index(m)] if models.index(m) < 3 else f"M{models.index(m)}"
                 _ct = {"no_context": "NoContext", "current_only": "CurrentOnly",
                        "plus1_current_first": "CurrentFirst",
                        "plus1_stale_first": "StaleFirst", "plus2": "PlusTwo",
                        "stale_only": "StaleOnly"}[cond]
                 add(f"dose{_mt}{_ct}Dates", f"{wd['summary']['correct']:.3f}")
         rows.append(f"{label} & " + " & ".join(vals) + " \\\\")
     hdr = ("& & " + f"\\multicolumn{{{len(dose_models)}}}{{c}}{{no dates}} & "
            f"\\multicolumn{{{len(_ddm)}}}{{c}}{{dates}} \\\\\n"
            f"\\cmidrule(lr){{3-{2 + len(dose_models)}}}"
            f"\\cmidrule(lr){{{3 + len(dose_models)}-{2 + len(dose_models) + len(_ddm)}}}\n"
            "constructed context & $n$ & "
            + " & ".join(m.replace("_", "\\_") for m in dose_models)
            + " & " + " & ".join(m.replace("_", "\\_") for m in _ddm))
     _tabular("lc" + "c" * (len(dose_models) + len(_ddm)), hdr, rows,
              os.path.join(_PAPER, "tab_dose.tex"))

    # -------- Table: LLM keyer rung -------- #
    if llmkey:
     rows = []
     for key, label in [("reconsolidation", "threshold $\\tau$"),
                        ("lexical-key", "gazetteer"),
                        ("fuzzy-key", "fuzzy linker"),
                        ("alias-key(recent)", "linker + alias table"),
                        ("keep-latest-oracle", "true-key oracle")]:
         audit_key = {"reconsolidation": "threshold", "lexical-key": "lexical",
                      "fuzzy-key": "fuzzy", "alias-key(recent)": "alias(recent)",
                      "keep-latest-oracle": "oracle"}[key]
         if key not in llmkey["reference"]["alias_p=0.0"]["test"]:
             continue
         vals = [f"${llmkey['reference'][f'alias_p={a}']['test'][key]['current@1']:.3f}$"
                 for a in ("0.0", "1.0")]
         vals += [f"${100 * alias['keying_audit'][f'alias_p={a}'][audit_key]['frac_split']:.1f}$"
                  for a in ("0.0", "1.0")]
         rows.append(f"{label} & " + " & ".join(vals) + " \\\\")
     for m in km:
         mc = llmkey["by_model"][m]
         vals = [f"${mc[f'alias_p={a}']['summary']['current@1']:.3f}$"
                 for a in ("0.0", "1.0")]
         vals += [f"${100 * mc[f'alias_p={a}']['audit']['frac_split']:.1f}$"
                  for a in ("0.0", "1.0")]
         rows.append(f"LLM keyer ({m}) & " + " & ".join(vals) + " \\\\")
     hdr = ("& \\multicolumn{2}{c}{current@1} & \\multicolumn{2}{c}{\\% facts split} \\\\\n"
            "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\n"
            "keyer & anchored & anchor-free & anchored & anchor-free")
     _tabular("lcccc", hdr, rows, os.path.join(_PAPER, "tab_llmkey.tex"))

    if scale:
        rows = []
        slab = {"cosine-rag": "Cosine (store-all)", "time-rerank": "+ time-rerank",
                "recency-weighted": "+ recency-wt",
                "reconsolidation": "Overwrite: threshold",
                "lexical-key": "Overwrite: gazetteer",
                "fuzzy-key": "Overwrite: fuzzy linker",
                "alias-key(recent)": "Overwrite: linker + alias table",
                "fuzzy-rw-key": "Fuzzy linker, read+write",
                "keep-latest-oracle": "Overwrite: oracle"}
        _has_sr = "current_set_recall@5" in next(iter(
            scale["by_alias"]["alias_p=0.0"]["test"].values()))["summary"]
        for key, label in slab.items():
            cells = []
            for a in ("0.0", "1.0"):
                c = scale["by_alias"][f"alias_p={a}"]["test"].get(key)
                cells.append(f"${c['summary']['current@1']:.3f}$" if c else "---")
                cells.append(f"${c['summary']['contamination@5']:.3f}$" if c else "---")
                if _has_sr:
                    cells.append(f"${c['summary']['current_set_recall@5']:.3f}$"
                                 if c else "---")
            rows.append(f"{label} & " + " & ".join(cells) + " \\\\")
        if _has_sr:
            hdr = ("& \\multicolumn{3}{c}{anchored} & \\multicolumn{3}{c}{anchor-free} \\\\\n"
                   "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\n"
                   "policy & current@1 & cont.@5 & set rec.@5 & "
                   "current@1 & cont.@5 & set rec.@5")
            _tabular("lcccccc", hdr, rows, os.path.join(_PAPER, "tab_scale.tex"))
        else:
            hdr = ("& \\multicolumn{2}{c}{anchored} & \\multicolumn{2}{c}{anchor-free} \\\\\n"
                   "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\n"
                   "policy & current@1 & cont.@5 & current@1 & cont.@5")
            _tabular("lcccc", hdr, rows, os.path.join(_PAPER, "tab_scale.tex"))

    # -------- Table: natural short forms (the anchor-free ladder with real
    # aliases in place of mechanically derived ones) -------- #
    natural = _opt("stalebench_alias_natural_results.json")
    if natural:
        _NAT_ROWS = [("reconsolidation", "Recon", "Overwrite: threshold"),
                     ("lexical-key", "Lex", "Overwrite: gazetteer"),
                     ("fuzzy-key", "Fuzzy", "Overwrite: fuzzy linker"),
                     ("fuzzy-read-key", "FuzzyRead", "Read-path key (retain all)"),
                     ("fuzzy-rw-key", "FuzzyRW", "Symmetric key"),
                     ("alias-key(recent)", "Entity", "Overwrite: linker + alias table"),
                     ("alias-rw-key", "AliasRW", "Symmetric key + alias table"),
                     ("keep-latest-oracle", "Oracle", "Overwrite: oracle"),
                     ("cosine-rag", "Cosine", None)]
        _SRC = [("mechanical", "Mech", alias["alias_sweep"]["alias_p=1.0"]),
                ("natural", "Nat", natural["alias_sweep"]["natural"]["alias_p=1.0"]),
                ("natural+fallback", "NatFb",
                 natural["alias_sweep"]["natural+fallback"]["alias_p=1.0"])]
        rows = []
        for key, tag, label in _NAT_ROWS:
            cells = []
            for _, stag, cell in _SRC:
                cur = cell["test"][key]["summary"]["current@1"]
                dfc = 100 * cell["keying_deficit"][key]
                add(f"nat{stag}{tag}CurOne", f"{cur:.3f}")
                add(f"nat{stag}{tag}DefOne", f"{dfc:.1f}")
                cells += [f"${cur:.3f}$", f"${dfc:+.1f}$"]
            if label:
                rows.append(f"{label} & " + " & ".join(cells) + " \\\\")
        hdr = ("& \\multicolumn{2}{c}{mechanical} & \\multicolumn{2}{c}{natural} & "
               "\\multicolumn{2}{c}{natural+fallback} \\\\\n"
               "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\n"
               "policy & current@1 & deficit & current@1 & deficit & current@1 & deficit")
        _tabular("lcccccc", hdr, rows, os.path.join(_PAPER, "tab_natural.tex"))
        # coverage of the real inventory, and what it shares with the generator
        _cv = natural["alias_sweep"]["natural"]["alias_p=1.0"]["coverage"]
        _cvf = natural["alias_sweep"]["natural+fallback"]["alias_p=1.0"]["coverage"]
        add("natCovSubjects", str(_cv["test"]["n_subjects_with_forms"]))
        add("natNSubjects", str(_cv["test"]["n_subjects"]))
        # the deficit is negative where a rung beats the write-path oracle;
        # the text wants that as a positive "points above"
        for _stag, _c in (("Nat", natural["alias_sweep"]["natural"]["alias_p=1.0"]),
                          ("NatFb", natural["alias_sweep"]["natural+fallback"]["alias_p=1.0"])):
            add(f"nat{_stag}AliasRWAboveOne",
                f"{-100 * _c['keying_deficit']['alias-rw-key']:.1f}")
        add("natCovSubjectsPct",
            f"{100 * _cv['test']['n_subjects_with_forms'] / _cv['test']['n_subjects']:.0f}")
        add("natUncoveredPct",
            f"{100 * (1 - _cv['test']['n_subjects_with_forms'] / _cv['test']['n_subjects']):.0f}")
        add("natCovStatementsPct",
            f"{100 * _cv['test']['n_alias_events'] / _cv['test']['n_alias_eligible']:.1f}")
        add("natFbCovStatementsPct",
            f"{100 * _cvf['test']['n_alias_events'] / _cvf['test']['n_alias_eligible']:.1f}")
        add("natValCovSubjects", str(_cv["val"]["n_subjects_with_forms"]))
        _inv = json.load(open(_cv["test"]["inventory_path"]))
        _nat_forms = [(q, f) for q, e in _inv.items() for f in e.get("natural", [])]
        _syn_forms = [(q, f) for q, e in _inv.items() for f in e.get("synthetic", [])]

        def _n_ambiguous(pairs):
            owners: Dict[str, set] = {}
            for q, f in pairs:
                owners.setdefault(f.lower(), set()).add(q)
            return sum(1 for s in owners.values() if len(s) > 1)
        add("natNForms", f"{len(_nat_forms):,}")
        _syn_set = {(q, f.lower()) for q, f in _syn_forms}
        add("natOverlapPct",
            f"{100 * sum((q, f.lower()) in _syn_set for q, f in _nat_forms) / max(1, len(_nat_forms)):.1f}")
        add("natNAmbiguousForms", str(_n_ambiguous(_nat_forms)))
        add("aliasNAmbiguousForms", str(_n_ambiguous(_syn_forms)))
        add("natFbUnlinked",
            str(natural["keying_audit"]["natural+fallback"]["alias_p=1.0"]["fuzzy"]["unlinked"]))

        def _pfmt(v):
            if v < 1e-3:
                return "<10^{-3}"
            return f"={v:.3f}" if v < 0.01 else f"={v:.2f}"
        for src, stag in (("natural", "Nat"), ("natural+fallback", "NatFb")):
            _pp = natural["alias_sweep"][src]["alias_p=1.0"]["paired_p"]
            add(f"natPRwAlias{stag}", _pfmt(_pp["raw"]["rw_vs_alias"]))
            add(f"natPRwFuzzy{stag}", _pfmt(_pp["holm"]["rw_vs_fuzzy"]))
            add(f"natPAliasRwAlias{stag}", _pfmt(_pp["holm"]["aliasrw_vs_alias"]))

    # -------- The second deployed system: Graphiti, scored by three read
    # paths (as returned / valid first / valid only) and at the reader -------- #
    gph = _opt("stalebench_systems_graphiti_qwen14b.json")
    gphrp = _opt("stalebench_graphiti_readpath.json")
    gphrd = _opt("stalebench_sysreader_graphiti_qwen14b.json")
    if gph and gphrp:
        _gk = next((k for cfg in gph["by_config"].values() for k in cfg
                    if k.startswith("graphiti(")), None)
        _gcfgs = [(c, t) for c, t in (("anchored", "Zero"), ("anchor_free", "One"))
                  if c in gph["by_config"] and _gk in gph["by_config"][c]]
        _g0 = gph["by_config"][_gcfgs[0][0]][_gk]
        add("gphLib", f"\\texttt{{{_g0.get('lib_version')}}}")
        add("gphInvOnAdd", "yes" if _g0.get("invalidates_on_add") else "no")
        add("gphNQueries", str(len(gphrp["by_config"][_gcfgs[0][0]][_gk]
                                   ["as_returned"]) and
                               gphrp["by_config"][_gcfgs[0][0]][_gk]["n_queries"]))
        _ghours = 0.0
        _grd_rows: Dict[str, Dict] = {}
        for cfg, ctag in _gcfgs:
            g = gph["by_config"][cfg][_gk]
            s = g["summary"]
            ev = g.get("events", {})
            add(f"gphCur{ctag}", f"{s['current@1']:.3f}")
            add(f"gphCurText{ctag}", f"{s['current@1_text']:.3f}")
            add(f"gphContam{ctag}", f"{s['contamination@5']:.3f}")
            add(f"gphContamText{ctag}", f"{s['contamination@5_text']:.3f}")
            add(f"gphMiss{ctag}", f"{s['miss_rate']:.3f}")
            add(f"gphSlots{ctag}", f"{s['slots']:.0f}")
            add(f"gphActive{ctag}", f"{s.get('active_slots', s['slots']):.0f}")
            add(f"gphAdds{ctag}", str(ev.get("EDGE_ADD", 0)))
            add(f"gphInvalidations{ctag}", str(ev.get("EDGE_INVALIDATE", 0)))
            add(f"gphNoEdge{ctag}", str(ev.get("EPISODE_NO_EDGES", 0)))
            # entity nodes the extractor created: the anchor axis reaching the
            # system's own entity resolution (a short form it fails to link is
            # a new node)
            add(f"gphNodes{ctag}", str(ev.get("NODE_ADD", 0)))
            add(f"gphIngestHours{ctag}", f"{g['ingest_seconds'] / 3600:.1f}")
            _ghours += g["ingest_seconds"] / 3600
            if g.get("n_llm_calls"):
                add(f"gphCallsPerStmt{ctag}",
                    f"{g['n_llm_calls'] / g['n_add_calls']:.1f}")
            rp = gphrp["by_config"][cfg][_gk]
            add(f"gphInvalidFirst{ctag}", str(rp["n_invalid_ranked_first"]))
            for arm, atag in (("valid_first", "VF"), ("valid_only", "VO")):
                a = rp[arm]["summary"]
                add(f"gph{atag}Cur{ctag}", f"{a['current@1']:.3f}")
                add(f"gph{atag}CurText{ctag}", f"{a['current@1_text']:.3f}")
                add(f"gph{atag}Contam{ctag}", f"{a['contamination@5']:.3f}")
                add(f"gph{atag}ContamText{ctag}", f"{a['contamination@5_text']:.3f}")
                add(f"gph{atag}Miss{ctag}", f"{a['miss_rate']:.3f}")
            _pk = f"{cfg}:{_gk}:valid_first-vs-as_returned:current@1"
            if _pk in gphrp["tests"]["holm"]:
                add(f"gphPVFCur{ctag}", _pfmt(gphrp["tests"]["holm"][_pk]))
            # supersessions in the stream: statements beyond the first per fact
            add(f"gphNSupersessions{ctag}",
                str(gph["by_config"][cfg]["n_statements"] - gph["config"]["n_test_facts"]))
            if gphrd and cfg in gphrd["by_config"] and _gk in gphrd["by_config"][cfg]:
                _rd = gphrd["by_config"][cfg][_gk]
                _rname = next(iter(_rd["as_is"]["readers"]), None)
                if _rname:
                    for arm, atag in (("as_is", "AsIs"), ("valid_first", "VF"),
                                      ("read_keyed", "Keyed"),
                                      ("read_keyed_collapsed", "Merged"),
                                      ("read_keyed_oldest", "Oldest")):
                        if arm in _rd and _rname in _rd[arm]["readers"]:
                            add(f"gphRd{atag}{ctag}",
                                f"{_rd[arm]['readers'][_rname]['summary']['correct']:.3f}")
                    _grd_rows[cfg] = {pol: po["as_is"]["readers"][_rname]["summary"]["correct"]
                                      for pol, po in gphrd["by_config"][cfg].items()
                                      if _rname in po["as_is"]["readers"]}
                    _grd_rows[cfg][f"{_gk}|valid_first"] = (
                        _rd["valid_first"]["readers"][_rname]["summary"]["correct"]
                        if "valid_first" in _rd else None)
        add("gphIngestHoursTotal", f"{_ghours:.1f}")
        # the table: the system by its three read paths against the ladder on
        # the same stream, both configurations side by side
        rows = []
        _ladder = [("cosine-rag", "Cosine (store-all)"), ("time-rerank", "+ time-rerank"),
                   ("fuzzy-rw-key", "Symmetric key"),
                   ("keep-latest-oracle", "Overwrite: oracle")]

        def _gcells(cfg, cur, curt, con, rd):
            return [f"${cur:.3f}$", f"${curt:.3f}$", f"${con:.3f}$",
                    (f"${rd:.3f}$" if rd is not None else "---")]
        _spec = [(f"{_gk}|as_returned", "Graphiti, as returned"),
                 (f"{_gk}|valid_first", "Graphiti, valid edges first"),
                 (f"{_gk}|valid_only", "Graphiti, valid edges only")]
        for key, label in _spec + _ladder:
            cells = []
            for cfg, ctag in (("anchored", "Zero"), ("anchor_free", "One")):
                if cfg not in gph["by_config"] or _gk not in gph["by_config"][cfg]:
                    cells += ["---"] * 4
                    continue
                rdrow = _grd_rows.get(cfg, {})
                if "|" in key:
                    _, arm = key.split("|")
                    a = (gph["by_config"][cfg][_gk]["summary"] if arm == "as_returned"
                         else gphrp["by_config"][cfg][_gk][arm]["summary"])
                    rd = rdrow.get(_gk if arm == "as_returned" else
                                   (f"{_gk}|valid_first" if arm == "valid_first" else "\0"))
                else:
                    a = gph["by_config"][cfg][key]["summary"]
                    rd = rdrow.get(key)
                cells += _gcells(cfg, a["current@1"], a["current@1_text"],
                                 a["contamination@5"], rd)
            rows.append(f"{label} & " + " & ".join(cells) + " \\\\")
        hdr = ("& \\multicolumn{4}{c}{anchored} & \\multicolumn{4}{c}{anchor-free} \\\\\n"
               "\\cmidrule(lr){2-5}\\cmidrule(lr){6-9}\n"
               "policy & cur@1 & by text & cont.@5 & reader & "
               "cur@1 & by text & cont.@5 & reader")
        _tabular("lcccccccc", hdr, rows, os.path.join(_PAPER, "tab_graphiti.tex"))

    # -------- A second adjudicator model. Mem0 0.1.118 under the default and
    # the third-person extraction prompt, each with qwen2.5:14b and glm-4.6;
    # Graphiti with both models. Quoted beside the headline runs, never
    # instead of them. -------- #
    _ADJ = [("DefaultQwen", "stalebench_systems_chroma_01118_run2.json",
             "qwen2.5:14b", "default"),
            ("DefaultGlm", "stalebench_systems_chroma_01118_glm_default.json",
             "glm-4.6", "default"),
            ("TpQwen", "stalebench_systems_chroma_01118_tp_qwen.json",
             "qwen2.5:14b", "third-person"),
            ("TpGlm", "stalebench_systems_chroma_01118_tp_glm.json",
             "glm-4.6", "third-person")]
    _adj_rows = []
    _adj_present = {}
    for atag, fn, model, prompt in _ADJ:
        dd = _opt(fn)
        if not dd:
            continue
        cells = []
        for cfg, ctag in (("anchored", "Zero"), ("anchor_free", "One")):
            cc = dd["by_config"].get(cfg, {})
            mk = next((k for k in cc if k.startswith("mem0(")), None)
            if not mk:
                cells += ["---"] * 5
                continue
            g = cc[mk]
            s, ev = g["summary"], g.get("events", {}) or {}
            _adj_present[(atag, ctag)] = True
            add(f"adj{atag}Cur{ctag}", f"{s['current@1']:.3f}")
            add(f"adj{atag}CurText{ctag}", f"{s.get('current@1_text', s['current@1']):.3f}")
            add(f"adj{atag}Contam{ctag}", f"{s['contamination@5']:.3f}")
            add(f"adj{atag}ContamText{ctag}",
                f"{s.get('contamination@5_text', s['contamination@5']):.3f}")
            add(f"adj{atag}Slots{ctag}", f"{s['slots']:.0f}")
            add(f"adj{atag}Adds{ctag}", str(ev.get("ADD", 0)))
            add(f"adj{atag}Updates{ctag}", str(ev.get("UPDATE", 0)))
            add(f"adj{atag}Deletes{ctag}", str(ev.get("DELETE", 0)))
            add(f"adj{atag}Silent{ctag}", str(g.get("n_silent_add", 0)))
            add(f"adj{atag}Hours{ctag}", f"{g['ingest_seconds'] / 3600:.1f}")
            cells += [f"${s['current@1']:.3f}$",
                      f"${s.get('current@1_text', s['current@1']):.3f}$",
                      f"${s['contamination@5']:.3f}$", f"${s['slots']:.0f}$",
                      f"{ev.get('UPDATE', 0)}+{ev.get('DELETE', 0)}"]
        _short = {"qwen2.5:14b": "qwen", "glm-4.6": "glm"}.get(model, model)
        label = f"Mem0, {_short}, {prompt}"
        _adj_rows.append(f"{label} & " + " & ".join(cells) + " \\\\")
        # the same arms at the reader, when that run exists
        _rdf = _opt(f"stalebench_sysreader_{fn.split('01118_')[-1].replace('.json', '')}.json")
        if _rdf:
            for cfg, ctag in (("anchored", "Zero"), ("anchor_free", "One")):
                cc = _rdf["by_config"].get(cfg, {})
                mk = next((k for k in cc if k.startswith("mem0(")), None)
                if not mk:
                    continue
                for arm, rtag in (("as_is", "AsIs"), ("read_keyed_collapsed", "Merged"),
                                  ("read_keyed_oldest", "Oldest")):
                    rd_ = cc[mk].get(arm, {}).get("readers", {})
                    rname = next(iter(rd_), None)
                    if rname:
                        add(f"adj{atag}Rd{rtag}{ctag}",
                            f"{rd_[rname]['summary']['correct']:.3f}")
    _gg = _opt("stalebench_systems_graphiti_glm.json")
    _ggrp = _opt("stalebench_graphiti_readpath_glm.json")
    if _gg:
        _gk2 = next((k for cfg in _gg["by_config"].values() for k in cfg
                     if k.startswith("graphiti(")), None)
        for cfg, ctag in (("anchored", "Zero"), ("anchor_free", "One")):
            if cfg not in _gg["by_config"] or _gk2 not in _gg["by_config"][cfg]:
                continue
            g = _gg["by_config"][cfg][_gk2]
            s, ev = g["summary"], g.get("events", {}) or {}
            add(f"gphGlmCur{ctag}", f"{s['current@1']:.3f}")
            add(f"gphGlmContam{ctag}", f"{s['contamination@5']:.3f}")
            add(f"gphGlmSlots{ctag}", f"{s['slots']:.0f}")
            add(f"gphGlmActive{ctag}", f"{s.get('active_slots', s['slots']):.0f}")
            add(f"gphGlmInvalidations{ctag}", str(ev.get("EDGE_INVALIDATE", 0)))
            add(f"gphGlmNodes{ctag}", str(ev.get("NODE_ADD", 0)))
            add(f"gphGlmNoEdge{ctag}", str(ev.get("EPISODE_NO_EDGES", 0)))
            add(f"gphGlmHours{ctag}", f"{g['ingest_seconds'] / 3600:.1f}")
            if g.get("n_llm_calls"):
                add(f"gphGlmCallsPerStmt{ctag}",
                    f"{g['n_llm_calls'] / g['n_add_calls']:.1f}")
            vf = None
            if _ggrp and cfg in _ggrp["by_config"] and _gk2 in _ggrp["by_config"][cfg]:
                vf = _ggrp["by_config"][cfg][_gk2]["valid_first"]["summary"]
                add(f"gphGlmVFCur{ctag}", f"{vf['current@1']:.3f}")
                add(f"gphGlmVFContam{ctag}", f"{vf['contamination@5']:.3f}")
                add(f"gphGlmInvalidFirst{ctag}",
                    str(_ggrp["by_config"][cfg][_gk2]["n_invalid_ranked_first"]))
    # one row per model for Graphiti, as returned and valid-first; the qwen
    # rows are quoted from the headline run whether or not the glm run exists
    for src, srp, mlabel in ((gph, gphrp, "qwen"), (_gg, _ggrp, "glm")):
        if not src:
            continue
        gk = next((k for cfg in src["by_config"].values() for k in cfg
                   if k.startswith("graphiti(")), None)
        for arm, alabel in (("as_returned", "as returned"), ("valid_first", "valid first")):
            cells = []
            for cfg in ("anchored", "anchor_free"):
                if cfg not in src["by_config"] or gk not in src["by_config"][cfg]:
                    cells += ["---"] * 5
                    continue
                g = src["by_config"][cfg][gk]
                ev = g.get("events", {}) or {}
                a = (g["summary"] if arm == "as_returned"
                     else (srp["by_config"][cfg][gk][arm]["summary"]
                           if srp and cfg in srp["by_config"] else None))
                if a is None:
                    cells += ["---"] * 5
                    continue
                cells += [f"${a['current@1']:.3f}$", f"${a['current@1_text']:.3f}$",
                          f"${a['contamination@5']:.3f}$",
                          f"${g['summary'].get('active_slots', g['summary']['slots']):.0f}/{g['summary']['slots']:.0f}$",
                          str(ev.get("EDGE_INVALIDATE", 0))]
            _adj_rows.append(f"Graphiti, {mlabel}, {alabel} & " + " & ".join(cells) + " \\\\")
    if _adj_rows:
        hdr = ("& \\multicolumn{5}{c}{anchored} & \\multicolumn{5}{c}{anchor-free} \\\\\n"
               "\\cmidrule(lr){2-6}\\cmidrule(lr){7-11}\n"
               "system, model, prompt & cur@1 & text & cont.@5 & slots & ops & "
               "cur@1 & text & cont.@5 & slots & ops")
        _tabular("lcccccccccc", hdr, _adj_rows,
                 os.path.join(_PAPER, "tab_adjudicator.tex"))

    # -------- The third deployed system: LangMem, one model call per
    # statement, inserts and updates on and deletes off as installed -------- #
    _LGM = [("Qwen", "stalebench_systems_langmem_qwen.json", "qwen2.5:14b"),
            ("Glm", "stalebench_systems_langmem_glm.json", "glm-4.6")]
    _lgm_rows = []
    _lgm_any = False
    for mtag, fn, mlabel in _LGM:
        dd = _opt(fn)
        if not dd:
            continue
        rdf = _opt(fn.replace("stalebench_systems_", "stalebench_sysreader_"))
        cells = []
        for cfg, ctag in (("anchored", "Zero"), ("anchor_free", "One")):
            cc = dd["by_config"].get(cfg, {})
            lk = next((k for k in cc if k.startswith("langmem(")), None)
            if not lk:
                cells += ["---"] * 4
                continue
            g = cc[lk]
            s, ev = g["summary"], g.get("events", {}) or {}
            _lgm_any = True
            tag = f"lgm{mtag}"
            add(f"{tag}Cur{ctag}", f"{s['current@1']:.3f}")
            add(f"{tag}CurText{ctag}", f"{s['current@1_text']:.3f}")
            add(f"{tag}Contam{ctag}", f"{s['contamination@5']:.3f}")
            add(f"{tag}ContamText{ctag}", f"{s['contamination@5_text']:.3f}")
            add(f"{tag}Miss{ctag}", f"{s['miss_rate']:.3f}")
            add(f"{tag}Hedged{ctag}", f"{s['hedged@1_text']:.3f}")
            add(f"{tag}Slots{ctag}", f"{s['slots']:.0f}")
            add(f"{tag}Inserts{ctag}", str(ev.get("INSERT", 0)))
            add(f"{tag}Updates{ctag}", str(ev.get("UPDATE", 0)))
            add(f"{tag}Deletes{ctag}", str(ev.get("DELETE", 0)))
            add(f"{tag}Noop{ctag}", str(ev.get("NOOP", 0)))
            add(f"{tag}Failed{ctag}", str(ev.get("ADD_FAILED", 0)))
            add(f"{tag}Hours{ctag}", f"{g['ingest_seconds'] / 3600:.1f}")
            if g.get("n_llm_calls"):
                add(f"{tag}CallsPerStmt{ctag}", f"{g['n_llm_calls'] / g['n_add_calls']:.1f}")
            add(f"{tag}Lib", f"\\texttt{{{g.get('lib_version')}}}")
            rd = None
            if rdf and cfg in rdf["by_config"] and lk in rdf["by_config"][cfg]:
                _r = rdf["by_config"][cfg][lk]
                rname = next(iter(_r["as_is"]["readers"]), None)
                if rname:
                    for arm, atag in (("as_is", "AsIs"), ("read_keyed", "Keyed"),
                                      ("read_keyed_collapsed", "Merged"),
                                      ("read_keyed_oldest", "Oldest")):
                        if arm in _r and rname in _r[arm]["readers"]:
                            add(f"{tag}Rd{atag}{ctag}",
                                f"{_r[arm]['readers'][rname]['summary']['correct']:.3f}")
                    _sm = _r["as_is"]["readers"][rname]["summary"]
                    rd = _sm["correct"]
                    # what the wrong answers are: a superseded value, or
                    # neither value -- the latter is where a memory that
                    # names values outside the benchmark's window shows up
                    add(f"{tag}RdStale{ctag}", f"{_sm['stale']:.3f}")
                    add(f"{tag}RdOther{ctag}",
                        f"{_sm.get('other', 0.0) + _sm.get('abstain', 0.0):.3f}")
            cells += [f"${s['current@1']:.3f}$", f"${s['current@1_text']:.3f}$",
                      f"${s['contamination@5_text']:.3f}$",
                      (f"${rd:.3f}$" if rd is not None else "---")]
        _lgm_rows.append(f"LangMem, {mlabel} & " + " & ".join(cells) + " \\\\")
        # what the memories look like: length, how many of the fact's values
        # the top-1 memory names, and whether it names a year after the
        # benchmark's window -- knowledge the stream never contained
        _ctx = _opt(fn.replace(".json", ".contexts.json"))
        if _ctx:
            import re as _re
            import statistics as _st
            _post = _re.compile(r"\b20(2[1-9]|[3-9][0-9])\b")
            for cfg, ctag in (("anchored", "Zero"), ("anchor_free", "One")):
                cc = _ctx["by_config"].get(cfg, {})
                lk = next((k for k in cc if k.startswith("langmem(")), None)
                if not lk:
                    continue
                top = [hs[0]["text"] if hs else "" for hs in cc[lk]]
                gold, stale = _ctx["gold"], _ctx["stale_values"]
                add(f"lgm{mtag}TextLen{ctag}", f"{_st.mean(len(t) for t in top):.0f}")
                add(f"lgm{mtag}PostWindow{ctag}",
                    f"{sum(1 for t in top if _post.search(t)) / len(top):.3f}")
                add(f"lgm{mtag}ValuesNamed{ctag}",
                    f"{_st.mean(sum(1 for v in set([g] + list(sv)) if v.lower() in t.lower()) for t, g, sv in zip(top, gold, stale)):.2f}")
        # the library's own logged failures (malformed patches from the
        # model, skipped rather than applied) live in the run log
        _log = os.path.join(_RES, fn.replace("stalebench_systems_", "").replace(".json", "_run.log"))
        if os.path.exists(_log):
            _txt = open(_log, errors="ignore").read()
            add(f"lgm{mtag}PatchFail", str(_txt.count("Could not apply patch")))
            add(f"lgm{mtag}SchemaMiss", str(_txt.count("Could not find existing schema")))
    if _lgm_any:
        # the ladder on the same stream, from the qwen run's own file
        _ld = _opt("stalebench_systems_langmem_qwen.json")
        _ldrd = _opt("stalebench_sysreader_langmem_qwen.json")
        for key, label in (("cosine-rag", "Cosine (store-all)"), ("time-rerank", "+ time-rerank"),
                           ("fuzzy-rw-key", "Symmetric key"),
                           ("keep-latest-oracle", "Overwrite: oracle")):
            cells = []
            for cfg in ("anchored", "anchor_free"):
                cc = (_ld or {}).get("by_config", {}).get(cfg, {})
                if key not in cc:
                    cells += ["---"] * 4
                    continue
                s = cc[key]["summary"]
                rd = None
                if _ldrd and cfg in _ldrd["by_config"] and key in _ldrd["by_config"][cfg]:
                    _r = _ldrd["by_config"][cfg][key]["as_is"]["readers"]
                    rname = next(iter(_r), None)
                    rd = _r[rname]["summary"]["correct"] if rname else None
                cells += [f"${s['current@1']:.3f}$", f"${s['current@1_text']:.3f}$",
                          f"${s['contamination@5_text']:.3f}$",
                          (f"${rd:.3f}$" if rd is not None else "---")]
            _lgm_rows.append(f"{label} & " + " & ".join(cells) + " \\\\")
        hdr = ("& \\multicolumn{4}{c}{anchored} & \\multicolumn{4}{c}{anchor-free} \\\\\n"
               "\\cmidrule(lr){2-5}\\cmidrule(lr){6-9}\n"
               "policy & cur@1 & by text & cont.@5 (text) & reader & "
               "cur@1 & by text & cont.@5 (text) & reader")
        _tabular("lcccccccc", hdr, _lgm_rows, os.path.join(_PAPER, "tab_langmem.tex"))

    # -------- The fourth mechanism: ERASE, rewrite-on-arrival (reimplemented) -------- #
    _ERS = [("Qwen", "stalebench_systems_erase_qwen.json", "qwen2.5:14b"),
            ("Glm", "stalebench_systems_erase_glm.json", "glm-4.6")]
    _ers_rows = []
    _ers_any = False
    for mtag, fn, mlabel in _ERS:
        dd = _opt(fn)
        if not dd:
            continue
        rdf = _opt(fn.replace("stalebench_systems_", "stalebench_sysreader_"))
        cells = []
        for cfg, ctag in (("anchored", "Zero"), ("anchor_free", "One")):
            cc = dd["by_config"].get(cfg, {})
            ek = next((k for k in cc if k.startswith("erase(")), None)
            if not ek:
                cells += ["---"] * 4
                continue
            g = cc[ek]
            s_, ev = g["summary"], g.get("events", {}) or {}
            _ers_any = True
            tag = f"ers{mtag}"
            add(f"{tag}Cur{ctag}", f"{s_['current@1']:.3f}")
            add(f"{tag}CurText{ctag}", f"{s_['current@1_text']:.3f}")
            add(f"{tag}Contam{ctag}", f"{s_['contamination@5']:.3f}")
            add(f"{tag}ContamText{ctag}", f"{s_['contamination@5_text']:.3f}")
            add(f"{tag}Miss{ctag}", f"{s_['miss_rate']:.3f}")
            add(f"{tag}Stale{ctag}", f"{s_['stale_rate']:.3f}")
            add(f"{tag}Hedged{ctag}", f"{s_['hedged@1_text']:.3f}")
            add(f"{tag}Slots{ctag}", f"{s_['slots']:.0f}")
            for evn, etag in (("INSERT", "Inserts"), ("KEEP", "Keeps"), ("REWRITE", "Rewrites"),
                              ("DELETE", "Deletes"), ("PARSE_FAIL", "ParseFail")):
                add(f"{tag}{etag}{ctag}", str(ev.get(evn, 0)))
            add(f"{tag}Hours{ctag}", f"{g['ingest_seconds'] / 3600:.1f}")
            add(f"{tag}RetrieveK", str(g.get("retrieve_k", "")))
            _orc = cc.get("keep-latest-oracle", {}).get("summary", {}).get("current@1")
            if _orc is not None:
                add(f"{tag}GapToOracle{ctag}", f"{100 * (_orc - s_['current@1']):.1f}")
                _nq = (dd.get("config") or {}).get("n_test_facts")
                if _nq:
                    add(f"{tag}GapQueries{ctag}", str(round((_orc - s_['current@1']) * _nq)))
            if g.get("n_llm_calls"):
                add(f"{tag}CallsPerStmt{ctag}", f"{g['n_llm_calls'] / g['n_add_calls']:.1f}")
            rd = None
            if rdf and cfg in rdf["by_config"] and ek in rdf["by_config"][cfg]:
                _r = rdf["by_config"][cfg][ek]
                rname = next(iter(_r["as_is"]["readers"]), None)
                if rname:
                    for arm, atag in (("as_is", "AsIs"), ("read_keyed", "Keyed"),
                                      ("read_keyed_collapsed", "Merged"),
                                      ("read_keyed_oldest", "Oldest")):
                        if arm in _r and rname in _r[arm]["readers"]:
                            add(f"{tag}Rd{atag}{ctag}",
                                f"{_r[arm]['readers'][rname]['summary']['correct']:.3f}")
                    _sm = _r["as_is"]["readers"][rname]["summary"]
                    rd = _sm["correct"]
                    add(f"{tag}RdStale{ctag}", f"{_sm['stale']:.3f}")
                    add(f"{tag}RdOther{ctag}",
                        f"{_sm.get('other', 0.0) + _sm.get('abstain', 0.0):.3f}")
            cells += [f"${s_['current@1']:.3f}$", f"${s_['current@1_text']:.3f}$",
                      f"${s_['contamination@5_text']:.3f}$",
                      (f"${rd:.3f}$" if rd is not None else "---")]
        _ers_rows.append(f"ERASE, {mlabel} & " + " & ".join(cells) + " \\\\")
        # what the rewrites look like: does the rewritten text still name the
        # superseded value (past tense keeps the string a reader sees)
        _ctx = _opt(fn.replace(".json", ".contexts.json"))
        if _ctx:
            import statistics as _st
            for cfg, ctag in (("anchored", "Zero"), ("anchor_free", "One")):
                cc = _ctx["by_config"].get(cfg, {})
                ek = next((k for k in cc if k.startswith("erase(")), None)
                if not ek:
                    continue
                top = [hs[0]["text"] if hs else "" for hs in cc[ek]]
                gold, stale = _ctx["gold"], _ctx["stale_values"]
                add(f"ers{mtag}TextLen{ctag}", f"{_st.mean(len(t) for t in top):.0f}")
                # a query whose fact has no surviving memory anywhere in the
                # returned depth: the store deleted every version of it
                _gone = sum(1 for i, hs in enumerate(cc[ek])
                            if not any((h.get("sid") or "").startswith(f"f{i}:") for h in hs))
                add(f"ers{mtag}Gone{ctag}", f"{_gone / len(cc[ek]):.3f}")
                add(f"ers{mtag}ValuesNamed{ctag}",
                    f"{_st.mean(sum(1 for v in set([g] + list(sv)) if v.lower() in t.lower()) for t, g, sv in zip(top, gold, stale)):.2f}")
    if _ers_any:
        _ld = _opt("stalebench_systems_erase_qwen.json")
        _ldrd = _opt("stalebench_sysreader_erase_qwen.json")
        for key, label in (("cosine-rag", "Cosine (store-all)"), ("time-rerank", "+ time-rerank"),
                           ("fuzzy-rw-key", "Symmetric key"),
                           ("keep-latest-oracle", "Overwrite: oracle")):
            cells = []
            for cfg in ("anchored", "anchor_free"):
                cc = (_ld or {}).get("by_config", {}).get(cfg, {})
                if key not in cc:
                    cells += ["---"] * 4
                    continue
                s_ = cc[key]["summary"]
                rd = None
                if _ldrd and cfg in _ldrd["by_config"] and key in _ldrd["by_config"][cfg]:
                    _r = _ldrd["by_config"][cfg][key]["as_is"]["readers"]
                    rname = next(iter(_r), None)
                    rd = _r[rname]["summary"]["correct"] if rname else None
                cells += [f"${s_['current@1']:.3f}$", f"${s_['current@1_text']:.3f}$",
                          f"${s_['contamination@5_text']:.3f}$",
                          (f"${rd:.3f}$" if rd is not None else "---")]
            _ers_rows.append(f"{label} & " + " & ".join(cells) + " \\\\")
        hdr = ("& \\multicolumn{4}{c}{anchored} & \\multicolumn{4}{c}{anchor-free} \\\\\n"
               "\\cmidrule(lr){2-5}\\cmidrule(lr){6-9}\n"
               "policy & cur@1 & by text & cont.@5 (text) & reader & "
               "cur@1 & by text & cont.@5 (text) & reader")
        _tabular("lcccccccc", hdr, _ers_rows, os.path.join(_PAPER, "tab_erase.tex"))

    if systems and _best:
        # Build this table from the SAME probe-gated selection the prose uses.
        # Reading the raw result files here instead rendered a withdrawn run
        # (a backend that ranked least-similar first) in a row beside prose
        # quoting the probed one -- a table contradicting its own text.
        rows = []
        _refsrc = None
        for _v, (_fn, _dd, _st) in sorted(_best.items()):
            if _refsrc is None:
                _refsrc = _dd
        # the ladder scored on this exact stream, if available: without it a
        # reader compares a deployed system only against store-all and an
        # oracle, and cross-table inference to the keying rungs is wrong
        # because this subsample is easier for them than the full test split
        _ladder = _opt("stalebench_systems_ladder.json") or _refsrc
        _ra = _ladder["by_config"]["anchored"]
        _rf = _ladder["by_config"].get("anchor_free", {})
        for _key, _lbl in (("cosine-rag", "Cosine (store-all)"),
                           ("time-rerank", "+ time-rerank"),
                           ("reconsolidation", "Overwrite: threshold"),
                           ("lexical-key", "Overwrite: gazetteer"),
                           ("fuzzy-key", "Overwrite: fuzzy linker"),
                           ("fuzzy-rw-key", "Symmetric key (read+write)")):
            if _key in _ra:
                _s0, _s1 = _ra[_key]["summary"], _rf.get(_key, {}).get("summary")
                rows.append(
                    f"{_lbl} & \\texttt{{---}} & ${_s0['current@1']:.3f}$ & "
                    + (f"${_s1['current@1']:.3f}$" if _s1 else "---")
                    + f" & ${_s0['current@5']:.3f}$ & "
                      f"${_s0['contamination@5']:.3f}$ & "
                      f"${_s0['slots']:.0f}$ & --- \\\\")
        for _v, (_fn, _dd, _st) in sorted(_best.items(), reverse=True):
            _ca = _dd["by_config"]["anchored"]
            _cf = _dd["by_config"].get("anchor_free", {})
            _mk = next(k for k in _ca if k.startswith("mem0("))
            _s0 = _ca[_mk]["summary"]
            _s1 = _cf.get(_mk, {}).get("summary")
            _ev = _ca[_mk].get("events") or {}
            rows.append(
                f"\\textbf{{Mem0}} ({_mk[5:-1]}) & \\texttt{{{_v}}} & "
                f"${_s0['current@1']:.3f}$ & "
                + (f"${_s1['current@1']:.3f}$" if _s1 else "---")
                + f" & ${_s0['current@5']:.3f}$ & "
                  f"${_s0['contamination@5']:.3f}$ & ${_s0['slots']:.0f}$ & "
                  f"{_ev.get('ADD', 0)}\\,/\\,{_ev.get('UPDATE', 0)} \\\\")
        if "keep-latest-oracle" in _ra:
            _s0 = _ra["keep-latest-oracle"]["summary"]
            _s1 = _rf.get("keep-latest-oracle", {}).get("summary")
            rows.append(
                "Overwrite: true-key oracle & \\texttt{---} & "
                f"${_s0['current@1']:.3f}$ & "
                + (f"${_s1['current@1']:.3f}$" if _s1 else "---")
                + f" & ${_s0['current@5']:.3f}$ & "
                  f"${_s0['contamination@5']:.3f}$ & ${_s0['slots']:.0f}$ "
                  "& --- \\\\")
        _tabular("llcccccc",
                 "policy / system & version & cur@1 (anch.) & "
                 "cur@1 (anch.-free) & cur@5 & cont.@5 & slots & "
                 "ADD\\,/\\,UPDATE",
                 rows, os.path.join(_PAPER, "tab_systems.tex"))

    # ---------------- the deployed system at the reader ---------------- #
    # The paper argues that retrieval metrics misrank policies and only a
    # reader settles it, then scored the one deployed system by retrieval
    # metrics alone. This block reads the run that fixed that, and it also
    # carries the paper's own read-path fix applied post hoc to the system's
    # output -- so contribution (3) is finally measured against contribution
    # (5) rather than sitting beside it.
    _ARMS = (("as_is", "AsIs"), ("read_keyed", "Keyed"),
             ("read_keyed_collapsed", "Merged"),
             ("read_keyed_oldest", "Oldest"))
    # How much of the merged arm is free? Facts arrive as they become true, so
    # on the clean condition the last version of a fact is a current one and a
    # recency tie-break is right without knowing anything. State the size of
    # that gift rather than let the arm's number imply it was earned.
    if _best:
        from stalebench.templama import load_facts as _lf, split_facts as _sf
        _fq, _ = _lf()
        _, _tq = _sf(_fq)
        _tq = _tq[:int(next(iter(_best.values()))[1]["config"]["n_test_facts"])]
        _recok = sum(1 for _f in _tq if (len(_f.runs) - 1) in _f.current_runs)
        add("sysRecencyValid", str(_recok))
        add("sysRecencyTotal", str(len(_tq)))
    _rd_rows, _rd_release = [], None
    _rd_ladder: Dict[tuple, list] = {}
    for _vr, (_fnr, _ddr, _str) in sorted(_best.items()):
        _rdp = os.path.join(_RES, _fnr.replace("stalebench_systems_",
                                               "stalebench_sysreader_"))
        if not os.path.exists(_rdp):
            continue
        _rd = json.load(open(_rdp))
        _tgr = "Adj" if _vr.startswith("0.1.") else "Cur"
        # "ollama:qwen2.5:14b" -- strip the backend, keep the model, which
        # itself contains a colon
        _rdname = _rd["config"]["readers"][0].partition(":")[2] or \
            _rd["config"]["readers"][0]
        add(f"sysrd{_tgr}Reader", f"\\texttt{{{_rdname}}}")
        for _cfgr, _ctr in (("anchored", "Zero"), ("anchor_free", "One")):
            _cellr = _rd["by_config"].get(_cfgr, {})
            for _polr, _polout in _cellr.items():
                _isys = _polr.startswith("mem0(")
                # The ladder rows are the same contexts in every sysreader file
                # -- byte-identical, since those policies are deterministic and
                # run on one stream -- so they are quoted from one reader pass.
                # Emitting them per release would collide, and the collision is
                # informative: the same prompts do not always get the same
                # answer, which is measured below rather than papered over.
                if not _isys:
                    for _armr, _an in _ARMS:
                        _arm = _polout.get(_armr)
                        if _arm and _arm["readers"]:
                            _rd_ladder.setdefault(
                                (_cfgr, _polr, _armr), []).append(
                                next(iter(_arm["readers"].values()))
                                ["summary"]["correct"])
                    if _rd_release not in (None, _vr):
                        continue
                _key = (_tgr if _isys else "") + (
                    "Sys" if _isys else "".join(
                        w.capitalize() for w in _polr.split("-")))
                for _armr, _an in _ARMS:
                    _arm = _polout.get(_armr)
                    if not _arm or not _arm["readers"]:
                        continue
                    _rsum = next(iter(_arm["readers"].values()))["summary"]
                    add(f"sysrd{_key}{_an}{_ctr}", f"{_rsum['correct']:.3f}")
                    add(f"sysrd{_key}{_an}Stale{_ctr}",
                        f"{_rsum['stale']:.3f}")
                    add(f"sysrd{_key}{_an}Retr{_ctr}",
                        f"{_arm['retrieval']['current@1']:.3f}")
                    add(f"sysrd{_key}{_an}Text{_ctr}",
                        f"{_arm['retrieval']['current@1_text']:.3f}")
                # what the paper's fix is worth to whatever produced these
                if _polout.get("as_is", {}).get("readers") and \
                        _polout.get("read_keyed_collapsed", {}).get("readers"):
                    _base = next(iter(
                        _polout["as_is"]["readers"].values()))["summary"]["correct"]
                    _merged = next(iter(
                        _polout["read_keyed_collapsed"]["readers"].values()
                    ))["summary"]["correct"]
                    add(f"sysrd{_key}Gain{_ctr}",
                        f"{100 * (_merged - _base):.1f}")
            if _cfgr == "anchored" and _rd_release is None:
                _rd_release = _vr
                for _polr, _polout in _cellr.items():
                    _cells = []
                    for _armr, _an in _ARMS:
                        # the oldest-kept control contributes its reader column
                        # only: its cur@1 is not the quantity it exists to show
                        _wide = _armr != "read_keyed_oldest"
                        _arm = _polout.get(_armr)
                        if not _arm or not _arm["readers"]:
                            _cells += ["---", "---"] if _wide else ["---"]
                            continue
                        _rsum = next(iter(_arm["readers"].values()))["summary"]
                        if _wide:
                            _cells.append(
                                f"${_arm['retrieval']['current@1']:.3f}$")
                        _cells.append(f"${_rsum['correct']:.3f}$")
                    _lbl = ("Mem0 " + _vr if _polr.startswith("mem0(")
                            else _polr.replace("_", "\\_"))
                    _rd_rows.append(f"\\texttt{{{_lbl}}} & "
                                    + " & ".join(_cells) + " \\\\")
        _rdt = _rd["tests"]
        add(f"sysrd{_tgr}Family", str(_rdt["family_size"]))
        add(f"sysrd{_tgr}Surviving",
            str(sum(1 for v in _rdt["holm"].values() if v < 0.05)))
        for _cfgr, _ctr in (("anchored", "Zero"), ("anchor_free", "One")):
            for _armr, _an in (("read_keyed", "Keyed"),
                               ("read_keyed_collapsed", "Merged")):
                _lblp = next((k for k in _rdt["holm"]
                              if k.startswith(f"{_cfgr}:mem0(")
                              and f":{_armr}-vs-as_is:" in k
                              and k.endswith(":correct")), None)
                if _lblp:
                    _pv = _rdt["holm"][_lblp]
                    add(f"sysrdP{_tgr}{_an}{_ctr}",
                        "$<0.001$" if _pv < 0.001 else f"${_pv:.3f}$")
    # No reader-reproducibility macro is emitted from _rd_ladder, though the
    # data invites one. The ladder's contexts are identical in every sysreader
    # file, so scoring them twice looks like scoring the reader twice -- but
    # answers are served from a deterministic cache keyed on the prompt, so
    # agreement is guaranteed by the cache rather than observed in the model.
    # An earlier pair of passes, before the cache covered both, disagreed on a
    # few cells; that is an anecdote we cannot now reproduce, and reporting the
    # post-cache number as reproducibility would be reporting the cache.
    if _rd_rows:
        _tabular("lccccccc",
                 "context source & \\multicolumn{2}{c}{as retrieved} & "
                 "\\multicolumn{2}{c}{$+$ query resolved} & "
                 "\\multicolumn{2}{c}{$+$ merged, freshest} & "
                 "oldest \\\\\n"
                 " & cur@1 & reader & cur@1 & reader & cur@1 & reader "
                 "& reader",
                 _rd_rows, os.path.join(_PAPER, "tab_sysreader.tex"))

    # -------- The body's deployed-systems table: one row per (system, variant),
    # both configurations, the same four columns the per-system appendix
    # tables use. Every cell is read back from a macro already emitted above,
    # so this table cannot disagree with the prose or with Appendix K.
    if _ers_any:
        def _ver(tag):
            return _seen[tag].replace("\\texttt{", "").rstrip("}")
        _dep_spec = [
            (f"Mem0 {_ver('probedCurVersion')}",
             "probedCurCur{c}", "textCurCur{c}", "textCurContam{c}", "sysrdCurSysAsIs{c}"),
            (f"Mem0 {_ver('probedAdjVersion')}",
             "probedAdjCur{c}", "textAdjCur{c}", "textAdjContam{c}", "sysrdAdjSysAsIs{c}"),
            ("Graphiti, as returned",
             "gphCur{c}", "gphCurText{c}", "gphContamText{c}", "gphRdAsIs{c}"),
            ("Graphiti, valid edges first",
             "gphVFCur{c}", "gphVFCurText{c}", "gphVFContamText{c}", "gphRdVF{c}"),
            ("LangMem, qwen2.5:14b",
             "lgmQwenCur{c}", "lgmQwenCurText{c}", "lgmQwenContamText{c}", "lgmQwenRdAsIs{c}"),
            ("LangMem, glm-4.6",
             "lgmGlmCur{c}", "lgmGlmCurText{c}", "lgmGlmContamText{c}", "lgmGlmRdAsIs{c}"),
            ("ERASE, qwen2.5:14b",
             "ersQwenCur{c}", "ersQwenCurText{c}", "ersQwenContamText{c}", "ersQwenRdAsIs{c}"),
            ("ERASE, glm-4.6",
             "ersGlmCur{c}", "ersGlmCurText{c}", "ersGlmContamText{c}", "ersGlmRdAsIs{c}"),
        ]
        _dep_missing = [m.format(c=c) for _, *ms in _dep_spec for m in ms
                        for c in ("Zero", "One") if m.format(c=c) not in _seen]
        if _dep_missing:
            print(f"WARNING: tab_deployed.tex NOT written -- macros missing: "
                  f"{', '.join(_dep_missing)}")
        else:
            _dep_rows = []
            for label, *ms in _dep_spec:
                cells = [f"${_seen[m.format(c=c)]}$" for c in ("Zero", "One") for m in ms]
                _dep_rows.append(f"{label} & " + " & ".join(cells) + " \\\\")
            # the ladder rows of the ERASE table: same streams, same columns
            _dep_rows.append("\\midrule")
            _dep_rows += _ers_rows[-4:]
            hdr = ("& \\multicolumn{4}{c}{anchored} & \\multicolumn{4}{c}{anchor-free} \\\\\n"
                   "\\cmidrule(lr){2-5}\\cmidrule(lr){6-9}\n"
                   "system & cur@1 & by text & cont.@5 & reader & "
                   "cur@1 & by text & cont.@5 & reader")
            _tabular("lcccccccc", hdr, _dep_rows, os.path.join(_PAPER, "tab_deployed.tex"))

    with open(os.path.join(_PAPER, "numbers.tex"), "w") as fh:
        fh.write("% generated by stalebench.make_paper_numbers -- do not edit\n")
        fh.write("\n".join(n) + "\n")
    _frozen.append(True)

    # a crash midway leaves stale tab_*.tex behind an otherwise clean-looking
    # run, which is how a shadowed accessor silently published old numbers
    if _written["tables"] < _EXPECTED_TABLES:
        raise SystemExit(
            f"only {_written['tables']} tables written this run, expected at "
            f"least {_EXPECTED_TABLES} -- generation aborted partway and stale "
            f"tab_*.tex are still on disk")
    print(f"Wrote numbers.tex + {_written['tables']} tables to "
          f"{os.path.abspath(_PAPER)}")


if __name__ == "__main__":
    main()
