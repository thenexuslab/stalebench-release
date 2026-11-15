"""StaleBench reader track: what stale retrieval costs at the end of the pipeline.

v0.2 measured contamination@k and *argued*, from the knowledge-conflict
literature, that a context holding both a current and a stale version harms the
reader. This track measures it, under three designs that answer three different
objections:

1. ``closed_book`` --- the question with no context at all. TempLAMA facts are
   real Wikidata facts, so a frontier reader may simply know them; without this
   control every context-conditioned number is confounded. It also bounds how
   much of the benchmark can be solved without the store.
2. ``policy_track`` --- each policy's own top-5 becomes the context. End-task
   accuracy per policy, plus the *within-policy* contrast between its
   contaminated and its clean queries (observational: contamination is not
   randomly assigned, so this is an association).
3. ``dose_response`` --- contexts constructed rather than retrieved: the current
   statement alone, plus one stale version (either order), plus two, or stale
   only. Contamination is now manipulated while everything else is held fixed,
   so the effect is causal.

Every design runs with and without the stored timestamp surfaced in the
context, which answers the reader-level form of the cheapest objection: *why
not show the model the dates and let it sort them out?*

Run:  python -m stalebench.run_reader [reader-spec ...]
      python -m stalebench.run_reader ollama:qwen2.5:14b openrouter:anthropic/claude-sonnet-4.6
"""

from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from stalebench.embed import MINILM, Embedder
from stalebench.llm import make_client
from stalebench.metrics import (bootstrap_ci, holm, paired_permutation,
                                score_ranking)
from stalebench.policies import (CosineRAG, KeepLatestOracle, LexicalKey,
                                 ReadWriteFuzzyKey, Reconsolidation, TimeRerank)
from stalebench.reader import build_prompt, score_answer, stale_values_of
from stalebench.templama import build_instance, load_facts, split_facts

_HERE = os.path.dirname(__file__)
_OUT = os.environ.get("STALEBENCH_READER_OUT") or os.path.join(
    _HERE, "..", "results", "stalebench_reader_results.json")

_PARA = 0.5          # headline condition of the retrieval track
_TOPK = 5            # context size == the k of contamination@5
_WORKERS = 8
_DEFAULT_READERS = ["ollama:qwen2.5:14b", "ollama:llama3.1:8b", "zai:glm-4.6"]
# smoke-test escape hatch: cap the query set before spending on a hosted model
_LIMIT = int(os.environ.get("STALEBENCH_READER_LIMIT", "0")) or None
# restrict the constructed-context conditions (a slow hosted reader can still
# contribute the policy track and the closed-book control)
_DOSE_ONLY = [c for c in os.environ.get("STALEBENCH_DOSE_ONLY", "").split(",") if c]
# the date-surfacing ablation is reported for the primary reader; a slow hosted
# reader can skip it and still contribute the policy track and closed book
_DATE_CONDS = ((False,) if os.environ.get("STALEBENCH_DATES") == "0"
               else (False, True))

# tuned on val in the main track; re-stated here so the reader track is a pure
# consumer of already-fixed hyperparameters (no tuning against reader accuracy)
_TUNED = {"time-rerank_k": 2, "tau": 0.6, "fuzzy_tau": 0.2}


def _mean_ci(recs: Sequence[Dict[str, float]], keys: Sequence[str]) -> Dict:
    return {k: bootstrap_ci([r[k] for r in recs], seed=1) for k in keys}


def _batch(client, prompts: List[str], max_tokens: int = 32) -> List[str]:
    """Generate for every prompt, tolerating individual failures.

    A hosted endpoint will occasionally reject one prompt outright (content
    filter, a malformed unicode sequence). Letting that abort a 10k-call sweep
    loses hours of completed work for one datapoint, so a failed prompt becomes
    an empty answer -- scored as ``other``, never as correct -- and the count is
    reported so a systematic failure cannot hide as a quality result.
    """
    def one(p):
        try:
            return client.generate(p, max_tokens)
        except Exception as e:                       # noqa: BLE001
            client.n_failed = getattr(client, "n_failed", 0) + 1
            client.last_error = f"{type(e).__name__}: {e}"
            return ""

    with ThreadPoolExecutor(getattr(client, "max_workers", _WORKERS)) as ex:
        return list(ex.map(one, prompts))


# --------------------------------------------------------------------------- #
# Context builders
# --------------------------------------------------------------------------- #


def _policy_contexts(policy_factory: Callable, inst, ev, qe, dates: bool,
                     reverse: bool = False
                     ) -> Tuple[List[str], List[Dict], List[List[str]]]:
    """Run a policy, then turn each query's top-k into a reader prompt.

    ``reverse`` presents the top-k in ascending relevance, i.e. the best hit
    *adjacent* to the question rather than farthest from it. The default
    (descending) is the common RAG convention and is also the arrangement the
    dose-response identifies as worst, so the comparison is the obvious
    challenge to this section's headline: is the transfer failure a property of
    the policies, or of one context convention?
    """
    p = policy_factory()
    p.reset()
    for (text, iid, t), e in zip(inst.events, ev):
        p.observe(e, iid, t, text)
    prompts, retr, tops = [], [], []
    for (qtext, key, cur), q in zip(inst.queries, qe):
        ranked = p.rank(q, qtext)
        top = ranked[:_TOPK]
        if reverse:
            top = top[::-1]
        texts = [inst.text_of[i] for i in top]
        years = [inst.year_of[i] for i in top] if dates else None
        prompts.append(build_prompt(texts, qtext, years))
        retr.append(score_ranking(ranked, key, cur, (_TOPK,)))
        tops.append(top)
    return prompts, retr, tops


def _dose_contexts(inst, facts, dates: bool) -> Dict[str, Tuple[List[int], List[str]]]:
    """Constructed contexts that vary contamination while holding content fixed."""
    conds: Dict[str, Tuple[List[int], List[str]]] = {
        c: ([], []) for c in ("no_context", "current_only", "plus1_current_first",
                              "plus1_stale_first", "plus2", "stale_only")}
    for qi, ((qtext, key, cur), f) in enumerate(zip(inst.queries, facts)):
        fi = key                                   # "f<idx>"
        cur_run = max(cur)                         # latest current version
        stale_runs = [r for r in range(len(f.runs)) if r not in cur]
        c_txt = inst.text_of[f"{fi}:{cur_run}"]
        c_yr = inst.year_of[f"{fi}:{cur_run}"]
        s_txt = [inst.text_of[f"{fi}:{r}"] for r in stale_runs]
        s_yr = [inst.year_of[f"{fi}:{r}"] for r in stale_runs]

        def add(name, texts, years):
            conds[name][0].append(qi)
            conds[name][1].append(build_prompt(texts, qtext,
                                               years if dates else None))

        add("no_context", [], [])
        add("current_only", [c_txt], [c_yr])
        if s_txt:
            # the single competing version is the *most recent* stale one: the
            # hardest and most realistic distractor a store would surface
            add("plus1_current_first", [c_txt, s_txt[-1]], [c_yr, s_yr[-1]])
            add("plus1_stale_first", [s_txt[-1], c_txt], [s_yr[-1], c_yr])
            add("stale_only", [s_txt[-1]], [s_yr[-1]])
        if len(s_txt) >= 2:
            add("plus2", [s_txt[0], c_txt, s_txt[-1]],
                [s_yr[0], c_yr, s_yr[-1]])
    return conds


# --------------------------------------------------------------------------- #
# Experiment
# --------------------------------------------------------------------------- #


def run(reader_specs: Sequence[str]) -> Dict:
    facts, stats = load_facts()
    val, test = split_facts(facts)
    if _LIMIT:
        test = test[:_LIMIT]
    inst = build_instance(test, paraphrase_p=_PARA, seed=0)
    emb = Embedder(MINILM)
    ev = emb.encode([e[0] for e in inst.events], f"test_ev_p{_PARA}")
    qe = emb.encode([q[0] for q in inst.queries], f"test_q_p{_PARA}")

    gold = inst.gold
    stale_vals = [stale_values_of(f.runs, f.current_value) for f in test]
    n_with_stale = sum(1 for s in stale_vals if s)
    # a few facts have a superseded value nested inside the current one
    # ("Democratic Party" inside "Democratic Party of Korea"), so answering the
    # gold value exactly scores as hedged. That puts a ceiling below 1.0 on the
    # reader metric, and the oracle reader should be read against it.
    n_unscorable = sum(1 for g, sv in zip(gold, stale_vals)
                       if score_answer(g, g, sv)["correct"] != 1.0)

    policies = {
        "cosine-rag": lambda: CosineRAG(),
        "time-rerank": lambda: TimeRerank(k=_TUNED["time-rerank_k"]),
        "reconsolidation": lambda: Reconsolidation(tau=_TUNED["tau"]),
        "lexical-key": lambda: LexicalKey([f.subject for f in test]),
        # a controlled pair: identical current@1, opposite contamination. The
        # read-path rung resolves the query but retains everything; the
        # symmetric rung also merges on write. A retrieval metric cannot tell
        # them apart, so what the reader does with them is the cleanest test
        # of whether contamination is real (S:anchor, S:reader)
        "fuzzy-read-key": lambda: ReadWriteFuzzyKey(
            [f.subject for f in test], _TUNED["fuzzy_tau"], write_keys=False),
        "fuzzy-rw-key": lambda: ReadWriteFuzzyKey(
            [f.subject for f in test], _TUNED["fuzzy_tau"]),
        "keep-latest-oracle": lambda: KeepLatestOracle(),
    }
    metrics = ("correct", "stale", "hedged", "other", "abstain")

    out: Dict = {"config": {"paraphrase_p": _PARA, "top_k": _TOPK,
                            "n_test_facts": len(test),
                            "n_facts_with_stale_value": n_with_stale,
                            "n_gold_unscorable": n_unscorable,
                            "metric_ceiling": 1.0 - n_unscorable / len(test),
                            "max_answer_tokens": 32,
                            "tuned": _TUNED, "readers": list(reader_specs)},
                 "readers": {}}

    for spec in reader_specs:
        client = make_client(spec)
        r_out: Dict = {"policy_track": {}, "policy_conditional": {},
                       "dose_response": {}, "contamination_effect": {}}
        print(f"\n=== reader {client.name} ===", flush=True)

        for dates in _DATE_CONDS:
            dk = "with_dates" if dates else "no_dates"
            r_out["policy_track"][dk] = {}
            r_out["policy_conditional"][dk] = {}
            base_tops = None
            for pname, fac in policies.items():
                prompts, retr, tops = _policy_contexts(fac, inst, ev, qe, dates)
                if pname == "cosine-rag":
                    base_tops = tops
                # how much of a policy's effect is re-ordering rather than
                # re-selecting? time-rerank(k) permutes the similarity top-k,
                # so for most queries its context is the SAME set of statements
                # as plain cosine, in a different order.
                same_set = float(np.mean([set(a) == set(b)
                                          for a, b in zip(tops, base_tops)]))
                same_set_diff_order = float(np.mean(
                    [set(a) == set(b) and list(a) != list(b)
                     for a, b in zip(tops, base_tops)]))
                answers = _batch(client, prompts)
                recs = [score_answer(a, g, s)
                        for a, g, s in zip(answers, gold, stale_vals)]
                r_out["policy_track"][dk][pname] = {
                    "summary": {m: float(np.mean([r[m] for r in recs]))
                                for m in metrics},
                    "ci": _mean_ci(recs, ("correct", "stale")),
                    "retrieval_current@1": float(np.mean([r["current@1"]
                                                          for r in retr])),
                    "retrieval_contamination@5": float(
                        np.mean([r[f"contamination@{_TOPK}"] for r in retr])),
                    "context_same_set_as_cosine": same_set,
                    "context_same_set_reordered": same_set_diff_order,
                    "per_query_correct": [r["correct"] for r in recs],
                }
                # within-policy association: same policy, same reader, queries
                # split by whether ITS top-k was contaminated
                cont = [r[f"contamination@{_TOPK}"] > 0 for r in retr]
                grp = {"contaminated": [x for x, c in zip(recs, cont) if c],
                       "clean": [x for x, c in zip(recs, cont) if not c]}
                r_out["policy_conditional"][dk][pname] = {
                    g: {"n": len(v),
                        "correct": float(np.mean([x["correct"] for x in v]))
                        if v else None,
                        "stale": float(np.mean([x["stale"] for x in v]))
                        if v else None}
                    for g, v in grp.items()}
                print(f"  [{dk}] {pname:20s} correct "
                      f"{r_out['policy_track'][dk][pname]['summary']['correct']:.3f} "
                      f"stale {r_out['policy_track'][dk][pname]['summary']['stale']:.3f}",
                      flush=True)

            # paired tests on the reader comparisons that carry the claim.
            # Same queries, same reader, different context -> sign-flip test.
            pq = {k: v["per_query_correct"]
                  for k, v in r_out["policy_track"][dk].items()}
            fam = {
                "rerank_vs_cosine": ("time-rerank", "cosine-rag"),
                "lexical_vs_rerank": ("lexical-key", "time-rerank"),
                "lexical_vs_cosine": ("lexical-key", "cosine-rag"),
                "oracle_vs_lexical": ("keep-latest-oracle", "lexical-key"),
            }
            raw = {name: paired_permutation(pq[a], pq[b], seed=3)
                   for name, (a, b) in fam.items()}
            r_out.setdefault("policy_paired_p", {})[dk] = {
                "raw": raw, "holm": holm(raw),
                "deltas": {name: float(np.mean(pq[a]) - np.mean(pq[b]))
                           for name, (a, b) in fam.items()}}

            # -- constructed contexts: contamination manipulated ------------ #
            conds = _dose_contexts(inst, test, dates)
            if _DOSE_ONLY:
                conds = {k: v for k, v in conds.items() if k in _DOSE_ONLY}
            per_cond_recs: Dict[str, Dict[int, Dict]] = {}
            r_out["dose_response"][dk] = {}
            for cname, (idx, prompts) in conds.items():
                if not prompts:
                    continue
                answers = _batch(client, prompts)
                recs = [score_answer(a, gold[i], stale_vals[i])
                        for a, i in zip(answers, idx)]
                per_cond_recs[cname] = dict(zip(idx, recs))
                r_out["dose_response"][dk][cname] = {
                    "n": len(recs),
                    "summary": {m: float(np.mean([r[m] for r in recs]))
                                for m in metrics},
                    "ci": _mean_ci(recs, ("correct", "stale")),
                }
                print(f"  [{dk}] dose {cname:20s} n={len(recs):3d} correct "
                      f"{np.mean([r['correct'] for r in recs]):.3f} stale "
                      f"{np.mean([r['stale'] for r in recs]):.3f}", flush=True)

            # position effect: the SAME two statements, order reversed. This is
            # the quantity that explains why a reranker can win retrieval and
            # lose the end task, and it is identified by construction.
            cf, sf = (per_cond_recs.get("plus1_current_first"),
                      per_cond_recs.get("plus1_stale_first"))
            if cf and sf:
                sh = sorted(set(cf) & set(sf))
                a2 = [sf[i]["correct"] for i in sh]     # current last
                b2 = [cf[i]["correct"] for i in sh]     # current first
                r_out.setdefault("position_effect", {})[dk] = {
                    "n_paired": len(sh),
                    "current_last_correct": float(np.mean(a2)),
                    "current_first_correct": float(np.mean(b2)),
                    "gain_from_current_last": float(np.mean(a2) - np.mean(b2)),
                    "ci": bootstrap_ci([x - y for x, y in zip(a2, b2)], seed=1),
                    "p_paired_permutation": paired_permutation(a2, b2, seed=2),
                }
                print(f"  [{dk}] position effect: current-first {np.mean(b2):.3f} "
                      f"vs current-last {np.mean(a2):.3f}", flush=True)

            # paired effect of adding one stale version to a correct context
            base, plus = per_cond_recs.get("current_only"), \
                per_cond_recs.get("plus1_stale_first")
            if base and plus:
                shared = sorted(set(base) & set(plus))
                a = [base[i]["correct"] for i in shared]
                b = [plus[i]["correct"] for i in shared]
                d = [x - y for x, y in zip(a, b)]
                r_out["contamination_effect"][dk] = {
                    "n_paired": len(shared),
                    "current_only_correct": float(np.mean(a)),
                    "plus1_stale_first_correct": float(np.mean(b)),
                    "drop": float(np.mean(d)),
                    "drop_ci": bootstrap_ci(d, seed=1),
                    "p_paired_permutation": paired_permutation(a, b, seed=2),
                }
                print(f"  [{dk}] contamination effect: {np.mean(a):.3f} -> "
                      f"{np.mean(b):.3f} (drop {np.mean(d):+.3f}, "
                      f"p={r_out['contamination_effect'][dk]['p_paired_permutation']:.4f})",
                      flush=True)

        # context-order ablation: same policies, same top-k, reversed order
        if os.environ.get("STALEBENCH_ORDER_ABLATION") == "1":
            r_out["reversed_order"] = {}
            only = [x for x in
                    os.environ.get("STALEBENCH_ORDER_POLICIES", "").split(",") if x]
            for pname, fac in policies.items():
                if only and pname not in only:
                    continue
                prompts, retr, _ = _policy_contexts(fac, inst, ev, qe, False,
                                                    reverse=True)
                answers = _batch(client, prompts)
                recs = [score_answer(a, g, sv)
                        for a, g, sv in zip(answers, gold, stale_vals)]
                r_out["reversed_order"][pname] = {
                    "summary": {m: float(np.mean([r[m] for r in recs]))
                                for m in metrics},
                    "ci": _mean_ci(recs, ("correct", "stale")),
                    "per_query_correct": [r["correct"] for r in recs]}
                # paired against the same policy's forward-order context:
                # identical statements, identical retrieval, order flipped
                fwd = r_out["policy_track"]["no_dates"][pname]["per_query_correct"]
                rev = r_out["reversed_order"][pname]["per_query_correct"]
                r_out["reversed_order"][pname]["vs_forward"] = {
                    "delta": float(np.mean(rev) - np.mean(fwd)),
                    "ci": bootstrap_ci([b - a for a, b in zip(fwd, rev)], seed=1),
                    "p_paired_permutation": paired_permutation(rev, fwd, seed=2),
                }
                v = r_out["reversed_order"][pname]["vs_forward"]
                print(f"  [reversed] {pname:20s} correct "
                      f"{r_out['reversed_order'][pname]['summary']['correct']:.3f} "
                      f"(forward {np.mean(fwd):.3f}, delta {v['delta']:+.3f}, "
                      f"p={v['p_paired_permutation']:.4f})", flush=True)

        # closed book: no context at all (parametric-knowledge control)
        cb = per_cond_recs.get("no_context")
        if cb:
            recs = list(cb.values())
            r_out["closed_book"] = {
                "n": len(recs),
                "summary": {m: float(np.mean([r[m] for r in recs]))
                            for m in metrics},
                "ci": _mean_ci(recs, ("correct", "stale")),
            }
        r_out["usage"] = {"n_calls": client.n_calls, "n_cached": client.n_cached,
                          "usd": round(client.usd, 4), "model": client.model,
                          "n_failed": getattr(client, "n_failed", 0),
                          "last_error": getattr(client, "last_error", None)}
        print(f"  usage: {client.n_calls} calls, {client.n_cached} cached, "
              f"{getattr(client, 'n_failed', 0)} failed, ${client.usd:.3f}",
              flush=True)
        out["readers"][client.name] = r_out
    return out


def merge(paths: Sequence[str], out_path: str = _OUT) -> Dict:
    """Combine per-reader result files into one.

    Readers differ in throughput by more than an order of magnitude, so a slow
    hosted model is run with a reduced set of constructed-context conditions
    while a local model runs all of them. Merging afterwards keeps each
    reader's scope explicit instead of forcing the slowest one to define it.
    """
    def _deep(a: Dict, b: Dict) -> Dict:
        """Union b into a, recursing into dicts so a partial re-run adds
        conditions instead of replacing everything the reader already had."""
        for k, v in b.items():
            if isinstance(v, dict) and isinstance(a.get(k), dict):
                _deep(a[k], v)
            else:
                a[k] = v
        return a

    merged: Optional[Dict] = None
    for p in paths:
        d = json.load(open(p))
        if merged is None:
            merged = d
        else:
            for name, r in d["readers"].items():
                if name in merged["readers"]:
                    _deep(merged["readers"][name], r)
                else:
                    merged["readers"][name] = r
            merged["config"]["readers"] = sorted(
                set(merged["config"]["readers"]) | set(d["config"]["readers"]))
    for name, r in merged["readers"].items():
        r["conditions_run"] = sorted(r["dose_response"].get("no_dates", {}))
    json.dump(merged, open(out_path, "w"), indent=1)
    return merged


def main():
    if sys.argv[1:2] == ["--merge"]:
        m = merge(sys.argv[2:])
        print(f"merged {len(m['readers'])} readers -> {os.path.abspath(_OUT)}")
        for name, r in m["readers"].items():
            print(f"  {name}: policies "
                  f"{len(r['policy_track']['no_dates'])}, dose conditions "
                  f"{len(r['conditions_run'])}")
        return
    specs = sys.argv[1:] or _DEFAULT_READERS
    res = run(specs)
    os.makedirs(os.path.dirname(_OUT), exist_ok=True)
    json.dump(res, open(_OUT, "w"), indent=1)
    print(f"\nWrote {os.path.abspath(_OUT)}")
    total = sum(r["usage"]["usd"] for r in res["readers"].values())
    print(f"total API cost: ${total:.2f}")


if __name__ == "__main__":
    main()
