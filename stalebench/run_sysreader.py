"""Put a deployed system on the reader axis, and apply the paper's own fix to it.

The reader track argues that retrieval metrics misrank policies and that only a
reader settles it. The systems track then scores the one *deployed* system by
retrieval metrics alone, under an id-attribution convention the paper itself
calls generous. This module closes that gap. It answers two questions on the
contexts a real run of the system actually produced:

1. **Reader track over the system's own output.** What does a model answer when
   handed the five memories Mem0 surfaced --- in Mem0's words, since it rewrites
   what it stores --- rather than the five statements a ladder policy selected?
   Whichever way it lands is quotable: a confirmed placement rests on the
   paper's strongest evidence axis, and a reversal shows the system's low
   contamination was partly definitional.

2. **The paper's cheapest fix, applied to the system.** Contribution (3) is that
   resolving the *query* through the same keyer as the stream is free and worth
   double digits. Nobody has applied it to a deployed system. Re-ranking what
   the system surfaced --- promoting hits whose text resolves to the query's
   entity, freshest first --- is exactly ``ReadKeyedRetention.rank`` operating
   on the system's output instead of on a store we control, and it costs no
   ingestion at all.

The run is a pure consumer of :mod:`stalebench.run_systems`' context sidecar, so
neither question costs a second pass over the LLM-bound ingestion. That also
means the arms are *paired by construction*: same stream, same queries, same
system state, so the sign-flip test is the right one.

The ladder policies from the same sidecar are re-ranked identically, which is
the control that licenses reading the system's number: post-hoc re-ranking of
``cosine-rag``'s hits should recover something close to the read-path rung the
paper measures directly. If it does not, the method is wrong before it is ever
pointed at Mem0.

Dates are never surfaced here. The ladder can show a stored year because the
benchmark wrote it; the deployed system surfaces no timestamp a reader could
use, so a with-dates arm would compare a policy's context against a system's
context plus metadata the system does not have.

Run:  python -m stalebench.run_sysreader <contexts.json> [reader-spec ...]
"""

from __future__ import annotations

import json
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from stalebench.llm import make_client
from stalebench.metrics import (bootstrap_ci, holm, paired_permutation,
                                score_ranking, score_texts)
from stalebench.policies import FuzzyKeyer
from stalebench.reader import build_prompt, score_answer, stale_values_of
from stalebench.run_reader import _batch
from stalebench.templama import build_instance, load_facts, split_facts

_HERE = os.path.dirname(__file__)
_OUT = os.environ.get("STALEBENCH_SYSREADER_OUT") or os.path.join(
    _HERE, "..", "results", "stalebench_sysreader_results.json")

_TOPK = 5
_KS = (5,)
_DEFAULT_READERS = ["ollama:qwen2.5:14b"]
_LIMIT = int(os.environ.get("STALEBENCH_SYSREADER_LIMIT", "0")) or None
# tuned on val in the main track and re-stated, as in the reader track: nothing
# here is tuned against a reader-side outcome
_FUZZY_TAU = 0.2
_METRICS = ("correct", "stale", "hedged", "other", "abstain")


# --------------------------------------------------------------------------- #
# Read-path resolution over what a store surfaced
# --------------------------------------------------------------------------- #


def _recency(hit: Dict) -> Tuple:
    """A sortable freshness key that works for both kinds of hit.

    A ladder hit carries the benchmark year it was written with. A system's hit
    carries the system's own ``updated_at``, which is the only recency signal
    that survives an UPDATE --- the payload metadata still names the memory's
    *original* writer, so reading a timestamp out of it would silently sort by
    the wrong event.
    """
    stamp = hit.get("updated_at") or hit.get("created_at")
    if stamp:
        return (1, str(stamp))
    t = hit.get("t")
    return (0, "") if t is None else (1, f"{float(t):020.4f}")


def read_key_rerank(hits: Sequence[Dict], query_text: str,
                    keyer: FuzzyKeyer) -> List[Dict]:
    """Promote the hits whose text resolves to the query's entity, freshest first.

    This is ``ReadKeyedRetention.rank`` with the store's ranking standing in for
    cosine order: same keyer, same tie-break, same fallback when the query does
    not resolve. It uses only the text the store chose to surface plus a list of
    canonical subject names --- no benchmark ids, no ground truth --- so it is a
    wrapper a deployer could actually write.
    """
    qkey = keyer.key_of(None, None, "", query_text or "")
    if qkey is None:
        return list(hits)
    keyed = [(h, keyer.key_of(None, None, "", h.get("text") or ""))
             for h in hits]
    match = [h for h, k in keyed if k == qkey]
    match.sort(key=_recency, reverse=True)
    return match + [h for h, k in keyed if k != qkey]


def collapse(hits: Sequence[Dict], keyer: FuzzyKeyer,
             keep: str = "freshest") -> List[Dict]:
    """Keep only the freshest hit per resolved entity, in the given order.

    Read-path resolution alone promotes *every* version of the queried fact, so
    the five slots a reader sees fill with the fact arguing with itself: the
    retrieval metric improves and the context gets worse. That is the paper's
    controlled pair seen from the read side, and it is why contribution (3) is
    stated as read *and* write keying rather than read keying alone. Collapsing
    to one memory per entity is the read-side image of merging on write, and it
    is the arm that makes the two-term decomposition available to a system whose
    write path we cannot change.

    ``keep`` selects which memory of an entity survives, and its control is
    necessary rather than decorative. Facts in this benchmark arrive as they
    become true, so on the clean condition the last version of a fact is a
    current one *by construction*, and any recency rule is right for free.
    Running the same merge backwards --- ``keep="oldest"`` --- separates what
    entity resolution contributes, namely pruning the context to one memory
    about the queried fact, from what recency contributes. Only the first is a
    claim about keying, and only the second would break under the corrupted
    timestamps of the shuffle track.
    """
    best: Dict = {}
    for h in hits:
        k = keyer.key_of(None, None, "", h.get("text") or "")
        if k is None:
            best[f"unresolved{id(h)}"] = h     # never merged away
        elif k not in best:
            best[k] = h
        elif ((_recency(h) < _recency(best[k])) if keep == "oldest"
              else (_recency(h) > _recency(best[k]))):
            best[k] = h
    kept = {id(h) for h in best.values()}
    return [h for h in hits if id(h) in kept]


# --------------------------------------------------------------------------- #
# Scoring one arm
# --------------------------------------------------------------------------- #


def _retrieval(hits_per_query: Sequence[Sequence[Dict]],
               queries: Sequence[Tuple], gold: Sequence[str],
               stale: Sequence[Sequence[str]]) -> List[Dict]:
    """Score an arm's contexts the two ways the systems track reports."""
    out = []
    for hits, (_, key, cur), g, sv in zip(hits_per_query, queries, gold, stale):
        ranked: List[str] = []
        for h in hits:
            if h.get("sid") and h["sid"] not in ranked:
                ranked.append(h["sid"])
        rec = score_ranking(ranked, key, cur, _KS)
        rec.update(score_texts([h.get("text") or "" for h in hits], g, sv, _KS))
        out.append(rec)
    return out


def _load(path: str) -> Dict:
    ctx = json.load(open(path))
    cfg = ctx["config"]
    facts, _ = load_facts()
    _, test = split_facts(facts)
    test = test[:cfg["n_test_facts"]] if cfg.get("subsampled") else test
    if len(test) != cfg["n_test_facts"]:
        raise SystemExit(f"{path}: cannot rebuild the {cfg['n_test_facts']}-fact "
                         f"split this file was written from")
    ctx["_facts"] = test
    return ctx


# --------------------------------------------------------------------------- #
# Experiment
# --------------------------------------------------------------------------- #


def run(ctx_path: str, reader_specs: Sequence[str]) -> Dict:
    ctx = _load(ctx_path)
    cfg = ctx["config"]
    test = ctx["_facts"]
    gaz = [f.subject for f in test]
    stale_all = [stale_values_of(f.runs, f.current_value) for f in test]
    keyer = FuzzyKeyer(gaz, _FUZZY_TAU)

    out: Dict = {"config": {"contexts": os.path.basename(ctx_path),
                            "source_config": cfg, "top_k": _TOPK,
                            "fuzzy_tau": _FUZZY_TAU,
                            "readers": list(reader_specs),
                            "dates_surfaced": False},
                 "by_config": {}}

    ques: Dict[str, Tuple] = {}
    for name, cell in ctx["by_config"].items():
        alias_p = 1.0 if name == "anchor_free" else 0.0
        inst = build_instance(test, paraphrase_p=cfg["paraphrase_p"], seed=0,
                              alias_p=alias_p)
        n = len(inst.queries)
        idx = list(range(n if _LIMIT is None else min(n, _LIMIT)))
        queries = [inst.queries[i] for i in idx]
        gold = [inst.gold[i] for i in idx]
        stale = [stale_all[i] for i in idx]
        ques[name] = (queries, gold, stale)
        cell_out: Dict = {}

        for pol, hits_per_query in cell.items():
            if len(hits_per_query) != n:
                raise SystemExit(f"{ctx_path}: {name}/{pol} holds "
                                 f"{len(hits_per_query)} queries, the rebuilt "
                                 f"instance has {n}")
            # the paper's contribution (3), applied post-hoc to whatever
            # produced these contexts, in its two halves: resolve the query,
            # then merge same-entity memories the way keying on write would
            rk = [read_key_rerank(hits_per_query[i], q[0], keyer)
                  for i, q in zip(idx, queries)]
            arms = {
                "as_is": [hits_per_query[i] for i in idx],
                "read_keyed": rk,
                "read_keyed_collapsed": [collapse(h, keyer) for h in rk],
                # the same merge, resolving ties the wrong way round: how much
                # of the gain is entity resolution and how much is recency
                # being correct by construction on the clean condition
                "read_keyed_oldest": [collapse(h, keyer, keep="oldest")
                                      for h in rk],
            }
            # an invalidating system carries its own verdict on every hit;
            # ranking non-invalidated edges first is the read path of the
            # paper's temporal-KG rung, applied to the system's output
            if any("invalid_at" in h for hits in arms["as_is"] for h in hits):
                arms["valid_first"] = [
                    sorted(hits, key=lambda h: 0 if h.get("invalid_at") is None else 1)
                    for hits in arms["as_is"]]
            pol_out: Dict = {}
            for arm, hits in arms.items():
                top = [h[:_TOPK] for h in hits]
                retr = _retrieval(top, queries, gold, stale)
                pol_out[arm] = {
                    "retrieval": {m: float(np.mean([r[m] for r in retr]))
                                  for m in retr[0]},
                    "retrieval_records": retr,
                    "prompts": [build_prompt([h.get("text") or "" for h in t],
                                             q[0])
                                for t, q in zip(top, queries)],
                    "readers": {},
                }
            cell_out[pol] = pol_out
        out["by_config"][name] = cell_out

    # -- readers ----------------------------------------------------------- #
    for spec in reader_specs:
        client = make_client(spec)
        print(f"\n=== reader {client.name} ===", flush=True)
        for name, cell_out in out["by_config"].items():
            _, gold, stale = ques[name]
            for pol, pol_out in cell_out.items():
                for arm, a in pol_out.items():
                    before = getattr(client, "n_failed", 0)
                    answers = _batch(client, a["prompts"])
                    recs = [score_answer(x, g, s)
                            for x, g, s in zip(answers, gold, stale)]
                    a["readers"][client.name] = {
                        "summary": {m: float(np.mean([r[m] for r in recs]))
                                    for m in _METRICS},
                        "ci": {m: bootstrap_ci([r[m] for r in recs], seed=1)
                               for m in ("correct", "stale")},
                        "records": recs,
                        # this arm's own failures, not the client's running total
                        "n_failed": getattr(client, "n_failed", 0) - before,
                    }
                    s = a["readers"][client.name]["summary"]
                    print(f"  [{name}] {pol:24s} {arm:11s} correct "
                          f"{s['correct']:.3f}  stale {s['stale']:.3f}  hedged "
                          f"{s['hedged']:.3f}  (retr cur@1 "
                          f"{a['retrieval']['current@1']:.3f} / text "
                          f"{a['retrieval']['current@1_text']:.3f})", flush=True)
        # prompts are large and reconstructible; keep the file readable
    for cell_out in out["by_config"].values():
        for pol_out in cell_out.values():
            for a in pol_out.values():
                a.pop("prompts", None)

    out["tests"] = _significance(out)
    return out


def _significance(out: Dict) -> Dict:
    """Paired sign-flip tests over the comparisons this track exists to make.

    Every pair below shares a stream, a query set and a store state, so the
    per-query differences are genuinely paired. Holm is applied once over the
    whole family rather than per config, because the family is what a reader
    would scan.
    """
    raw: Dict[str, float] = {}
    detail: Dict[str, Dict] = {}

    def add(label, a, b, metric):
        if not a or not b:
            return
        x = [r[metric] for r in a]
        y = [r[metric] for r in b]
        raw[label] = paired_permutation(x, y, seed=7)
        detail[label] = {"metric": metric, "n": len(x),
                         "mean_a": float(np.mean(x)),
                         "mean_b": float(np.mean(y)),
                         "delta": float(np.mean(x) - np.mean(y))}

    for cfgname, cell in out["by_config"].items():
        readers = set()
        for pol_out in cell.values():
            readers |= set(pol_out["as_is"]["readers"])
        sysnames = [p for p in cell if p.startswith(("mem0", "graphiti", "langmem", "erase"))]
        for pol, pol_out in cell.items():
            # did resolving the query -- and then merging -- help whatever
            # produced these contexts, at retrieval and at the reader?
            for arm in ("read_keyed", "read_keyed_collapsed",
                        "read_keyed_oldest", "valid_first"):
                if arm not in pol_out:
                    continue
                for metric in ("current@1", "current@1_text",
                               "contamination@5_text"):
                    add(f"{cfgname}:{pol}:{arm}-vs-as_is:{metric}",
                        pol_out[arm]["retrieval_records"],
                        pol_out["as_is"]["retrieval_records"], metric)
                for rd in readers:
                    a = pol_out[arm]["readers"].get(rd)
                    b = pol_out["as_is"]["readers"].get(rd)
                    if a and b:
                        add(f"{cfgname}:{pol}:{arm}-vs-as_is:{rd}:correct",
                            a["records"], b["records"], "correct")
        # the system against every ladder rung on the same stream, at the reader
        for sysname in sysnames:
            for pol in cell:
                if pol == sysname:
                    continue
                for rd in readers:
                    a = cell[sysname]["as_is"]["readers"].get(rd)
                    b = cell[pol]["as_is"]["readers"].get(rd)
                    if a and b:
                        add(f"{cfgname}:{sysname}-vs-{pol}:{rd}:correct",
                            a["records"], b["records"], "correct")
                        add(f"{cfgname}:{sysname}-vs-{pol}:{rd}:stale",
                            a["records"], b["records"], "stale")
    return {"raw": raw, "holm": holm(raw) if raw else {},
            "family_size": len(raw), "detail": detail}


def main() -> None:
    args = [a for a in sys.argv[1:]]
    if not args:
        raise SystemExit(__doc__.strip().splitlines()[-1])
    ctx_path, specs = args[0], (args[1:] or _DEFAULT_READERS)
    res = run(ctx_path, specs)
    os.makedirs(os.path.dirname(_OUT), exist_ok=True)
    json.dump(res, open(_OUT, "w"), indent=1)
    print(f"\nWrote {os.path.abspath(_OUT)}")


if __name__ == "__main__":
    main()
