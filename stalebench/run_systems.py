"""Place a shipped agent-memory system on the keying ladder.

The paper prices same-fact detection with idealized rungs and cites production
systems as the motivation. This runner closes that loop by scoring Mem0 --- whose
ADD/UPDATE/DELETE loop is exactly the decision being priced --- under the same
harness, on the same streams, with the same encoder as the dense baselines.

Four things are reported, and the first is the weakest of them:

* the usual retrieval metrics, so the system takes its place in the ladder;
* **Mem0's own event counts**. A system whose UPDATE operation never fires is
  a store-all policy regardless of what its retrieval scores look like, and an
  event histogram says that in a way no aggregate metric can;
* the same contexts scored by **what they say** rather than by which statement
  id they carry, because a system that merges memories is not honestly scored
  by attribution alone (see :func:`stalebench.metrics.score_texts`); and
* the contexts themselves, written to a sidecar file, so the reader track can
  ask what a model answers from this system's output without paying for the
  ingestion twice.

Because the decision is made by an LLM, the system is run with more than one
model where possible: "this system does not supersede" and "the model we gave
it does not supersede" are different findings, and only a second model
separates them.

A ``graphiti:ollama:<model>`` spec scores Graphiti, the engine under Zep, under
the identical harness. Its histogram counts facts added and invalidated rather
than ADD/UPDATE, and because it supersedes by invalidation rather than deletion
its store has an active size beside its total.

Run:  python -m stalebench.run_systems [--limit N] [--models ollama:qwen2.5:14b,zai:glm-4.6,graphiti:ollama:qwen2.5:14b]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List

import numpy as np

from stalebench.embed import MINILM, Embedder
from stalebench.metrics import bootstrap_ci, score_ranking, score_texts
from stalebench.policies import (CosineRAG, FuzzyKey, KeepLatestOracle,
                                 LexicalKey, ReadWriteFuzzyKey,
                                 Reconsolidation, TimeRerank)
from stalebench.reader import stale_values_of
from stalebench.templama import build_instance, load_facts, split_facts

_HERE = os.path.dirname(__file__)
_OUT = os.environ.get("STALEBENCH_SYS_OUT") or os.path.join(
    _HERE, "..", "results", "stalebench_systems_results.json")

_PARA = 0.5
_KS = (5,)
# how deep to record what each policy surfaced. The reader consumes five, but
# read-path keying re-ranks from depth, so a context file truncated at five
# cannot be used to ask what resolving the query would have done -- which is
# the paper's own cheapest fix applied to a deployed system.
_DEPTH = 20


def _hits(policy, inst, q, qtext, depth: int) -> List[Dict]:
    """What this policy surfaces for one query, in its own order.

    A wrapped system may rewrite the statement it was given, so its hits carry
    their own text; a ladder policy returns the corpus verbatim, so its text is
    looked up. Recording both in one shape lets the reader track and the
    text-level scorer treat them identically.
    """
    if hasattr(policy, "hits"):
        return [dict(h) for h in policy.hits(qtext)[:depth]]
    return [{"id": sid, "sid": sid, "text": inst.text_of.get(sid, ""),
             "t": inst.year_of.get(sid)}
            for sid in policy.rank(q, qtext)[:depth]]


def _score(policy, inst, ev, qe, gold=None, stale_vals=None,
           depth: int = _DEPTH) -> Dict:
    policy.reset()
    t0 = time.time()
    for (text, iid, t), e in zip(inst.events, ev):
        policy.observe(e, iid, t, text)
    ingest = time.time() - t0
    records, contexts = [], []
    for i, ((qtext, key, cur), q) in enumerate(zip(inst.queries, qe)):
        hits = _hits(policy, inst, q, qtext, depth)
        ranked = []
        for h in hits:
            if h.get("sid") and h["sid"] not in ranked:
                ranked.append(h["sid"])
        rec = score_ranking(ranked, key, cur, _KS)
        if gold is not None:
            # the id-level ranking drops hits the wrapper cannot attribute; the
            # text-level one must not, because a reader sees them
            rec.update(score_texts([h["text"] for h in hits], gold[i],
                                   stale_vals[i], _KS))
        records.append(rec)
        contexts.append(hits)
    summary = {m: float(np.mean([r[m] for r in records])) for m in records[0]}
    summary["slots"] = float(policy.slots)
    out = {"summary": summary,
           "records": records,
           "contexts": contexts,
           "ci": {m: bootstrap_ci([r[m] for r in records], seed=1)
                  for m in ("current@1", "contamination@5")},
           "ingest_seconds": round(ingest, 1),
           "events": getattr(policy, "events", {}),
           "n_add_calls": getattr(policy, "n_add", None),
           "lib_version": getattr(policy, "lib_version", None),
           "vector_store": getattr(policy, "vector_store", None),
           "n_silent_add": getattr(policy, "n_silent_add", 0),
           "adjudicates_on_add": getattr(policy, "adjudicates_on_add", None),
           # which extraction prompt and which LLM backend produced this cell
           "extraction_prompt": getattr(policy, "extraction_prompt_label", None),
           "llm_backend": getattr(policy, "llm_backend", None)}
    if hasattr(policy, "invalidates_on_add"):
        # a graph system supersedes by invalidation, so its store has two
        # sizes, and its analogue of the adjudication flag is its own
        summary["active_slots"] = float(policy.active_slots)
        out.update({"invalidates_on_add": policy.invalidates_on_add,
                    "graph_backend": policy.graph_backend,
                    "n_llm_calls": policy.n_llm_calls})
    if hasattr(policy, "rewrites_on_add"):
        out.update({"rewrites_on_add": policy.rewrites_on_add,
                    "retrieve_k": policy.retrieve_k,
                    "n_llm_calls": policy.n_llm_calls})
    if hasattr(policy, "updates_on_add"):
        out.update({"updates_on_add": policy.updates_on_add,
                    "deletes_on_add": policy.deletes_on_add,
                    "n_llm_calls": policy.n_llm_calls})
    return out


_ZAI_URL = "https://api.z.ai/api/paas/v4"


def _policy_for(spec: str):
    """``ollama:<m>`` and ``zai:<m>`` drive Mem0; ``graphiti:ollama:<m>`` and
    ``graphiti:zai:<m>`` drive Graphiti. A ``@third-person`` suffix on a Mem0
    spec swaps the user-centric extraction prompt for one that admits facts
    about any named entity (see ``THIRD_PERSON_EXTRACTION_PROMPT``)."""
    from stalebench.systems import (THIRD_PERSON_EXTRACTION_PROMPT,
                                    GraphitiPolicy, Mem0Policy)
    spec, _, variant = spec.partition("@")
    if variant not in ("", "third-person"):
        raise ValueError(f"unknown variant {variant!r}")
    prompt = THIRD_PERSON_EXTRACTION_PROMPT if variant else None
    backend, _, model = spec.partition(":")
    if backend == "ollama":
        return Mem0Policy(llm_model=model, extraction_prompt=prompt)
    if backend == "zai":
        pol = Mem0Policy(llm_model=model, extraction_prompt=prompt)
        pol.use_openai_compatible(base_url=_ZAI_URL,
                                  api_key=os.environ.get("ZAI_API_KEY", ""))
        return pol
    if backend == "graphiti":
        llm_backend, _, model = model.partition(":")
        if llm_backend == "ollama":
            return GraphitiPolicy(llm_model=model)
        if llm_backend == "zai":
            return GraphitiPolicy(llm_model=model, llm_backend="zai",
                                  hosted_base_url=_ZAI_URL,
                                  hosted_api_key=os.environ.get("ZAI_API_KEY", ""))
        raise ValueError(f"graphiti runs against ollama or zai, not {llm_backend!r}")
    if backend == "langmem":
        # langmem:<ollama|zai>:<model>; a ``@deletes`` variant would have been
        # parsed off above as ``variant``, but LangMem's switch is separate
        from stalebench.systems_langmem import LangMemPolicy
        llm_backend, _, model = model.partition(":")
        if llm_backend not in ("ollama", "zai"):
            raise ValueError(f"langmem runs against ollama or zai, not {llm_backend!r}")
        return LangMemPolicy(llm_model=model, llm_backend=llm_backend,
                             enable_deletes=os.environ.get("STALEBENCH_LANGMEM_DELETES") == "1")
    if backend == "erase":
        # erase:<ollama|zai>:<model>: rewrite-on-arrival, reimplemented
        from stalebench.systems_erase import ErasePolicy
        llm_backend, _, model = model.partition(":")
        if llm_backend not in ("ollama", "zai"):
            raise ValueError(f"erase runs against ollama or zai, not {llm_backend!r}")
        return ErasePolicy(llm_model=model, llm_backend=llm_backend)
    raise ValueError(f"unknown backend {backend!r}")


def _ctx_path(out_path: str) -> str:
    """Sidecar for the surfaced contexts.

    They are two orders of magnitude larger than the summaries and are consumed
    by a different runner, so they do not belong in the file the paper's number
    generator reads on every build.
    """
    base, _, _ = out_path.rpartition(".json")
    return (base or out_path) + ".contexts.json"


def run(limit: int, models: List[str]) -> Dict:
    facts, stats = load_facts()
    val, test = split_facts(facts)
    test = test[:limit] if limit else test
    emb = Embedder(MINILM)
    gold = [f.current_value for f in test]
    stale_vals = [stale_values_of(f.runs, f.current_value) for f in test]
    out: Dict = {"config": {"paraphrase_p": _PARA, "n_test_facts": len(test),
                            "subsampled": bool(limit), "models": models,
                            "depth": _DEPTH},
                 "by_config": {}}
    ctx: Dict = {"config": dict(out["config"]),
                 "gold": gold, "stale_values": stale_vals,
                 "by_config": {}}
    # tuned on val at each regime, as everywhere else in this paper; the
    # previous hard-coded k=2 made the methods guardrail false even though the
    # tuned value is the same
    gaz_val = [f.subject for f in val]
    for alias_p, cfg in ((0.0, "anchored"), (1.0, "anchor_free")):
        inst_v = build_instance(val, paraphrase_p=_PARA, seed=0, alias_p=alias_p)
        vtag = f"sysval_n{len(val)}_a{alias_p}"
        ev_v = emb.encode([e[0] for e in inst_v.events], f"{vtag}_ev")
        qv = emb.encode([q[0] for q in inst_v.queries], f"{vtag}_q")

        def _c1(pol):
            return _score(pol, inst_v, ev_v, qv)["summary"]["current@1"]
        tuned = {
            "k": max([2, 3, 5, 10, 20], key=lambda k: _c1(TimeRerank(k=k))),
            "tau": max([round(float(x), 2) for x in np.arange(0.50, 0.91, 0.05)],
                       key=lambda t: _c1(Reconsolidation(tau=t))),
            "ftau": max([round(float(x), 2) for x in np.arange(0.20, 0.66, 0.05)],
                        key=lambda t: _c1(FuzzyKey(gaz_val, t))),
        }
        print(f"[{cfg}] tuned on val: {tuned}", flush=True)
        inst = build_instance(test, paraphrase_p=_PARA, seed=0, alias_p=alias_p)
        tag = f"n{len(test)}_a{alias_p}"
        ev = emb.encode([e[0] for e in inst.events], f"sys_ev_{tag}")
        qe = emb.encode([q[0] for q in inst.queries], f"sys_q_{tag}")
        # one query per fact, in fact order, so the value lists line up by index
        assert inst.gold == gold, "query order diverged from the fact order"
        cell: Dict = {"n_statements": len(inst.events)}
        ccell: Dict = {}
        # reference points from the paper's own ladder, same stream
        # The paper's own ladder belongs on this stream. Without it a reader
        # compares a deployed system against store-all and an oracle only, and
        # cross-table inference to the keying rungs is wrong because this
        # subsample is easier for them than the full test split.
        gaz = [f.subject for f in test]
        for name, fac in (("cosine-rag", CosineRAG),
                          ("time-rerank", lambda: TimeRerank(k=tuned["k"])),
                          ("reconsolidation",
                           lambda: Reconsolidation(tau=tuned["tau"])),
                          ("lexical-key", lambda: LexicalKey(gaz)),
                          ("fuzzy-key", lambda: FuzzyKey(gaz, tuned["ftau"])),
                          ("fuzzy-rw-key",
                           lambda: ReadWriteFuzzyKey(gaz, tuned["ftau"])),
                          ("keep-latest-oracle", KeepLatestOracle)):
            cell[name] = _score(fac(), inst, ev, qe, gold, stale_vals)
            ccell[name] = cell[name].pop("contexts")
        for spec in models:
            pol = _policy_for(spec)
            print(f"[{cfg}] {pol.name} with {spec} over {len(inst.events)} "
                  f"statements", flush=True)
            cell[pol.name] = _score(pol, inst, ev, qe, gold, stale_vals)
            c = cell[pol.name]
            ccell[pol.name] = c.pop("contexts")
            active = (f" active {c['summary']['active_slots']:.0f}"
                      if "active_slots" in c["summary"] else "")
            print(f"  current@1 {c['summary']['current@1']:.3f}  contam@5 "
                  f"{c['summary']['contamination@5']:.3f}  slots "
                  f"{c['summary']['slots']:.0f}{active}  events {c['events']}  "
                  f"({c['ingest_seconds']:.0f}s)", flush=True)
            print(f"  text-level: current@1 "
                  f"{c['summary']['current@1_text']:.3f}  hedged@1 "
                  f"{c['summary']['hedged@1_text']:.3f}  contam@5 "
                  f"{c['summary']['contamination@5_text']:.3f}", flush=True)
        out["by_config"][cfg] = cell
        ctx["by_config"][cfg] = ccell
        # write after every config: these runs take an hour each and have twice
        # been lost whole to a crash in the second one. A partial file is
        # usable; the paper's tables already skip configs that are absent.
        try:
            os.makedirs(os.path.dirname(_OUT), exist_ok=True)
            json.dump(out, open(_OUT, "w"), indent=1)
            json.dump(ctx, open(_ctx_path(_OUT), "w"))
            print(f"  [checkpoint] wrote {cfg} to {os.path.abspath(_OUT)}"
                  f" (+ contexts)", flush=True)
        except OSError as e:
            print(f"  [checkpoint failed] {e}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=150,
                    help="facts to use (0 = all); systems are slow to ingest")
    ap.add_argument("--models", default="ollama:qwen2.5:14b",
                    help="comma-separated: ollama:<m> / zai:<m> score Mem0, "
                         "graphiti:ollama:<m> scores Graphiti")
    a = ap.parse_args()
    res = run(a.limit, [m for m in a.models.split(",") if m])
    os.makedirs(os.path.dirname(_OUT), exist_ok=True)
    json.dump(res, open(_OUT, "w"), indent=1)

    print("\n" + "=" * 74)
    print(f"Production systems on StaleBench "
          f"({res['config']['n_test_facts']} test facts)")
    print("=" * 74)
    for cfg, cell in res["by_config"].items():
        print(f"\n[{cfg}]  {cell['n_statements']} statements")
        for name, r in cell.items():
            if not isinstance(r, dict) or "summary" not in r:
                continue
            s = r["summary"]
            ev = f"  events {r['events']}" if r.get("events") else ""
            if "active_slots" in s:
                ev += f"  active {s['active_slots']:.0f}"
            if "current@1_text" in s:
                ev += (f"  [text cur@1 {s['current@1_text']:.3f} hedged "
                       f"{s['hedged@1_text']:.3f} contam@5 "
                       f"{s['contamination@5_text']:.3f}]")
            print(f"  {name:24s} cur@1 {s['current@1']:.3f}  stale "
                  f"{s['stale_rate']:.3f}  contam@5 {s['contamination@5']:.3f}"
                  f"  slots {s['slots']:.0f}{ev}")
    print(f"\nWrote {os.path.abspath(_OUT)}")


if __name__ == "__main__":
    main()
    # Both configs of a completed run sat on disk for half an hour while the
    # interpreter refused to exit: the wrapped library's ingestion thread pool
    # and its telemetry client leave non-daemon threads that the shutdown
    # handler joins forever, so a chained job waiting on this PID waits on
    # nothing. Everything is written and flushed by the line above, and this
    # runs only after main() returns normally -- an exception still propagates
    # and still fails the process.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
