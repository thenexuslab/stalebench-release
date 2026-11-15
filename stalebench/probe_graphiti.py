"""Graphiti's gate: does it write, does it retrieve, does invalidation fire?

Every (release, backend) pair in the systems section is hand-probed before its
numbers are used (Table tab:gates), because a pair can produce plausible-looking
metrics while measuring the backend rather than the system, in either half. This
is that probe for Graphiti, in two sizes:

* the paper's two-statement probe --- Sagan moves clubs; ask which club now. A
  system that supersedes ends with one *active* fact and ranks the new club
  first. For a temporal graph "supersedes" means the old edge gains an
  ``invalid_at`` while staying in the store, so the probe reports both store
  sizes and the validity flag on every hit rather than a slot count alone.
* a six-statement probe --- two subjects, three versions each, interleaved ---
  the smallest stream on which "every write is stored" and "the queried fact
  comes first" can fail independently of each other, and on which invalidation
  has to fire twice per subject.

The gate is the paper's: writes and retrieval. Whether the *current* version
ranks first, and whether invalidation fires, are findings the full run measures
at scale; they are printed here because two sentences show them more clearly
than three hundred, and they are not what decides whether the run is worth an
evening of local inference.

Writes ``results/graphiti_probe.json`` and exits non-zero if the gate fails, so
a chained full run does not start on a broken stack.

Run:  python -m stalebench.probe_graphiti
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.request
import warnings
from typing import Dict, List, Sequence, Tuple

import numpy as np

_HERE = os.path.dirname(__file__)
_OUT = os.path.join(_HERE, "..", "results", "graphiti_probe.json")

# the Policy interface wants an embedding; the system computes its own
_Z = np.zeros(384, dtype=np.float32)

# the paper's probe: one fact, two versions, chronological. The query names no
# club, so nothing in it favours either version.
_PAIR = [("Peter Sagan plays for Bora-Hansgrohe.", "f0:0", 0.0),
         ("Peter Sagan plays for TotalEnergies.", "f0:1", 1.0)]
_PAIR_QUERIES = [("Which team does Peter Sagan currently play for?", "f0:1")]

_SIX = [("Peter Sagan plays for Liquigas.", "f0:0", 0.0),
        ("Valentino Rossi rides for Ducati.", "f1:0", 1.0),
        ("Peter Sagan plays for Tinkoff.", "f0:1", 2.0),
        ("Valentino Rossi rides for Yamaha.", "f1:1", 3.0),
        ("Peter Sagan plays for Bora-Hansgrohe.", "f0:2", 4.0),
        ("Valentino Rossi rides for Petronas Yamaha SRT.", "f1:2", 5.0)]
_SIX_QUERIES = [("Which team does Peter Sagan currently play for?", "f0:2"),
                ("Which team does Valentino Rossi currently ride for?", "f1:2")]


def _watch_prompt_sizes(pol, sizes: List[Tuple[int, int]]) -> None:
    """Record every call's token usage, which the library discards.

    Ollama serves a fixed context window and truncates a longer prompt from the
    front, where the instructions are; a run whose prompts exceed it would be
    measuring truncation. The probe records the largest prompt it sends so that
    number can be set against the window the server actually loaded.
    """
    llm = pol._graph().llm_client
    orig = llm.client.chat.completions.create

    async def create(**kw):
        r = await orig(**kw)
        if getattr(r, "usage", None):
            sizes.append((r.usage.prompt_tokens, r.usage.completion_tokens))
        return r

    llm.client.chat.completions.create = create


def _context_length(ollama_url: str) -> int:
    try:
        ps = json.load(urllib.request.urlopen(f"{ollama_url}/api/ps", timeout=5))
        return int(ps["models"][0]["context_length"])
    except Exception:
        return -1


def _probe(pol, stream: Sequence[Tuple[str, str, float]],
           queries: Sequence[Tuple[str, str]],
           sizes: List[Tuple[int, int]]) -> Dict:
    pol.reset()
    _watch_prompt_sizes(pol, sizes)
    t0 = time.time()
    calls0 = pol.n_llm_calls
    for text, sid, t in stream:
        pol.observe(_Z, sid, t, text)
    ingest = time.time() - t0
    written = pol.written()
    out_q = []
    for qtext, current in queries:
        hits = pol.hits(qtext)
        ranking: List[str] = []
        active: List[str] = []
        for h in hits:
            if h["sid"] and h["sid"] not in ranking:
                ranking.append(h["sid"])
            if h["sid"] and not h["invalid_at"] and h["sid"] not in active:
                active.append(h["sid"])
        key = current.split(":")[0]
        out_q.append({
            "query": qtext, "current": current,
            "ranking": ranking, "active_ranking": active,
            # the backend half of the gate: the first thing back is about the
            # fact that was asked for
            "rank1_is_queried_fact": bool(ranking and ranking[0].split(":")[0] == key),
            # the system half: is it the current version, as returned, and
            # once the facts the system itself invalidated are dropped
            "rank1_is_current": bool(ranking and ranking[0] == current),
            "active_rank1_is_current": bool(active and active[0] == current),
            "hits": hits})
    return {"n_statements": len(stream),
            "ingest_seconds": round(ingest, 1),
            "seconds_per_statement": round(ingest / len(stream), 1),
            "llm_calls": pol.n_llm_calls - calls0,
            "events": dict(pol.events),
            "slots": pol.slots, "active_slots": pol.active_slots,
            "n_silent_add": pol.n_silent_add,
            "written": written,
            "all_written": len(written) == len(stream),
            "queries": out_q}


def _show(label: str, r: Dict) -> None:
    print(f"\n[{label}] {r['n_statements']} statements in {r['ingest_seconds']}s "
          f"({r['seconds_per_statement']}s/stmt, {r['llm_calls']} llm calls)")
    print(f"  events {r['events']}  slots {r['slots']}  active {r['active_slots']}"
          f"  silent {r['n_silent_add']}  written {r['written']}")
    for q in r["queries"]:
        print(f"  Q: {q['query']}  current={q['current']}")
        print(f"     ranking {q['ranking']}  active {q['active_ranking']}  "
              f"queried-fact-first={q['rank1_is_queried_fact']} "
              f"current-first={q['rank1_is_current']} "
              f"active-current-first={q['active_rank1_is_current']}")
        for i, h in enumerate(q["hits"], 1):
            flag = "INVALID" if h["invalid_at"] else "valid  "
            va = (h["valid_at"] or "")[:10]
            ia = (h["invalid_at"] or "")[:10]
            print(f"     {i:2d}. {flag} sid={h['sid']} first={h['meta_sid']} "
                  f"valid_at={va or None} invalid_at={ia or None}  {h['text']!r}")


def run() -> Dict:
    warnings.filterwarnings("ignore")
    from stalebench.systems import GraphitiPolicy

    # STALEBENCH_GRAPHITI_LLM=zai:glm-4.6 probes the hosted adjudicator instead
    spec = os.environ.get("STALEBENCH_GRAPHITI_LLM", "ollama:qwen2.5:14b")
    llm_backend, _, model = spec.partition(":")
    if llm_backend == "zai":
        pol = GraphitiPolicy(llm_model=model, llm_backend="zai",
                             hosted_base_url="https://api.z.ai/api/paas/v4",
                             hosted_api_key=os.environ.get("ZAI_API_KEY", ""))
    else:
        pol = GraphitiPolicy(llm_model=model)
    sizes: List[Tuple[int, int]] = []
    pair = _probe(pol, _PAIR, _PAIR_QUERIES, sizes)
    _show("pair", pair)
    six = _probe(pol, _SIX, _SIX_QUERIES, sizes)
    _show("six", six)
    pol.close()

    probes = (pair, six)
    writes_ok = all(r["all_written"] and r["n_silent_add"] == 0 for r in probes)
    retrieval_ok = all(q["hits"] and q["rank1_is_queried_fact"]
                       for r in probes for q in r["queries"])
    verdict = {
        "writes_ok": writes_ok,
        "retrieval_ok": retrieval_ok,
        "gate_pass": writes_ok and retrieval_ok,
        "invalidation_fires": all(r["events"].get("EDGE_INVALIDATE", 0) > 0
                                  for r in probes),
        "current_first": all(q["rank1_is_current"]
                             for r in probes for q in r["queries"]),
        "active_current_first": all(q["active_rank1_is_current"]
                                    for r in probes for q in r["queries"]),
    }
    prompt = [p for p, _ in sizes]
    return {"lib_version": pol.lib_version,
            "invalidates_on_add": pol.invalidates_on_add,
            "graph_backend": pol.graph_backend,
            "llm_model": pol.llm_model, "embed_model": pol.embed_model,
            "prompt_tokens": {"n": len(prompt),
                              "max": max(prompt) if prompt else None,
                              "mean": round(float(np.mean(prompt)), 1) if prompt else None,
                              "ollama_context_length": _context_length(pol.ollama_url)},
            "pair": pair, "six": six, "verdict": verdict}


def main() -> None:
    res = run()
    os.makedirs(os.path.dirname(_OUT), exist_ok=True)
    json.dump(res, open(_OUT, "w"), indent=1)
    v = res["verdict"]
    print(f"\ngraphiti-core {res['lib_version']} on {res['graph_backend']} with "
          f"{res['llm_model']} (invalidates_on_add={res['invalidates_on_add']})")
    print(f"prompt tokens: max {res['prompt_tokens']['max']} mean "
          f"{res['prompt_tokens']['mean']} over {res['prompt_tokens']['n']} calls; "
          f"server context {res['prompt_tokens']['ollama_context_length']}")
    print(f"VERDICT  writes {'OK' if v['writes_ok'] else 'FAIL'}  retrieval "
          f"{'OK' if v['retrieval_ok'] else 'FAIL'}  invalidation "
          f"{'fires' if v['invalidation_fires'] else 'does NOT fire'}  current-first "
          f"{v['current_first']}  active-current-first {v['active_current_first']}")
    print(f"Wrote {os.path.abspath(_OUT)}")
    if not v["gate_pass"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
