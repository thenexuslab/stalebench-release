"""The gate for LangMem, mirroring the Graphiti probe.

Two statements about one fact, then six about two: does the library store
every write, does its own search return the queried fact first, and does an
UPDATE (or a DELETE, where enabled) fire when a fact is superseded? A system
whose numbers are quoted must pass this first, on the model it is quoted with.

Run:  STALEBENCH_LANGMEM_LLM=ollama:qwen2.5:14b python -m stalebench.probe_langmem
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

from stalebench.systems_langmem import LangMemPolicy

_HERE = os.path.dirname(__file__)
_OUT = os.environ.get("STALEBENCH_LANGMEM_PROBE_OUT") or os.path.join(
    _HERE, "..", "results", "langmem_probe.json")

_PAIR = [("Peter Sagan plays for Bora-Hansgrohe.", "f0:0", 0.0),
         ("Peter Sagan plays for TotalEnergies.", "f0:1", 1.0)]
_SIX = [("Peter Sagan plays for Liquigas.", "f0:0", 0.0),
        ("Valentino Rossi plays for Ducati.", "f1:0", 0.0001),
        ("Peter Sagan plays for Tinkoff.", "f0:1", 1.0),
        ("Valentino Rossi plays for Yamaha.", "f1:1", 1.0001),
        ("Peter Sagan plays for Bora-Hansgrohe.", "f0:2", 2.0),
        ("Valentino Rossi plays for Petronas Yamaha.", "f1:2", 2.0001)]
_Q = {"f0": "Which team does Peter Sagan currently play for?",
      "f1": "Which team does Valentino Rossi currently play for?"}


def _policy() -> LangMemPolicy:
    spec = os.environ.get("STALEBENCH_LANGMEM_LLM", "ollama:qwen2.5:14b")
    backend, _, model = spec.partition(":")
    return LangMemPolicy(llm_model=model, llm_backend=backend,
                         enable_deletes=os.environ.get("STALEBENCH_LANGMEM_DELETES") == "1")


def _stream(pol: LangMemPolicy, statements):
    pol.reset()
    t0 = time.time()
    for text, iid, t in statements:
        pol.observe(np.zeros(384, np.float32), iid, t, text)
    secs = time.time() - t0
    rankings = {k: pol.rank(np.zeros(384, np.float32), q) for k, q in _Q.items()
                if any(iid.startswith(k) for _, iid, _ in statements)}
    hits = {k: [(h["sid"], h["text"][:60]) for h in pol.hits(q)[:5]] for k, q in _Q.items()
            if k in rankings}
    return {"seconds": round(secs, 1), "per_statement": round(secs / len(statements), 1),
            "llm_calls": pol.n_llm_calls, "events": dict(pol.events), "slots": pol.slots,
            "rankings": rankings, "hits": hits}


def run():
    pol = _policy()
    pair = _stream(pol, _PAIR)
    six = _stream(pol, _SIX)
    current = {"f0": "f0:2", "f1": "f1:2"}
    verdict = {
        "writes_ok": pair["slots"] >= 1 and six["slots"] >= 2
                     and not pair["events"].get("ADD_FAILED") and not six["events"].get("ADD_FAILED"),
        "retrieval_ok": all(r and r[0].startswith(k) for k, r in six["rankings"].items()),
        "update_fires": bool(pair["events"].get("UPDATE") or six["events"].get("UPDATE")),
        "current_first": all(r and r[0] == current[k] for k, r in six["rankings"].items()),
    }
    verdict["gate_pass"] = verdict["writes_ok"] and verdict["retrieval_ok"]
    return {"lib_version": pol.lib_version, "llm_model": pol.llm_model,
            "llm_backend": pol.llm_backend, "enable_deletes": pol.enable_deletes,
            "pair": pair, "six": six, "verdict": verdict}


def main() -> None:
    res = run()
    os.makedirs(os.path.dirname(_OUT), exist_ok=True)
    json.dump(res, open(_OUT, "w"), indent=1)
    v = res["verdict"]
    print(f"\nlangmem {res['lib_version']} with {res['llm_backend']}:{res['llm_model']} "
          f"(deletes={'on' if res['enable_deletes'] else 'off'})")
    print(f"pair: {res['pair']['per_statement']}s/stmt {res['pair']['llm_calls']} calls "
          f"events {res['pair']['events']} slots {res['pair']['slots']} ranking {res['pair']['rankings']}")
    print(f"six:  {res['six']['per_statement']}s/stmt {res['six']['llm_calls']} calls "
          f"events {res['six']['events']} slots {res['six']['slots']} ranking {res['six']['rankings']}")
    for k, hs in res["six"]["hits"].items():
        print(f"  hits {k}: {hs}")
    print(f"VERDICT  writes {'OK' if v['writes_ok'] else 'FAIL'}  retrieval "
          f"{'OK' if v['retrieval_ok'] else 'FAIL'}  update {'fires' if v['update_fires'] else 'does NOT fire'}"
          f"  current-first {v['current_first']}")
    print(f"Wrote {os.path.abspath(_OUT)}")
    if not v["gate_pass"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
