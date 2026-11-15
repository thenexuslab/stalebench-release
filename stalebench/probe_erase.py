"""The gate for ERASE, mirroring the LangMem probe.

Run:  STALEBENCH_ERASE_LLM=ollama:qwen2.5:14b python -m stalebench.probe_erase
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

from stalebench.systems_erase import ErasePolicy
from stalebench.probe_langmem import _PAIR, _SIX, _Q

_HERE = os.path.dirname(__file__)
_OUT = os.environ.get("STALEBENCH_ERASE_PROBE_OUT") or os.path.join(
    _HERE, "..", "results", "erase_probe.json")


def _policy() -> ErasePolicy:
    spec = os.environ.get("STALEBENCH_ERASE_LLM", "ollama:qwen2.5:14b")
    backend, _, model = spec.partition(":")
    return ErasePolicy(llm_model=model, llm_backend=backend)


def _stream(pol: ErasePolicy, statements):
    pol.reset()
    t0 = time.time()
    for text, iid, t in statements:
        pol.observe(pol._embed(text), iid, t, text)
    secs = time.time() - t0
    keys = {iid.rsplit(":", 1)[0] for _, iid, _ in statements}
    rankings = {k: pol.rank(np.zeros(384, np.float32), q) for k, q in _Q.items() if k in keys}
    hits = {k: [(h["sid"], h["text"][:70]) for h in pol.hits(q)[:5]] for k, q in _Q.items() if k in keys}
    return {"seconds": round(secs, 1), "per_statement": round(secs / len(statements), 1),
            "llm_calls": pol.n_llm_calls, "events": dict(pol.events), "slots": pol.slots,
            "rankings": rankings, "hits": hits}


def run():
    pol = _policy()
    pair = _stream(pol, _PAIR)
    six = _stream(pol, _SIX)
    current = {"f0": "f0:2", "f1": "f1:2"}
    edits = lambda ev: ev.get("REWRITE", 0) + ev.get("DELETE", 0)
    verdict = {
        "writes_ok": pair["slots"] >= 1 and six["slots"] >= 2,
        "retrieval_ok": all(r and r[0].startswith(k) for k, r in six["rankings"].items()),
        "edit_fires": bool(edits(pair["events"]) or edits(six["events"])),
        "current_first": all(r and r[0] == current[k] for k, r in six["rankings"].items()),
    }
    verdict["gate_pass"] = verdict["writes_ok"] and verdict["retrieval_ok"]
    return {"lib_version": pol.lib_version, "llm_model": pol.llm_model,
            "llm_backend": pol.llm_backend, "retrieve_k": pol.retrieve_k,
            "pair": pair, "six": six, "verdict": verdict}


def main() -> None:
    res = run()
    os.makedirs(os.path.dirname(_OUT), exist_ok=True)
    json.dump(res, open(_OUT, "w"), indent=1)
    v = res["verdict"]
    print(f"\nerase ({res['lib_version']}) with {res['llm_backend']}:{res['llm_model']} k={res['retrieve_k']}")
    for name in ("pair", "six"):
        r = res[name]
        print(f"{name}: {r['per_statement']}s/stmt {r['llm_calls']} calls events {r['events']} "
              f"slots {r['slots']} ranking {r['rankings']}")
        for k, hs in r["hits"].items():
            print(f"  hits {k}: {hs}")
    print(f"VERDICT  writes {'OK' if v['writes_ok'] else 'FAIL'}  retrieval "
          f"{'OK' if v['retrieval_ok'] else 'FAIL'}  edit {'fires' if v['edit_fires'] else 'does NOT fire'}"
          f"  current-first {v['current_first']}")
    print(f"Wrote {os.path.abspath(_OUT)}")
    if not v["gate_pass"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
