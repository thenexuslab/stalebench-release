"""Can a benchmark id be attached to a memory at all? Asked per release.

Every number the systems track reports rests on one assumption: that a
statement's id, handed to the system as metadata, comes back attached to the
memory that statement produced. If it does not, an id-level metric measures the
wrapper's bookkeeping rather than the system, and it does so silently --- a
mis-attributed memory looks exactly like a retrieval miss.

This probe tests the assumption directly and cheaply. It streams a handful of
statements about *distinct* subjects, each carrying its own id, then reads the
store back and asks, for every memory the system holds, whether the subject
named in its text is the subject of the statement whose id it carries. No
retrieval, no ranking, no scoring conventions: just whether the label survived
the write.

A release that batches or defers extraction can attach the calling statement's
metadata to memories derived from earlier buffered statements. That is not a
retrieval failure and must not be reported as one.

Writes ``results/metadata_probe/<version>_<store>.json``.

Run under each interpreter you want to compare:

    python -m stalebench.diag_metadata
    /path/to/venv-with-0.1.x/bin/python -m stalebench.diag_metadata
"""

from __future__ import annotations

import json
import os
import warnings
from typing import Dict, List

import numpy as np

_HERE = os.path.dirname(__file__)
_OUT_DIR = os.path.join(_HERE, "..", "results", "metadata_probe")

# distinct subjects, distinct values, one statement each: any memory naming
# subject S must carry S's id, whatever the system did to the wording
_STREAM: List[tuple] = [
    ("Valentino Rossi plays for Yamaha Motor Racing.", "s0", "Valentino Rossi"),
    ("Peter Sagan plays for Bora-Hansgrohe.", "s1", "Peter Sagan"),
    ("Dida plays for S.C. Internacional.", "s2", "Dida"),
    ("Jens Lehmann plays for Arsenal F.C.", "s3", "Jens Lehmann"),
    ("Sam Bradford plays for Philadelphia Eagles.", "s4", "Sam Bradford"),
    ("Edwin van der Sar plays for VV Noordwijk.", "s5", "Edwin van der Sar"),
]


def run() -> Dict:
    warnings.filterwarnings("ignore")
    from stalebench.systems import Mem0Policy

    pol = Mem0Policy()
    pol.reset()
    zero = np.zeros(384, dtype=np.float32)
    for text, sid, _subject in _STREAM:
        pol.observe(zero, sid, 0.0, text)

    subject_of = {sid: subj for _t, sid, subj in _STREAM}
    mem = pol._memory()
    kw = {pol._cap_kwarg(mem.get_all): 1_000_000}
    got = mem.get_all(**pol._scope(mem.get_all), **kw)
    rows = (got or {}).get("results", []) or []

    checked: List[Dict] = []
    for row in rows:
        text = row.get("memory") or row.get("text") or ""
        meta_sid = (row.get("metadata") or {}).get("sid")
        latest_sid = pol._sid_of.get(row.get("id"))
        # which statement's subject does this memory actually talk about?
        named = [sid for _t, sid, subj in _STREAM if subj.lower() in text.lower()]
        checked.append({
            "text": text, "meta_sid": meta_sid, "latest_sid": latest_sid,
            "subject_sid": named[0] if len(named) == 1 else None,
            "meta_correct": bool(named) and meta_sid == named[0],
            "latest_correct": bool(named) and latest_sid == named[0],
        })

    attributable = [row for row in checked if row["subject_sid"]]
    return {
        "lib_version": pol.lib_version,
        "vector_store": pol.vector_store,
        "adjudicates_on_add": pol.adjudicates_on_add,
        "n_statements": len(_STREAM),
        "n_memories": len(rows),
        "n_identifiable": len(attributable),
        "n_meta_correct": sum(r["meta_correct"] for r in attributable),
        "n_latest_correct": sum(r["latest_correct"] for r in attributable),
        "events": pol.events,
        "subject_of": subject_of,
        "memories": checked,
    }


def main() -> None:
    res = run()
    os.makedirs(_OUT_DIR, exist_ok=True)
    path = os.path.join(_OUT_DIR,
                        f"{res['lib_version']}_{res['vector_store']}.json")
    json.dump(res, open(path, "w"), indent=1)
    n_id = res["n_identifiable"]
    print(f"mem0 {res['lib_version']} / {res['vector_store']}: "
          f"{res['n_statements']} statements -> {res['n_memories']} memories, "
          f"{n_id} name exactly one subject")
    print(f"  metadata id correct        : {res['n_meta_correct']}/{n_id}")
    print(f"  latest-writer id correct   : {res['n_latest_correct']}/{n_id}")
    for row in res["memories"]:
        mark = "ok " if row["meta_correct"] else "BAD"
        print(f"  [{mark}] meta={str(row['meta_sid']):5s} "
              f"latest={str(row['latest_sid']):5s} "
              f"subject={str(row['subject_sid']):5s}  {row['text']!r}")
    print(f"Wrote {os.path.abspath(path)}")


if __name__ == "__main__":
    main()
