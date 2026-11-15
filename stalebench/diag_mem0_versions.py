"""The mem0 version difference in its smallest form: two statements, one fact.

The full systems run (:mod:`stalebench.run_systems`) measures each release over
hundreds of statements, which is the number the paper reports. But the
difference between the two release lines is visible in two sentences, and a
two-sentence demonstration is worth having for three reasons: it is the clearest
statement of the finding, it costs four LLM calls instead of an hour, and it
isolates the claim from anything the larger stream might confound.

Sagan moves clubs. A system that supersedes ends with one memory and answers
with the new club; one that does not ends with two and ranks by similarity.

Writes ``results/mem0_versions/<version>.json`` so both lines can be scored
without either overwriting the other, and so the paper's numbers stay generated
rather than transcribed from a log.

Run under each interpreter you want to compare:

    python -m stalebench.diag_mem0_versions
    /path/to/venv-with-0.1.x/bin/python -m stalebench.diag_mem0_versions
"""

from __future__ import annotations

import json
import os
import warnings
from typing import Dict

import numpy as np

_HERE = os.path.dirname(__file__)
_OUT_DIR = os.path.join(_HERE, "..", "results", "mem0_versions")

# one fact, two versions, chronological. The query is value-agnostic, so nothing
# in it favours either club.
_STREAM = [("Peter Sagan plays for Bora-Hansgrohe.", "f0:0", 0.0),
           ("Peter Sagan plays for TotalEnergies.", "f0:1", 1.0)]
_QUERY = "Which team does Peter Sagan currently play for?"
_CURRENT_ID = "f0:1"


def run() -> Dict:
    warnings.filterwarnings("ignore")
    from stalebench.systems import Mem0Policy

    pol = Mem0Policy()
    pol.reset()
    for text, sid, t in _STREAM:
        pol.observe(np.zeros(384, dtype=np.float32), sid, t, text)
    ranking = pol.rank(np.zeros(384, dtype=np.float32), _QUERY)
    return {
        "lib_version": pol.lib_version,
        "adjudicates_on_add": pol.adjudicates_on_add,
        "n_statements": len(_STREAM),
        "events": pol.events,
        "slots": pol.slots,
        "ranking": ranking,
        "rank1_is_current": bool(ranking and ranking[0] == _CURRENT_ID),
        "query": _QUERY,
        "stream": [t for t, _, _ in _STREAM],
    }


def main() -> None:
    res = run()
    os.makedirs(_OUT_DIR, exist_ok=True)
    path = os.path.join(_OUT_DIR, f"{res['lib_version']}.json")
    json.dump(res, open(path, "w"), indent=1)
    print(f"mem0 {res['lib_version']}: events={res['events']} "
          f"slots={res['slots']} rank1_current={res['rank1_is_current']} "
          f"(adjudicates_on_add={res['adjudicates_on_add']})")
    print(f"Wrote {os.path.abspath(path)}")


if __name__ == "__main__":
    main()
