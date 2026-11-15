"""Stamp library versions into a systems results file, after the fact.

The Mem0 result is uninterpretable without the ``mem0ai`` version that produced
it, because the ADD/UPDATE adjudication this benchmark prices is on the
``add()`` path in the 0.1.x line and not in the 2.0.x line. The wrapper records
it now, but a run that predates that change has the field empty, and re-running
costs an hour of a laptop's LLM time to recover one string.

So it is recovered by inspecting the interpreter that produced the run, and
marked as exactly that: ``version_source: "post-hoc environment capture"``. It
is provenance, not a measurement, and labelling it lets a reader discount it
accordingly rather than assuming it was recorded at run time.

Refuses to overwrite a version that is already present, so it cannot silently
relabel a run.

Run:  python -m stalebench.annotate_env results/stalebench_systems_results.json
"""

from __future__ import annotations

import json
import sys
from typing import Dict


def _installed() -> Dict[str, object]:
    import inspect

    import mem0
    from mem0.memory import main as _m
    try:
        adjudicates = "get_update_memory_messages" in inspect.getsource(_m)
    except Exception:
        adjudicates = None
    return {"lib_version": getattr(mem0, "__version__", "unknown"),
            "adjudicates_on_add": adjudicates}


def annotate(path: str) -> int:
    doc = json.load(open(path))
    env = _installed()
    n = 0
    for cfg in doc.get("by_config", {}).values():
        for name, cell in cfg.items():
            if not isinstance(cell, dict) or not name.startswith("mem0("):
                continue
            if cell.get("lib_version"):
                continue                    # recorded at run time; leave alone
            cell.update(env)
            cell["version_source"] = "post-hoc environment capture"
            n += 1
    if n:
        json.dump(doc, open(path, "w"), indent=1)
    return n


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    for path in sys.argv[1:]:
        n = annotate(path)
        env = _installed()
        print(f"{path}: stamped {n} cell(s) with mem0 "
              f"{env['lib_version']} (adjudicates_on_add="
              f"{env['adjudicates_on_add']})")


if __name__ == "__main__":
    main()
