"""Pinned StaleBench evaluator --- the leaderboard contract.

A submission is JSONL, one object per query:

    {"query_id": "test/anchored/q0", "ranking": ["s3f9a1c0e2b7", ...]}

``ranking`` is the ordered list of statement ids the system would place in a
reader's context, best first. Systems consume the statement stream and may
store, merge, overwrite or discard whatever they like; only the ranking is
scored.

    python evaluate.py --key key_anchored_test.jsonl \
                       --submission my_system.jsonl [--k 5]

Statement ids are opaque salted hashes with no ordinal or grouping structure,
and the statement files carry no fact key, version, or currency field: which
statements belong to one fact, and which of them are current, is known only to
this key file. Reading the key file at inference time is cheating.

Reported metrics (see the paper for definitions):

  current@1        top-1 belongs to the queried fact AND is a current version
  stale_rate       top-1 belongs to the queried fact but is superseded
  miss_rate        top-1 belongs to another fact (or nothing was returned)
  current@k        some current version appears in the top-k
  contamination@k  the top-k holds BOTH a current and a stale version

This file is frozen: metric definitions must not change across submissions.
"""

import argparse
import json


def score_one(ranking, fact_ids, current_ids, k):
    rec = {}
    top = ranking[0] if ranking else None
    if top is None or top not in fact_ids:
        rec.update({"current@1": 0.0, "stale_rate": 0.0, "miss_rate": 1.0})
    elif top in current_ids:
        rec.update({"current@1": 1.0, "stale_rate": 0.0, "miss_rate": 0.0})
    else:
        rec.update({"current@1": 0.0, "stale_rate": 1.0, "miss_rate": 0.0})
    has_cur = has_stale = False
    for item_id in ranking[:k]:
        if item_id not in fact_ids:
            continue
        if item_id in current_ids:
            has_cur = True
        else:
            has_stale = True
    rec["current@%d" % k] = float(has_cur)
    rec["contamination@%d" % k] = float(has_cur and has_stale)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", required=True)
    ap.add_argument("--submission", required=True)
    ap.add_argument("--k", type=int, default=5)
    a = ap.parse_args()

    gold = {}
    for line in open(a.key):
        q = json.loads(line)
        gold[q["query_id"]] = (set(q["fact_ids"]), set(q["current_ids"]))
    sub = {}
    for line in open(a.submission):
        s = json.loads(line)
        sub[s["query_id"]] = s.get("ranking", [])

    missing = set(gold) - set(sub)
    if missing:
        raise SystemExit("submission is missing %d queries (e.g. %s)"
                         % (len(missing), sorted(missing)[0]))

    recs = [score_one(sub[qid], f, c, a.k) for qid, (f, c) in gold.items()]
    out = {m: sum(r[m] for r in recs) / len(recs) for m in recs[0]}
    out["n_queries"] = len(recs)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
