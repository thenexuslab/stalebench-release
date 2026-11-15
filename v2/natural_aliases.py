"""Natural alias inventory for the anchor-free track.

The submitted benchmark withdraws the lexical anchor using *mechanically
derived* short forms (head token, leading token, initialed form, drop-last).
The obvious reviewer objection is that the axis is synthetic. This job builds
the ecological counterpart from two public sources that require no annotation:

* Wikidata ``aliases`` (the "also known as" field) for each subject, and
* English Wikipedia redirect titles pointing at the subject's article.

Both are real surface forms real writers used. The output is an inventory in
the same shape as the synthetic one, plus a comparison that answers the three
questions a reviewer will ask: how many subjects have natural aliases at all,
how much the natural forms overlap the synthetic ones, and how much *more*
ambiguous the natural inventory is (collisions across subjects), since ambiguity
is what the keying ladder is actually priced against.

Redirects are filtered: case/punctuation variants of the full name carry no
anchor-withdrawal signal, and neither do forms that contain the full name.

The default covers the test split. ``--split val`` builds the same inventory
for the val subjects (written under a ``_val`` suffix), which the keying ladder
needs because every threshold is tuned on val: without it the natural
configuration would be tuned on a stream that is still anchored.

Usage:  python3 natural_aliases.py [--limit N] [--split test|val]
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wikidata as wd  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RELEASE = os.path.join(HERE, "..", "release")
OUT = os.path.join(HERE, "out")

# mirrors stalebench/alias.py: forms no writer uses as a standalone reference
TYPE_WORDS = {
    "fc", "sc", "ac", "cf", "line", "ministry", "party", "club", "university",
    "college", "school", "team", "council", "of", "the", "and", "national",
    "city", "county", "district", "department", "association", "federation",
    "company", "corporation", "group", "institute", "the university",
}


def norm(s: str) -> str:
    return " ".join((s or "").lower().replace(".", " ").split())


def synthetic_forms(name: str) -> list:
    """Reimplements the released alias axis so the two can be compared."""
    toks = name.split()
    out = []
    if len(toks) >= 2:
        out += [toks[-1], toks[0], f"{toks[0][0]}. {toks[-1]}"]
    if len(toks) >= 3:
        out.append(" ".join(toks[:-1]))
    return [f for f in dict.fromkeys(out)
            if norm(f) not in TYPE_WORDS and len(f) > 1]


def useful_redirect(title: str, full_name: str) -> bool:
    """Keep only redirects that are genuinely different, shorter references."""
    t, f = norm(title), norm(full_name)
    if not t or t == f:
        return False
    if f in t:                      # "Valentino Rossi (rider)" -> no new signal
        return False
    if re.search(r"\d{4}", title):  # disambiguation by year
        return False
    if len(t) > len(f):             # longer than the anchor: not a short form
        return False
    if norm(t) in TYPE_WORDS:
        return False
    # drop pure case/spacing/diacritic variants: same letters when squashed
    if re.sub(r"[^a-z]", "", t) == re.sub(r"[^a-z]", "", f):
        return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="0 = all test subjects")
    ap.add_argument("--no-redirects", action="store_true")
    ap.add_argument("--split", choices=("test", "val"), default="test")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    suffix = "" if args.split == "test" else f"_{args.split}"

    keys = [json.loads(l) for l in
            open(os.path.join(RELEASE, f"key_anchored_{args.split}.jsonl"))]
    subjects = {}
    for k in keys:
        qid = k["fact_key"].split("_")[0]
        subjects.setdefault(qid, k["subject"])
    items = list(subjects.items())
    if args.limit:
        items = items[:args.limit]
    print(f"{len(items)} {args.split} subjects", flush=True)

    print("warming entity cache ...", flush=True)
    wd.prefetch([q for q, _ in items], pause=0.2)

    # one request per subject is unusably slow once Wikimedia throttles;
    # the API takes 50 titles at a time, so fetch redirects up front in batches
    redirect_map = {}
    if not args.no_redirects:
        titles = {}
        for qid, _ in items:
            t = wd.enwiki_title(qid)
            if t:
                titles[qid] = t
        print(f"fetching redirects for {len(titles)} articles in batches ...",
              flush=True)
        by_title = wd.wikipedia_redirects_batch(list(titles.values()))
        redirect_map = {qid: by_title.get(t, []) for qid, t in titles.items()}
        print(f"  got redirects for "
              f"{sum(1 for v in redirect_map.values() if v)} articles", flush=True)

    inventory, stats_rows = {}, []
    for i, (qid, name) in enumerate(items, 1):
        wd_aliases = [a for a in wd.aliases(qid)
                      if useful_redirect(a, name) or norm(a) != norm(name)]
        wd_aliases = [a for a in wd_aliases if useful_redirect(a, name)]

        redirects = [r for r in redirect_map.get(qid, [])
                     if useful_redirect(r, name)]

        natural = list(dict.fromkeys(wd_aliases + redirects))
        synth = synthetic_forms(name)
        inventory[qid] = {
            "subject": name,
            "wikidata_aliases": wd_aliases,
            "wikipedia_redirects": redirects,
            "natural": natural,
            "synthetic": synth,
        }
        ns, nn = {norm(x) for x in synth}, {norm(x) for x in natural}
        stats_rows.append({
            "qid": qid, "n_natural": len(natural), "n_synthetic": len(synth),
            "overlap": len(ns & nn), "natural_only": len(nn - ns),
            "synthetic_only": len(ns - nn),
        })
        if i % 50 == 0:
            print(f"  {i}/{len(items)}", flush=True)

    # ambiguity: one surface form pointing at several distinct subjects
    def ambiguity(field):
        m = collections.defaultdict(set)
        for qid, rec in inventory.items():
            for f in rec[field]:
                m[norm(f)].add(qid)
            m[norm(rec["subject"])].add(qid)
        colliding = {k: sorted(v) for k, v in m.items() if len(v) > 1}
        subj_hit = {q for v in colliding.values() for q in v}
        return {
            "n_forms": len(m),
            "n_colliding_forms": len(colliding),
            "pct_colliding_forms": round(100 * len(colliding) / max(1, len(m)), 2),
            "n_subjects_touched": len(subj_hit),
            "pct_subjects_touched": round(100 * len(subj_hit) / max(1, len(inventory)), 2),
            "examples": dict(list(colliding.items())[:8]),
        }

    with_nat = sum(1 for r in inventory.values() if r["natural"])
    with_syn = sum(1 for r in inventory.values() if r["synthetic"])
    tot_ov = sum(r["overlap"] for r in stats_rows)
    tot_nat = sum(r["n_natural"] for r in stats_rows)

    summary = {
        "n_subjects": len(inventory),
        "subjects_with_natural_alias": with_nat,
        "pct_subjects_with_natural_alias": round(100 * with_nat / len(inventory), 1),
        "subjects_with_synthetic_alias": with_syn,
        "pct_subjects_with_synthetic_alias": round(100 * with_syn / len(inventory), 1),
        "total_natural_forms": tot_nat,
        "total_synthetic_forms": sum(r["n_synthetic"] for r in stats_rows),
        "natural_forms_also_synthetic": tot_ov,
        "pct_natural_covered_by_synthetic": round(100 * tot_ov / max(1, tot_nat), 1),
        "ambiguity_synthetic": ambiguity("synthetic"),
        "ambiguity_natural": ambiguity("natural"),
        "source": "wikidata aliases + enwiki redirects",
        "split": args.split,
    }

    with open(os.path.join(OUT, f"alias_inventory_natural{suffix}.json"), "w") as f:
        json.dump(inventory, f, indent=1, ensure_ascii=False)
    with open(os.path.join(OUT, f"alias_natural_summary{suffix}.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n=== natural alias inventory ===")
    print(f"subjects with >=1 natural alias : {with_nat}/{len(inventory)} "
          f"({summary['pct_subjects_with_natural_alias']}%)")
    print(f"subjects with >=1 synthetic     : {with_syn}/{len(inventory)} "
          f"({summary['pct_subjects_with_synthetic_alias']}%)")
    print(f"natural forms total             : {tot_nat}")
    print(f"  of which synthetic also has   : {tot_ov} "
          f"({summary['pct_natural_covered_by_synthetic']}%)")
    print(f"ambiguous forms  synthetic      : "
          f"{summary['ambiguity_synthetic']['pct_colliding_forms']}% of forms, "
          f"{summary['ambiguity_synthetic']['pct_subjects_touched']}% of subjects")
    print(f"ambiguous forms  natural        : "
          f"{summary['ambiguity_natural']['pct_colliding_forms']}% of forms, "
          f"{summary['ambiguity_natural']['pct_subjects_touched']}% of subjects")
    print(f"\nwrote {OUT}/alias_inventory_natural{suffix}.json + "
          f"alias_natural_summary{suffix}.json")


if __name__ == "__main__":
    main()
