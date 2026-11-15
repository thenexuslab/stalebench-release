"""Materialize StaleBench as a distributable dataset + pinned evaluator.

A benchmark paper's contribution is an instrument other people can run. This
module writes the whole thing to a directory that can be uploaded as-is:

    statements_<config>_<split>.jsonl   the chronological stream (opaque ids)
    queries_<config>_<split>.jsonl      value-agnostic queries (no labels)
    key_<config>_<split>.jsonl          the answer key, read only by the scorer
    baseline.py                         reference baselines over public files
    alias_inventory.json                the derivable alias table (v1.1 track)
    llm_cache.jsonl.gz                  every reader/keyer judgment, so the LLM
                                        numbers regenerate with no API key
    evaluate.py                         pinned scorer; the leaderboard contract
    README.md                           dataset card (HF front-matter)
    DATASHEET.md                        datasheet for datasets
    manifest.json                       config hashes + row counts

Three configs, matching the paper's conditions:

``anchored``     paraphrase 0.5, no aliasing --- the v0.2 headline regime, where
                 subjects appear verbatim and a gazetteer keyer nearly matches
                 the oracle.
``anchor_free``  paraphrase 0.5, aliasing on --- non-introducing statements name
                 the subject by a short form; the regime where the open problem
                 lives.
``full_corpus``  anchored, plus all non-queried TempLAMA fact keys as
                 distractors --- a realistic retrieval pool.

Currency labels are copied from the source (Wikidata validity years via
TempLAMA); nothing in the release is derived from a retrieval score, so a
submission cannot be scored against its own behaviour.

Run:  python -m stalebench.release [out_dir]
"""

from __future__ import annotations

import glob
import gzip
import hashlib
import json
import os
import shutil
import sys
from typing import Dict, List, Optional

from stalebench.alias import alias_stats, build_alias_map
from stalebench.templama import (RELATIONS, build_instance,
                                 load_all_changing_facts,
                                 load_distractor_facts, load_facts,
                                 split_facts, subjects_with_class)

_HERE = os.path.dirname(__file__)
_OUT = os.path.join(_HERE, "..", "release")
_LLM_CACHE = os.path.join(_HERE, "..", "results", "llm_cache")
_ASSETS = os.path.join(_HERE, "release_assets")

CONFIGS = {
    "anchored":    {"paraphrase_p": 0.5, "alias_p": 0.0, "distractors": False},
    "anchor_free": {"paraphrase_p": 0.5, "alias_p": 1.0, "distractors": False},
    "full_corpus": {"paraphrase_p": 0.5, "alias_p": 0.0, "distractors": True},
    "full_corpus_anchor_free": {"paraphrase_p": 0.5, "alias_p": 1.0,
                                "distractors": True},
    # v1.1: every changing key, not only those single-valued in every year, so
    # a fact may have several simultaneously-valid current values. The key
    # format already carries current ids as a set, so the evaluator is
    # unchanged; what changes is which facts are queried.
    "scale": {"paraphrase_p": 0.5, "alias_p": 0.0, "distractors": False,
              "all_changing": True},
    "scale_anchor_free": {"paraphrase_p": 0.5, "alias_p": 1.0,
                          "distractors": False, "all_changing": True},
    # v1.2: the anchor-free axis with the short forms people actually use
    # (Wikidata aliases + Wikipedia redirects) instead of mechanically derived
    # ones. A subject with no real short form stays anchored under ``natural``
    # and takes the mechanical forms under ``natural+fallback``; the inventory
    # is split-specific so no test alias informs val tuning.
    "anchor_free_natural": {"paraphrase_p": 0.5, "alias_p": 1.0,
                            "distractors": False, "alias_source": "natural"},
    "anchor_free_natural_fallback": {"paraphrase_p": 0.5, "alias_p": 1.0,
                                     "distractors": False,
                                     "alias_source": "natural+fallback"},
}

_NATURAL_INVENTORY = {
    "val": os.path.join(_HERE, "..", "v2", "out", "alias_inventory_natural_val.json"),
    "test": os.path.join(_HERE, "..", "v2", "out", "alias_inventory_natural.json"),
}

_EVALUATOR = '''"""Pinned StaleBench evaluator --- the leaderboard contract.

A submission is JSONL, one object per query:

    {"query_id": "test/anchored/q0", "ranking": ["s3f9a1c0e2b7", ...]}

``ranking`` is the ordered list of statement ids the system would place in a
reader's context, best first. Systems consume the statement stream and may
store, merge, overwrite or discard whatever they like; only the ranking is
scored.

    python evaluate.py --key key_anchored_test.jsonl \\
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
'''


# Salt for opaque statement ids. Fixed so the release is reproducible, but the
# ids carry no fact key, no version ordinal, and no grouping structure: a
# system cannot recover "these two statements are the same fact" from them, and
# cannot recover "this one is current" at all. v0.2 shipped ids of the form
# "f171:0", which made the test split solvable by string manipulation alone.
_ID_SALT = "stalebench-v1.0"


def _opaque_id(config: str, split: str, fact_key: str, version: int) -> str:
    h = hashlib.sha1(
        f"{_ID_SALT}|{config}|{split}|{fact_key}|{version}".encode()).hexdigest()
    return "s" + h[:12]


def _rows(facts, cfg, split, name, distractors=None):
    src = cfg.get("alias_source", "mechanical")
    inst = build_instance(facts, paraphrase_p=cfg["paraphrase_p"], seed=0,
                          alias_p=cfg["alias_p"], distractors=distractors,
                          alias_source=src,
                          alias_inventory=(_NATURAL_INVENTORY[split]
                                           if src != "mechanical" else None))
    fact_by_id = {f"f{i}": f for i, f in enumerate(facts)}
    dist_by_id = {f"d{i}": f for i, f in enumerate(distractors or [])}
    statements: List[Dict] = []
    ids_of_fact: Dict[str, List[str]] = {}
    current_of_fact: Dict[str, List[str]] = {}
    for text, iid, t in inst.events:
        key, _, ver = iid.rpartition(":")
        f = fact_by_id.get(key) or dist_by_id.get(key)
        sid = _opaque_id(name, split, f.key if f else key, int(ver))
        statements.append({
            "id": sid,
            "split": split,
            "config": name,
            "text": text,
            "year": inst.year_of[iid],
            "t": t,
        })
        if key in fact_by_id:
            ids_of_fact.setdefault(key, []).append(sid)
            if int(ver) in fact_by_id[key].current_runs:
                current_of_fact.setdefault(key, []).append(sid)
    queries: List[Dict] = []
    keys: List[Dict] = []
    for qi, ((qtext, key, cur), f) in enumerate(zip(inst.queries, facts)):
        qid = f"{split}/{name}/q{qi}"
        # public: everything a system may condition on
        queries.append({
            "query_id": qid,
            "question": qtext,
            "relation": f.relation,
            "config": name,
            "split": split,
        })
        # private: the answer key, scored by evaluate.py and nothing else
        keys.append({
            "query_id": qid,
            "fact_ids": ids_of_fact.get(key, []),
            "current_ids": current_of_fact.get(key, []),
            "gold_value": f.current_value,
            "fact_key": f.key,
            "subject": f.subject,
            "n_versions": len(f.runs),
            "label_source": "wikidata_validity_year",
            "final_year": max(f.by_year),
        })
    return statements, queries, keys, inst


def _write_jsonl(path: str, rows: List[Dict]) -> str:
    with open(path, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    return hashlib.md5(open(path, "rb").read()).hexdigest()[:12]


def export(out_dir: str = _OUT) -> Dict:
    os.makedirs(out_dir, exist_ok=True)
    facts, stats = load_facts()
    val, test = split_facts(facts)
    dist = load_distractor_facts()
    dist_v, dist_t = split_facts(dist)

    manifest: Dict = {"source": "TempLAMA (Dhingra et al., 2022) over Wikidata",
                      "filters": stats, "configs": {},
                      "splits": {"val": len(val), "test": len(test)},
                      "relations": sorted(RELATIONS)}
    scale_facts, scale_stats = load_all_changing_facts()
    scale_val, scale_test = split_facts(scale_facts)
    manifest["scale_filters"] = scale_stats
    manifest["scale_splits"] = {"val": len(scale_val), "test": len(scale_test)}
    for name, cfg in CONFIGS.items():
        entry = {"params": cfg, "files": {}}
        _big = cfg.get("all_changing")
        _pairs = (("val", scale_val, dist_v), ("test", scale_test, dist_t)) \
            if _big else (("val", val, dist_v), ("test", test, dist_t))
        for split, fs, ds in _pairs:
            st, qs, ks, inst = _rows(fs, cfg, split, name,
                                     ds if cfg["distractors"] else None)
            sp = os.path.join(out_dir, f"statements_{name}_{split}.jsonl")
            qp = os.path.join(out_dir, f"queries_{name}_{split}.jsonl")
            kp = os.path.join(out_dir, f"key_{name}_{split}.jsonl")
            entry["files"][f"key_{split}"] = {
                "path": os.path.basename(kp), "rows": len(ks),
                "md5": _write_jsonl(kp, ks)}
            entry["files"][f"statements_{split}"] = {
                "path": os.path.basename(sp), "rows": len(st),
                "md5": _write_jsonl(sp, st),
                "n_alias_statements": inst.n_alias_events,
                "n_distractor_statements": inst.n_distractor_events}
            entry["files"][f"queries_{split}"] = {
                "path": os.path.basename(qp), "rows": len(qs),
                "md5": _write_jsonl(qp, qs)}
        manifest["configs"][name] = entry

    cls = subjects_with_class(facts)
    amap = build_alias_map(cls)
    json.dump({"alias_to_subjects": {a: sorted(o) for a, o in sorted(amap.items())},
               "stats": alias_stats(cls)},
              open(os.path.join(out_dir, "alias_inventory.json"), "w"),
              ensure_ascii=False, indent=1)
    # the real short forms the natural configurations drew from, per split, so
    # a submission's linker can be handed exactly what the paper's was
    natural = {split: json.load(open(p)) for split, p in _NATURAL_INVENTORY.items()}
    json.dump(natural, open(os.path.join(out_dir, "alias_inventory_natural.json"), "w"),
              ensure_ascii=False, indent=1)

    manifest["llm_cache"] = export_llm_cache(out_dir)
    with open(os.path.join(out_dir, "evaluate.py"), "w") as fh:
        fh.write(_EVALUATOR)
    for asset in ("baseline.py", "trivial_baselines.py"):
        shutil.copyfile(os.path.join(_ASSETS, asset),
                        os.path.join(out_dir, asset))
    manifest["trivial_baselines"] = _score_trivial(out_dir)
    with open(os.path.join(out_dir, "README.md"), "w") as fh:
        fh.write(_readme(manifest))
    with open(os.path.join(out_dir, "DATASHEET.md"), "w") as fh:
        fh.write(_datasheet(manifest))
    json.dump(manifest, open(os.path.join(out_dir, "manifest.json"), "w"), indent=1)
    return manifest


def export_llm_cache(out_dir: str, cache_dir: Optional[str] = None) -> Dict:
    """Ship every LLM judgment as one compressed JSONL.

    The reader and keyer numbers are the only ones in the paper that depend on
    a model outside the repository. Shipping the greedy-decoded responses --
    keyed by the same hash the client uses -- makes them regenerable with no
    API access and no key, which is the difference between "reproducible" and
    "reproducible if you pay". 5.5k judgments compress to well under a
    megabyte, so there is no reason not to.
    """
    cache_dir = cache_dir or _LLM_CACHE
    rows, path = [], os.path.join(out_dir, "llm_cache.jsonl.gz")
    for fp in sorted(glob.glob(os.path.join(cache_dir, "*.json"))):
        try:
            d = json.load(open(fp))
        except (json.JSONDecodeError, OSError):
            continue
        rows.append({"key": os.path.basename(fp)[:-5], "model": d.get("model"),
                     "prompt": d.get("prompt"), "text": d.get("text")})
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    by_model: Dict[str, int] = {}
    for r in rows:
        by_model[r["model"]] = by_model.get(r["model"], 0) + 1
    return {"path": os.path.basename(path), "entries": len(rows),
            "by_model": by_model,
            "bytes": os.path.getsize(path)}


def import_llm_cache(archive: str, cache_dir: Optional[str] = None) -> int:
    """Expand a shipped ``llm_cache.jsonl.gz`` back into the client's cache."""
    cache_dir = cache_dir or _LLM_CACHE
    os.makedirs(cache_dir, exist_ok=True)
    n = 0
    with gzip.open(archive, "rt", encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            dest = os.path.join(cache_dir, f"{r['key']}.json")
            if not os.path.exists(dest):
                json.dump({"model": r["model"], "prompt": r["prompt"],
                           "text": r["text"]}, open(dest, "w"))
            n += 1
    return n


def _score_trivial(out_dir: str) -> Dict:
    """Score the model-free reference rules, so the release states its own floor."""
    import importlib.util

    def _load(name):
        spec = importlib.util.spec_from_file_location(
            name, os.path.join(out_dir, f"{name}.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    tb, ev = _load("trivial_baselines"), _load("evaluate")
    out: Dict = {}
    for cfg in CONFIGS:
        st = [json.loads(l) for l in
              open(os.path.join(out_dir, f"statements_{cfg}_test.jsonl"))]
        qs = [json.loads(l) for l in
              open(os.path.join(out_dir, f"queries_{cfg}_test.jsonl"))]
        key = {}
        for line in open(os.path.join(out_dir, f"key_{cfg}_test.jsonl")):
            k = json.loads(line)
            key[k["query_id"]] = (set(k["fact_ids"]), set(k["current_ids"]))
        st.sort(key=lambda s: s["t"])
        out[cfg] = {}
        for mode in ("verbatim", "trigram"):
            subs = {r["query_id"]: r["ranking"] for r in tb.run(st, qs, mode)}
            recs = [ev.score_one(subs[q], f, c, 5) for q, (f, c) in key.items()]
            out[cfg][mode] = {m: sum(r[m] for r in recs) / len(recs)
                              for m in recs[0]}
    return out


def _hf_config_block() -> str:
    """HF front-matter configs, generated from CONFIGS.

    Hand-maintaining this list silently omitted a shipped config, so
    ``load_dataset(..., name)`` failed for the configuration the card called
    the hardest. Generating it makes that impossible.
    """
    out = []
    for name in CONFIGS:
        out.append(f"- config_name: {name}\n"
                   f"  data_files:\n"
                   f"  - split: validation\n"
                   f"    path: queries_{name}_val.jsonl\n"
                   f"  - split: test\n"
                   f"    path: queries_{name}_test.jsonl\n")
    return "".join(out)


def _readme(m: Dict) -> str:
    n = m["filters"]["single_valued_changing"]
    _hf_configs = _hf_config_block()
    return f"""---
license: cc0-1.0
task_categories:
- question-answering
language:
- en
tags:
- agent-memory
- retrieval
- temporal
- knowledge-conflict
pretty_name: StaleBench
size_categories:
- 1K<n<10K
configs:
{_hf_configs}---

# StaleBench

An agent that stores facts over time eventually stores several versions of the
same fact. Asked a **value-agnostic** question --- *which team does X play for
now?* --- a similarity retriever cannot separate the current version from stale
ones, because every version matches the entity the query names.

StaleBench isolates that failure. It supplies a chronological stream of
statements and one current-value query per fact, with currency labels taken
from **Wikidata validity years** (via TempLAMA) --- never from any retrieval
score, so no system can be scored against its own behaviour.

## Contents

{n} single-valued changing facts over {len(m['relations'])} relations, split
val/test by fact-key hash ({m['splits']['val']}/{m['splits']['test']}). The
configurations:

| config | statements carry | why it exists |
|---|---|---|
| `anchored` | the subject verbatim | control: a gazetteer keyer nearly matches the oracle here |
| `anchor_free` | mechanical short forms after the first mention | where same-fact detection is still open |
| `anchor_free_natural` | real short forms (Wikidata aliases, Wikipedia redirects); subjects without one stay anchored | the ecological form of the same axis |
| `anchor_free_natural_fallback` | real short forms, mechanical ones where none exist | like-for-like with `anchor_free` |
| `full_corpus` | + all non-queried TempLAMA facts as distractors | realistic retrieval pool |
| `full_corpus_anchor_free` | both | hardest configuration; most headroom |
| `scale`, `scale_anchor_free` | every changing key, set-valued gold | the multi-valued track |

**Rank on `anchor_free` (or `full_corpus_anchor_free`), not `anchored`.** The
anchored configurations are deliberately easy and are near-saturated on
current@1: a rule that recovers the subject from the question, keeps every
statement containing it, and answers most-recent-first scores close to the
true-key oracle there. That is the control working as intended --- it is the
measurement that shows templated text makes same-fact detection look solved ---
but it means `anchored` is a diagnostic, not a leaderboard. The same rule
collapses on the anchor-free configurations, and it is maximally contaminated
(contamination@5 near 1) at full store footprint in every case, so report
contamination@5 and slots alongside current@1 rather than current@1 alone.

## Task

Consume `statements_<config>_<split>.jsonl` in `t` order. Store, merge,
overwrite or discard whatever you like. For each query in
`queries_<config>_<split>.jsonl`, return the ordered statement ids you would
place in a reader's context.

```
{{"query_id": "test/anchored/q0", "ranking": ["s3f9a1c0e2b7", "s7b21e4d0aa19"]}}
```

Score with the pinned evaluator, and reproduce the reference rows first:

```bash
python baseline.py --config anchored --split test
python evaluate.py --key key_anchored_test.jsonl \
                   --submission sub_anchored_test_cosine.jsonl
```

**What the public files do and do not contain.** Statement ids are salted
opaque hashes with no ordinal or grouping structure, and statement records
carry only `text`, `year` and `t`. Which statements describe one fact, and
which of them are current, appears *only* in `key_<config>_<split>.jsonl`,
which `evaluate.py` reads and a system must not. An earlier revision encoded
the fact key and version directly in the statement id, which made the test
split solvable by string manipulation; that is fixed, and it is the reason the
key is now a separate file.

**Tune on `val` only.** Test labels exist to be reported, not searched.

## Reproducing the paper's LLM numbers offline

`llm_cache.jsonl.gz` holds every greedy-decoded reader and keyer response used
in the paper, keyed by the same hash the harness uses. Expand it with
`stalebench.release.import_llm_cache` and the reader track replays with no API
key and no cost.

## Metrics

`current@1`, `stale_rate`, `miss_rate`, `current@k`, and **`contamination@k`**
--- the top-k holding both a current and a stale version, i.e. a context that
argues with itself. Timestamp reranking can fix `current@1` while leaving
`contamination@k` untouched, which is why both are reported.

## Provenance and license

Facts derive from Wikidata (CC0) via TempLAMA (Dhingra et al., 2022);
verbalization templates and alias inventory are contributed by this work and
released CC0 likewise. `alias_inventory.json` is derivable from the subject
strings, and is shipped so that entity-linking baselines are reproducible.
"""


def _datasheet(m: Dict) -> str:
    f = m["filters"]
    return f"""# Datasheet: StaleBench

Following Gebru et al., *Datasheets for Datasets*.

## Motivation

**Why was the dataset created?** To isolate *stale retrieval* in agent memory:
the failure in which a store holding several versions of a fact surfaces an
outdated one for a value-agnostic query. Existing freshness benchmarks ask
whether a *model* is current with the world; knowledge-conflict work studies a
*reader* given conflicting context. Neither measures which version reaches the
context, which is the decision an agent's memory actually makes.

**Who created it?** The authors of the accompanying paper. No external funding
specific to this dataset.

## Composition

**What do instances represent?** Two kinds. A *statement* is one natural
language assertion of a (subject, relation, value) triple that was valid over a
run of consecutive years, carrying its year as metadata rather than in the text.
A *query* is a value-agnostic current-value question naming only the subject.

**How many instances?** {f['fact_keys']} TempLAMA fact keys reduce to
{f['single_valued']} that are single-valued in every year and
{f['single_valued_changing']} whose value also changes at least once --- the
queried set. Each config materializes those facts into ~1k statements; the
`full_corpus` config adds the {f['fact_keys'] - f['single_valued']} multi-valued
keys as unqueried distractors.

**Is any information missing?** Facts that are multi-valued in some year (a
footballer playing for club and country) are excluded from the queried set
because they defeat crisp top-1 scoring. They appear only as distractors. This
is a real restriction on coverage, stated in the paper's limitations.

**Is the data linked to people?** Yes --- public figures (athletes, politicians,
executives) as recorded in Wikidata. No private individuals, no sensitive
attributes beyond public employment, office, party and education.

**Labels.** `gold_value`, and the `current_ids` set in the key file, come from
the Wikidata validity year recorded in TempLAMA, i.e. the value valid in the final year of the window.
They are not produced by any model or retrieval score.

## Collection

**How was the data acquired?** Derived from the public TempLAMA release
(Dhingra et al., 2022), itself built from Wikidata start/end qualifiers.
Verbalization uses hand-written per-relation templates (one canonical, three
paraphrases, three question forms); the canonical statement template is the
TempLAMA cloze with the value filled in.

**Sampling.** No sampling: every TempLAMA key passing the single-valued and
changing filters is included. Val/test is a deterministic MD5 parity split on
the fact key, so it is reproducible and cannot be re-drawn to taste.

## Preprocessing

Consecutive years with the same value collapse into one *run*, so a fact
contributes one statement per distinct value interval rather than one per year.
This prevents inflating `n` by repetition and keeps version counts meaningful.
The alias inventory is derived mechanically from subject strings (head token,
leading token, initialed form, drop-last phrase), filtered for organizational
type words so that no alias is a form no writer would use. Ambiguous aliases
(one form, several subjects) are deliberately **kept**: entity-resolution
ambiguity is part of what the benchmark prices.

## Uses

**Intended use.** Evaluating retention and retrieval policies for agent memory:
which version of a fact reaches the reader's context, at what memory footprint.

**Out of scope.** It is not a test of world knowledge, not a QA benchmark for
readers in general, and not a measure of factual accuracy at large. Statements
are templated, English-only, and cover nine relations; results do not transfer
automatically to naturalistic dialogue.

**Risks.** Facts about named public figures are as of 2020 and are now
outdated; the dataset must not be used as a current-facts source. The "current"
label means *current within the 2010--2020 window*.

## Distribution and maintenance

Released under CC0-1.0, matching Wikidata's dedication. Versioned with the
paper; `manifest.json` carries per-file row counts and MD5s so a run can prove
which revision it scored. Statement records deliberately carry no fact key,
version, or currency field: an earlier revision encoded them in the statement
id, which made the test split solvable by string manipulation. The evaluator (`evaluate.py`) is frozen: metric
definitions must not change across submissions, and any future config is added
as a new config rather than by editing an existing one.
"""


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else _OUT
    m = export(out)
    print(f"wrote release to {os.path.abspath(out)}")
    for name, cfg in m["configs"].items():
        rows = {k: v["rows"] for k, v in cfg["files"].items()}
        print(f"  {name:12s} {rows}")
    lc = m["llm_cache"]
    print(f"  llm cache    {lc['entries']} judgments, {lc['bytes'] / 1e6:.2f} MB "
          f"({lc['by_model']})")


if __name__ == "__main__":
    main()
