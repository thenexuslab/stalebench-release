---
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
- config_name: anchored
  data_files:
  - split: validation
    path: queries_anchored_val.jsonl
  - split: test
    path: queries_anchored_test.jsonl
- config_name: anchor_free
  data_files:
  - split: validation
    path: queries_anchor_free_val.jsonl
  - split: test
    path: queries_anchor_free_test.jsonl
- config_name: full_corpus
  data_files:
  - split: validation
    path: queries_full_corpus_val.jsonl
  - split: test
    path: queries_full_corpus_test.jsonl
- config_name: full_corpus_anchor_free
  data_files:
  - split: validation
    path: queries_full_corpus_anchor_free_val.jsonl
  - split: test
    path: queries_full_corpus_anchor_free_test.jsonl
- config_name: scale
  data_files:
  - split: validation
    path: queries_scale_val.jsonl
  - split: test
    path: queries_scale_test.jsonl
- config_name: scale_anchor_free
  data_files:
  - split: validation
    path: queries_scale_anchor_free_val.jsonl
  - split: test
    path: queries_scale_anchor_free_test.jsonl
- config_name: anchor_free_natural
  data_files:
  - split: validation
    path: queries_anchor_free_natural_val.jsonl
  - split: test
    path: queries_anchor_free_natural_test.jsonl
- config_name: anchor_free_natural_fallback
  data_files:
  - split: validation
    path: queries_anchor_free_natural_fallback_val.jsonl
  - split: test
    path: queries_anchor_free_natural_fallback_test.jsonl
---

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

892 single-valued changing facts over 9 relations, split
val/test by fact-key hash (424/468). The
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
{"query_id": "test/anchored/q0", "ranking": ["s3f9a1c0e2b7", "s7b21e4d0aa19"]}
```

Score with the pinned evaluator, and reproduce the reference rows first:

```bash
python baseline.py --config anchored --split test
python evaluate.py --key key_anchored_test.jsonl                    --submission sub_anchored_test_cosine.jsonl
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
