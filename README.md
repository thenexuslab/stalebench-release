# StaleBench

**A benchmark that isolates the stale-retrieval failure of agent memory on real
updating facts, prices every fix, and measures what each one costs the reader.**

An agent that stores facts over time eventually stores several versions of the
same fact. Asked a *value-agnostic* question ("which team does X play for
**now**?"), a similarity retriever cannot prefer the current version, because
every version matches the entity the query names. StaleBench measures that
failure on 892 real updating facts, with currency labels defined by the data
source (Wikidata validity years via TempLAMA) and never by a retrieval score.

Paper: *StaleBench: Isolating Stale Retrieval in Agent Memory, and Pricing the
Fix*, Findings of EACL 2027.
Archive: https://doi.org/10.5281/zenodo.24237231

## Results (TempLAMA test split, 468 changing facts, paraphrase p=0.5)

Retrieval:

| policy | current@1 | stale@1 | contam@5 | slots |
|---|---|---|---|---|
| BM25 (store-all) | 0.400 | 0.592 | 0.991 | 995 |
| Cosine (store-all) | 0.464 | 0.530 | 0.974 | 995 |
| Cosine + time-rerank | 0.925 | 0.049 | 0.974 | 995 |
| Cosine + recency-weight | 0.784 | 0.203 | 0.936 | 995 |
| Overwrite: threshold τ | 0.859 | 0.073 | 0.137 | 513 |
| Overwrite: gazetteer keyer | 0.972 | 0.004 | 0.000 | 467 |
| Keep-latest oracle (upper bound) | 0.979 | 0.000 | 0.000 | 468 |

Reader accuracy on each policy's own top-5, no dates shown:

| policy | current@1 | contam@5 | qwen2.5:14b | glm-4.6 |
|---|---|---|---|---|
| Cosine (store-all) | 0.464 | 0.974 | 0.558 | 0.611 |
| + time-rerank | 0.925 | 0.974 | 0.329 | 0.677 |
| Overwrite: threshold | 0.859 | 0.137 | 0.874 | 0.872 |
| Overwrite: gazetteer | 0.972 | 0.000 | 0.979 | 0.974 |
| Read-path key (no overwrite) | 0.989 | 0.998 | 0.327 | 0.684 |
| Symmetric key (read + write) | 0.989 | 0.000 | 0.983 | 0.979 |
| Overwrite: oracle | 0.979 | 0.000 | 0.985 | 0.981 |

Findings:

1. **The failure is real outside a generator.** Store-all similarity retrieval
   returns a stale version for 53–59% of value-agnostic queries.
2. **Retrieval metrics can rank memory policies backwards.** The read-path key
   and the symmetric key return the identical statement at rank 1 for every
   test query, yet differ by 65.6 and 29.5 points of reader accuracy on the two
   readers. The difference is contamination@5, the one retrieval-side number
   that ranks them correctly. Showing the reader a trustworthy date closes the
   gap, so retention and surfaced dates are alternatives.
3. **Timestamp reranking fixes the top-1 symptom only.** It lifts current@1
   from 0.464 to 0.925 and leaves every stale version in the context.
4. **Same-fact detection has a price.** When later mentions use short forms
   instead of the full name, the gazetteer keyer falls 65.8 points short of the
   oracle and embedding-threshold keying 31.4. Entity resolution applied to both
   the write path and the query recovers most of the loss.
5. **Deployed systems each handle supersession differently**, and scoring any
   of them by a single convention misdescribes it (150-fact subsample, anchored):

| system | current@1 by id | by text | contam@5 | reader |
|---|---|---|---|---|
| Mem0 2.0.11 | 0.180 | 0.480 | 0.947 | 0.627 |
| Mem0 0.1.118 | 0.913 | 0.573 | 0.593 | 0.720 |
| Graphiti, as returned | 0.347 | 0.347 | 0.947 | 0.593 |
| Graphiti, valid edges first | 0.833 | 0.833 | 0.120 | 0.893 |
| LangMem, qwen2.5:14b | 0.773 | 0.773 | 0.173 | 0.833 |
| LangMem, glm-4.6 | 0.800 | 0.147 | 0.753 | 0.613 |
| ERASE, qwen2.5:14b | 0.853 | 0.860 | 0.100 | 0.867 |
| ERASE, glm-4.6 | 0.953 | 0.953 | 0.060 | 0.980 |
| Cosine (store-all) | 0.460 | 0.460 | 0.973 | 0.573 |
| Overwrite: oracle | 0.987 | 0.987 | 0.007 | 1.000 |

## Score your own system

`release/` is self-contained: statements, queries and answer keys for eight
configurations, a datasheet, a manifest, and a frozen scorer.

```bash
pip install -r requirements.txt
cd release
python baseline.py --config anchored --split test      # writes sub_anchored_test_cosine.jsonl
python evaluate.py --key key_anchored_test.jsonl --submission sub_anchored_test_cosine.jsonl
```

That reproduces the store-all row above exactly. A submission is one ranking
of statement ids per query; see `release/README.md` for the format.

## Reproduce the paper

```bash
mkdir -p data && for s in train val test; do
  curl -o data/templama_$s.json https://storage.googleapis.com/gresearch/templama/$s.json
done
python -m pytest stalebench/tests/ -q      # 63 pass
python -m stalebench.make_paper_numbers    # results/ -> paper/numbers.tex and every table
```

`results/` holds the result records behind every number in the paper, so the
last command needs no model and no API key. To re-run the experiments:

```bash
python -m stalebench.run                   # controlled regime map -> results/
python -m stalebench.run_real              # TempLAMA headline track (+ clock x arrival corruption grid)
python -m stalebench.run_alias             # anchor-free + full-corpus + mechanism
python -m stalebench.run_alias_natural     # anchor-free with Wikidata/Wikipedia short forms
python -m stalebench.run_scale             # every changing key; set recall over multi-valued gold
python -m stalebench.run_systems --models ollama:qwen2.5:14b            # Mem0 on the ladder's streams
python -m stalebench.probe_graphiti                                       # gate: does invalidation fire?
python -m stalebench.run_systems --models graphiti:ollama:qwen2.5:14b   # Graphiti (FalkorDB on :6379)
python -m stalebench.diag_graphiti results/<contexts.json>               # as returned / valid first / valid only
python -m stalebench.probe_langmem                                        # gate: does the update fire?
python -m stalebench.run_systems --models langmem:ollama:qwen2.5:14b     # LangMem (LangChain) on the same streams
python -m stalebench.probe_erase                                          # gate: do the edits fire?
python -m stalebench.run_systems --models erase:ollama:qwen2.5:14b       # ERASE (rewrite-on-arrival, reimplemented)
python -m stalebench.run_sysreader results/<contexts.json> ollama:qwen2.5:14b   # a system's contexts at the reader
python -m stalebench.run_reader ollama:qwen2.5:14b     # reader track (cached on disk)
python -m stalebench.run_reader zai:glm-4.6            # second reader
python -m stalebench.run_reader --merge a.json b.json  # combine per-reader runs
python -m stalebench.run_llmkey zai:glm-4.6            # LLM as a same-fact keyer
python -m stalebench.release               # materialize dataset + pinned evaluator
```

Every LLM call is greedy-decoded and cached; the cache ships in
`release/llm_cache.jsonl.gz`, so reader numbers regenerate offline.

## Layout

```
stalebench/      harness: generator, TempLAMA track, policies (keyer x mechanism),
                 metrics and statistics, reader, runners for Mem0, Graphiti,
                 LangMem and ERASE, probe gates, release export, tests
release/         the benchmark: eight configurations, alias inventories, datasheet,
                 manifest, frozen evaluate.py, reference baselines, LLM cache
results/         result records the paper's numbers are generated from
v2/              builder for the natural alias inventory (Wikidata aliases and
                 Wikipedia redirects) and the inventories it produced
```

## Guardrails

Labels are source-defined (generator schedule or Wikidata years), never derived
from retrieval scores. Every hyperparameter is tuned on a validation split
only, and re-tuned separately in every regime, so the store-all baselines get
their best shot everywhere. Reader-track hyperparameters are inherited from
retrieval tuning and never tuned against reader accuracy. Released statement
ids are opaque and label-free, and the answer key is a separate file, so
nothing in the public data reveals which statements share a fact or which are
current.

## Citation and licence

```bibtex
@inproceedings{murshed2027stalebench,
  title     = {{StaleBench}: Isolating Stale Retrieval in Agent Memory, and Pricing the Fix},
  author    = {Murshed, S M Mahbub},
  booktitle = {Findings of the Association for Computational Linguistics: EACL 2027},
  year      = {2027}
}
```

Code is licensed under Apache-2.0 (`LICENSE`). Data is CC0 1.0, except the
natural alias inventories, which include Wikipedia redirect titles and are
CC BY-SA 4.0; see `LICENSE-DATA.md`.
