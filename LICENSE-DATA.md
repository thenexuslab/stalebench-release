# Data licences

The code in this repository (the `stalebench/` package, the `v2/` scripts,
`release/evaluate.py`, `release/baseline.py`, `release/trivial_baselines.py`) is licensed under the
Apache License 2.0; see `LICENSE`.

The data in `release/` is licensed as follows.

- **Statement, query and key files, verbalization templates and
  `alias_inventory.json`**: CC0 1.0 Universal
  (https://creativecommons.org/publicdomain/zero/1.0/). Facts derive from
  Wikidata (CC0) via TempLAMA (Dhingra et al., 2022), which its authors
  distribute under the Apache License 2.0.
- **`release/alias_inventory_natural.json` and `v2/out/alias_inventory_natural*.json`**:
  CC BY-SA 4.0 (https://creativecommons.org/licenses/by-sa/4.0/). They combine
  Wikidata aliases (CC0) with Wikipedia redirect page titles, which carry
  Wikipedia's licence.
- **`release/llm_cache.jsonl.gz` and `results/`**: model outputs and result
  records kept for offline reproduction, released CC0 1.0.
