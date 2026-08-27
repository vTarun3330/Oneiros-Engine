# Data directory

The raw datasets and generated corpora are intentionally not stored in Git.
The local directory is about 2 GB and contains derived data, repository
checkouts, and evaluation partitions that should not be copied casually.

The tracked files under `corpus/v4_unified_prompt_candidate/` identify the exact Phase
3 corpus without publishing its records:

- `manifest.json` stores record/split counts, quality gates, and SHA-256 hashes.
- `training_exclusions.json` records the 36 retained corpus items excluded from
  training because their completions exceed the canonical token limit.
- `reverification_exclusions.json` records the 150 V3 records rejected because
  no current-policy, reference-valid mutation-killing assertion remained.

Current canonical corpus:

- version: `v4_unified_prompt_candidate`
- total records: 8,237
- train/validation/test: 6,646 / 781 / 810
- stored tests: 30,105

See `dataset_documentation.md` for the provenance, leakage gates, rebuild
process, and sealed-test policy. Copy an authorized corpus into
`data/corpus/v4_unified_prompt_candidate/` before running preflight or training. The
hashes must match the tracked manifest.
