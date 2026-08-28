# Data directory

Raw datasets, generated corpus records, repository checkouts, and sealed
evaluation partitions are intentionally excluded from Git. The small tracked
artifacts under `corpus/v4_1_research_hardened_candidate/` identify and audit
the active corpus without publishing its records.

Active V4.1 identity:

- corpus: `v4_1_research_hardened_candidate`
- parent: frozen V4 at commit `1d2cca8`
- total records: 8,237
- train / ablation-dev / validation / test: 6,052 / 594 / 781 / 810
- retained model-fitting records after the training exclusion ledger: 5,941
- prompt schema: `oneiros_unified_test_generation_v2`

Tracked evidence includes the manifest, ablation-dev manifest, prompt-lineage
audit, training-exclusion ledger, and JSON/CSV/human-readable exclusion
analysis. `records.json`, `splits.json`, repository caches, and checkpoints
remain local; their hashes are locked in the manifest.

Rebuild and verify with the exact commands in `../V4_1_NEXT_RUN.md`. Never edit a
released corpus in place, never train from the sealed `test` split, and never
describe the frozen V4 or archived V3 runbooks as the active V4.1 protocol.
The methodology and visible/hidden field policy are documented in
`../research/v4_1/METHODOLOGY.md`.
