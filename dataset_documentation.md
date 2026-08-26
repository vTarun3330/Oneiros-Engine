# Oneiros Phase 3 corpus

## Training source of truth

The current SFT-first pipeline uses the immutable corpus version
`v3_final_candidate`. The version is centralized in
`config.CANONICAL_CORPUS_VERSION` so local preflight, Modal training,
evaluation, and audit scripts select the same files.

| Partition | Records | Independent groups |
|---|---:|---:|
| Train | 6,755 | 764 |
| Validation | 799 | 84 |
| Test | 833 | 95 |
| **Total** | **8,387** | **943** |

The corpus stores 30,956 tests. Its task composition, taken from the tracked
manifest, is:

| Task family | Records |
|---|---:|
| Hidden mutation reproduction | 7,786 |
| Official repository SWE-bench reproduction | 324 |
| Official repository pytest reproduction | 154 |
| Official repository unittest reproduction | 115 |
| Curated fixed/bug reproduction seeds | 8 |

The exact hashes and quality-gate flags are in
`data/corpus/v3_final_candidate/manifest.json`. Do not train from raw mutation
files, legacy split files, or a different corpus directory while describing
the run as the 8,387-record V3 experiment.

## Quality and leakage controls

`harness/corpus.py` verifies the corpus before training and fails closed if a
record, split, or manifest hash changes. The canonical manifest asserts:

- every training record has verified supervision and a reference oracle;
- train, validation, and test are group-disjoint;
- semantic duplicates and conflicting supervision are rejected;
- repository projects are disjoint across splits;
- official real-bug records require fixed-pass/buggy-fail evidence;
- overlong retained records are explicitly excluded from training; and
- external evaluation remains locked.

Reference code is oracle-only. It is never included in SFT, DPO, or generation
prompts.

## Execution modes

Function-level records use `function_assertion`. A generated candidate is
accepted only when it is one bounded assertion that calls the required entry
point, passes on the reference, and fails on the target mutant. It is executed
in a restricted child process with a parent timeout.

Repository records use pytest, unittest, or SWE-bench fragments. Their
official evidence comes from the real project environment. These records may
be used as supervised context, but generated repository-fragment system
evaluation is not yet implemented and is excluded from the function-level
kill rate.

## Locked validation and test policy

Validation may be used for checkpoint selection, SFT/DPO comparison, and
hyperparameter decisions. The test partition must remain sealed until every
model-selection decision is frozen. `dpo_eval` refuses to open the test split
unless `--confirm-final-test` is explicitly supplied.

The manifest also keeps 225 BugsInPy repository tasks locked for external
evaluation. Materialized training records and locked evaluation tasks must not
be silently mixed.

## Local storage and Git policy

The raw `data/` tree is about 2 GB and includes derived corpora and external
repository checkouts, so Git tracks only:

- `data/README.md`;
- the canonical manifest; and
- the canonical training-exclusion ledger.

Obtain the authorized corpus separately and copy it into
`data/corpus/v3_final_candidate/`. Verify it before any experiment:

```powershell
py -X utf8 -c "from pathlib import Path; from harness.corpus import verify_corpus; print(verify_corpus(Path('data/corpus/v3_final_candidate'))['training_records'])"
```

The command must print `8387` and must not report a hash or split failure.

## Rebuilding safely

The scripts in `scripts/` preserve the staged history from raw HumanEval/MBPP
mutations through BugsInPy and SWE-bench verified repository records. Rebuild
only when the source data or schema deliberately changes. Create a new version
directory and new manifest; never edit `v3_final_candidate` in place. Every new
corpus version requires a fresh SFT adapter because the run fingerprint
includes the record and split hashes.

Before a performance GPU run, execute the exact GPU-free readiness preflight.
The 800-record scope is required to exercise the checkpoint-monitoring
schedule; a 32-record selection is an integration smoke only.

```powershell
py scripts/preflight_sft_run.py --corpus-version v3_final_candidate --max-pairs 800 --epochs 1 --batch-size 1 --learning-rate 0.00005 --lr-scheduler-type constant_with_warmup --repository-completion-token-limit 1024 --min-function-kill-rate 0.58
```

## Training

Use the canonical version explicitly in named experiments:

```powershell
py scripts/modal_train.py --fresh --phase sft --corpus-version v3_final_candidate --run-name v3_hardened_integration_32_seed42 --seed 42 --max-pairs 32
```

DPO cannot start until SFT and its immutable reference adapter are complete,
and their recorded checksums match. A bounded smoke adapter cannot be reused
as a full production adapter because the training scope is fingerprinted.
