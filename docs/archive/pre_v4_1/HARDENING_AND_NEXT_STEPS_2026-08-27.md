# Oneiros hardening report and next steps

Date: 27 August 2026

## Outcome

The code defects found in the audit have been corrected locally. The corpus and saved checkpoints were not deleted or overwritten, no paid cloud run was launched, and the sealed final test split was not used.

The historical 67/100 SFT smoke result must not yet be presented as the current hardened result. The evaluator now rejects unsafe or non-target assertions and runs candidates in isolated restricted processes, so the saved adapter must be re-evaluated on validation data under the new evaluator.

## Corrections completed

| Area | Correction |
|---|---|
| Candidate validity | A model candidate must be exactly one bounded `assert`, call the requested entry point, and avoid imports, introspection, file/process/network primitives, comprehensions, and oversized literals. |
| Execution isolation | Function-level code now runs in a fresh Python process with restricted built-ins/imports, a temporary working directory, deterministic random seeding, bounded output, parent-enforced timeout, and POSIX resource limits when available. |
| Differential accounting | A candidate is counted only when it passes on the reference. Reference failures and policy failures are invalid candidates, not losers or kills. |
| Legacy execution paths | The benchmark runner, grammar/static baselines, coverage fuzzer, BugsInPy loader, and general execution harness no longer execute candidates directly in the main process. |
| DPO reference | DPO now requires a frozen SFT reference, `sft_metadata.json`, and a matching adapter checksum. Base-model fallback is disabled. |
| DPO prompt alignment | DPO uses the same 512-token head/tail prompt compaction strategy as SFT and live generation. Completions remain fail-closed and cannot be silently truncated. |
| Reproducibility | New runs record source-tree, dependency-specification, runtime-package, Python, model, corpus, adapter, panel, and outcome fingerprints. |
| Base model | `microsoft/Phi-3-mini-4k-instruct` is pinned to revision `f39ac1d28e925b323eae81227eaba4464caced4e`. |
| Cloud environment | Previously floating training packages are pinned, and the full source/test/config set is baked into the Modal image. |
| Evaluation seeds | `--seed` is supported; hardened validation results use separate seed-specific filenames. |
| Bounded checkpoints | Evaluation-only and DPO-only launches reuse the SFT adapter's frozen bounded/full training scope instead of incorrectly recomputing it. |
| Final-test protection | `dpo_eval` now refuses to open the sealed test split unless `--confirm-final-test` is explicitly supplied. |
| Repository results | Results explicitly state that repository-fragment system evaluation is not implemented and is excluded from the function kill rate. |
| Documentation/defaults | Phase 3 is documented as SFT-first, defaults point to canonical V3, and the old 67/100 result is labelled historical pending re-evaluation. |

## Verification completed

- 84 automated tests passed.
- All Python modules compiled successfully.
- Artifact audit: 125 invariants passed, zero failures, two historical-result warnings.
- Canonical corpus: 8,387 records, 30,956 stored tests, train/validation/test counts 6,755/799/833, and no final-test artifacts.
- The exact 800-record hardened SFT preflight passed all readiness gates. It
  retained 1,657 generation-compatible synthetic examples and 78 repository
  examples before balancing, recorded 62 selected records with no
  policy-and-reference-valid winner as an immutable hashed exclusion, found
  zero sequence overflows, and planned 130 optimizer steps with three locked
  validation checkpoints.
- Hardened sample check on the first 100 function-level V3 validation records:
  - 591 stored assertions inspected.
  - 567 policy- and reference-valid assertions.
  - 381 verified mutation-killing assertions.
  - All 100 records retained at least one verified winner.
  - The 24 exclusions all failed because they did not call the target entry point; there were zero reference-execution compatibility failures.
- Whole-repository test coverage is approximately 18%. This remains an open engineering gap even though the corrected critical paths have direct tests.

## What is not yet confirmed

- The corrected Modal image has not been built and exercised on a GPU.
- The historical 67/100 adapter has not been re-measured under the hardened evaluator.
- A new provenance-complete SFT adapter has not yet been trained.
- Repository-level pytest/unittest execution environments are not implemented for generated tests.
- DPO has not demonstrated a repeatable validation improvement over hardened SFT.
- The final test split remains sealed and must stay sealed until all model-selection decisions are frozen.

## Required next execution order

All commands below consume Modal compute except the local tests/preflight. Run them only after cost approval.

### 1. Put the corrected source under version control

Initialize a Git repository if this folder is intended to be the authoritative project, commit the hardened source, and tag the exact code used for every cloud run. Do not mix later edits into an existing named run; the new fingerprint gate will reject them.

### 2. Re-evaluate the saved 67/100 smoke adapter on validation only

Run the same immutable adapter with three generation seeds. These commands do not train and do not touch the final test split.

```powershell
.\venv\Scripts\python.exe scripts\modal_train.py --phase sft_eval --corpus-version v3_final_candidate --run-name v3_repo1024_aligned_constant_lr_smoke_800_20260821_1 --seed 42
.\venv\Scripts\python.exe scripts\modal_train.py --phase sft_eval --corpus-version v3_final_candidate --run-name v3_repo1024_aligned_constant_lr_smoke_800_20260821_1 --seed 43
.\venv\Scripts\python.exe scripts\modal_train.py --phase sft_eval --corpus-version v3_final_candidate --run-name v3_repo1024_aligned_constant_lr_smoke_800_20260821_1 --seed 44
```

Report the full-validation function kill rate, candidate kill rate, parse rate, invalid rate, Wilson interval, and per-function outcomes for every seed. Report the mean and range across seeds. Do not quote 67% as current unless this re-evaluation supports it.

### 3. Run a small integration-only SFT smoke

This verifies that the pinned image, tokenizer, model revision, provenance files, checkpoint save, and isolated evaluator work together. It is not a performance experiment.

```powershell
.\venv\Scripts\python.exe scripts\modal_train.py --fresh --phase sft --corpus-version v3_final_candidate --run-name v3_hardened_integration_32_seed42 --seed 42 --max-pairs 32 --sft-epochs 1 --sft-learning-rate 0.00005 --sft-lr-scheduler-type constant_with_warmup --sft-batch-size 1 --sft-repository-completion-token-limit 1024 --no-sft-monitor-kill-rate
```

Proceed only if the run produces matching source/model/dependency fingerprints, a verified SFT marker, matching root/reference adapter checksums, finite loss/gradient metrics, and no execution-policy infrastructure errors.

### 4. Reproduce the 800-record SFT smoke under the hardened pipeline

```powershell
.\venv\Scripts\python.exe scripts\modal_train.py --fresh --phase sft --corpus-version v3_final_candidate --run-name v3_hardened_repo1024_constant_smoke_800_seed42 --seed 42 --max-pairs 800 --sft-epochs 1 --sft-learning-rate 0.00005 --sft-lr-scheduler-type constant_with_warmup --sft-batch-size 1 --sft-repository-completion-token-limit 1024 --sft-monitor-validation-functions 100 --sft-monitor-patience 3
```

Then repeat validation-only seeds 42, 43, and 44 on this new run, as in step 2. Select a checkpoint using validation data only. A 100-function smoke result is preliminary; the selected adapter must also be run on the complete function-level validation split.

### 5. Train the full SFT candidate only after the smoke is stable

Use a new run name, remove `--max-pairs`, keep the chosen hyperparameters frozen, and use the full locked validation monitor. Run validation-only evaluation across the predeclared seeds. Record aggregate, family-level, and per-function outcomes.

### 6. Treat DPO as an experiment, not a required final component

Start a bounded DPO smoke only after hardened SFT has a complete comparable validation baseline. DPO must use the same corpus, adapter, validation panel, candidate count, runtime, and seed. Accept DPO only if paired per-function diagnostics show a repeatable net improvement with controlled regressions; otherwise deploy the SFT adapter without DPO.

### 7. Implement repository-native system evaluation

Build disposable BugsInPy/SWE-bench project environments, generate framework-specific pytest/unittest tests, run fixed and buggy revisions under hard OS/container limits, and store commands, dependency lockfiles, logs, exit codes, and per-record evidence. Until this exists, repository records must remain excluded from the headline function kill rate and must not be described as completed system testing.

### 8. Raise engineering assurance

Increase tests around the cloud launcher, full training orchestration, checkpoint resume/restart, repository runners, and error paths. Add CI with compilation, unit tests, corpus verification, secret scanning, and a coverage threshold for critical modules.

### 9. Open the final test split exactly once

Only after the SFT/DPO choice, hyperparameters, checkpoint, seeds, metrics, and paper tables are frozen should the team run `dpo_eval --confirm-final-test`. The final result is for reporting, not for further tuning.

## Honest readiness statement

The local code and data invariants are now substantially safer and internally consistent, but the project is not yet fully validated or deployment-ready. The immediate next milestone is hardened, multi-seed validation of the saved SFT adapter—not DPO and not final-test execution.
