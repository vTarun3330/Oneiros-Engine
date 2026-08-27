# Hardened SFT Re-evaluation — 27 August 2026

## Decision

**Do not merge this experiment into `main`. Do not start DPO from this adapter yet.**

The acceptance rule was fixed before reading the results: all three validation-only seeds must complete on the same locked panel and each completed seed must achieve at least a 58% function kill rate. Seeds 42 and 43 completed below 58%. Seed 44 was safely checkpointed at 234/775 when both configured Modal workspaces became unavailable because of spend controls. Completing seed 44 cannot reverse the already failed per-seed gate.

The historical 67/100 checkpoint-monitor result remains a smoke result, not a full validation result. The mean of the two complete hardened runs is 55.68%, 11.32 percentage points below that smoke result.

## Git and evaluation identity

- Baseline tag: `phase3-hardened-v1`
- Baseline commit: `fb0baf7fd68a3150919507255e0a9244a6dca47e`
- Experiment branch: `experiment/hardened-sft-reeval-seeds-42-44`
- Corpus: `v3_final_candidate`, 8,387 verified records
- Evaluation split: locked validation split only; final test access was `false`
- Validation records: 799 total = 775 executable function records + 24 repository records held for native-project evaluation
- Candidates per function: 8 (6,200 requested candidates per complete seed)
- Adapter SHA-256: `41ea59c4d297bd2530d778f4e6582eb5d9c53d39573d2bc70bd43496be23e4cf`
- Evaluation scope SHA-256: `cac7553b8e242437c22d30165c3064527b6cf12c2ca650349ef626a25d5775ba`
- Source-tree SHA-256: `59ad95e255a4dd1bfbd5728c4f4ed206525e60eb410a859ae1cc7e619d1298cd`
- Base model: `microsoft/Phi-3-mini-4k-instruct` at revision `f39ac1d28e925b323eae81227eaba4464caced4e`
- Runtime: 4-bit NF4, BF16 compute, eager attention, one A10/A10G GPU

## Results

| Seed | Status | Functions killed | Function kill rate | Wilson 95% interval | Parse success | Valid/requested candidates | End-to-end candidate kill rate |
|---:|---|---:|---:|---:|---:|---:|---:|
| 42 | Complete | 424/775 | 54.71% | 51.19–58.18% | 92.19% | 2,138/6,200 (34.48%) | 23.45% |
| 43 | Complete | 439/775 | 56.65% | 53.13–60.09% | 92.23% | 2,199/6,200 (35.47%) | 24.87% |
| 44 | Partial, resumable | 141/234 | 60.26% prefix only | Not a terminal result | 90.97% | 867/1,872 (46.31%) | 29.49% |

Two-complete-seed mean: **55.68%**. Range: **54.71–56.65%**. The partial seed-44 prefix is excluded from all final aggregates.

Raw local artifact checksums:

- Seed 42 JSON: `22b087198d78e41802ccc42672c1feea444026a3af7a2d79efb6cb94ca47f4e8`
- Seed 43 JSON: `5e355114ac7d6c190f4f858d9ad468419894025bde6128a4812b4611633aecc0`
- Seed 44 progress at 234: `7829e3755216fd0a7beef6b5a87d1ff47bfce56463649c60b1084929196b4cbd`

Raw checkpoints and per-function outputs remain outside Git under `results/v3_repo1024_aligned_constant_lr_smoke_800_20260821_1/`.

## Why the 100-function smoke result was optimistic

The full panel exposes a large benchmark shift that a small ordered prefix hides:

| Benchmark | Complete-seed observations | Killed | Function kill rate | Parse success | Valid/requested candidates |
|---|---:|---:|---:|---:|---:|
| HumanEval | 112 | 101 | 90.18% | 100.00% | 96.76% |
| MBPP | 1,438 | 762 | 52.99% | 91.60% | 30.16% |

HumanEval represents only 56 of the 775 functions per seed, but it occurs at the start of this deterministic validation order and is much easier for the adapter. The MBPP-heavy remainder produces far more assertions that either fail to parse or do not execute successfully against the reference implementation. Therefore, 67/100 measured an easy, unrepresentative prefix and cannot be presented as the final project kill rate.

Across the two completed seeds, the weakest sufficiently represented bug families were:

| Bug family | Observations | Function kill rate | Parse success | Valid/requested candidates |
|---|---:|---:|---:|---:|
| Index | 44 | 34.09% | 77.27% | 19.60% |
| Negate removal | 16 | 43.75% | 99.22% | 21.09% |
| Logical | 36 | 47.22% | 89.24% | 46.53% |
| Off by one | 118 | 53.39% | 95.23% | 30.83% |
| Boundary | 688 | 54.94% | 91.32% | 32.92% |
| Arithmetic | 416 | 56.01% | 96.09% | 33.86% |

## Pipeline fixes made on the experiment branch

Three operational defects were found and fixed before the accepted runs:

1. The Windows launcher now configures UTF-8 before printing status labels, preventing a CP1252 `UnicodeEncodeError` before launch.
2. The Modal runtime now adds `/root/oneiros` to `sys.path` before importing project configuration, preventing remote `ModuleNotFoundError: config` retries.
3. Hugging Face model shards now use the persistent Modal volume. After the first cache population, the same pinned model loaded in roughly 6–10 seconds for later seeds instead of being downloaded again.

The repository test suite passes: **84/84 tests**.

## Required next work

1. Restore Modal billing/spend authority, then resume seed 44 with the unchanged command and source revision. The evaluator should print `[VALIDATION RESUME]` and continue from function 235.
2. Keep the final test split sealed. Do not run `dpo_eval` or use `--confirm-final-test`.
3. Train a new SFT adapter on a substantially larger, balanced portion of the 6,728 eligible training records instead of the current 800-record smoke scope.
4. Improve training examples for exact assertion syntax, entry-point invocation, and reference-valid inputs. The main weakness is candidate validity, especially on MBPP and index mutations.
5. Rebalance the training curriculum by benchmark and bug family, emphasizing index, negate-removal, logical, off-by-one, and boundary cases. Derive new examples only from the training split; do not train on validation failures or their record contents.
6. Re-run the same locked seeds and require the predeclared full-panel gate before model selection.
7. Begin DPO only after SFT passes the repeated full-validation gate. DPO is an optional refinement stage, not a substitute for inadequate SFT validity.

Resume command (primary profile after spend access is restored):

```powershell
py -3.12 scripts/modal_train.py --phase sft_eval --corpus-version v3_final_candidate --run-name v3_repo1024_aligned_constant_lr_smoke_800_20260821_1 --seed 44
```

The experiment branch should remain unmerged until a later adapter passes the full gate.
