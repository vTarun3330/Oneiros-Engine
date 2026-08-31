# Oneiros V4.1 ablation results

No GPU ablation result is recorded yet. This file intentionally preserves every predeclared experiment as **INCONCLUSIVE / NOT RUN** rather than inventing metrics or selecting a change from the locked validation set.

All design experiments use the fixed, training-only `ablation_dev` split (`415f6ac27cbf17f3dd1cc289495e08d6844b57a6a689d265958b5aceed19034e`) with seeds 42, 43, and 44. Candidate count, generation order, sampling configuration, execution policy, and Kill@k definitions are frozen in `research/v4_1/FROZEN_EVALUATION_CONFIG.json`.

| Group | Control | Treatment | Status | Decision |
| --- | --- | --- | --- | --- |
| A | code only | + specification; + legitimate context | not run | INCONCLUSIVE |
| B | safely reproduced legacy format | unified V4.1 format | not run | INCONCLUSIVE |
| C | old one-test wording | self-contained test-case wording | not run | INCONCLUSIVE |
| D | diagnostic token head/tail | section-aware AST-unit compaction | local safety tests only | INCONCLUSIVE |
| E | oracle localization, diagnostic only | public/buggy-side localization | not run | INCONCLUSIVE |
| F | synthetic only | 10%, 20%, 30% repository supervision | not run | INCONCLUSIVE |
| G | proportional sampling | dataset and family balancing | not run | INCONCLUSIVE |
| H | prior verified policy | strict reference-pass/buggy-fail supervision | not run | INCONCLUSIVE |
| I | 800 examples | ~2k, ~4k, full eligible train | not run | INCONCLUSIVE |
| J | 512/768 rejected by promptability gate | admissible 1024 vs 1280 | local CPU evidence only | INCONCLUSIVE |

## Group J — function prompt budget (prerequisite)

Group J was added after a local audit showed the frozen 512-token function
prompt budget is smaller than the prompt's own required sections. Under
fail-closed section-aware compaction this is not a tuning knob but a
prerequisite: at 512 tokens most records cannot be prompted at all, so they
yield zero candidates regardless of the model.

Measured locally on CPU with no model loaded. Every function record on each
panel was attempted, so these counts are exhaustive rather than sampled.

Unpromptable function records by budget:

| Panel | Records | 512 | 768 | 1024 | 1280 | 1536 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `ablation_dev` | 542 | 388 (71.6%) | 28 (5.2%) | 0 | 0 | 0 |
| `val` (locked) | 757 | 583 (77.0%) | 0 | 0 | 0 | 0 |

Usable synthetic training records, from the real preflight runs:

| Budget | Prompt-compatible synthetic train pairs (of 5,595) |
| --- | ---: |
| 512 | 1,930 |
| 768 | 5,396 |

**Gate conclusion.** Promptability is a hard prerequisite, not a performance
result. 512 fails on both panels. 768 clears `val` but leaves 28 unpromptable
`ablation_dev` records. **1024 is the smallest tested budget that clears
`evaluation_panel_fully_promptable` on both panels**, and 1024 + 128 leaves ample
room in the 2,048 sequence budget. Dropping the 28 records to make 768 pass would
silently change the panel and is not permitted.

This narrows the admissible budgets to 1024 and above. It does **not** select
one: which admissible budget yields better Kill@k and reference-validity is
still an open GPU question, and **no model has been trained at any budget**.

The prompt budget is part of the evaluation scope hash. Results at a new budget
are not comparable with V4 numbers produced at 512 unless the V4 adapter is
re-evaluated under the same budget.

Future rows must report requested, parse-valid, execution-valid, reference-valid, and killing candidates; Kill@1/2/4/8; Wilson intervals per seed; dataset and mutation-family slices; token use; unique/effective examples; and an ACCEPT, REJECT, or INCONCLUSIVE decision. Negative results must remain in both artifacts.
