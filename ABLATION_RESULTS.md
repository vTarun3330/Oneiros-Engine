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
| J | 512-token function prompt budget | 768, 1024 | local CPU evidence only | INCONCLUSIVE |

## Group J — function prompt budget (prerequisite)

Group J was added after a local audit showed the frozen 512-token function
prompt budget is smaller than the prompt's own required sections. Under
fail-closed section-aware compaction this is not a tuning knob but a
prerequisite: at 512 tokens most records cannot be prompted at all, so they
yield zero candidates regardless of the model.

Measured locally on CPU, with no model loaded:

| Panel | Function records | Unpromptable at 512 | Share |
| --- | ---: | ---: | ---: |
| `ablation_dev` | 542 | 388 | 71.6% |
| `val` (locked) | 757 | 583 | 77.0% |

Sampled over 400 synthetic training records, the share whose required sections
fit is 34.5% at 512, 97.2% at 768, and 99.8% at 1024. The sequence budget is
2,048 and function completions are capped at 128, so 768 and 1,024 both fit
without touching evaluator semantics.

No GPU comparison has been run, so **no budget has been selected**. The prompt
budget is part of the evaluation scope hash: results at a new budget are not
comparable with V4 numbers produced at 512 unless the V4 adapter is
re-evaluated under the same budget.

Future rows must report requested, parse-valid, execution-valid, reference-valid, and killing candidates; Kill@1/2/4/8; Wilson intervals per seed; dataset and mutation-family slices; token use; unique/effective examples; and an ACCEPT, REJECT, or INCONCLUSIVE decision. Negative results must remain in both artifacts.
