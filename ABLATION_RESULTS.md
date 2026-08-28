# Oneiros V4.1 ablation results

No GPU ablation result is recorded yet. This file intentionally preserves every predeclared experiment as **INCONCLUSIVE / NOT RUN** rather than inventing metrics or selecting a change from the locked validation set.

All design experiments use the fixed, training-only `ablation_dev` split (`415f6ac27cbf17f3dd1cc289495e08d6844b57a6a689d265958bb5aceed19034e`) with seeds 42, 43, and 44. Candidate count, generation order, sampling configuration, execution policy, and Kill@k definitions are frozen in `research/v4_1/FROZEN_EVALUATION_CONFIG.json`.

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

Future rows must report requested, parse-valid, execution-valid, reference-valid, and killing candidates; Kill@1/2/4/8; Wilson intervals per seed; dataset and mutation-family slices; token use; unique/effective examples; and an ACCEPT, REJECT, or INCONCLUSIVE decision. Negative results must remain in both artifacts.
