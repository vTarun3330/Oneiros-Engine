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
| G | proportional sampling | dataset and family balancing | CPU audit; unmatched GPU controls withheld | INCONCLUSIVE |
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

## Group G — upstream datasets and achieved weights

Sampling labels now follow `source.upstream_then_source.name_v1`, keeping an
upstream dataset such as HumanEval or MBPP separate from the ingestion source
`oneiros_clean_mutations`. Dataset identity is metadata only and is never added
to a model prompt. Reports retain ingestion-source metrics and add distinct
upstream `dataset_metrics` and `equal_weight_dataset_macro` fields; an unknown
dataset makes the macro explicitly incomplete rather than dropping functions.

Every SFT/preflight sampling report records raw, unique and effective examples,
repeat histograms, and achieved weights per dataset, mutation family and
dataset-family pair. `results/v4_1_dataset_sampling_audit.json` audits only the
eligible training split and records sources that would have merged datasets.
It finds 5,941 eligible training records: 4,650 MBPP, 938 HumanEval, 234
SWE-bench Verified, 112 BugsInPy, and 7 manual examples, with zero unknown
dataset labels. `oneiros_clean_mutations` is the ingestion source that would
have collapsed multiple upstream datasets.

The CPU diagnostic rejected the old proposed G matrix. G0 changed
deduplication and repository composition in addition to sampling. G1/G2
requested twice as many examples with a two-repeat cap, so every example was
repeated twice and proportions did not change. No GPU result exists, and these
commands are withheld. G remains INCONCLUSIVE until all treatments share one
verified pool, effective budget, optimizer settings and repository composition.

## Group L — base model and attention backend (screening)

**Hypothesis.** A smaller code-specialised base model with an available fused
attention backend can raise reference-valid, execution-valid generation and cut
generation cost with **no change to the evaluation protocol**.

**Status.** Zero-shot screen on the fixed 32-function `ablation_dev` monitor
panel (`selection_sha256 0ed8b64f…fafe907`), seed 42, 8 candidates/function,
1024-token prompt budget, plus one 32-pair integration run for L2. **This is not
a trained-model comparison**; no research-scale SFT exists for either base model.

| Arm | Model / backend | Kill rate | Kill@1 | Kill@8 | Parse-valid | Exec-valid | Ref-valid | s/function |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| L0 (control) | Phi-3-mini-4k `@f39ac1d2`, eager | 0.594 (19/32) | 0.156 | 0.594 | 0.508 | 0.410 | 0.270 | 6.53 |
| L1 | Phi-3-mini-4k, SDPA | — | — | — | — | — | — | — |
| L2 (treatment) | Qwen2.5-Coder-1.5B `@2e1fd397`, SDPA | 0.719 (23/32) | 0.375 | 0.719 | 0.980 | 0.980 | 0.566 | 1.28 |

Wilson 95% intervals: L0 [0.423, 0.745], L2 [0.546, 0.844] — **they overlap**, so
the kill-rate gap alone is not decisive at n=32. The candidate-validity gaps are
measured over 256 candidates per arm and are far larger than that noise floor.

**L1 — negative result, retained.** `transformers` 4.48.3 raises
`ValueError: Phi3ForCausalLM does not support an attention implementation through
torch.nn.functional.scaled_dot_product_attention`. `flash_attention_2` is also
unavailable for Phi-3's sliding-window attention here. **Eager is the only
admissible Phi-3 backend in this environment**, so no SDPA speedup is reachable
for the incumbent model.

**L2 integration evidence.** `local_j1024_integration_32_seed42_qwen_v1`:
baseline 21/32 → terminal 23/32, planned checkpoints `[6]` matched evaluated
checkpoints `[6]`, loss 0.7749, **215.5 s total vs 1710.5 s** for the equivalent
Phi-3 run. The monitor **did not promote** the checkpoint:
`end_to_end_candidate_kill_rate` fell 0.2773 → 0.2539, outside the 0.01
candidate-health tolerance. McNemar exact p = 0.7266. A 6-optimizer-step run
carries no training signal in either direction; the gate behaved correctly and
the dip is recorded rather than explained away.

**Reproducibility caveat, retained.** Repeated identical-configuration runs did
not reproduce bit-exactly at fixed seed: L0 Kill@1 moved 7/32 → 5/32 and the L2
baseline moved 23/32 → 21/32 across processes. **Single-run deltas below roughly
two functions on this panel are not evidence.**

**Decision: ACCEPT_FOR_SCREENING_ONLY.** L2 becomes the base model for
subsequent experiments because the decision hierarchy (§53) ranks reference- and
execution-validity above the headline metric, and those gaps are large and
low-noise. This is not a claim that Qwen trains better — that requires the
800-pair comparison. A base-model swap also changes the tokenizer, so **the
prompt-budget admissibility sweep (Group J) and every V4 comparison must be
re-derived** before any headline claim.

## Group K — complex-function mix (design input, not yet a result)

Complexity is derived **only** from the buggy-side localized AST
(`harness/function_complexity.py`, `COMPLEXITY_POLICY_VERSION
oneiros_buggy_ast_complexity_v1`); it never reads reference code, gold patch,
gold test, or oracle output.

| Split | Records | Complex | Moderate | Simple | Repository (no function AST) |
| --- | ---: | ---: | ---: | ---: | ---: |
| train | 5,941 | 2,754 | 1,999 | 842 | 346 |
| ablation_dev | 580 | 326 | 129 | 87 | 38 |

The bounded SFT sampler enforces a **0.60 minimum complex share**; the 32-pair
selection audited at **0.6207** (18 complex / 3 moderate / 8 simple / 3
repository). K0 (natural mix) versus K1 (0.60 floor) is predeclared in
`research/v4_1/ablation_plan.json` and **has not been run**.

**Decision: INCONCLUSIVE.** Complex functions are present and the floor is
enforced and audited, but no evidence yet shows the floor improves the objective.
The share is a design input, not a result.

Future rows must report requested, parse-valid, execution-valid, reference-valid, and killing candidates; Kill@1/2/4/8; Wilson intervals per seed; dataset and mutation-family slices; token use; unique/effective examples; and an ACCEPT, REJECT, or INCONCLUSIVE decision. Negative results must remain in both artifacts.
