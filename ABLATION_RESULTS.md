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

## Group I — training scale (200 vs 800 pairs)

Qwen, LR 5e-5, seed 42, 100-function `ablation_dev` panel.

| Step | 200 pairs killed | ref-valid | | Step | 800 pairs killed | ref-valid |
| ---: | ---: | ---: | --- | ---: | ---: | ---: |
| 0 | 65 | 0.609 | | 0 | 65 | 0.609 |
| **17** | **70** | 0.580 | | **50** | **73** | 0.449 |
| 34 | 69 | 0.446 | | 100 | 64 | 0.444 |
| 35 | 64 | 0.431 | | 142 | 69 | 0.479 |

Quadrupling supervision (553 → 2,260 examples, 35 → 142 steps) moved the
selected peak from 70 to 73 — at or barely above the two-function noise floor —
while reference-validity still collapsed inside the first 50 steps.

**Decision: REJECT.** The policy is **not data-starved** at this learning rate.
This is a negative result against the undertraining hypothesis that motivated
the training-scale ablation, and it is retained as such.

Both predictions in `results/v4_1_prediction_qwen_sft800_seed42.json`, recorded
**before** the run, were confirmed: the monitor selected an early checkpoint
(50, not terminal 142), and reference-validity at 142 ended below step 0.

## Group M — learning rate (5e-5 vs 1e-5)

Identical 2,260 examples, seed 42, 800 pairs. Only the rate differs.

| Step | 5e-5 killed | 5e-5 ref-valid | 1e-5 killed | 1e-5 ref-valid |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 65 | 0.609 | 65 | 0.609 |
| **50** | **73** | 0.449 | **72** | **0.541** |
| 100 | 64 | 0.444 | 62 | 0.464 |
| ~142 | 69 | 0.479 | 65 | 0.464 |
| loss | 0.7696 | | 0.8690 | |

At the selected checkpoint the kill rate is unchanged within noise (72 vs 73)
while reference-validity recovers 0.449 → 0.541 against a 0.609 baseline, about
57% of the damage 5e-5 caused.

**Decision: ACCEPT 1e-5.** Section 53 ranks reference-validity above the
headline metric, and this buys a large gain there for no measurable kill-rate
cost. Single seed, so directional until confirmed on 43 and 44.

**Limitation, stated plainly:** the lower rate *slowed* the degradation without
preventing it. Both arms still peak at the first checkpoint and decline. The
learning rate is not the root cause.

## Group K — complex-function mix (measured)

Qwen, LR 1e-5, 800 pairs, seed 42. Complexity is derived only from the
buggy-side localized AST.

| Step | K0 (49.8% complex) | ref-valid | K1 (60.1% complex) | ref-valid |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 65 | 0.609 | 65 | 0.609 |
| **50** | **68** | 0.543 | **72** | 0.541 |
| 100 | 64 | 0.468 | 62 | 0.464 |
| ~143 | 66 | 0.456 | 65 | 0.464 |
| loss | 0.8867 | | 0.8690 | |

K1 beats K0 by four functions at the selected checkpoint, with reference-validity
identical and slightly lower loss.

**Decision: WEAK ACCEPT.** Directional, not established, for three reasons:

1. Four functions sits just above the two-function noise floor.
2. One seed.
3. **The contrast is narrower than the labels suggest.** Removing the floor
   still leaves 49.8% complex records, because the corpus is naturally
   complex-heavy. This measures a ten-point shift, not zero-versus-sixty. A
   stronger test needs a deliberately simple-heavy control.

So: complex functions *are* in the training mix, the 0.60 floor is enforced and
audited, and raising the share does appear to help — but report it as
directional.

## Cross-cutting observation — form is learned before behaviour

Not a controlled ablation; a pattern present in every completed Qwen arm.

Parse-validity and execution-validity rise **monotonically** (0.991 → 0.999 and
0.968 → 0.991) while reference-validity **collapses** (0.609 → 0.43–0.48). The
policy learns to emit well-formed, runnable tests that assert **incorrect
expected values**. It acquires form faster than behaviour.

Every arm also peaks at its *first* monitored checkpoint and declines, which
means **the true maximum has never been observed** — a 50-step checkpoint
interval is too coarse to locate it.

Phi-3 is the informative exception: under the same recipe its reference-validity
*rose* (0.295 → 0.466). So the collapse is a property of a policy already near
its ceiling, not of the recipe.

Neither more data (Group I) nor a gentler learning rate (Group M) fixed this;
both only slowed it. That points at **supervision content** rather than quantity
or optimisation speed.

**Decision: INCONCLUSIVE**, with two concrete follow-ups: finer checkpointing to
locate the real peak, and ablation A to test whether the behavioural
specification is being used at all.

## Group N — checkpoint resolution sweep (hypothesis refuted)

**Hypothesis, recorded before the run.** Every arm peaked at its first
monitored checkpoint — step 17 at 200 pairs, step 50 at 800. We had never
observed the actual maximum, only the first place we happened to look. If the
true peak sat at step 20 with reference-validity still near 0.58, then every
number recorded so far would understate the model.

Qwen, LR 1e-5, 800 pairs, seed 42, 100-function `ablation_dev` panel. Only the
monitor interval changed (50 → 10); early stopping was disabled so no
checkpoint could be skipped. Run `20260902-231747-sweep_fine_ckpt`, 4,123 s,
16 checkpoints.

| Step | Killed/100 | Parse | Exec | Ref-valid | Cand-kill |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 65 | 0.9912 | 0.9675 | **0.6088** | 0.2537 |
| 10 | 64 | 0.9938 | 0.9663 | 0.6012 | 0.2562 |
| 20 | 64 | 0.9938 | 0.9688 | 0.6012 | 0.2625 |
| 30 | 61 | 0.9988 | 0.9675 | 0.5800 | 0.2462 |
| 40 | 67 | 0.9975 | 0.9762 | 0.5813 | 0.2762 |
| 50 | 68 | 0.9962 | 0.9725 | 0.5575 | 0.2737 |
| **60** | **72** | 0.9975 | 0.9712 | 0.5375 | 0.2900 |
| 70 | 67 | 0.9988 | 0.9775 | 0.5500 | 0.2825 |
| 80 | 65 | 0.9988 | 0.9738 | 0.5038 | 0.2800 |
| 90 | 65 | 0.9975 | 0.9750 | 0.4838 | 0.2938 |
| 100 | 63 | 0.9962 | 0.9750 | 0.4650 | 0.2850 |
| 110 | 63 | 0.9962 | 0.9725 | 0.4750 | 0.2938 |
| 120 | 67 | 0.9975 | 0.9788 | 0.4775 | 0.3050 |
| 130 | 64 | 0.9988 | 0.9812 | 0.4662 | 0.2988 |
| 140 | 68 | 0.9988 | 0.9750 | 0.4738 | 0.3137 |
| 142 | 69 | 0.9988 | 0.9762 | 0.4788 | **0.3175** |

### Both predictions falsified

**"The peak lies strictly before step 50" — FALSIFIED.** The peak is step 60 at
72/100. The best checkpoint strictly before 50 is step 40 at 67, *worse* than
the 68 we already had at step 50. Steps 10, 20 and 30 (64, 64, 61) are at or
**below the untrained baseline of 65** — the model gets worse before it gets
better.

**"Peak reference-validity exceeds 0.541" — FALSIFIED where it matters.**
Reference-validity never rises above its untrained 0.6088. The highest
post-training value is 0.6012 at steps 10–20, and those checkpoints kill fewer
bugs than doing nothing at all. At the best kill-rate checkpoint (60),
reference-validity is 0.5375 — *lower* than the 0.541 at step 50.

**Decision: the understatement hypothesis is REJECTED.** The coarse interval was
not hiding a better model. No previously recorded number was an understatement.
Finer checkpointing cost 69 minutes and bought +4 functions — barely above the
±2 noise floor — at a checkpoint with worse reference-validity than the one it
replaces. A 10-step interval is not worth its monitoring cost as a default.

### What the resolution did reveal

The form-before-behaviour pattern starts at the **very first checkpoint**, not
after some threshold. By step 10, parse-validity is already climbing
(0.9912 → 0.9938) and reference-validity is already falling (0.6088 → 0.6012).
The two move in opposite directions for the entire run.

The two success layers do not even peak together: candidate kill rate rises
almost monotonically to its maximum at the *terminal* step (0.3175), while
function kill rate peaks at 60 and then falls.

**The practical consequence is a genuine trade-off, not a sampling artifact.**
There is no checkpoint that is simultaneously good at killing bugs and faithful
to the reference. Select on kill rate and you take step 60 with
reference-validity 0.5375; select on reference-validity and you take step 10–20
with a kill rate at or below the untrained baseline.

## Group O — multi-seed confirmation (corrects two earlier claims)

Qwen, LR 1e-5, 800 pairs, 100-function `ablation_dev` panel. Only the seed
changed.

| Seed | Baseline | Best | At step | Paired gain | Baseline 95% CI | Best 95% CI |
| ---: | ---: | ---: | ---: | ---: | :--- | :--- |
| 42 | 65 | 72 | 50 | **+7** | [0.553, 0.736] | [0.625, 0.799] |
| 43 | 58 | 73 | 142 | **+15** | [0.482, 0.672] | [0.636, 0.807] |
| 44 | 62 | 73 | 142 | **+11** | [0.522, 0.709] | [0.636, 0.807] |

| | Values | Mean | Range | Stdev |
| --- | --- | ---: | ---: | ---: |
| Untrained baseline | 65, 58, 62 | 61.67 | **7** | 3.51 |
| Trained best | 72, 73, 73 | 72.67 | **1** | 0.58 |
| Paired gain | 7, 15, 11 | 11.0 | 8 | 4.00 |

### The noise floor was wrong

The ±2 figure quoted throughout earlier sections came from repeated runs at the
**same seed**. Across seeds the untrained baseline moves by **7 functions** on
the identical panel with nothing changed but the seed — more than three times
larger. Two prior claims have to be re-judged against that.

**Two floors apply and were being conflated.** A *paired* within-seed
comparison (baseline vs trained inside one run) cancels seed choice, so its
floor is same-seed variability, about 2. An *unpaired* comparison between two
separately trained runs does not cancel it, so its floor is the across-seed
range, 7. Judging a paired gain against the unpaired range discards real
effects; judging an unpaired difference against the paired floor accepts noise.

### What this establishes

**SFT works, and now on evidence rather than one seed.** The paired gain is
positive in all three seeds (+7, +15, +11) and exceeds the applicable floor in
each. **ACCEPTED.**

**SFT sharply reduces seed sensitivity — a new finding.** The three untrained
baselines are spread across 7 functions; after training they converge to 72–73,
a range of 1. That stability is a stronger argument for keeping SFT than the raw
gain, and it was invisible at one seed.

### What this withdraws

**The complexity-floor WEAK ACCEPT is downgraded to INCONCLUSIVE.** K1 vs K0 was
+4 on seed 42 alone — an *unpaired* comparison between two separately trained
runs, against an across-seed range of 7. Complex functions remain in the corpus
and the 0.60 floor remains enforced and audited; what is withdrawn is the claim
that raising the share demonstrably helps. Running K0 and K1 on seeds 43 and 44
would settle it.

**Checkpoint selection is not stable across seeds.** Best step is 50, 142, 142.
The step-60 peak from Group N was already marginal (+4 against a floor of 2);
combined with this, selecting a checkpoint by peak kill rate on a single seed is
not reliable.

Future rows must report requested, parse-valid, execution-valid, reference-valid, and killing candidates; Kill@1/2/4/8; Wilson intervals per seed; dataset and mutation-family slices; token use; unique/effective examples; and an ACCEPT, REJECT, or INCONCLUSIVE decision. Negative results must remain in both artifacts.
