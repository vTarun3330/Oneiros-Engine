# SFT root-cause protocol

Frozen on 2026-09-28 at commit `e56af51808562ec179d238425881e0c51fa4eb52`, before
any result it governs was computed. The machine-readable state lives in
`results/sft_root_cause_state.json` and the hypotheses in
`results/sft_root_cause_hypotheses.json`. Nothing here may change after a result it
governs has been seen. Amendments are appended below with a date and a reason; the
original text stays.

## Objective

Determine *causally* why the arm A SFT adapter (checkpoint 431) changes behaviour
without establishing robust generalisation. The objective is not a higher Kill@8. The
loop ends in exactly one terminal state:

* **A. Dominant cause supported.** Every condition in *Terminal rule A* holds.
* **B. Underdetermined.** The hypotheses are bounded, but no dominant cause is
  identifiable within the sample and compute budget below. A final report then lists
  what was ruled out, what remains plausible, and what evidence would decide it.

## Interpretation carried forward (not re-litigated)

- SFT changed behaviour. Locked-validation Kill@8 rose from 0.594452 to 0.628798
  (+3.4346 points). McNemar p = 0.078 against the frozen p < 0.01, so the result is
  positive but insufficient.
- Reference validity fell from 0.452114 to 0.333388 (−11.8726 points).
- In an earlier base-model artifact, 70.7% of MBPP `wrong_expected_value` candidates on
  non-killed functions were wrong oracles on inputs that already distinguished reference
  from mutant (`results/v4_2_oracle_vs_input_base_val_s42.json`). The project report
  cites 73.1% for the dominant failure.
- Leading diagnosis: partial learning. Input selection improves; semantic oracle
  prediction does not generalise.
- It is **not** "SFT learned nothing".

## Frozen identity

| Item | Value |
|---|---|
| Branch / HEAD / origin | `experiment/research-eval-ablations` / `e56af51` / equal; clean tree |
| GPU | NVIDIA RTX 4500 Ada, 24,570 MiB, idle (521 MiB used, 0%) at freeze |
| Base model | `Qwen/Qwen2.5-Coder-1.5B-Instruct` @ `2e1fd397ee46e1388853d2af2c993145b0f1098a`, 4-bit NF4, bf16 compute |
| SFT adapter (arm A, checkpoint 431) | sha256 `e67dd599a37cbf2a738791c5cdf889cb4938a2ad53c991dca2fefae503b6f9e7` |
| Generation | temperature 0.7, top_p 0.9, sampling, 8 candidates per function, 1,024 new tokens, 1,024 prompt tokens, seed 42 |
| Parser / contract | `whole_output` parse; test-function candidates allowed; prompt schema `oneiros_unified_test_generation_v2` (`full` / `self_contained`) |
| Evaluator | `metrics/research_evaluation.py` raw sha256 `649e96fa…`; `harness/safe_execution.py` `1eb3927a…`; per-test timeout 0.5 s |
| Locked-validation artifacts | base `64beb7d8…`, arm A `700a233b…`; identical generation settings and evaluation scope `51ae08b6…` |

Any new generation reuses these settings unless the probe's single declared factor
changes one of them.

## Data-use status

| Data | Status | Permitted use in this loop |
|---|---|---|
| `val` (locked validation, 781 records / 757 function records) | **spent** | Descriptive decomposition of the retained raw outputs only. No checkpoint, hyperparameter or prompt selection. |
| `ablation_dev` (500 function records in the SFT monitor) | **spent** (it selected checkpoint 431) | Descriptive trajectory analysis of the retained monitor outputs only. |
| `test` / sealed final | **consumed** | Never read, never rerun. |
| Reserved confirmation IDs, `results/sealed_final*` | protected | Never read. |
| `train` | available | Diagnostic probes (Phase 3) and training arms (Phase 4). |
| Fresh cohorts (Phase 6) | not yet created | Confirmation only; failures never return to training. |

The spent splits may inform *which hypothesis* to test next. That choice is
confirmed only on disjoint cohorts (Phase 6), so it cannot become model selection.

## Hypotheses

The seven hypotheses H1–H7, each with the mediator signature that would support or
weaken it, are in `results/sft_root_cause_hypotheses.json`:

1. specification insufficiency
2. objective mismatch
3. adapter or model capacity
4. dataset-distribution shift
5. contract degradation
6. overfitting or forgetting
7. insufficient evaluation power

## Predeclared effect sizes

| Quantity | Minimum meaningful effect |
|---|---|
| Kill@8 improvement | +3 percentage points |
| `P(correct oracle | discriminating input)` improvement | +5 points |
| Reference-validity regression | at most 3 points (a larger drop fails) |
| Diversity (exact-unique ratio) regression | at most 10% relative |

## Inference

- **Unit of resampling.** Every interval is a **paired cluster bootstrap** over the
  record's semantic `group_id`. Validation has 781 records in 82 groups (up to 55
  mutants per group); ablation_dev has 594 in 63. Candidates, and even functions, are
  not independent. 10,000 resamples, seed 20260928, percentile 95% intervals.
  Function-level Wilson intervals are reported only alongside, never instead.
- **Decision regions for a difference `d`:**
  - *supported*: the lower 95% bound is at or above the minimum meaningful effect;
  - *directional*: the lower bound is above 0 but below the minimum;
  - *null-compatible*: the interval includes 0;
  - *reversed*: the upper bound is below 0.
- **Looks.** Each probe and arm has one fixed sample size and **one look**. There are
  no interim analyses, so no alpha spending is needed. If an interim look is ever
  added, it must first be declared here with an O'Brien–Fleming spending function.
- **Multiplicity.** Holm correction across the primary mediators of the probes run in
  one phase.

## Terminal categories (Phase 1)

Every requested candidate receives exactly one category, by this precedence
(`harness/causal_ledger.py`, tested exhaustive and exclusive):

1. `parse_failure`
2. `duplicate`: the same normalised code as an earlier-ranked candidate for the same
   function
3. `valid_kill`
4. `execution_failure`
5. `non_discriminating_input`
6. `discriminating_wrong_oracle`
7. `correct_content_invalid_contract`
8. `valid_non_kill` (a weak oracle)

"Discriminating" means that one of the candidate's **own** calls to the function under
test behaves differently on the mutant: the status and the sha256 of the result's
`repr` are compared. For a candidate that fails on the reference, only the calls made
up to and including its first failing assertion count, because a real test stops
there. The evaluator's verdicts are re-executed and checked, and disagreements are
reported.

## Phase plan and budget

| Phase | Content | Compute | Fixed size |
|---|---|---|---|
| 0 | freeze | none | – |
| 1 | causal ledger over retained outputs: locked validation (both arms) and the arm A monitor trajectory (ablation_dev, steps 0–431) | CPU | all retained candidates |
| 2 | decomposition and paired transitions | CPU | as Phase 1 |
| 3 | cheap causal probes (A specification ladder, B constrained output, C capacity), each on one frozen cohort of train functions | generation only; ≤ 12 GPU-hours total | 240 train functions, at most 3 per semantic group, drawn once (seed 20260928) and hash-frozen before generation |
| 4 | one single-factor training arm, chosen by expected information gain | ≤ 3 arms, ≤ 8 GPU-hours each | matched examples, tokens, optimiser, seed, learning rate, steps |
| 5 | mediator gate | CPU | – |
| 6 | independent confirmation (function-lineage-disjoint and repository-disjoint cohorts) | ≤ 6 GPU-hours | fixed before generation; no N = 400 native claim (A′ admitted 5 of 120) |

If the budget is exhausted before terminal rule A is met, the loop ends in terminal
state B.

GPU jobs run detached through `scripts/gpu_run.py` with logs, a heartbeat,
checkpoints, verified resume, PID and GPU-utilisation records, and a completion
receipt. Only one competing GPU job runs at a time.

## Terminal rule A (all required)

1. The predicted mediator changes (the *supported* region).
2. Kill@8 changes in the predicted direction.
3. Removing the intervention (ablation) removes the effect.
4. The effect reproduces on two independent cohorts.
5. The intervals clear the minimum meaningful effects.
6. Validity and diversity stay within the regression bounds.

Otherwise, each hypothesis is closed as one of:

- rejected;
- inconclusive for insufficient power;
- invalid because of a pipeline failure;
- under-specified by the available prompt;
- not identifiable with the current data and compute.

## After every phase

1. Verify the artifacts and their hashes.
2. Run the relevant tests after writing has finished.
3. Write a receipt: objective, command, data, model, hashes, duration, results,
   hypothesis update and next decision.
4. Commit and push code, tests, documentation and small receipts. Large ledgers stay
   local under `results/sft_root_cause/`, with their hashes in the committed receipts.
5. Quarantine invalid artifacts; never overwrite them.
6. Stop only for an integrity failure, a need for new authority, a destructive action,
   or a terminal conclusion.

## Amendments

**2026-09-28, after Phase 2: oracle mediator refined (exploratory; nothing
removed).**

- **Finding.** SFT moved about 99.5% of policy-valid assertions to exact equality; the
  base used 72.5%, with 1,308 `!=` assertions. A weak oracle holds on the reference far
  more often than an exact value, so the frozen `P(correct oracle | discriminating
  input)` mixes value-prediction ability with assertion form.
- **Refinement.** From Phase 3 on, the primary oracle mediator is **exact-value accuracy
  on a fixed discriminating input**. The frozen unstratified mediator is still reported
  alongside it. Phase 2 also reports equality-stratified accuracy.
- **Superseded receipt.** The first Phase 2 receipt is quarantined, with its sha256, in
  `results/sft_root_cause_state.json`.
- **Guardrail for user decision (not changed here).** The frozen
  reference-validity bound (at most 3 points of regression) is also form-confounded. An
  intervention that replaces weak oracles with correct exact ones can lower reference
  validity while making tests stronger.

**2026-09-28, amendment 2: interpretation correction and Phase 3 rules (prospective,
after review).**

*Phase 2 interpretation.* The dated receipt
`results/sft_root_cause_phase2_interpretation_correction_2026-09-28.json` supersedes the
interpretive wording of amendment 1 and of the Phase 2 receipt. Neither earlier receipt is
modified.

- Value prediction: "No value-prediction improvement is demonstrated; a ≥5-point
  improvement is excluded, but meaningful degradation remains possible."
- Assertion form and validity: "The regression is consistent with the large
  assertion-form composition shift, but causal attribution is not possible from this
  post-hoc, model-selected stratum."
- `P(kill | reference-valid)` is descriptive only. It conditions on a treatment-affected
  outcome, which produces survivor/selection bias.
- H5 is narrowed to syntactic/structural contract degradation (weakened). H6 is narrowed
  to progressive degradation after step 100 (weakened). A rapid early shift or
  forgetting remains open.
- Power is reported as two separate questions: (A) the effect is greater than zero;
  (B) the lower bound is at least +3 points. The ~296× multiplier answers only B, at 50%
  power.

*Guardrail.* The original three-point reference-validity guardrail is kept unchanged for
every historical and future promotion decision. From Phase 3 on, these are reported
alongside it and never replace it:

- overall reference-validity rate;
- exact-equality reference-validity rate;
- fixed-input exact-output accuracy;
- function-level probability of at least one valid killing candidate;
- assertion-form distribution;
- diversity;
- nonanswer rate.

*H4 decision rules (fixed-input exact-output accuracy, SFT minus base):*

1. If SFT does not improve on exposed training functions, weaken distribution shift and
   strengthen the objective and capacity explanations.
2. If SFT improves on exposed functions but not on lineage-disjoint unexposed functions,
   support a memorisation/exposure failure.
3. If SFT improves on both train cohorts but not on a genuinely external cohort, support
   distribution shift.
4. If no adequate unseen external cohort exists, H4's distribution-shift arm stays open
   and is not described as tested.

"Improves" means the paired cluster-bootstrap lower bound is above 0. The threshold for
"supported" is +5 points.

*Fixed-input probe requirements (Phase 3A and 3C):*

- **Input selection:** inputs are chosen by a deterministic, arm-blind rule before any
  generation. The rule never inspects model outputs and never picks inputs because a
  model produced them.
- **Verification:** each input is mechanically verified to discriminate buggy from fixed.
- **Concealment:** A0–A3 never show the fixed implementation, the mutation diff or the
  correct output.
- **Freezing and identity:** the cohort, inputs and expected outputs are hashed and
  frozen before generation. Every model and specification level receives identical
  inputs.
- **Group cap:** representation of each semantic group is capped.
- **Answer format:** answers use a short machine-readable schema, and the answer rate,
  truncation and nonanswers are reported.
- **A4:** the level showing behavioural evidence is diagnostic and non-deployable. It
  never enters training or confirmation.
- **Probe 3B:** the constrained-test probe is replaced by the output-schema/extraction
  control inside the fixed-input probe.
- **Probe 3C:** runs only after 3A is verified. A larger model is interpreted only if it
  passes an answer-rate non-inferiority gate. A non-local snapshot is reported
  (download size, disk space, time) before any download, and no revision is ever
  silently substituted.

**2026-09-28, amendment 3: Phase 3 correction (CPU only; nothing overwritten).**

The v2 receipts `results/sft_root_cause_phase3a_result_receipt_v2.json` and
`results/sft_root_cause_phase3c_result_receipt_v2.json` supersede the v1 receipts, which
remain unchanged. The dated receipt
`results/sft_root_cause_phase3_interpretation_correction_2026-09-28.json` lists every
withdrawn claim.

- **Schema rule.** The schema-control rule is applied to the exact strata a hypothesis
  decision reads. Every stratum reports SFT − base under both schemas and their
  difference-in-differences, with a semantic-group-clustered interval. The exposed
  stratum flips sign, so **H4 is `open`**, qualified as a schema-dependent exposure
  result.
- **Status values.** Every status must be one of `status_values`, and a test enforces
  this. Qualifiers go in a separate field. H1 is `strengthened` (scale qualification),
  H3 is `strengthened` (model-scale association), and H4 is `open`.
- **A3** is a diagnostic, oracle-derived partial behavioural hint (the correct type and,
  where sized, length). It is non-deployable and prohibited from training and
  confirmation. Its null result means only that this hint was insufficient.
- **A4** is access to the complete fixed implementation. Its gain does not independently
  establish natural-language specification insufficiency.
- **P-values.** Finite-resample p-values use a paired semantic-group sign-flip test with
  p = (extreme + 1) / (resamples + 1), with Holm correction across frozen primaries. A
  p-value is never reported as zero, and bootstrap tail proportions are not reported as
  p-values.
- **Phase 4.** The Phase 3 unexposed cohort is development evidence only. A fresh
  lineage-disjoint mediator-gate cohort must be frozen before any Phase 4 training (see
  `results/sft_root_cause_phase4_cohort_census.json` and
  `docs/SFT_ROOT_CAUSE_PHASE4_DESIGN_DRAFT.md`). Phase 6 confirmation stays separate.

**2026-09-28, amendment 4: v3 analyses and correction addendum (additive; amendment 3's text
is kept as written).**

Amendment 3 is superseded where it conflicts with this amendment:

- **Hypotheses:**
  - Amendment 3's "H1 `strengthened`" is superseded: **H1 is `open`**. A1 and A2 were not
    run, A3 is an oracle-derived type/length hint, and A4 exposes the complete fixed
    implementation. None of these identifies natural-language specification
    insufficiency.
  - **H3 is only a model-scale association.** Its adapter/parameter-capacity subclaim is
    `open`.
  - **H4 remains `open`** and schema-dependent. The route-exact rule stops at the exposed
    schema check.
- **Current receipts:** `results/sft_root_cause_phase3a_result_receipt_v3.json` and
  `results/sft_root_cause_phase3c_result_receipt_v3.json`, interpreted by
  `results/sft_root_cause_phase3_interpretation_correction_addendum_2026-09-28_v2.json`.
  The v1 and v2 receipts and the first correction receipt remain unchanged history.
- **Analysis rules from v3 on:**
  - Every paired contrast validates exact item identity.
  - The Phase 3C gate is recomputed from rows and enforced before interpretation.
  - Monte Carlo p-values are reported at their resolution, and a Holm-adjusted
    zero-effect p never establishes a ≥ 5-point effect.
  - The frozen ≥ 5-point rule uses per-contrast (unadjusted) intervals and supports no
    familywise claim.
- **No hypothesis is terminally supported.**

**2026-09-28, amendment 5: Phase 4 redesign (draft; nothing frozen or launched).**

- **Current design:** `docs/SFT_ROOT_CAUSE_PHASE4_DESIGN_DRAFT_V2.md`. The V1 draft is
  superseded because it changed input conditioning and loss masking together. It is kept
  byte-identical and recorded as superseded in `results/sft_root_cause_state.json`.
- **Primary comparison:** C and T share one fixed-call prompt, one verified
  (call, value) pair, one target text and identical `input_ids`. **Only the label mask
  differs**: C supervises every completion token; T supervises only the value span and
  EOS.
- **Optional three-arm decomposition** (C0/C1/T): documented only; needs separate
  authorisation.
- **Corrected confirmatory five-point rule:**
  - the declared lower 95% bound is above +5 under **both** schemas;
  - no schema sign flip;
  - the answer-rate gate passes;
  - the validity and diversity guardrails pass.

  The earlier rule (point estimate ≥ 5 and lower bound > 0) is superseded. It shows a
  positive effect, not an effect of at least 5.
- **Exploratory pilot outcomes:** `promising`, `harm` or `inconclusive_power`. An
  underpowered non-pass never automatically rejects the intervention.
- **Power:** `results/sft_root_cause_phase4_power_analysis_v1.json` (simulation from
  train-side Phase 3A outcomes; training-seed variance not modelled). A true 5-point
  effect cannot pass the confirmatory rule at any feasible size.
- **Pool labels:**
  - The strict untouched pool (33 feasible groups) is unusable as a gate.
  - The 352-group arm-A-exposed remainder is internal and exploratory only. It was used
    by arm A, is not untouched and is not repository-disjoint.
  - The 150/150/rest proposal is not frozen.
- **Repository-disjoint Phase 6** remains an unresolved blocker. No internal split can
  satisfy it.
- **Next step:** Choice A or Choice B (`docs/SFT_ROOT_CAUSE_PHASE4_CHOICES.md`) is the
  user's decision. No GPU step (including a timing run) happens without separate
  authorisation.

**2026-09-29, amendment 6: bounded Phase 4 preparation (additive).**

- **Reporting:** `results/sft_root_cause_reporting_addendum_2026-09-29_v1.json`.
  - The Phase 3A six-test Holm family is a post-hoc operationalisation with no
    familywise claim; the H1 and H4 decisions are unchanged.
  - The Phase 4 power analysis is planning evidence, not a guarantee.
  - Acquisition totals are explicit: about 16,800 candidates for the gate plus Phase 6,
    and at least 68 repositories.
- **Trainer:** `engine/sft_trainer.py` has `objective_mode` (`None` legacy,
  `full_completion`, `value_only`).
  - Explicit modes carry labels built from the trainer's own tokenisation, and the
    collator preserves them.
  - Value-only spans are positional, and they are refused unless they start after
    `' == '`, end on a token boundary and end the completion.
  - Tests pass real prepared examples through the production dataset, collator and a
    TRL `SFTTrainer` dataloader.
- **Power sensitivity:** `results/sft_root_cause_phase4_power_sensitivity_v1.json`,
  covering cross-schema coupling and effect level. The exact bootstrap gate agrees with
  the fast method on 93.5–100% of datasets.
- **Native rehearsal:** `results/sft_root_cause_phase4_native_rehearsal_receipt_v1.json`.
  - 26 development-retrospective targets were run in WSL with uv 0.12.7 and CPython
    3.13; 24 of 26 natively qualified.
  - Blocker: fixed-call construction succeeded for 0 of 24.
- **Objective smoke:** `results/sft_root_cause_phase4_objective_smoke_v1.json`,
  ACCEPTED. The Trainer batches differ only in labels. This is not efficacy training.
- **Decision:** `results/sft_root_cause_decision_receipt_2026-09-29_v1.json`. Choice B
  screening comes first, only as inexpensive screening, and it needs user
  authorisation. Choice A remains required for any generalisation claim.
- **No root cause is established.**

**2026-09-29, amendment 7: corrections after the Phase 4 preparation (additive).**

- **Execution-dose provenance:** `results/v4_3_execution_dose_source_provenance_v1.json`.
  - Historical source bindings are verified at their frozen commit (`41e821a`), not
    against the working tree; all 27 bindings reproduce there.
  - The Phase 4 trainer change (`efa8d9a`) is recorded as explicit drift.
  - The artifacts are historical, valid for their original run, and **not
    current-run-ready**. Their launch guard still refuses them, and a new execution-dose
    run needs a new source-bound preflight.
- **Power:** `results/sft_root_cause_phase4_power_sensitivity_v2.json` supersedes v1 for
  planning (supersession addendum `..._power_sensitivity_v1_supersession.json`; v1
  unchanged).
  - Partial coupling now uses one common selector.
  - Required-group ranges are unchanged.
  - One exact-vs-fast cell differs by more than 5 points.
  - Lower bound > 0 is a positive-effect screen; lower bound > +5 is exceeds-five. At
    about 200–350 groups the design is an exploratory positive-effect screen, **not**
    confirmation of a five-point effect.
- **Native rehearsal v2:** `results/sft_root_cause_phase4_native_rehearsal_receipt_v2.json`.
  - Environment success 25/26; semantic qualification 24/25; operational throughput
    24/26.
  - `choice_A_ready=false` (0/24 safe fixed calls).
  - Portable evidence bundles are `..._phase4_native_evidence_v1.json` and
    `..._phase4_smoke_evidence_v1.json`.
- **Objective contrast:** full-completion versus value-only changes both the positions
  and the mass of supervised tokens (952 versus 220 in the smoke).
  - Interpretation **A is frozen**: the Choice B screen estimates the composite
    "value-only masking plus reduced supervised-token mass".
  - A token-count-matched arm (interpretation B) was not added.
- **Decision:** `results/sft_root_cause_decision_receipt_2026-09-29_v2.json`
  (v1 unchanged).
- **No root cause is established.**

**2026-09-29, amendment 8: Choice B made launchable (not launched) and Choice A capture pilot.**

- **Choice B runner:** `scripts/phase4_choice_b.py` (freeze / preflight / train / evaluate
  / analyse).
  - Split v2 is `results/sft_root_cause_phase4_choice_b_split_v2.json`.
    - Gate: 200 Arm-A-exposed-remainder groups (400 fixed-input items), fixed before any
      outcome.
    - Training: 3,907 verified rows from 187 other groups.
    - v1 was refused by its own preflight because 37 prompts would be compacted; the
      refusal is preserved.
  - Preflight `results/sft_root_cause_phase4_choice_b_preflight_receipt.json` is
    **ready**.
    - The arms differ only in label mask, objective mode and output directory.
    - Supervised tokens: 98,187 (control) versus 27,571 (treatment) — the composite
      estimand.
    - 490 steps per arm; about 2.35 GPU-hours estimated.
  - **Training and evaluation require explicit user authorisation.** The train, evaluate
    and analyse paths have not been executed.
- **Choice A argument capture:** `results/sft_root_cause_phase4_argument_capture_receipt_v1.json`.
  - Method: profile-hook capture of real arguments during the difference-exposing
    official tests, then independent literal replay on both revisions.
  - Yield: 214 calls observed; 37 serialisable; 37 replayable on both revisions; 18 short
    verified oracles.
  - 3/24 qualified targets have a usable fixed call, all in **one** repository (humanize).
  - Most calls fail on instance-method receivers (131) or non-literal types (31).
  - **Decision:** usable yield is negligible for a repository-diverse cohort. Mass
    acquisition stays stopped, and no target is admitted: any use needs isolation-v6
    revalidation first.
  - Next: a mediator representation that does not require a literal-only free-function
    call. The candidate is receiver-aware replay, where the receiver is rebuilt from a
    captured, literal-only constructor call, or the official test's own asserted
    expression value as the oracle.
- **No root cause is established.**

**2026-09-29, amendment 9: Choice B v2 (stage-aware lifecycle; evaluation integrity; re-freeze).**

- **Supersessions:** the v1 Choice B preflight is historical and superseded, and is
  preserved byte-for-byte. Decision receipt v3 corrects Choice A to 3/24 targets, 1/8
  repositories, a descriptive interval, not ready.
- **Runner:** `scripts/phase4_choice_b_v2.py` with `harness/choice_b_lifecycle.py`.
  - Stage order: preflighted → control_trained → treatment_trained → control_evaluated →
    treatment_evaluated → analysed.
  - A bound source identity excludes results, so result-only commits are accepted and
    source changes are refused.
  - Both arms train before either gate look.
- **Data:** split v3 removes 4 training groups (91 rows) found by an outcome-blind
  near-duplicate audit. 3,816 training rows remain; the gate is unchanged.
- **Evaluation spec v2**, frozen before any outcome:
  - Guardrails are **point-estimate operational safety thresholds**, with bootstrap
    intervals reported alongside.
  - Outcomes: statistically supported adverse evidence, operational guardrail stop,
    promising, or inconclusive power, in that order of precedence.
  - The estimand is the composite described in the spec, not a scalar gradient-dose
    reduction.
  - At 200 groups, a 5-point effect has low useful detection power (33–55% by the exact
    bootstrap), but detection is not impossible.
- **Integrity:**
  - exact key sets;
  - contract-bound, strictly resumable raw outputs with quarantine;
  - Kill@8 rows must exactly equal the gate records;
  - tracked compact evidence sufficient to recompute the analysis on another clone.
- **GPU integration smoke:** passed. It is operational evidence only.
- **Claims:** no root-cause or generalisation claim is made.

**2026-09-29, amendment 10: Choice B v2 result (one look) and next step.**

- **Run:** the single predeclared screen ran through the stage-aware lifecycle.
  - control trained, then treatment trained (478 steps each);
  - control gate look, then treatment gate look;
  - one frozen analysis, which reproduces from tracked evidence.
- **Outcome: `inconclusive_power`.**
  - Answer schema: +1.25 points [−0.25, +3.00].
  - Prefilled assertion: 0.00 [−1.25, +1.25].
  - No sign flip; every operational guardrail passes.
  - This is not "no effect". No seeds were added and the gate was not reused.
- **Record:** decision receipt v4.
- **Next (needs approval):** the Choice A receiver-aware replay pilot in
  `docs/SFT_ROOT_CAUSE_CHOICE_A_RECEIVER_REPLAY_PROTOCOL.md`, whose thresholds were
  declared before running.
- **Claims:** no root-cause or generalisation claim is made.

**2026-09-29, amendment 11: Choice B reporting corrections (additive; decision receipt v5).**

- **Outcome:** the frozen outcome stays `inconclusive_power`, and no predeclared stopping
  criterion was met.
- **Parse success:** −0.812 points, nominal paired 95% interval [−1.562, −0.125].
  - This is a statistically distinguishable **adverse exploratory secondary** signal.
  - It was not a stopping guardrail, is one of several secondary metrics, and is not
    adjusted for multiplicity.
- **Related downward trends (descriptive):**
  - execution success: −1.06 [−2.25, +0.13];
  - exact-equality validity: −2.44 [−4.94, 0.00];
  - reference validity: −1.50 [−3.88, +0.81].
- **Hypothesis (plausible, unproven, not causal):** value-only masking may remove useful
  structural supervision.
- **Seed variance:** unmeasured. Within-seed intervals do not show that more seeds would
  add little information. Choice A is next because it is cheaper and addresses feasibility.
- **Memory:** 4,128 MiB is PyTorch peak *allocated* memory; about 14,440 MiB is device
  *used* telemetry. They are different quantities.
- **Power:** the planning study used 400 simulations and 2,000 resamples per cell; the
  final analysis used 10,000 resamples.
- **Paths:** future receipts store repository-relative paths. The historical absolute path
  is not rewritten.
