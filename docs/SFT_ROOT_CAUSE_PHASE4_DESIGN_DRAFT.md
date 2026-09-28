# Phase 4 design draft: matched supervision-objective experiment

**Status: DRAFT, not frozen, not executed.** Nothing here has been trained, generated or
frozen. Freezing and launch both require explicit authorisation. Written 2026-09-28,
after the Phase 3 correction
(`results/sft_root_cause_phase3_interpretation_correction_2026-09-28.json`).

## Question

Does changing **only the supervision objective** change fixed-input output prediction at
1.5B? Compare:

- **control:** ordinary whole-test assertion SFT;
- **treatment:** fixed-input verified-output supervision.

Phase 2 showed that arm A's SFT moved input choice and assertion form. Phase 3A could not
resolve whether it moved output prediction, because the result depends on the answer
schema. Phase 3C associates larger Qwen scale with much better output prediction. The
open causal question is whether the objective, rather than the model, is what failed to
teach values at 1.5B.

## Arms (both start from the immutable base)

The base is `Qwen/Qwen2.5-Coder-1.5B-Instruct` @ `2e1fd397…`, 4-bit NF4, bf16 compute.
Neither arm starts from arm A.

| | Control C: whole-test SFT | Treatment T: fixed-input verified-output SFT |
|---|---|---|
| Prompt | production prompt (buggy code and specification) | production prompt plus "Write the test as one assertion using exactly this call: `<call>`" |
| Target | `assert <call> == <value>` | `assert <call> == <value>` |
| Loss | every completion token | the `<value>` tokens and end-of-sequence only |
| (call, value) pairs | the verified killing upstream tests, first 3 per record, as for arm A | **identical pairs** |

The only intended factor is the objective. The model is told which input to use and is
trained only on the value. Two unavoidable differences, both reported:

- T's prompts are longer by about 15 instruction tokens.
- T has fewer loss-bearing tokens.

Declared limitation: in T, "conditioning on a given input" and "value-only loss" change
together as one objective factor. Separating them would need a further arm.

## Held constant

- **Data:** the same records, semantic groups and (call, value) pairs, and the same
  number of optimiser examples.
- **Optimisation:** the same optimiser, learning rate (1e-5, `constant_with_warmup`,
  25 warm-up steps), effective batch 16, 1 epoch, seed 42 and checkpoint cadence.
- **LoRA:** as arm A (r = 16, alpha 32, dropout 0.05, `q_proj/k_proj/v_proj/o_proj`), with
  no replay or regularisation in either arm.
- **Prompt and evaluation:** the same prompt layout and system prompt, and the same
  evaluation protocol.

## Cohorts (from `results/sft_root_cause_phase4_cohort_census.json`)

- **Strict pool** (no arm A training record, in neither Phase 3 cohort): 42 groups, of
  which 33 are feasible for the fixed-input probe. Its 80%-power MDE is about 15.7 points,
  which is **too small for a 5-point gate on its own**.
- **Arm A training groups not in Phase 3:** 401 groups, 352 feasible. Base-initialised
  C and T never saw these groups, so they are lineage-disjoint *for the Phase 4 arms*.
  They are not usable for comparing arm A itself.
- **Proposed partition** of the 352 feasible groups, by a seeded hash before any
  training:
  - **Phase 5 mediator gate:** 150 groups;
  - **Phase 6 confirmation:** 150 further groups, sealed and not inspected until Phase 6;
  - the remaining groups, the 120 Phase 3 exposed-cohort groups and the non-feasible
    arm A groups form the **training pool**.
- **Phase 3 unexposed cohort (120 groups):** a secondary development panel only. It is
  never trained on, and never the sole acceptance or confirmation panel.
- **Strict pool:** a small supplementary panel, reported descriptively.
- **Estimated gate power:** about 2.6 points SE and an 80%-power MDE of about 7 points at
  150 groups with 2 items each. Adding more fixed inputs per function would reduce this
  only partly, because items within a group are correlated.
- **Repository-disjoint cohort:** none is currently available. The A′ acquisition
  admitted 5 of 120 candidates, and native qualification has not been run. The Phase 6
  repository-disjoint requirement therefore **cannot yet be met**, and this is declared
  now.

## Primary mediator and gates (to be frozen before training)

- **Primary:** fixed-input exact-output accuracy on the Phase 5 gate cohort, T minus C,
  as a paired semantic-group cluster bootstrap. It is measured **under both** the
  prefilled-assertion schema and the `ANSWER:` schema.
- **Pass:**
  - lower 95% bound > 0 **and** point estimate ≥ +5 points under **both** schemas;
  - no schema sign flip;
  - the answer-rate non-inferiority gate holds.
- **Why both schemas:** T trains in the prefill format, so a prefill-only gain could be a
  format advantage. Phase 3A showed schema dependence.
- **Downstream measures:** production generation (8 candidates, temperature 0.7) on the
  gate functions. Report Kill@8 and the **original 3-point reference-validity guardrail**,
  plus the supplementary metrics:
  - exact-equality validity;
  - function-level P(at least one valid killing candidate);
  - assertion-form distribution;
  - diversity;
  - nonanswer rate.
- **Mechanistic rule:** if the mediator does not pass, the arm is rejected even if Kill@8
  rises.

## Seeds and budget

- **Seeds:** one seed as a bounded feasibility pilot. More seeds only if the mediator
  passes the gate.
- **Budget:** at most 8 GPU-hours per arm.
  - Arm A's training time was not recorded by `scripts/gpu_run.py`, so a bounded timing
    step (a few optimiser steps) is needed before any launch.
  - Its ten monitor evaluations took 850–1,280 s each. Phase 4 would evaluate only the
    selected checkpoints on the gate cohort.
- **Execution:** durable GPU runs with logs, PID, heartbeat, checkpoints, verified
  resume and completion receipts. One GPU job at a time.

## Data never used

Validation, ablation_dev, consumed test, sealed-final, reserved confirmation and A′
confirmation records are never used. Phase 5 and Phase 6 failures never return to
training.

## Decisions needed before freezing

1. Approve or amend the gate/confirmation/training partition of the 401 remaining arm A
   groups (150/150/rest).
2. Accept that Phase 6 cannot currently include a repository-disjoint cohort, or ask for
   native qualification first.
3. Authorise the timing step and the single-seed pilot, or not.
