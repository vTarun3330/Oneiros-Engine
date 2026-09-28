# Phase 4 design draft V2: a single-factor loss-mask experiment

**Status: DRAFT V2, not frozen, not executed.** No cohort has been frozen, no training
manifest has been emitted, and no training, timing, generation or evaluation job has run.
It was written on 2026-09-28 at the end of the CPU-only Phase C.

**It supersedes** `docs/SFT_ROOT_CAUSE_PHASE4_DESIGN_DRAFT.md` (commit `2e7ddba`). That
draft is kept byte-identical as history. Its supersession is recorded here, in protocol
amendment 5 and in `results/sft_root_cause_state.json`, not by editing it.

## Why V1 was invalid

The V1 draft changed **two** factors at once:

1. whether the prompt supplies the call/input;
2. whether the loss covers the full completion or only the value.

Its control-versus-treatment contrast therefore could not attribute an effect to either
factor. It was not a single-factor experiment.

## Primary design: two arms, only the loss mask differs

Both arms start from the same immutable base, `Qwen/Qwen2.5-Coder-1.5B-Instruct` @
`2e1fd397ee46e1388853d2af2c993145b0f1098a`.

| | Control C | Treatment T |
|---|---|---|
| Prompt | fixed-call prompt: production system and user prompt plus "Write the test as one assertion using exactly this call: `<call>`" | **identical** |
| Pair | verified (call, value) | **identical** |
| Target text | `assert <call> == <value>` + EOS | **identical** |
| Labels | every completion token (assertion keyword, call, `==`, value, EOS) | **only** the complete value span and EOS; prompt, `assert`, call and `==` tokens masked |

**Identity matrix: identical in C and T.**

- **Model and tokens:** base model name and immutable revision; tokenizer and its
  immutable revision; chat template; system prompt; user prompt; call; expected value;
  target text; `input_ids`; `attention_mask`; unmasked sequence length.
- **Data:** example order; record IDs; semantic groups; group weights; number of
  examples.
- **Optimisation:** optimiser; learning rate (1e-5); scheduler (`constant_with_warmup`,
  25 warm-up steps); effective batch 16 (gradient accumulation 16 × batch 1); epochs;
  optimiser steps; seed.
- **Adapter and runtime:** LoRA modules (`q_proj/k_proj/v_proj/o_proj`), rank 16,
  alpha 32, dropout 0.05; 4-bit NF4 quantisation; bf16 precision.
- **Procedure:** checkpoint cadence; evaluator and evaluation settings.

**Intentionally different: the label tensor only.**

**Implementation and proof.** `harness/objective_masking.py` builds one token sequence
and derives both label tensors from it. The value span is located from character offsets
by position, never by substring search. An example is refused if a token straddles `==`
and the value, or if the last value token does not end at the value's end.

`tests/test_objective_masking.py` uses the pinned tokenizer and synthetic data only. It
proves:

- identical `input_ids`, attention masks and target text;
- that only the labels differ;
- that control supervises every completion token;
- that treatment supervises exactly the value span and EOS, with the decoded supervised
  labels shown in any failure message.

The tested values are:

- a positive integer, a negative integer, a float, a boolean and `None`;
- an escaped string, a Unicode string and an empty string;
- a list, a tuple, a dictionary and a nested container;
- a multi-token value;
- a call containing the value's own literal text, and a call whose argument equals the
  value.

**Declared limitation.** Supervised token counts necessarily differ, because that *is*
the manipulation. The input and optimiser exposure are identical.

## Optional three-arm decomposition (documented, NOT launched, needs separate authorisation)

| Arm | Prompt | Loss |
|---|---|---|
| C0 | production prompt (model chooses the input) | full completion |
| C1 | fixed-call prompt | full completion |
| T | fixed-call prompt | value span and EOS only |

- C1 − C0 estimates the effect of fixed-call conditioning.
- T − C1 estimates the effect of value-only loss masking.
- C0 versus T alone is **not** a single-factor contrast and is never reported as one.

## Pre-launch matched-manifest preflight (C4)

The validator exists, and its rules are tested on toy fixtures. No real manifest is
frozen or emitted in this task, because the cohort choice (A or B) is unresolved.
Before any launch, the concrete manifest must prove that:

- every row has a verified executable (call, value) pair that passes on the reference,
  checked in the restricted worker;
- every call discriminates buggy from fixed behaviour where required;
- no repository fragment or other non-function record is included without a valid
  training representation;
- no gate or confirmation group appears in training;
- no repeated row inflates counts, and semantic-group caps are enforced;
- source, family and complexity distributions are reported;
- token counts are reported for both arms;
- input-sequence hashes are identical across arms and label-mask hashes differ;
- examples, optimiser steps and ordering are matched.

## Checkpoint and evaluation rules (to be frozen before launch)

- **Checkpoint:** the final checkpoint of the fixed schedule (the last optimiser step).
  There is no gate-driven checkpoint selection and no cherry-picking; intermediate
  checkpoints are never evaluated on the gate.
- **One look:** the gate cohort is evaluated once per arm.
- **Answer-rate gate:** answer-rate non-inferiority between arms (T ≥ C − 2 points) at
  each schema before any accuracy is read.
- **Both schemas:** the mediator is measured under the prefilled-assertion schema and
  the `ANSWER:` schema.
- **Primary mediator:** fixed-input exact-output accuracy, T minus C, as a paired
  semantic-group cluster bootstrap on exactly paired items.
- **Downstream measures:**
  - Kill@8 from production generation (8 candidates, temperature 0.7);
  - the **original 3-point reference-validity guardrail**;
  - exact-equality validity;
  - function-level P(at least one valid killing candidate);
  - assertion-form distribution;
  - diversity and duplication;
  - nonanswer rate.
- **Outcomes** (formal rules in the Phase D power analysis):
  - *Confirmatory 5-point pass:*
    - the effect is in the correct direction under both schemas, with no schema sign
      flip;
    - the answer-rate gate passes;
    - the declared lower confidence bound exceeds +5 under **both** schemas;
    - the validity and diversity guardrails pass.
  - *Exploratory pilot outcomes:* `promising`, `harm` or `inconclusive_power`. An
    underpowered non-pass never automatically rejects the intervention.
  - *Rollback/rejection:* treatment is rejected if the mediator shows harm (upper
    bound < 0 under both schemas) or the validity guardrail fails.
- **Seeds:** one seed is a feasibility pilot only and cannot support a final efficacy
  or generalisation conclusion.

## Cohort and confirmation (unresolved; see `results/sft_root_cause_phase4_power_analysis_v1.json`)

- The cohort requires the user's Choice A or Choice B. The earlier 150/150/rest proposal
  is **not** frozen.
- A repository-disjoint Phase 6 cohort does not exist. No internal split of arm A's
  source distribution can substitute for it.
