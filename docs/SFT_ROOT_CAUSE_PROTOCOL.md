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
