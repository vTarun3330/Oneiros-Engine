# Execution-intervention and retention pilot — frozen protocol

First frozen 2026-09-24 and re-frozen the same day after a claims audit and a
token-matched-control study, in both cases before any training. This document
covers CPU design and preflight only. **No GPU training or evaluation has been
launched. Training needs explicit approval.**

**Design class: composite efficacy pilot.** The treatment is a
**25%-example / 58%-supervised-token execution intervention**, compared with
the frozen control:

| measure | treatment | frozen control |
|---|---:|---:|
| execution-supervision examples | 256 / 1,024 (25%) | 0 |
| execution-supervision target tokens | 73,665 / 127,108 (≈ 58.0%) | 0 |
| total supervised tokens | 127,108 | 105,457 |
| treatment ÷ control supervised tokens | 1.205306× | — |

A token-matched control proved infeasible (see below). The following
limitation is therefore frozen and is carried in the analysis output:

> Any observed effect is attributable to the combined 25%-example, 58%-token
> execution-supervision intervention and cannot isolate supervision type from
> supervised-token exposure.

Never write "execution traces caused the improvement". Never attribute a gain
to execution supervision alone while the total-token difference remains.

## Question

The ordered-trace pilot (commit `a776313`) was a null result. Its only tested
intervention was 122 execution examples: 11.9% of examples and 23.1% of
supervised tokens (30,842 of 133,603). Outputs were about 95% unchanged, so that
result cannot separate "execution supervision does not help" from "the
intervention was too small to move the model".

This pilot asks: **does a materially larger execution-supervision intervention
move execution prediction without collapsing canonical test generation?** A
positive answer would show efficacy of the composite intervention. It would not
isolate the mechanism.

It does not revisit temporal order. The completed experiment found no order
benefit, so only the ordered representation is used. Its builder
(`scripts/build_execution_trace_ab.py`, `ordered=True`) is unchanged.

## Arms

| arm | what it is | trained here? |
|---|---|---|
| control | the frozen 1,024-example canonical control (adapter `c1fe9e52…`, trained at `767d5a9`) | **no** — reused unchanged |
| dose_treatment | the same 1,024 positions, with 256 replaced by verified ordered-trace examples | once, after approval |
| base | untrained Qwen2.5-Coder-1.5B-Instruct | no (descriptive retention reference only) |

**Held constant:**
- model snapshot `2e1fd397…`;
- seed 42 and learning rate 1e-5;
- one epoch at batch 1 × 16 accumulation, which gives 64 optimizer steps;
- constant-with-warmup schedule, with 25 warmup steps;
- 4-bit NF4 quantisation with SDPA attention;
- prompt budgets of 1,024 tokens (function) and 2,048 (repository);
- a 3,072-token sequence limit.

The preflight checks that `engine/`, `config/` and the execution harness are
byte-identical to the commit that trained the control, so LoRA, quantisation and
regularisation are the control's own settings.

**Prompt compaction (exact preflight evidence).** The trainer's existing
section-aware prompt compaction (`section_aware_ast_units_before_chat_v4_1`)
applies to both arms, exactly as it did when the control was trained.

| trainer preparation | control | treatment |
|---|---:|---:|
| examples retained / dropped | 1,024 / 0 | 1,024 / 0 |
| malformed prompts admitted | 0 | 0 |
| `prompt_compacted_examples` | 141 | 92 |
| `prompt_truncated_examples` (the trainer's recorded field) | 141 | 92 |
| support/context units removed by compaction | 7,203 | 4,438 |
| target code units dropped | 0 | 0 |

Compaction removes support and context units from long repository prompts. The
trainer records those examples as `prompt_truncated_examples`, and that field is
preserved as recorded. No target code unit and no supervised completion token
is removed.

## Phase gates (preparation)

| phase | gate | result |
|---|---|---|
| 1 census | pool materialised from the 385 train lineages only; ≥ 256 unique verified rows | **pass** — 8,854 candidates → 5,578 trace-eligible rows, 1,636 records, 346 lineages; 0 held-out lineages executed |
| 2 design | at least one predeclared example share meets every hard constraint | **pass** — 25% of examples feasible; 50% infeasible |
| 3 build | arm invariants, balanced replay, leakage checks, budgets | **pass** |
| 4 retention panel | train-derived, disjoint from both arms and every protected split | **pass** — 613 records, 100 lineages |
| 5 protocol | analysis, evaluators and runner written and tested before training | **pass** |
| 6 claims audit | example share, token share, compaction and replay minimum stated exactly | **pass** (this revision) |
| 7 matched control | token-matched replay control within 1%, 2% or 5% | **infeasible at all three** → composite pilot |
| 8 preflight | 1,024/1,024 prepared, none dropped; token ratio ≤ 1.25; 64-step plan; retention prompts within budget | see receipt |

## Design selection

Two designs were predeclared, by example share: 25% (256 examples) and 50%
(512). The rule is **the largest design that meets every hard constraint**:

- **Pool.** Execution rows come from the Phase-1 pool: verified, train lineages only, trace ≤ 1,024 tokens, one call per record, unique targets.
- **Content mix.** The tier mix reproduces the 12% arm's (simple 37, moderate 46, complex 39, scaled by largest remainder).
- **Lineage cap.** At most 4 rows per lineage, using the smallest cap that fills. No bug family may exceed 35% of the rows.
- **Replay.** Balanced replay must be satisfiable.
- **Budgets.** No token budget is exceeded.

**50% is infeasible.** It needs 164 complex-tier rows, but at caps 1–4 the
complex tier runs out at 42, 79, 108 and 132 rows. **25% is feasible at cap 3.**

| | 12% arm (failed) | treatment |
|---|---:|---:|
| execution examples | 122 (11.9%) | 256 (25.0%) |
| execution supervised tokens | 30,842 (23.1%) | 73,665 (≈ 58.0%) |
| total supervised tokens | 133,603 | 127,108 (control: 105,457) |
| tiers simple / moderate / complex | 37 / 46 / 39 | 78 / 96 / 82 |
| sources MBPP / HumanEval / curated | 77 / 43 / 2 | 199 / 56 / 1 |
| unique lineages | — | 160 (max 3 per lineage) |

## Balanced replay (exact result)

The treatment keeps the control's canonical row, byte for byte, at each of the
other 768 positions. The replaced positions follow two rules:
- **Exact quotas.** Removal quotas over (source × complexity tier) are exact, by largest remainder.
- **Bounded loss.** No bug family or execution mode may lose more than 1.15× its proportional share.

**What holds:**
- No dataset, complexity tier, bug family, execution mode or repository category was eliminated.
- Real-repository rows go from 284 to 212 (74.6%).
- Large groups stay close to 75% represented: MBPP 75.1%, HumanEval 75.2%, SWE-bench Verified 74.8%, BugsInPy 74.4%, every tier 74.8–75.3%, function-assertion rows 75.1%.

**What does not hold:** a 75% floor for every subgroup. Kept counts are
integers, so a cell of *n* rows can keep only *k/n*. A 3-row cell keeps 2/3 or
3/3; it can never keep 75%.
- **Observed minimum: 66.7%**, in two 3-row source × tier cells (HumanEval/simple and curated/moderate, 3 → 2 each).
- **Next lowest:** `repository_unittest_fragment` at 70.3% (37 → 26) and the `membership` family at 70.6% (17 → 12).

The builder records the exact minimum and this granularity note in the manifest.

## Supervised token mass

The trainer normalises loss per supervised token over each 16-example window.
So an auxiliary task's gradient share is its token share, and total mass
matters. Two levers pull the treatment toward the control's mass, applied in a
fixed order:
1. **Shortest call per record.** Each record uses its verified call with the shortest ordered trace.
2. **Longest canonical rows first.** Within each exact replay quota, the longest canonical completions are replaced first.

The treatment still carries **1.205306× the control's supervised tokens**. The
hard ceiling is 1.25. Lever 2 also shortens the retained repository completions:
SWE-bench complex median 347 → 288 tokens and BugsInPy complex 133 → 91. The
manifest records this.

## Token-matched control study (infeasible)

**Goal.** A fresh control with no execution supervision and 1,024 unique
examples. It would preserve representation, stay disjoint from both evaluation
panels and every protected split, and match 127,108 supervised tokens within 1%,
2% or 5%. It would also avoid model outcomes, padding and repetition. See
`results/v4_3_execution_dose_matched_control.json` and
`scripts/build_execution_dose_matched_control.py`.

**The only valid construction.** To differ from the treatment only through the
intervention, the control must share the treatment's 768 replay rows. Every
change must therefore happen at the 256 intervention positions. Each of those
positions holds the frozen control's row, or an unused gold test from the
identical source × tier × mode × bug-family cell:
- no model outcomes are used;
- the frozen control's lineage caps and token limits apply;
- prompts must be compactable within budget;
- rows carrying exact or whitespace-normalised duplicates are rejected.

| tolerance | feasible | supervised tokens | ratio to treatment | rows swapped |
|---|---|---:|---:|---:|
| 1% | **no** | 108,694 | 0.8551 | 78 |
| 2% | **no** | 108,694 | 0.8551 | 78 |
| 5% | **no** | 108,694 | 0.8551 | 78 |

**The best attempt still fell short.** Representation was identical to the
frozen control, with 336 lineages and at most 4 rows per function lineage. There
were 0 exact and 0 near duplicates, and no evaluation-panel lineage. Preparation
dropped nothing, compacted 141 prompts, removed 7,296 support units and 0 code
units. Even so, it reaches only 0.855.

**The ceiling is structural, not a search artefact.** A conflict-free upper
bound (longest candidates onto shortest rows, ignoring lineage caps) is **0.890**
of the treatment's mass. The 5% band needs at least 120,753 tokens.
- **Why:** the 256 positions already hold the control's longest same-cell rows. The remaining gold tests within the frozen token limits are shorter.
- **Why the relaxed bound doesn't help:** swapping rows at any of the 1,024 positions could reach 1.166 in principle. But a swap at a shared replay position must appear in both arms, which adds equal mass to each. Made in the control alone, it changes the replay, and the arms would no longer differ only by the intervention.

**Consequence.** No matched control is fabricated. The frozen control stays the
comparator, and the pilot is a composite efficacy pilot, bound by the inference
limitation above.

## Outcomes and thresholds (frozen)

### Mechanism gate

The panel is the same 97 train-derived items as every earlier pilot, with the
same greedy decoding (128 tokens, batch 8). The treatment is compared against
the frozen control's existing artifact.

| check | threshold |
|---|---|
| primary outcome | **lenient semantic correctness**, per requested item |
| gain, intended output **and** shown code's actual output | ≥ +5 pp in each |
| paired interval | Newcombe (1998) method-10 score interval, 90%, lower bound ≥ 0, in each condition |
| format guard | strict answer rate (intended) may not fall more than 2 pp |
| runaway guard | zero generations at the 128-token limit |

Strict formatting metrics are reported separately and never count as a semantic
gain. Newcombe replaces the Wald interval for this new pilot only, because the
Wald interval degenerates to [0, 0] without discordant pairs. The completed
experiment's gate is not revisited.

### Retention gate

Canonical Kill@8 on the 613-record train-derived panel. It contains every
function-mode train record in the 100 pilot-development lineages, at most 10
per lineage. Both arms were trained without these lineages. Generation uses the
unchanged successor settings: 8 samples, temperature 0.7, top-p 0.9, seed 42,
batch 2, and the whole-output parser.

| check | threshold |
|---|---|
| primary | treatment − control Kill@8, paired by record |
| interval | 90% percentile, lineage-cluster bootstrap, 10,000 replicates, seed 20260924 |
| pass | lower bound ≥ −3 pp |
| fail | upper bound < −3 pp |
| otherwise | inconclusive — **does not pass** |

**No optional stopping.** Both gates are always measured: the mechanism
evaluation and all three retention runs execute even if the mechanism gate
fails, and the analysis refuses to run on a partial set. This is predeclared
and is not to be changed to save runtime.

## Result-dependent path (frozen)

| mechanism | retention | outcome label | next permitted step |
|---|---|---|---|
| pass | pass | `composite_intervention_mechanism_and_retention_passed` | stop; **request explicit authorization** before opening the 100-lineage confirmation panel |
| pass | fail / inconclusive | `composite_intervention_gain_with_retention_<verdict>` | stop; gain not accepted |
| fail | any | `composite_intervention_null_5pp_gain_excluded` if both Newcombe upper bounds are < 5, else `composite_intervention_null_inconclusive` | stop this line; no escalation, extra epochs, mixture change or re-thresholding |

Every outcome carries the intervention label and the inference limitation. The
analysis sets `causal_attribution_to_execution_supervision_permitted: false`.
No outcome promotes a model or opens confirmation, validation, ablation_dev,
test or sealed-final data.

## Stopping rules and sequencing (for the approved GPU run)

The jobs run strictly one at a time through `scripts/gpu_run.py`, with no
pytest or other heavy job alongside:
1. train `dose_treatment`;
2. run the mechanism evaluation;
3. run retention Kill@8 for the control, then the treatment, then the base;
4. run the frozen analysis on the CPU.

**Stopping rules:**
- **Failed training.** A non-finite loss, a dropped example or a step count other than 64 ends the run as a failure.
- **Retries.** The only permitted retry is an identical rerun after an infrastructure crash that produced no adapter.
- **Selection.** Training loss is never used to select anything.
- **Raw artifacts.** Every runner and evaluator refuses to overwrite an existing raw artifact.
- **Launch gate.** Every runner and evaluator refuses unless HEAD differs from the preflight commit only by the committed preflight receipt, and every bound source still has its recorded LF-canonical hash.

**Estimate** (unchanged by this revision, because no extra arm is trained): about 8.3 min training, 1 min mechanism, about 11 min per retention arm × 3; about 42 min of GPU time and about 190 MB of new storage. The exact figures are in the receipt.

## Leakage boundary

**Nothing protected was opened.** No step opened validation, ablation_dev,
test, sealed-final or confirmation data, or the canonical `records.json`. Every
script opens only the hash-verified train shard of the development view, and
refuses unless the corpus manifest certifies group-, semantic-group- and
project-disjoint splits.

**Disjointness:**
- The lineage split is reused, not re-drawn.
- Treatment lineages are disjoint from the pilot-development and confirmation lineages.
- Treatment records are disjoint from both evaluation panels.
- The matched-control candidates exclude the same lineages.

All results are train-derived: not generalization, model selection or
final-test results.

## Artifacts

**Tracked:**
- `results/v4_3_execution_dose_census.json`
- `results/v4_3_execution_dose_design.json`
- `results/v4_3_execution_dose_dataset_manifest.json`
- `results/v4_3_execution_dose_matched_control.json`
- `results/v4_3_execution_dose_retention_panel.json`
- `results/v4_3_execution_dose_preflight_receipt.json`

**Ignored, bound by hash:** under `results/v4_3_execution_dose_v1/`, the pool,
the treatment arm and its manifest. No matched-control arm exists, because none
was feasible.
