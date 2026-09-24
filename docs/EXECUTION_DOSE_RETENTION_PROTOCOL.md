# Execution-dose and retention experiment — frozen protocol

Frozen 2026-09-24, before the treatment is trained. CPU design and preflight
only. **No GPU training has been launched. Training needs explicit approval.**

## Question

The ordered-trace pilot (commit `a776313`) was a null result. Its only tested
dose was 122 execution examples: 11.9% of examples and 23.1% of supervised
tokens (30,842 of 133,603). Outputs were about 95% unchanged, so that result
cannot separate "execution supervision does not help" from "the dose was too
small to move the model".

This experiment asks one question: **does a materially larger dose of the same
supervision move execution prediction, without collapsing canonical test
generation?**

It does not revisit temporal order. The completed experiment found no order
benefit, so only the ordered representation is used. Its builder
(`scripts/build_execution_trace_ab.py`, `ordered=True`) is unchanged.

## Arms

| arm | what it is | trained here? |
|---|---|---|
| control | the frozen 1,024-example canonical control (adapter `c1fe9e52…`, trained at `767d5a9`) | **no** — reused unchanged |
| dose_treatment | the same 1,024 positions; 256 replaced by verified ordered-trace examples | once, after approval |
| base | untrained Qwen2.5-Coder-1.5B-Instruct | no (descriptive retention reference only) |

The following are held constant:
- **Training setup:** model snapshot `2e1fd397…`, seed 42, learning rate 1e-5, one epoch, batch size 1 × 16 accumulation, 64 optimizer steps, constant-with-warmup schedule (25 steps), 4-bit NF4 with SDPA attention.
- **Budgets:** prompt budgets of 1,024 (function) and 2,048 (repository) tokens, a 3,072-token sequence limit, and no truncation.

The preflight also checks that `engine/`, `config/` and the execution harness are byte-identical to the commit that trained the control. So LoRA, quantisation and regularisation are the control's own settings, not settings claimed to match it.

## Phase gates (this preparation task)

| phase | gate | result |
|---|---|---|
| 1 census | pool materialised from the 385 train lineages only; ≥ 256 unique verified rows | **pass** — 8,854 candidates → 5,578 trace-eligible rows, 1,636 records, 346 lineages; 0 held-out lineages executed; 203 s |
| 2 design | at least one predeclared dose meets every hard constraint | **pass** — 25% feasible; 50% infeasible |
| 3 build | arm invariants, balanced replay, leakage checks, budgets | **pass** |
| 4 retention panel | train-derived, disjoint from both arms and every protected split | **pass** — 613 records, 100 lineages |
| 5 protocol | analysis, evaluators and runner written and tested before training | **pass** |
| 6 preflight | trainer prepares 1,024/1,024 with no drop; token mass ≤ 1.25× control; 64-step plan; retention prompts all within budget | see receipt |

## Dose selection (Phase 2)

Two designs were predeclared: 25% (256 examples) and 50% (512). The rule is
**the largest dose that meets every hard constraint**.

The hard constraints are:
1. Execution rows come from the Phase-1 pool. That means verified, train lineages only, trace ≤ 1,024 tokens, one call per record, and unique targets. The candidate rule is the first pilot's, unchanged.
2. The tier mix reproduces the 12% arm's (simple 37, moderate 46, complex 39, scaled by largest remainder). **The dose changes; the content mix does not.**
3. At most 4 rows per lineage, the O1 dataset's own cap, using the smallest cap that fills. No bug family may exceed 35% of the rows.
4. Balanced replay (below) is satisfiable.
5. No token budget is exceeded.

**50% is infeasible.** At 512 rows the 12% arm's mix needs 164 complex-tier rows. Only 270 complex records exist, in 42 lineages. At caps 1–4 the complex tier runs out at 42, 79, 108 and 132 rows. Meeting it would require concentrating 5–6 rows per lineage, or changing the content mix, which would confound dose with content.

**25% is feasible at a cap of 3** and is selected:

| | 12% arm (failed) | 25% treatment |
|---|---:|---:|
| execution examples | 122 (11.9%) | 256 (25.0%) |
| execution supervised tokens | 30,842 (23.1%) | 73,665 (58.0%) |
| total supervised tokens | 133,603 | 127,108 (control: 105,457) |
| tiers simple / moderate / complex | 37 / 46 / 39 | 78 / 96 / 82 |
| sources MBPP / HumanEval / curated | 77 / 43 / 2 | 199 / 56 / 1 |
| unique lineages | — | 160 (max 3 per lineage) |

The treatment is 2.1× the failed arm's example share and 2.5× its token share.

## Balanced replay and token mass

The treatment keeps the control's canonical row, byte for byte, at each of the other 768 positions. The replaced positions are chosen as follows:
- **Exact quotas:** largest-remainder removal quotas over (source × complexity tier).
- **Bounded loss:** no bug family or execution mode may lose more than 1.15× its proportional share, and each keeps at least one row.

Every source, tier, family and execution mode is retained, including the real-repository rows (284 → 212). Every kept fraction lies between 0.70 and 1.00, against a 0.75 target.

The trainer normalises loss per supervised token over each 16-example window. So an auxiliary task's gradient share is its share of tokens, and the total mass matters. Two levers pull the treatment's mass toward the control's, applied in a fixed order:
1. **Shortest call per record.** Each record uses the verified call with the shortest ordered trace. This does not change which functions or records appear.
2. **Longest canonical rows removed first.** Within each exact replay quota, the longest canonical completions are replaced first. This does not change any count.

Achieved: **1.205× the control's supervised tokens**, measured again by the trainer in the preflight; the hard limit is 1.25.

The residual mismatch is declared. Removing it would require picking short traces *within* a tier, which confounds dose with difficulty. Lever 2 shortens the retained repository completions: SWE-bench complex median 347 → 288 tokens, BugsInPy complex 133 → 91. That shift is recorded in the manifest.

## Outcomes and thresholds (frozen)

**Mechanism gate.** Uses the same 97-item train-derived panel and the same greedy decoding (128 tokens, batch 8) as every earlier pilot. Treatment is compared against the frozen control's existing artifact.

| check | threshold |
|---|---|
| primary outcome | **lenient semantic correctness**, per requested item |
| gain, intended output **and** shown code's actual output | ≥ +5 pp in each |
| paired interval | Newcombe (1998) method-10 score interval, 90%, lower bound ≥ 0, in each condition |
| format guard | strict answer rate (intended) may not fall by more than 2 pp |
| runaway guard | zero generations at the 128-token limit |

The frozen Wald interval of the earlier pilots collapses to [0, 0] when no pairs are discordant. It is replaced *for this new experiment only* by Newcombe's interval, declared before any data exist. The completed experiment's gate is not revisited.

Strict formatting metrics are reported in a separate section and never count as a semantic gain.

**Retention gate.** Canonical Kill@8 on the 613-record train-derived panel. The panel covers every function-mode train record in the 100 pilot-development lineages, at most 10 per lineage by stable rank. Both arms were trained without these lineages. Generation uses the unchanged successor settings: 8 samples, temperature 0.7, top-p 0.9, seed 42, batch 2, and the whole-output parser. Scoring uses the sealed evaluator's pure helpers.

| check | threshold |
|---|---|
| primary | treatment − control Kill@8, paired by record |
| interval | 90% percentile, lineage-cluster bootstrap, 10,000 replicates, seed 20260924 |
| pass | lower bound ≥ −3 pp |
| fail | upper bound < −3 pp |
| otherwise | inconclusive — **does not pass** |

The retention margin is 3 pp absolute. Both gates must pass.

## Result-dependent path (frozen)

| mechanism | retention | outcome | next permitted step |
|---|---|---|---|
| pass | pass | mechanism supported at 25% dose | stop; **request explicit authorization** before opening the 100-lineage confirmation panel |
| pass | fail / inconclusive | gain not accepted | stop; any follow-up needs a new decision |
| fail | any | null at 25% dose. It is recorded as "≥ 5 pp excluded" if both Newcombe upper bounds are < 5, otherwise as inconclusive | stop this line; no dose escalation, extra epochs, mixture change or re-thresholding |

No outcome promotes a model or opens confirmation, validation, ablation_dev,
test or sealed-final data.

## Stopping rules and sequencing (for the approved GPU run)

The GPU jobs run strictly one at a time through `scripts/gpu_run.py`, with no pytest or other heavy job alongside:
1. train `dose_treatment`;
2. run the mechanism evaluation;
3. run retention Kill@8 for control, then treatment, then base;
4. run the frozen analysis on the CPU.

These rules are fixed before the run:
- **Analysis inputs.** The analysis refuses to run on a partial set, so one gate's result cannot decide whether the other gate is measured.
- **Failed training.** A non-finite loss, a dropped example or a step count other than 64 ends the run as a failure. The only permitted retry is an identical rerun after an infrastructure crash that produced no adapter.
- **Selection.** Training loss is never used to select anything.
- **Raw artifacts.** Every runner and evaluator refuses to overwrite an existing raw artifact.
- **Launch gate.** Every runner and evaluator refuses unless HEAD differs from the preflight commit only by the committed preflight receipt, and every bound source still hashes to what the receipt recorded.

**Estimate:** about 8.3 min to train, 1 min for the mechanism evaluation, and about 11 min per retention arm, for roughly 43 min of GPU time in total. New storage is about 190 MB, most of it the treatment checkpoint including resume state. The exact figures are in the receipt.

## Leakage boundary

No step in this preparation opened validation, ablation_dev, test, sealed-final or confirmation data, and the canonical `records.json` stayed closed. Every script opens the hash-verified train shard of the development view only, and refuses unless the corpus manifest certifies group-, semantic-group- and project-disjoint splits. Beyond that:
- the lineage split is reused, not re-drawn, and a fresh re-derivation must equal it;
- treatment lineages are disjoint from the pilot-development and confirmation lineages;
- treatment records are disjoint from both evaluation panels.

Every result from this experiment is train-derived. It is not a generalization, model-selection or final-test result.

## Artifacts

- **Tracked:**
  - `results/v4_3_execution_dose_census.json`
  - `results/v4_3_execution_dose_design.json`
  - `results/v4_3_execution_dose_dataset_manifest.json`
  - `results/v4_3_execution_dose_retention_panel.json`
  - `results/v4_3_execution_dose_preflight_receipt.json`
- **Ignored, bound by hash:** under `results/v4_3_execution_dose_v1/` — the pool (5,859 rows), the treatment arm and its manifest; later, the evaluation artifacts.
