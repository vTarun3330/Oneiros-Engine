# Oneiros — research status after the sealed-final incident

**Written 2026-09-17; section 7 added 2026-09-24.** This is the current
standing of every result in the project. Where a figure is quoted it is quoted
from a committed receipt, not recomputed. Nothing here changes a number.

**The one-sentence version:** the sealed final test was attempted, failed after
authorization with zero output, and is consumed; the project's evidence is
locked validation and development evaluation, and it does not support a
final-test claim of any kind.

Incident account: [`SEALED_FINAL_INCIDENT.md`](SEALED_FINAL_INCIDENT.md).

---

## 1. Established results

Measured under the frozen successor protocol, seed 42, 8 candidates,
`whole_output` parsing, on `Qwen/Qwen2.5-Coder-1.5B-Instruct @ 2e1fd397`.

### Locked validation — the project's strongest evidence

One measurement on `val` (757 function records), decision rule frozen before
the run.

| arm | Kill@8 | functions killed |
|---|---|---|
| immutable base | 0.594452 | 450 / 757 |
| Arm A @ checkpoint 431 | 0.628798 | 476 / 757 |

Paired: **+114 gained, −88 lost, net +26**, Kill@8 **+3.4346 points**,
exact two-sided McNemar **p = 0.0783**.

**Decision: retain the base model.** The required threshold was p < 0.01.
Criterion 1 failed on significance. Criterion 5 failed on a contract-identity
defect of my own making, documented in
[`LOCKED_VALIDATION_INTEGRITY_DEFECT.md`](LOCKED_VALIDATION_INTEGRITY_DEFECT.md)
and recorded as a failure rather than reinterpreted.

Receipt: `results/v4_2_locked_validation_result_receipt.json`
(`f7cdf84df8e65c51…`).

### Development evaluation — selection only, not generalization

Four arms on `ablation_dev` (542 function records).

| arm | Kill@8 |
|---|---|
| base | 0.605166 |
| Arm A @ 150 | 0.690037 |
| Arm A @ 431 | 0.695572 |
| Arm B @ 150 | 0.658672 |

Primary comparison B@150 vs A@150: **−3.14 points**, +44/−61, net −17,
p = 0.118. **Arm B and the 15.066% O1 sidecar were rejected.**

Receipts: `results/v4_2_frozen_development_evaluation_receipt.json`
(`062774028693cfcb…`), `results/v4_2_development_selection_receipt.json`
(`0188324fe9eb2a51…`).

### Two methodological findings that hold independently

- **Development overstates arms by roughly 3.2×.** +10.5 points on
  `ablation_dev` became +3.3 on locked `val`, with non-overlapping ranges. Any
  development figure is a selection statistic, not a generalization estimate.
- **Every SFT arm cost reference validity.** −11.87 points on the locked split
  (0.4521 → 0.3334); functions with any reference-valid candidate fell 675 → 566.
  The same regression appeared in all four development arms.

---

### The execution path is demonstrated at full scale (2026-09-17)

**The full-scale permitted-split execution path is now demonstrated: admission,
scoping, base-model loading, generation, whole-output parsing, safe execution,
raw-output integrity, progress artifacts, and scoring reproduced the historical
base result exactly. This validates pipeline operation, not model quality.**

One GPU operational rehearsal on `ablation_dev`, base Qwen at the immutable
revision, no adapter, seed 42, 542 targets, 4,336 candidates, 577.8 s, exit 0.
Kill@8 came out at **0.605166** — bit-for-bit the figure the four-arm
development evaluation recorded for the base model on the same split.

That exact match is the finding, and it is a finding about *code*: the rehearsal
path is not a lookalike of the measured pipeline, it computes the same number.
It says nothing about the model. `ablation_dev` selected the checkpoints it
scored, so the rehearsal's Kill@k values are operational facts and **not** a
generalization result, a model-performance result, a basis for selection, an
Oneiros-versus-Atheris comparison, or a final-test result.

This is what the sealed final test never got: the path executed end to end, on
real data, at full scale, before anything irreversible depended on it.

Sanitised evidence: `results/v4_2_rehearsal_execution_receipt.json`. The raw
artifact is retained locally, Git-ignored, and referenced by SHA-256
`3d72891452c586820300079ae83bf998415975eb808bea425e1a0a4d859e5c77`.

## 2. Inconclusive results

- **Whether SFT generalizes.** Locked validation produced a **positive but not
  significant** gain (+3.43 points, p = 0.0783). That is not evidence of no
  effect; it is insufficient evidence of an effect at the predeclared
  threshold. The honest reading is *undetermined at this sample size*, and it
  cannot be resolved by re-measuring on `val` — that split has already been
  used for selection.
- **Real-repository performance.** No repository-native execution was ever
  performed. Repository records are excluded from every reported kill rate. No
  real-repository figure is claimed.
- **HumanEval figures.** Carry the prompt-copying caveat throughout.

---

## 3. Invalidated / unavailable: final-test evidence

**There is no final-test evidence, and there will be none from the `test`
split.**

| | |
|---|---|
| attempted | 2026-09-16, 18:11:52 UTC |
| status | **`failed_after_authorization`** |
| sealed candidates generated | **0** |
| reportable metrics | **0** |
| authorization | spent, `may_never_run_again: true` |
| receipt / bundle | `4a2bc52a…` / `c16b13eb…` |

The evaluator loaded the corpus and traversed sealed records, then raised while
admitting them — before any prompt was rendered, any sequence generated, or any
candidate scored. It produced **zero sealed candidates and zero reportable
metrics**. The cause was a loader defect, not the data: the sealed
loader required `entry_point` and `specification` to be non-empty on every
record, while repository-style records carry them blank by design and the
established evaluator has always loaded the whole split and *then* scoped the
kill rate to function-mode records. The same rejection reproduces on `val` (27
records) and `ablation_dev` (52 records).

**The consumed split must not be rerun** — not with a corrected loader, not
under a new receipt, not under any circumstances. A split that has been opened
is no longer a held-out split, regardless of what it produced.

**Consequently:**

- No final-test Kill@1, Kill@2, Kill@4 or Kill@8 exists.
- **No final-test Oneiros claim is supported.**
- **No Oneiros-versus-Atheris claim is supported**, and none would have been
  even on success: the run was scoped to the immutable base model only, with no
  adapter and no bundled baseline. The Atheris figures elsewhere in the project
  are development-panel comparisons and were never final-test evidence.

Evidence: [`evidence/sealed_final_incident/`](evidence/sealed_final_incident/),
hashes recorded in the incident report.

---

## 4. Known limitations

1. **No model was promoted.** The reported candidate is the untrained base
   model. Nothing supports a claim of robust Oneiros SFT generalization.
2. **Both held-out panels are used up.** `val` is spent on selection; `test` is
   consumed by the failed attempt. The project has no unused evaluation set.
3. **Repository records are excluded from every kill rate**, so all reported
   figures are function-level only.
4. **The defect class is not fully closed.** Six receipt generations and three
   audits hardened the receipt layer; the failure was one layer below, in a
   component deliberately designed to be unreachable before authorization — and
   that design is what kept it untested. The same load-everything-then-validate
   shape exists elsewhere (see item 4 of
   [`LOCAL_WINDOWS_GPU.md`](LOCAL_WINDOWS_GPU.md)).
5. **`results/v4_2_final_research_summary.md` is generated.** Its
   post-incident amendment is hand-written and would be dropped by a
   regeneration of that file.

---

## 5. The decision needed

**A new, independently constructed final set is required for any future final
measurement. None has been created, and none will be without an explicit
decision by the project owner.**

What that decision has to settle:

1. **Whether to have a final measurement at all.** Reporting on locked
   validation alone is a legitimate outcome, and locked validation has already
   decided the selection question.
2. **Where an independent set comes from.** `train`, `ablation_dev` and `val`
   are all selection-contaminated; `test` is consumed. A new set must be drawn
   from sources disjoint from all four, with disjointness *proved* by the
   existing near-duplicate machinery rather than asserted.
3. **Whether the protocol gains a full dress rehearsal gate.** The failure
   happened because the pre-authorization smoke proved the *generator* on two
   synthetic records, not the *pipeline* on a real split. A rehearsal that runs
   every stage end-to-end on a permitted split, at full scale, and whose result
   is discarded, is the gate that was missing.
4. **Whether the token becomes two-phase.** Today it is spent at the moment of
   access, so any downstream failure burns it. Reserving on open and committing
   only once the first batch has scored would have made this incident
   recoverable.

None of this is implemented. No replacement evaluator has been written and no
new split has been created or evaluated.

---

## 6. Where the evidence lives

| artifact | sha256 |
|---|---|
| `results/v4_2_frozen_development_evaluation_receipt.json` | `062774028693cfcbf5346eafeed232027748720940af7a47b5ac1a591055c4fe` |
| `results/v4_2_development_selection_receipt.json` | `0188324fe9eb2a51eebef71bbd4cc3ed45584f12393bf0fa6539817a09e58969` |
| `results/v4_2_locked_validation_preflight.json` | `97be6855d41b1186a5c8a59a8a89efa71e1f40f5e1c7f29ab387402b7598912c` |
| `results/v4_2_locked_validation_result_receipt.json` | `f7cdf84df8e65c51bd22410759c6fd54952ace3bfa80c6ab620674b53e288af9` |
| `results/v4_2_sealed_final_executable_receipt_v6.json` | `4a2bc52ae1448e12ba31fd2ad68ff0b26b54dd9507456574ea66cb3936af7c8e` |

Sealed-final receipts v1–v5 are retained as superseded evidence with
`.SUPERSEDED.md` markers; all are refused by the entrypoint by schema version.
The guard state files under `results/` are uncommitted by design; redacted
copies are committed under `docs/evidence/sealed_final_incident/`.

---

## 7. Why fine-tuning did not help: the mechanism chain (2026-09-19 → 2026-09-24)

Five diagnostics, each gated before its data was generated. All five ran on
**train-derived** panels only. None opened `val`, `ablation_dev`, `test`, the
sealed split or the 100-lineage confirmation panel, and **none is a
generalization, model-selection or final-test result**.

| step | question | predeclared gate | verdict |
|---|---|---|---|
| Gate 1 — TAP, base vs A@431 | did ordinary SFT change output prediction? | TOST, ±5 pp | not equivalent — driven by non-answers |
| Gate 2 — TAP, 7B vs 1.5B base | is model capacity the ceiling? | TAP-mut 95% lower bound > 0; answer-rate non-inferiority | **fail** |
| Pilot 1 — prompt-only execution supervision | does re-framing 128 of 1,024 examples as output prediction help? | +5 pp intended | **fail** |
| Pilot 2 — execution events, unordered vs ordered | does execution-event supervision, or its temporal order, help? | +5 pp in both conditions, 90% lower bound ≥ 0 | **fail, both arms** |
| Pilot 3 — composite execution intervention | does a 25%-example / 58%-supervised-token execution intervention move execution prediction without losing test generation? | mechanism +5 pp in both conditions, Newcombe 90% lower ≥ 0; Kill@8 retention, cluster-bootstrap 90% lower ≥ −3 pp | **mechanism fail; retention inconclusive** |

TAP asks a model to state what a call returns, given the correct code (TAP-ref)
or the mutated code plus specification (TAP-mut). Primary panel: 584 train items;
16 pilot items excluded as design data.

### Gate 1 — ordinary SFT (A@431) versus base

Per-requested accuracy, TAP-ref, with A@431 given its own training system
prompt: **48.1% base vs 32.0% A@431, −16.10 pp, 95% CI [−20.08, −12.11]**. The
loss is concentrated in non-answers (36 vs 201 of 584). On the 363 items both
answered, the difference is −1.10 pp and equivalent within ±5 pp. That subset
is self-selected, so it is diagnostic only.

**Reading:** an execution-reasoning gain from SFT was not demonstrated. SFT
damaged output-contract adherence, most severely on inputs that resemble its
training task: median output length on TAP-mut was 293 characters against
49 for base.

### Gate 2 — 7B base versus 1.5B base

TAP-mut per-requested: **9.2% vs 43.8%, −34.59 pp, 95% CI [−39.12, −30.06]**;
answer rate 15.6% vs 95.0%. TAP-ref: 10.8% vs 48.1%. Gate failed.

**Reading:** the 7B model ignored the assertion-only output contract on most
items. This rules out scaling *under the deployed contract*. It is **not**
evidence about 7B semantic capacity, because its worst-case bounds are wide.

### Pilot 1 — prompt-only execution supervision

The treatment re-framed 128 of 1,024 training examples as output prediction,
with every completion byte identical to control. On the 97-item train-derived
mechanism panel, strict intended accuracy moved from 1/97 to 4/97 (+3.09 pp,
90% CI [+0.20, +5.98]), below the +5 pp gate. Lenient intended accuracy:
**base 25/97, control 26/97, treatment 25/97**. The answer-rate gain (+10.3 pp)
was a formatting change.

### Pilot 2 — execution events, temporal order removed vs preserved

Both arms: 1,024 examples, of which 122 (11.9%) are verified execution-output
examples carrying an identical event multiset. Events are sorted in one arm and
kept in true execution order in the other. 64 optimizer steps; supervised token
mass 133,605 vs 133,603. Judged against Pilot 1's control on the same panel.

| lenient, per requested (n = 97) | control | unordered | ordered |
|---|---:|---:|---:|
| intended output | 26 | 26 | 25 |
| shown actual output | 23 | 25 | 25 |

| paired comparison | Δ pp | 90% CI (frozen Wald) | gained / lost |
|---|---:|---:|---:|
| unordered − control, intended | 0.00 | [0.00, 0.00] | 0 / 0 |
| unordered − control, shown actual | +2.06 | [−0.31, +4.44] | 2 / 0 |
| ordered − control, intended | −1.03 | [−2.72, +0.66] | 0 / 1 |
| ordered − control, shown actual | +2.06 | [−0.31, +4.44] | 2 / 0 |
| ordered − unordered, intended | −1.03 | [−2.72, +0.66] | 0 / 1 |
| ordered − unordered, shown actual | 0.00 | [0.00, 0.00] | 0 / 0 |

Neither arm passed. Strict answer rate was about 1% in every arm: most answers
are wrapped in Markdown bold (67–80 of 97).

### Post-gate diagnosis (exploratory — does not alter any gate)

- **The intervention barely moved the model.** Raw outputs were byte-identical
  across arms on 92.8–96.9% of items.
- **The model answers from a behavioural prior, not by executing the shown
  code.** On the 71 items where the shown code's result differs from the
  intended one, control was asked what the *shown* code returns. It gave the
  intended value 21 times and the shown code's actual value 14 times; 32
  answers matched neither and 4 had no usable answer. A model that executed the
  code would almost always give the actual value. Both trace arms shifted this
  by two items (16 actual). This independently reproduces the original TAP
  finding: 63.2% identical predictions whether shown correct or mutated code.
- **Interval robustness.** The frozen paired Wald interval collapses to [0, 0]
  when a comparison has no discordant pairs. Unordered/intended passed its
  "lower bound ≥ 0" check this way, and that pass is not evidence of a
  direction. Re-checked with Newcombe's paired score interval and an exact
  one-sided bound on the gain rate:
  - *intended output, and ordered vs unordered in both conditions:* a +5 pp
    gain is excluded by all three methods (exact gain bound ≤ 3.04 pp);
  - *shown actual output vs control:* +5 pp is excluded by Wald (+4.44) and
    Newcombe (+4.86) but **not** by the exact gain bound (+6.35). A small
    benefit there cannot be firmly ruled out on 97 items.

### What the chain establishes

1. No fine-tuning variant tested improved output prediction. On the 97-item
   panel, none beat the untrained base model.
2. Temporal order added nothing measurable over the unordered event multiset.
   A ≥ 5 pp order effect is excluded under every interval method.
3. Output-contract adherence (Markdown wrapping, non-answers) is the dominant
   *measured* failure in every step.

### What it does not establish

- That execution supervision cannot work. Only one dose was tested: 122
  examples, which were 11.9% of the 1,024 examples and 23.1% of the supervised
  tokens (30,842 of 133,603), with 64 steps and LoRA on 1.5B. Outputs were ~95%
  unchanged, so the result cannot separate "the signal does not help" from
  "the dose was too small to move the model".
- Anything about 7B semantic capacity.
- Anything about generalization. Every panel here is train-derived.

**Decision.** This intervention is stopped and the prior control is retained as
the reference. The 100-lineage confirmation panel remains unopened, and no
canonical Kill@8 was run for either trace arm. No further training, extra
epochs, mixture change or re-thresholding of this intervention will be run
without an explicit new decision.

### Proposed paper wording

Use this wording, or something no stronger. It reports null results at the
two tested intervention sizes. It does not claim that execution supervision
generally fails, and it never attributes an effect to supervision type alone.

> In two train-derived mechanism pilots on Qwen2.5-Coder-1.5B, we replaced part
> of a 1,024-example fine-tuning mixture with verified execution-trace
> supervision.
>
> In the first pilot it made up 12% of examples and 23% of supervised tokens.
> In the second it made up 25% of examples and 58% of supervised tokens. The
> second pilot also carried 1.2× the control's total supervised tokens, so it
> is a composite intervention.
>
> Neither pilot reached the predeclared +5 pp gain in intended-output or
> actual-output prediction on a 97-item panel. The largest change was +3.1 pp
> (3 items gained, 0 lost). A ≥ 5 pp gain on actual-output prediction could
> not be firmly excluded. Preserving temporal order gave no benefit over an
> unordered event multiset. About 94–95% of model outputs were unchanged by
> training.
>
> In the larger intervention, canonical test-generation Kill@8 moved by
> −0.3 pp, but noninferiority within 3 pp could not be confirmed.
>
> We therefore do not conclude that execution supervision fails in general.
> The second pilot cannot separate supervision type from total supervised-token
> exposure. All panels were derived from training data, so none of these
> results is a generalization estimate.

Avoid these phrasings:
- "execution supervision does not help";
- "trace training fails";
- "fine-tuning cannot teach execution".

Also avoid any wording that drops the dose, the model size, or the fact that
the panels are train-derived.

### Pilot 3 — composite execution intervention (run 2026-09-24)

The protocol is in
[EXECUTION_DOSE_RETENTION_PROTOCOL.md](EXECUTION_DOSE_RETENTION_PROTOCOL.md),
frozen at `b65ade4` before training.

**The treatment** was a **25%-example / 58%-supervised-token execution
intervention**:
- 256 of 1,024 examples;
- 73,665 of 127,108 supervised target tokens;
- 1.205306× the frozen control's 105,457 supervised tokens.

A token-matched control was infeasible at 1%, 2% and 5%, so the pilot is a
**composite efficacy pilot**. Any effect would be attributable only to the
combined intervention. It cannot isolate supervision type from supervised-token
exposure.

**Training** (run `20260924-194214-…`, 492.6 s):
- 1,024 of 1,024 examples retained, 0 dropped, 0 malformed;
- 92 prompts compacted (the trainer's `prompt_truncated_examples` = 92), 0 code units dropped;
- 64 of 64 steps, adapter `2c094981…`.

**Mechanism gate — fail.** The panel is the same 97 train-derived items, scored
by lenient semantic correctness against the frozen control.

| condition | control | treatment | Δ pp | gained / lost | Newcombe 90% | exact McNemar p |
|---|---:|---:|---:|---:|---:|---:|
| intended output | 26 | 25 | −1.03 | 0 / 1 | [−3.19, +1.02] | 1.00 |
| shown actual output | 23 | 26 | +3.09 | 3 / 0 | [+0.04, +6.34] | 0.25 |

- The strict answer rate was unchanged (1.03% in both arms).
- There were 0 completion-limit hits.
- Against the failed 12% ordered arm, the change was 0.00 pp on intended output and +1.03 pp on shown actual output.

**Retention gate — inconclusive, which does not pass.** Canonical Kill@8 on the
613-record train-derived panel:

| arm | Kill@1 | Kill@2 | Kill@4 | Kill@8 |
|---|---:|---:|---:|---:|
| control | 0.282 | 0.382 | 0.515 | 0.633 |
| treatment | 0.245 | 0.380 | 0.499 | 0.630 |
| base | 0.248 | 0.388 | 0.520 | 0.626 |

- Treatment − control Kill@8 was −0.33 pp, with 64 records gained and 66 lost.
- The lineage-cluster bootstrap 90% interval was [−3.55, +2.93]. Its lower bound fell below the −3 pp margin.
- The record-level Newcombe interval was [−3.38, +2.73].
- There were no prompt-budget failures in any arm.

**Frozen outcome:** `composite_intervention_null_inconclusive`.
- It is "inconclusive" rather than "≥ 5 pp excluded" because the shown-actual upper bound (6.34 pp) exceeds 5.
- On intended output, a +5 pp gain is excluded: the upper bound is 1.02 and the exact gain bound is 3.04.

**Exploratory, non-gating:** 91 of 97 mechanism outputs (93.8%) were
byte-identical between control and treatment in each condition. That holds even
with 58% of supervised tokens coming from execution supervision.

**Decision.** This intervention line stops, and the frozen control remains the
reference. The following will not happen without an explicit new decision:
- dose escalation;
- extra epochs;
- a mixture change;
- re-thresholding;
- promotion;
- opening confirmation or validation.

**What this adds to the chain.**
- Raising the execution share from 12% of examples (23% of tokens) to 25% (58%) still left the model's outputs almost unchanged, and it did not reach the predeclared gain.
- Canonical test generation was not measurably damaged, but its retention could not be confirmed within the 3 pp margin.
- The result does **not** show that execution supervision cannot work: other representations, longer training, larger models, or a design that can isolate supervision type were not tested.

### Evidence (sha256 of the committed bytes)

| artifact | sha256 |
|---|---|
| `results/tap_capacity_gate_analysis.json` | `c3a37d0b5d77e0d3908043241b5827e45b63eda1f1189be2fe677b41e7a8404d` |
| `results/tap_capacity_gate_receipt.json` | `a4ee8c8ee0f95e7b6f016d247d88de43af31521e8be80faa30d8ff2381f8e53a` |
| `results/tap_7b_capacity_analysis.json` | `a2fcedc69b31426bfe2b999b5689c01120b9348629841e2b4aaab8912de30251` |
| `results/tap_7b_capacity_receipt.json` | `c528ab6da93c2d09721232d068e63d2afe56a6f3c1c95911d88f30d9d2b555d1` |
| `results/v4_3_execution_supervision_pilot_analysis.json` | `83d8315847bb13d9935ece79761a11fe3a596f281cfb156792c72ca4702c8047` |
| `results/v4_3_execution_supervision_pilot_diagnosis.json` | `59ce7290167de0cb35d5c96deeb40f0ca864b40e849df9948bdc1a662ef272a6` |
| `results/v4_3_execution_trace_pilot_analysis.json` | `ee5a666bf4117ff09eff7f745ea4a298cd29e252887a61973992428d9a8f94ff` |
| `results/v4_3_execution_trace_pilot_diagnosis.json` | `91201a7ebffbd1e5c5800a98164e2db955ecca7109e6fc1cbcc639e2f5427805` |
| `results/v4_3_execution_trace_pilot_decision_receipt.json` | `65144bcc2244498b12ba65ef21183ea555cb63b98f7e21c106e2e7c92128bc27` |
| `results/v4_3_execution_dose_matched_control.json` | `3c3d92c7f41987817c4b81fc6a88870db296fa5cf0048bc945babf7f386f343b` |
| `results/v4_3_execution_dose_preflight_receipt.json` | `22df107c97e0ebe240e70730cba2d011f9f573a9c6d92e778bb7ecaf7921712c` |
| `results/v4_3_execution_dose_analysis.json` | `5b1a718710e42b3091f9417f4e879f0c82848aaab0936680d4cddcc1aabbf27e` |
| `results/v4_3_execution_dose_decision_receipt.json` | `3e6236e4596c5381635cd559987549ee877a3e90f73c166542e572e663743033` |
| `results/v4_3_tool_assisted_design_receipt.json` | `d9d248984e627d3e7d04820b10de676f0ebcb090e5055b6220408fcbe3fa8fbb` |
| `results/v4_3_tool_assisted_analysis.json` | `38995b78d81aa8ba34eb4f040b82929871d92621ebfd3bb6abf8d365844d3d41` |
| `results/v4_3_tool_assisted_lineage_manifest.json` | `dd5d9a548c5ec927467941388de43417bffdc056cf08b6e0e75be0db4a285d40` |
| `results/v4_3_tool_assisted_decision_receipt.json` | `78a7f6285b56e43c809e89d2811c205c77ae58184c5eadf295ec786cb8fe9011` |

Each decision receipt binds every training result, adapter, evaluation artifact,
run ID and source file by hash. Adapters, raw evaluations and datasets stay in
ignored local storage on the GPU machine.

## 8. Execution feedback at inference (run 2026-09-25: FAIL)

The execution-supervision SFT line is closed and will not be reopened: there
will be no more epochs, dose increases, mixture changes, re-thresholding or new
execution-supervision adapters.

The next hypothesis is different in kind. Execution feedback at inference time
may improve test generation beyond sham self-conditioning under matched calls,
sequences, input tokens and output-token caps. It writes no weights, and the model is never asked to simulate the
program. Candidates are executed in the sandbox against the code under test,
and only demonstrably invalid artifacts receive feedback: parse and shape
failures, prohibited constructs, invented names, malformed calls, and duplicates.

Some outcomes get no feedback at all:
- assertion failures;
- clean passes;
- exceptions or timeouts inside the code under test.

These are kept silently, because each could be, or would reveal, a kill.

The full design is in [TOOL_ASSISTED_REPAIR_PROTOCOL.md](TOOL_ASSISTED_REPAIR_PROTOCOL.md).
Its main features:
- **Arms:** A is the canonical control. B is a sham-feedback control. On every repaired slot it gets the same prompt, the same echoed candidate and a neutral sentence, padded to C's exact rendered input-token count. C is the repair arm.
- **Matching:** B and C are matched in calls, sequences, caps and rendered input tokens. Actual output tokens and wall time are measured outcomes and may differ.
- **Pairing:** B and C share round 1 and every non-repaired slot. They differ only where C gets feedback and B gets the matched sham. Both spend 16 sequences per target.
- **Final slots:** chosen by a policy that is blind to assertion outcomes.
- **Gate:** C − B Kill@8 must gain at least +5 pp with a lower bound above 0, and reference validity must stay within −3 pp.

**Panel.** The panel is a *qualified exploratory* train-derived set: 264 MBPP
records in 110 lineages. It is disjoint from every training arm and from every
earlier panel. It is not untouched: base-model generations on every train
lineage were labelled in the O1 oracle artifact. It also has no complex-tier,
HumanEval or repository records. Results on it cannot be confirmatory.

**Readiness.**
- **Actual Atheris:** the installed package is the real one and its instrumentation is active, but no permitted comparison panel exists yet.
- **Native repository:** execution is not ready for a reportable comparison. Only 3 of 5 official tests reproduce, and no generated test has been run natively.

### Result (run 2026-09-25): FAIL

**Evidence status.** This is an exploratory pilot on training-derived data. It
is not confirmation, generalization, validation or final-test evidence.

**Runs.** There were four GPU/CPU stages, run one after another:

| run | stage | duration |
|---|---|---:|
| `20260925-103511-…` | generate A | 298.5 s |
| `20260925-104038-…` | generate B/C | 3,168.2 s |
| `20260925-113357-…` | score B | 42.1 s |
| `20260925-113501-…` | score C | 44.1 s |

All four exited with code 0.

**Integrity.** B/C pairing held on all 264 targets, with 0 problems.
- B and C used identical input tokens: 1,769,147 each.
- Output tokens are a measured outcome, not matched: 89,937 for B and 90,022 for C (ratio 1.0009).
- Generation wall time was also measured: 2,269.8 s for B and 2,270.1 s for C (ratio 1.0001).

**Repairs.** 182 of 264 records (68.9%) got at least one repair.
- 450 were attempted, 449 delivered, and 1 skipped because matching was infeasible.
- **93% of delivered repairs were duplicate-candidate feedback** (419 of 449).
- Only 30 addressed a genuinely invalid artifact: 16 invalid shapes, 7 malformed calls, 5 missing calls, 1 prohibited construct and 1 syntax error.

**Primary comparison (C − B).** Paired across 110 lineage clusters, with 90% bootstrap intervals.

| metric | B | C | Δ | 90% CI |
|---|---:|---:|---:|---:|
| Kill@1 | 95/264 | 95/264 | 0.00 pp | [0.00, 0.00] |
| Kill@4 | 163/264 | 164/264 | +0.38 pp | [−0.74, +1.50] |
| **Kill@8** | **191/264** | **192/264** | **+0.38 pp** | **[−1.07, +1.77]** |
| reference-valid per requested | 0.480 | 0.479 | −0.14 pp | [−0.66, +0.34] |
| functions with ≥1 reference-valid | 232 | 233 | +0.38 pp | [0.00, +1.12] |
| execution success | 0.966 | 0.968 | +0.19 pp | [+0.04, +0.39] |
| exact-unique ratio | 0.969 | 0.975 | +0.006 | [+0.002, +0.010] |

The frozen verdict is **FAIL**. The Kill@8 upper bound (+1.77 pp) is below the
+5 pp minimum important gain, so a practically important benefit from execution
feedback is excluded. Most repaired duplicates became distinct tests, but this
did not change which mutants were killed.

**Secondary, not matched.** Against the canonical 8-sample protocol (A: 174 of
264 killed):
- B − A = **+6.44 pp** [+2.46, +10.59];
- C − A = **+6.82 pp** [+2.75, +10.89].

These gains come from 16 sequences plus text-level de-duplicating selection, not
from execution feedback. They are an uncontrolled observation that motivates
nothing on their own.

**Decision.** This intervention is stopped: no extra repairs, calls, tokens,
prompt changes or thresholds. The frozen model-only control remains the
reference, and confirmation stays closed. The receipts are
`results/v4_3_tool_assisted_analysis.json` and
`results/v4_3_tool_assisted_decision_receipt.json`.

## 9. Next direction (design only, 2026-09-25; isolation bound, exact diff, crash-safe publication)

**Recommended primary line: an independently constructed repository-native
evaluation.** It uses new bugs from repositories outside every indexed source,
reproduced natively on buggy and fixed revisions, with a dress rehearsal on a
separate pool before any one-time run.

**Optional secondary line: a sampling-budget study.** It asks whether 16 samples
with frozen text-only selection of 8 beat the nested first 8. It is exploratory
and takes about 20 GPU minutes.

**Isolation claim (narrow; D7 accepted 2026-09-25 under exactly this claim).** Disjointness is *enforced* against the
complete indexed source universe under the recorded repository, fork, commit,
patch, issue and function-similarity checks. It does not claim that no model
has seen a repository, and the 2025 fix cutoff is contamination-risk mitigation
only.

- **Bound decisions.** Every decision is made against a `FrozenReferenceUniverse`. Loading it re-verifies the recomputed collections, the collection hashes, the internal receipt hash, every input file and every canonical source against the receipt (`40ce55b0…`), which is published in the crash-safe bundle `results/next_direction_bundle`. The receipt hash comes from that object; no caller can supply one.
- **Bound coverage inputs.** The receipt binds the corpus manifest, `utils/dataset_identity.py`, the curated definition and the train-view loader. All ten curated definitions are indexed, a conservative superset of the eight corpus seeds.
- **Exact diff (policy A′).** Policy A failed its acquisition-feasibility pilot (admission 1/124, and that one admission was vendored Click code), so A′ was designed from that pilot, and those 124 candidates cannot confirm it. A′ admits a direct single-parent fix whose only changed production file is the target. Exactly one production function changes, with every executable hunk inside it, and only recognised test and documentation paths may change alongside, each with authenticated evidence. Patch lineage comes only from the derived target diff, and a submitted patch only has to carry the same derived added and removed lines per file. Vendored and generated code is excluded as a conservative mitigation. An independent confirmation pilot is pending.
- **Temporal rule.** The fixed-commit committer timestamp must be on or after 2025-01-01. It is enforced at admission and frozen in the receipt, and is contamination-risk mitigation only.
- **Separate stages.** Candidate evidence goes through schema validation, then offline authentication, then source-universe overlap. Network acquisition is a later, separately approved step.
- **Integrity.** `record_sha256` is a self-hash, not a signature. Downstream use must revalidate each record from its evidence sidecar, byte for byte.

**Power.** Evidence paths are repository-relative, and every evidence file is
hash-checked against the closed pilots' decision receipts. The artifact is
verified by full recomputation before the design accepts it
(`results/v4_3_repository_native_power_analysis.json`). Under the unchanged
+5 pp gate:
- a true gain of exactly +5 pp passes with probability at most 50% at any N;
- **N = 400 is recommended**, powered for a true +8 pp gain at ρ = 0.05 with at most 12 targets per repository (power 0.86 analytic, 0.82 clustered Monte Carlo);
- N = 200 is only a lower-power feasibility compromise (power about 0.64).

Artifacts are published as one immutable, manifest-verified generation selected by a single pointer. A crash leaves either the complete old generation or the complete new one.

The design, estimates, blockers (B1–B9) and required decisions (D1–D9) are in
[NEXT_DIRECTION_DECISION_MEMO.md](NEXT_DIRECTION_DECISION_MEMO.md).

**Nothing has been mined or installed, no split exists, no network call was
made, and no model has run.**
