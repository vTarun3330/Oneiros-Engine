# Oneiros — research status after the sealed-final incident

**Written 2026-09-17.** This is the current standing of every result in the
project. Where a figure is quoted it is quoted from a committed receipt, not
recomputed. Nothing here changes a number.

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
