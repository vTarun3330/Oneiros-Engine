# Claims traceability

Every claim any draft is permitted to make, with its evidence key (see
`evidence_ledger.md` §A) and the venues that use it. Claims are graded:

- **E** — established: a value or verdict read directly from a committed receipt.
- **I** — inconclusive: measured, but the predeclared rule did not reach a verdict.
- **U** — unavailable: no evidence exists; stated only as an absence.

A claim not in this table does not appear in any draft.

---

## Established (E)

| # | Claim | Evidence | Venues |
|---|---|---|---|
| E1 | The measured system is `Qwen2.5-Coder-1.5B-Instruct` pinned at revision `2e1fd397…1098a`, evaluated under `oneiros_successor_generation_protocol_v1` (digest `4c703a75…b54a8`). | `DEV-SEL`, `REH-EXEC` | all |
| E2 | All four development arms ran under one identical frozen contract (`6a87294c…6cdef`) and one identical evaluation scope (`d37f75d6…1f6ca`). | `DEV-SEL` | RENE, ICST |
| E3 | Development Kill@8: base 0.605166, A@150 0.690037, A@431 0.695572, B@150 0.658672. | `DEV-SEL` | all |
| E4 | The development panel selected the checkpoints it scored; its figures are selection statistics, not generalization estimates. | `DEV-SEL` › `interpretation_limits[0]` | all |
| E5 | The predeclared matched-duration comparison (B@150 vs A@150) gave −3.1365 points, net −17 functions, McNemar p = 0.118; the O1 sidecar was rejected. | `DEV-SEL` | RENE, ICST |
| E6 | Locked-validation Kill@8: base 450/757 = 0.594452; A@431 476/757 = 0.628798. | `LV-RES` | all |
| E7 | Paired locked difference: +114 gained, −88 lost, net +26, Kill@8 +3.4346 points, McNemar exact two-sided p = 0.0783200108. | `LV-RES` | all |
| E8 | The predeclared criterion required p < 0.01. The observed p did not meet it, so criterion 1 failed and the base model was retained. | `LV-RES` | all |
| E9 | Reference validity fell 0.452114 → 0.333388 on the locked panel, a −11.8726 point change; functions with any reference-valid candidate fell 675 → 566. | `LV-RES` | all |
| E10 | Reference validity fell in every development SFT arm relative to base (0.565498 → 0.458487 / 0.455028 / 0.448801). | `DEV-SEL` | RENE, ICST, CAIN |
| E11 | Parse and execution validity did **not** regress under SFT (+0.1486 and +1.3870 points on the locked panel). | `LV-RES` | RENE, ICST |
| E12 | The rehearsal scoped 594 requested records to 542 function-mode targets, excluding 52 repository-mode records, with 0 admission errors. | `REH-EXEC` | all |
| E13 | The rehearsal generated 4,336 candidates (542 × 8) in 271 batches of 2, exit status 0, 577.753 s. | `REH-EXEC` | RENE, ICST, CAIN |
| E14 | Raw-output integrity was complete: 4,336 outputs retained, 4,336 hashes, 0 missing, 0 mismatched, 0 mismatches on independent recomputation. | `REH-EXEC` | all |
| E15 | The rehearsal reproduced the historical development base Kill@8 exactly (0.605166 = 0.605166, `exact_match: true`) under the same evaluation-scope digest. | `REH-EXEC` | all |
| E16 | The rehearsal wrote no weights, loaded no adapter, and performed no training. | `REH-EXEC` › `model_identity` | all |
| E17 | The rehearsal receipt carries `final_test_measurement: false`, `eligible_for_model_selection: false`, `supports_performance_claim: false` as machine-checkable fields. | `REH-EXEC` | all |
| E18 | The sealed final attempt reached `failed_after_authorization`, generating 0 candidates and producing 0 reportable metrics; the split may never be rerun. | `INCIDENT` | all |
| E19 | The failure cause was a record-admission rule applied to every record *before* scope filtering, where the established evaluator filters scope first. | `INCIDENT` | all |
| E20 | Repository-mode records are excluded from every reported kill rate (38 on the development panel, 24 on the locked panel, 52 under the post-incident rehearsal rule). | `DEV-SEL`, `LV-RES`, `REH-EXEC` | RENE, ICST |

## Inconclusive (I)

| # | Claim | Evidence | Venues |
|---|---|---|---|
| I1 | Whether supervised fine-tuning improves Kill@8 on unseen data is **undetermined at this sample size**. The locked point estimate is positive (+3.4346) and the Wilson intervals of the two arms overlap; the predeclared significance threshold was not met. This is insufficient evidence of an effect, **not** evidence of no effect. | `LV-RES` | all |
| I2 | Whether the reference-validity regression would persist under a different supervision recipe is unmeasured; it was observed in all four arms of one recipe family. | `DEV-SEL`, `LV-RES` | RENE, ICST |
| I3 | Whether the ≈2.6× development-to-locked shrinkage generalises beyond this task, model and corpus is unmeasured — it is a single paired observation. | `DEV-SEL`, `LV-RES` | all |

## Unavailable (U) — stated only as absences

| # | Statement | Evidence | Venues |
|---|---|---|---|
| U1 | No final-test result exists for any arm. | `INCIDENT` | all |
| U2 | No native real-repository kill rate exists; repository records are excluded everywhere. | `DEV-SEL`, `LV-RES`, `REH-EXEC` | all |
| U3 | No valid Oneiros-versus-Atheris comparison exists; no baseline was bundled or executed in any measurement reported. | `REH-EXEC`, `STATUS` | all |
| U4 | Both held-out panels are now expended: `val` on selection, the final split by the failed attempt. | `STATUS` | RENE, ICST, CAIN |

---

## Prohibited statements — mechanically excluded

The following strings and their paraphrases must not appear in any draft. The
structural validator greps for each.

| Prohibited | Why |
|---|---|
| "Oneiros beats Atheris" / any Oneiros-vs-Atheris superiority | U3 — no such measurement exists |
| "SFT generalizes" | I1 — not established |
| "SFT is better than the base model" | E8 — the predeclared rule retained base |
| Any final-test metric | U1 |
| Any real-repository performance figure | U2 |
| Any value derived from the consumed split | E18 |
| The rehearsal as a model-quality or model-performance result | E17 — the receipt denies it in three fields |
| "proves there is no effect" / "no difference" about the locked result | I1 — absence of significance is not evidence of absence |

---

## Wording rules applied in all drafts

1. The locked result is reported as **positive but not significant under a
   predeclared threshold**, never as "no effect" and never as "an improvement".
2. Development figures are always introduced with the word *selection*.
3. The rehearsal is always introduced as *operational* or *pipeline fidelity*,
   and its Kill@k values are never compared against an arm.
4. The consumed split is described by mechanism only; no size, membership or
   identifier appears.
5. First person plural is used for actions taken; no venue draft names a person,
   institution, repository host or URL that would identify an author.
