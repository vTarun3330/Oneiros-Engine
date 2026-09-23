# Figure and table inventory

What each venue's CFP formally requires, what the science actually needs, where
every value comes from, and what has to be cut to fit each page limit.

Source keys refer to `evidence_ledger.md` §A.

---

## 1. Does any CFP formally require figures, tables, artifacts or a screencast?

Checked against the official CFP pages recorded in each venue's
`submission_checklist.md`.

| Venue | Figures/tables required? | Artifact required? | Screencast required? |
|---|---|---|---|
| SANER 2027 RENE | **No.** No figure or table mandate stated. | **Effectively yes.** "Availability of instruments and complementary artifacts" is an explicit evaluation criterion, and papers should "provide the links to all the artifacts in the submission". | No. |
| ICST 2027 Research | **No.** Page limit counts "all text, figures, tables, and appendices"; no mandate. | **Yes, expected.** "Submissions must supply all information needed to replicate the results and therefore are expected to include or point to an anonymized replication package". | No. |
| SANER 2027 SP&P | **No.** | **No, explicitly relaxed.** PC members are "not [to] require full availability of artifacts at submission time"; sharing via Zenodo/Figshare/Archive.org is encouraged. | No. |
| CAIN 2027 Research (short) | **No.** | **Yes, by default.** "Sharing is expected to be the default, and non-sharing needs to be justified", archived on an institutional or open platform. | No. |

**Conclusion:** no venue mandates a specific figure or table. Every visual below
is included because it carries information the prose cannot, and is cut when the
page budget is tighter than its value.

---

## 2. Master inventory

All five items are original, data-driven, and built only from committed
artifacts. None contains a screenshot, a raw prompt, a raw completion, candidate
code, project source, sealed data, or any third-party image.

### T1 — Main results table (`booktabs`)

Development arms and locked-validation arms, with the paired difference.

| Value | Source |
|---|---|
| Dev Kill@8 for base / A@150 / A@431 / B@150 and Kill@8 function counts | `DEV-SEL` `0188324f…09e58969` |
| Dev Wilson 95% intervals | `DEV-SEL` |
| Locked Kill@8 for base / A@431, counts 450 and 476, n = 757 | `LV-RES` `f7cdf84d…3e288af9` |
| Locked Wilson 95% intervals | `LV-RES` |
| Paired: +114 / −88 / net +26, Kill@8 +3.4346 pts, McNemar p = 0.0783200108 | `LV-RES` › `paired_comparison` |
| Predeclared threshold p < 0.01 and criterion outcome "failed" | `LV-RES` › `promotion_criteria` |

Design rule: development rows and locked rows sit in **separately headed
blocks** with the development block labelled *selection-only*. No column invites
a cross-panel subtraction.

### T2 — Candidate-validity table (`booktabs`)

| Value | Source |
|---|---|
| Dev parse / execution / reference validity for all four arms | `DEV-SEL` |
| Locked parse 0.987285 → 0.988771; execution 0.928336 → 0.942206; reference 0.452114 → 0.333388 | `LV-RES` |
| Functions with any reference-valid candidate 675 → 566 | `LV-RES` |
| Deltas +0.1486 / +1.3870 / −11.8726 points | `LV-RES` › `paired_comparison` |

Design rule: the reference-validity column is the one emphasised; parse and
execution are shown precisely because they did *not* regress, which is what makes
the reference-validity drop specific rather than general degradation.

### T3 — Claim-status table (`booktabs`)

Three blocks — established / inconclusive / unavailable — transcribed from
`claims_traceability.md`. Contains **no** final-test row, **no** real-repository
row, and **no** baseline-comparison row other than as explicit absences.

| Value | Source |
|---|---|
| Established rows E1–E20 | as per `claims_traceability.md` |
| Inconclusive rows I1–I3 | `LV-RES`, `DEV-SEL` |
| Unavailable rows U1–U4 | `INCIDENT` `6d2e8729…b9b4c267`, `STATUS` `58d12e2c…fa53de15` |

### F1 — Split and protocol diagram (TikZ)

Four panels — `train`, `ablation_dev`, `val`, and the final split — with the
role of each annotated: training; **selection-only**; locked, read once; and the
final split drawn with a hatched/struck style annotated *consumed — 0 candidates,
0 metrics, never re-runnable*.

| Value | Source |
|---|---|
| Panel roles and the locked-once property | `LV-RES` › `interpretation_limits`, `DEV-SEL` › `interpretation_limits` |
| Consumed-split status, zero candidates, zero metrics | `INCIDENT` |

Design rule, stated precisely:

- **Prohibited** on the final-split panel: its size, its membership, its record
  counts, and any performance score. None of these appears in any draft.
- **Permitted and stated**: the historical fact that the authorized attempt
  generated **zero candidates and zero reportable metrics**. This is not a
  measurement of the split; it is a record of an attempt that produced nothing,
  and omitting it would misrepresent what happened.

So the panel does carry two zeros, and they are the only numbers on it.

### F2 — Pipeline-integrity / rehearsal diagram (TikZ)

Linear stage chain: receipt verification → admission → mode-aware scoping →
generation → whole-output parsing → safe execution → raw-output retention →
progress writing → scoring → artifact digest. Annotated with the rehearsal's
verified counts and a footer stating it is an operational check, not selection.

| Value | Source |
|---|---|
| 594 → 542 targets, 52 repository excluded, 0 admission errors | `REH-EXEC` `06aca918…7136b59f` |
| 4,336 candidates, 271 batches of 2 | `REH-EXEC` |
| Raw integrity 0 missing / 0 mismatched / 0 recomputed mismatches | `REH-EXEC` |
| 55 progress files, monotonic 10 → 542 | `REH-EXEC` |
| Kill@8 0.605166 exactly matching the historical value | `REH-EXEC` › `reproduction_check` |
| `final_test_measurement: false`, `eligible_for_model_selection: false`, `supports_performance_claim: false` | `REH-EXEC` |

### F3 — Development-versus-locked comparison figure (PGFPlots)

Kill@8 with Wilson 95% intervals for the development arms and the locked arms on
one shared axis, with the predeclared threshold annotated and the locked pair
marked *positive, not significant under the predeclared rule*.

| Value | Source |
|---|---|
| All point estimates and intervals | `DEV-SEL`, `LV-RES` |
| Data file | `papers/common/data/kill_at_8.csv` |

Design rule: the two panels are visually separated and the caption states that
a development bar and a locked bar are **not** comparable quantities.

### F4 — Incident-to-safeguard causal diagram (TikZ)

Five nodes: predeclared freeze → authorized final attempt → admission-rule
failure → permanent refusal of the consumed split → full-scale permitted-split
rehearsal as a standing gate. No sealed record, identifier or payload appears.

| Value | Source |
|---|---|
| Failure mode and its cause | `INCIDENT` |
| Rehearsal as the resulting safeguard | `STATUS`, `REH-EXEC` |

### T4 — Risk-to-mitigation table (CAIN only, `booktabs`)

Evaluation-integrity risks for AI-enabled testing systems, each paired with the
control this project actually implemented and the artifact that evidences it.

| Value | Source |
|---|---|
| Every mitigation row must name a real implemented control | `DEV-SEL`, `LV-RES`, `REH-EXEC`, `INCIDENT` |

---

## 3. Per-venue allocation and page budget

Page limits are taken from the official CFPs (see each `submission_checklist.md`).

### SANER 2027 RENE — 10 pages + 2 reference-only pages

| Item | Include | Est. column-inches | Note |
|---|---|---|---|
| T1 main results | **yes** | ~0.45 page | core evidence |
| T2 candidate validity | **yes** | ~0.35 page | RQ2 |
| T3 claim status | **yes** | ~0.40 page | the track's central contribution |
| F1 split diagram | **yes**, single column | ~0.30 page | needed to show selection-only |
| F4 incident→safeguard | **yes**, single column | ~0.30 page | required by venue positioning |
| F2 pipeline diagram | **condensed** into F4's tail | — | avoids duplication |
| F3 dev-vs-locked figure | **no** | — | T1 carries the same numbers; cut to protect page budget |

Visual budget ≈ 1.8 pages of 10. **Fits.**

### ICST 2027 Research — 10 pages + 2 reference-only pages

| Item | Include | Est. column-inches | Note |
|---|---|---|---|
| T1 main results | **yes** | ~0.45 page | |
| T2 candidate validity | **yes** | ~0.35 page | |
| T3 claim status | **yes**, compact | ~0.30 page | |
| F2 full workflow diagram | **yes**, double column | ~0.45 page | required by venue positioning |
| F3 dev-vs-locked figure | **yes**, single column | ~0.35 page | must show "positive, not significant" |
| F1 split diagram | **merged** into F2's left region | — | |
| F4 incident diagram | **no** | — | prose only; ICST framing is empirical, not incident-led |

Visual budget ≈ 1.9 pages of 10, plus a replication-package section (~0.5 page).
**Fits.**

### SANER 2027 SP&P — 6 pages *including* references

This is the tightest budget: references are **inside** the six pages.

| Item | Include | Note |
|---|---|---|
| T1 main results, **reduced** to locked rows + the two development rows that bracket them | **yes** | ~0.30 page |
| T3 claim status, **reduced** to three short blocks | **yes** | ~0.25 page |
| F3 dev-vs-locked figure, single column | **yes** | ~0.30 page |
| T2 candidate validity | **cut** — the −11.8726 reference-validity change is stated inline in prose instead | |
| F1, F2, F4 | **cut** | |

Two tables + one figure ≈ 0.85 page of 6, leaving ~1 page for references.
**Fits**, and the CFP explicitly warns against dense figures being unreadable at
this length, so the retained figure uses at most two panels.

### CAIN 2027 Research (short) — 5 pages + 2 reference-only pages

| Item | Include | Note |
|---|---|---|
| F2 as an **evaluation-safety architecture diagram**, double column | **yes** | ~0.40 page |
| T4 risk-to-mitigation | **yes** | ~0.40 page |
| T1 main results, **reduced** to the locked pair plus one development row | **yes**, compact | ~0.20 page |
| T2, T3, F1, F3, F4 | **cut**; claim status compressed into three prose sentences | |

Visual budget ≈ 1.0 page of 5. **Fits.**

---

## 4. Data files

| File | Contents | Source |
|---|---|---|
| `data/kill_at_8.csv` | Kill@8 point estimates and Wilson bounds for every arm in both panels, plus the rehearsal | `DEV-SEL`, `LV-RES`, `REH-EXEC` |
| `data/validity.csv` | parse / execution / reference validity per arm per panel | `DEV-SEL`, `LV-RES` |
| `data/rehearsal_counts.csv` | scope, candidate, integrity and failure-taxonomy counts | `REH-EXEC` |

Each file carries a header comment naming its source receipt and that receipt's
SHA-256, so a figure can be traced to an artifact without opening the paper.

---

## 5. Prohibitions applied to every visual

1. No invented, estimated, interpolated or smoothed value. Every plotted number
   appears verbatim in a receipt.
2. No axis that invites subtracting a development figure from a locked figure.
3. No number of any kind attached to the consumed split.
4. No screenshot of prompts, completions, candidate code, project source or
   sealed data.
5. No third-party or copyrighted image.
6. No artifact link, repository host, or filename pattern that could identify an
   author under double-anonymous review.
