# ICST 2027 — Research Papers Track submission checklist

**Retrieved from the official CFP on 2026-09-18. Re-check before submitting.**

## Venue facts

| Item | Value |
|---|---|
| Official CFP URL | https://conf.researchr.org/track/icst-2027/icst-2027-research-papers |
| Conference | 20th IEEE International Conference on Software Testing, Verification and Validation, San Sebastián, Spain |
| Paper deadline | **2 November 2026**, AoE (UTC−12) |
| Abstract deadline | **Not stated** on the track page. **TODO — confirm whether a mandatory abstract precedes the paper deadline.** |
| Initial notification | Tue 22 Dec 2026 |
| Major-revision submission | Sun 31 Jan 2027 |
| Final notification | Sat 20 Feb 2027 |
| Camera ready | Thu 18 Mar 2027 |
| Page limit | **10 pages** including all text, figures, tables and appendices, **plus two additional pages containing only references** |
| Formatting | "two-column IEEE conference publication format"; "conform to the IEEE Conference Proceedings Formatting Guidelines (**use the letter format template and conference option**)" |
| Anonymity | **Double-anonymous.** "No submission may reveal its authors' identities… the authors' names must be omitted from the submission, and references to their prior work should be in the third person. **All artifacts, such as replication packages and tools, associated with the submission must also be anonymized.**" |
| Artifact expectation | **Expected.** "Submissions must supply all information needed to replicate the results and therefore are expected to include or point to an anonymized replication package with the necessary software, data, and instructions." |
| Figures/tables formally required? | **No** mandate; they count against the 10 pages. |
| Screencast required? | No (a separate Artifact Evaluation track exists). |
| Registration/presentation | "If a paper is accepted, at least one author of the paper is required to register for ICST 2027 and present the paper. We expect that the conference will be in-person." |
| Generative-AI policy | **Not stated** on the track page. **TODO — check the ICST 2027 main site and the IEEE Policy on Authorship for a generative-AI disclosure requirement before submitting.** |

Note: this track runs a **major-revision** cycle. A revise decision at the
December notification requires a resubmission by 31 January 2027.

## Planned tables, figures and page budget

| Item | Type | Est. share of 10 pages |
|---|---|---|
| Figure 1 — evaluation workflow and split roles | TikZ, **double column** | ~0.45 |
| Figure 2 — development vs. locked Kill@8 with Wilson intervals | TikZ, single column | ~0.35 |
| Table I — main results | `booktabs` | ~0.45 |
| Table II — candidate validity | `booktabs` | ~0.35 |
| Table III — rehearsal outcome | `booktabs` | ~0.30 |
| Table IV — receipted artifacts | `booktabs`, small | ~0.15 |
| Section IX — replication package | prose | ~0.50 |

Visual + replication budget ≈ **2.6 of 10 pages**. **Cut if over:** Table IV
first (its digests move to the package manifest), then Table III (values move
into prose).

## Pre-submission checks

- [ ] **Anonymity, including the artifact.** ICST explicitly requires the
      replication package to be anonymized. No repository host, account,
      personal URL, funder, institution or acknowledgment anywhere.
- [ ] **PDF metadata scrubbed.**
- [ ] **Letter paper** — `\documentclass[10pt,conference,letterpaper]{IEEEtran}`
      is set; confirm the output PDF is US Letter, not A4.
- [ ] **Page count** — body ≤ 10, references ≤ 2 extra pages, references only.
- [ ] **No padding or spacing manipulation.** Cut content instead.
- [ ] **Replication package complete** — frozen protocol receipt, decision-rule
      receipt, both result receipts, rehearsal receipt and execution receipt,
      per-figure data files with source digests, and commands that re-derive
      every table.
- [ ] **Raw outputs excluded but digested** — the package must not redistribute
      model completions in volume; publish their digests instead.
- [ ] **No material from the consumed split** — none exists; confirm none is
      implied.
- [ ] **Bibliography verified** against DOI / publisher / proceedings pages.
      Resolve every `% VERIFY-DOI` marker.
- [ ] **No citation to unpublished assistant transcripts.**
- [ ] **Evidence check** — every number matches `../common/evidence_ledger.md`.
- [ ] **Figure 2 coordinates re-derived** from `../common/data/kill_at_8.csv`.
      The mapping is `x = (Kill@8 − 0.50) × 25` cm. An earlier draft of this
      figure carried an incorrect Wilson offset; do not trust the coordinates
      without recomputing them.
- [ ] **Prohibited-claim grep** — see `../common/claims_traceability.md`.
- [ ] **Limitations are explicit**, as this venue's positioning demands: the
      locked result is positive but not significant under a predeclared rule;
      no final-test result; no repository-level rate; no baseline comparison.
- [ ] **TODO sweep** — no `TODO` marker in the PDF.
- [ ] **Compiles cleanly**; consider submitting to the separate Artifact
      Evaluation track after acceptance.

## Concurrent submission warning

> **Do not submit this manuscript to ICST while any of the SANER RENE, SANER
> SP&P or CAIN drafts in `papers/` is under review, and vice versa.** All four
> describe the same study from the same evidence. Concurrent submission of the
> same or substantially similar work violates IEEE policy and every venue's
> submission rules, and is grounds for desk rejection at all of them.
>
> Pick one venue. Submit. Wait for the decision.
