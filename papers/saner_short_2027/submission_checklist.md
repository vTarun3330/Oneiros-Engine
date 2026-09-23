# SANER 2027 — Short Papers and Posters (SP&P) submission checklist

**Retrieved from the official CFP on 2026-09-18. Re-check before submitting.**

## Venue facts

| Item | Value |
|---|---|
| Official CFP URL | https://conf.researchr.org/track/saner-2027/saner-2027-short-papers-and-posters-track |
| Abstract deadline | **Mon 19 Oct 2026**, AoE (UTC−12) |
| Paper deadline | **Fri 23 Oct 2026**, AoE (UTC−12) |
| Notification | Tue 8 Dec 2026 |
| Camera ready | Fri 8 Jan 2027 |
| Author registration deadline | Fri 8 Jan 2027 |
| Page limit | **6 pages for Short Papers, INCLUDING all text, figures, references and appendices.** (2 pages for poster extended abstracts — not used here.) |
| Reference-page rule | **None. References are inside the 6 pages.** This is the binding constraint on this draft. |
| Formatting | "IEEE Conference Proceedings Formatting Guidelines (title in 24pt font and full text in 10pt font, LaTeX users must use `\documentclass[10pt,conference]{IEEEtran}`)" |
| Anonymity | **Double-anonymous.** "Author names and affiliations must be omitted"; "References to authors' own related work must be in the third person." |
| Artifact expectation | **Relaxed at submission time.** "Program committee members are asked to keep into account the double-blind policy when reviewing papers, and therefore not require full availability of artifacts at submission time." Sharing via Zenodo, Figshare or Archive.org is encouraged. |
| Figures/tables formally required? | **No.** |
| Screencast required? | No. |
| Registration/presentation | "At least one author of each accepted paper must register (**full registration, not student registration**) to and attend the conference in order to present their paper." |
| Generative-AI policy | **Stated.** "Papers must comply with the **IEEE Policy on Authorship**, which includes guidelines on the use of generative AI." Read that policy and comply; disclose AI assistance where it requires. |

## Planned tables, figures and page budget

Six pages *including* references makes this the tightest budget of the four.

| Item | Type | Est. share of 6 pages |
|---|---|---|
| Table I — Kill@8 by panel (reduced to 2 dev rows + 2 locked rows) | `booktabs` | ~0.30 |
| Table II — claim status (3 compact blocks) | `booktabs` | ~0.25 |
| Figure 1 — development vs. locked with Wilson intervals | TikZ, single column | ~0.30 |
| References | — | ~0.80–1.00 |

**Deliberately cut** relative to the full papers: the candidate-validity table
(the −11.8726 point change is stated inline instead), the split diagram, the
pipeline diagram, and the incident causal diagram.

The CFP warns implicitly against unreadable density at this length; Figure 1 is
kept to four rows with one separator, not the six-row version used in the ICST
draft.

## Pre-submission checks

- [ ] **Anonymity** — no author name, affiliation, acknowledgment, institution,
      funder, repository host or identifying URL. Self-citation in third person.
- [ ] **PDF metadata scrubbed.**
- [ ] **Page count ≤ 6 INCLUDING references.** Verify by counting the compiled
      PDF, not by estimating. This is the most likely failure mode for this
      draft.
- [ ] **Trim the bibliography if needed** — cut citations rather than shrinking
      margins or spacing. A short paper may reasonably cite 12–18 works.
- [ ] **No padding or formatting manipulation** in either direction.
- [ ] **IEEE Policy on Authorship reviewed**, including its generative-AI
      guidance, and any required disclosure prepared.
- [ ] **Artifact** — optional at submission. If linked, it must be anonymous
      (Zenodo restricted / Anonymous GitHub), never a personal repository.
- [ ] **Bibliography verified** against DOI / publisher / proceedings pages;
      resolve `% VERIFY-DOI` markers for every entry that survives the trim.
- [ ] **No citation to unpublished assistant transcripts.**
- [ ] **Evidence check** — every number matches `../common/evidence_ledger.md`.
- [ ] **Figure 1 coordinates re-derived** from `../common/data/kill_at_8.csv`
      using the mapping `x = (Kill@8 − 0.54) × 22` cm stated in the source
      comment. Do not trust them without recomputing.
- [ ] **Prohibited-claim grep** — see `../common/claims_traceability.md`.
- [ ] **Self-contained despite the length** — a reader who has not seen the full
      version must be able to understand the setting, the decision rule, and why
      the locked result is not a null result.
- [ ] **TODO sweep.**
- [ ] **Compiles cleanly.**
- [ ] **Full (non-student) registration available** for an author if accepted.

## Concurrent submission warning

> **This short paper overlaps heavily with the SANER RENE, ICST and CAIN drafts
> in `papers/`.** Submitting it while any of those is under review — or
> submitting two of them to the same conference — is concurrent submission of
> substantially similar work. It violates IEEE policy and every venue's rules,
> and is grounds for desk rejection at all of them.
>
> Note especially: **SANER RENE and SANER SP&P are tracks of the same
> conference.** Submitting to both is not a hedge; it is a policy violation.
> Choose one SANER track.
