# CAIN 2027 — Research Track (short paper) submission checklist

**Retrieved from the official CFP on 2026-09-18. Re-check before submitting.**

## Venue facts

| Item | Value |
|---|---|
| Official CFP URL | https://conf.researchr.org/track/cain-2027/cain-2027-call-for-papers |
| Conference | 6th International Conference on AI Engineering — Software Engineering for AI (CAIN 2027), co-located with ICSE 2027 |
| Submission deadline | **Fri 30 Oct 2026**, AoE (UTC−12). The CFP states the deadline is **firm, with no extensions.** |
| Notification | Mon 11 Jan 2027 |
| Camera ready | Fri 29 Jan 2027 |
| Page limit (short) | **5 pages plus a maximum of 2 pages containing ONLY references** ("up to 5+2 pages") |
| Formatting | "All submissions must conform to the IEEE conference proceedings template… (title in 24pt font and full text in 10pt type, LaTeX users must use `\documentclass[10pt,conference]{IEEEtran}`)" |
| Anonymity | **Double-anonymous.** "Paper review will employ a double-anonymous review process." |
| Artifact expectation | **Sharing is the default.** "Sharing is expected to be the default, and non-sharing needs to be justified." Artifacts should go to "an institutional or open platform committed to long-term archiving." |
| Figures/tables formally required? | **No.** |
| Screencast required? | No. |
| Registration/presentation | "at least one author needs to register for CAIN'27 to present the paper **in person**" for inclusion in the proceedings. |
| Generative-AI policy | **No explicit author-facing rule was found on the CFP page.** The CFP's author-facing text points to the ACM authorship policy, which covers AI use. The sentences about disclosing AI use "in the review form" and about not uploading papers to public AI platforms appear in the **reviewer** instructions and are **not** author obligations. **Author-facing generative-AI requirements must be confirmed from the current IEEE/ACM authorship policies and CAIN author instructions before submission.** |
| Scope warning | "Submissions that report predominantly on data science or AI/ML algorithms without any or only minor connection to software engineering for AI-enabled systems will be **desk-rejected**." |

### Scope compliance note

The CAIN scope warning is the main desk-rejection risk for this study, because
the underlying work involves fine-tuning a model. The draft is therefore
positioned as **evaluation infrastructure and quality assurance for an
AI-enabled testing system**, not as a new model or a machine-learning result:

- the model is used unchanged and no modelling contribution is claimed;
- the contribution is the harness — receipts, split isolation, provenance,
  retained raw outputs, and a full-scale rehearsal gate;
- Section VII argues explicitly why this is an AI-engineering contribution;
- the defects reported are software-ordering and access-control defects with
  software remedies.

**Before submitting, re-read the CAIN scope page and confirm the framing still
lands on the software-engineering side of that line.**

## Planned tables, figures and page budget

| Item | Type | Est. share of 5 pages |
|---|---|---|
| Figure 1 — evaluation-safety architecture with control annotations | TikZ, **double column** | ~0.40 |
| Table I — headline measurement (compact) | `booktabs` | ~0.20 |
| Table II — risk → implemented control | `booktabs`, small | ~0.40 |

Visual budget ≈ **1.0 of 5 pages**; references use the separate 2-page
allowance. **Deliberately cut:** the candidate-validity table, the claim-status
table, the split diagram, the development-vs-locked figure, and the incident
causal diagram — claim status is compressed into prose in Sections III and IX.

## Pre-submission checks

- [ ] **Anonymity** — no author name, affiliation, acknowledgment, institution,
      funder, repository host or identifying URL anywhere.
- [ ] **PDF metadata scrubbed.**
- [ ] **Page count** — body ≤ 5, references ≤ 2 additional pages, references
      only on those pages.
- [ ] **No padding or spacing manipulation.** Cut content if over.
- [ ] **Author-facing generative-AI requirements confirmed** from the current
      IEEE/ACM authorship policies and the CAIN author instructions. Do not rely
      on this checklist for that: the CFP's explicit AI sentences are addressed
      to reviewers, and the author-facing route is the ACM authorship policy,
      which must be read directly.
- [ ] **Confidentiality best practice** — do not upload an under-review
      manuscript to a public generative-AI service that may train on its
      content. *(Stated here as prudent practice. The CFP's explicit prohibition
      on this is a reviewer instruction, not an author rule.)*
- [ ] **Artifact archived** on an institutional or open long-term platform, and
      **anonymized** for review. Non-sharing would require justification; here
      sharing is straightforward, since the package is receipts and data files.
- [ ] **Scope check** — the paper reads as AI engineering, not as an ML paper.
      No claim of a modelling contribution.
- [ ] **Bibliography verified** against DOI / publisher / proceedings pages;
      resolve `% VERIFY-DOI` markers.
- [ ] **No citation to unpublished assistant transcripts.**
- [ ] **Evidence check** — every number matches `../common/evidence_ledger.md`.
- [ ] **Table II honesty check** — the two-phase authorization row is labelled
      *designed, not yet implemented*. Every other row must name a control that
      actually exists in the system. Do not promote a plan to a control.
- [ ] **Prohibited-claim grep** — see `../common/claims_traceability.md`.
- [ ] **Non-significance wording** — never "no effect" or "no difference".
- [ ] **TODO sweep.**
- [ ] **Compiles cleanly.**
- [ ] **In-person presentation feasible** — CAIN requires in-person presentation
      for inclusion in the proceedings, at ICSE 2027. Confirm travel is possible
      before submitting.

## Concurrent submission warning

> **Do not submit this paper to CAIN while the SANER RENE, SANER SP&P or ICST
> drafts in `papers/` are under review anywhere, and vice versa.** All four are
> derived from one study and one evidence base; they are substantially similar
> work. Concurrent submission violates IEEE and ACM policy and every venue's
> rules, and risks desk rejection at all of them simultaneously.
>
> CAIN's deadline (30 Oct 2026) falls **after** the SANER deadline
> (23 Oct 2026) and **before** ICST (2 Nov 2026), so these windows overlap.
> That makes an accidental double submission easy. Decide the order in advance
> and submit to exactly one.
