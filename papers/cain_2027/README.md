# CAIN 2027 — Research Track short paper draft

**Short paper. 5 pages + 2 reference-only pages. Double-anonymous.
In-person presentation required.**

*Evaluation Integrity as an Engineering Concern: Receipts, Split Isolation and
Rehearsal for AI-Enabled Testing Systems*

## Positioning

This is **not framed as a machine-learning paper**, deliberately. CAIN
desk-rejects submissions that "report predominantly on data science or AI/ML
algorithms without any or only minor connection to software engineering for
AI-enabled systems", and the underlying study does involve fine-tuning a model.

So the model here is used unchanged and no modelling contribution is claimed.
The contribution is the **harness**: frozen hash-bound receipts, per-file source
binding, structural split isolation, retained raw outputs, explicit generation
semantics, and a full-scale rehearsal gate. Section VII argues directly why
trustworthy evaluation infrastructure is an AI-engineering deliverable — the
harness decides whether any model claim is admissible; the defects encountered
were software-ordering and access-control defects with software remedies; and an
irreversible evaluation resource needs release engineering.

## Contents

| Item | Location |
|---|---|
| Figure 1 | evaluation-safety architecture with control annotations (double column) |
| Table I | headline measurement, compact |
| Table II | risk → implemented control |

## Honesty constraint on Table II

Every row must name a control that **actually exists in the system**. The
two-phase authorization row is labelled *designed, not yet implemented* and must
stay labelled. Do not promote a plan to a control.

## Two CFP items that are easy to miss

- **The CFP's explicit generative-AI sentences are reviewer instructions, not
  author rules.** "AI use must be disclosed in the review form" and the
  prohibition on uploading papers to public AI platforms both sit in the review
  instructions. The author-facing route is the ACM authorship policy the CFP
  links. **Confirm author-facing generative-AI requirements from the current
  IEEE/ACM authorship policies and the CAIN author instructions before
  submitting** — an earlier version of this checklist wrongly stated the
  reviewer rule as an author obligation.
- The deadline is **firm, with no extensions**, and it falls between the SANER
  and ICST deadlines — which makes an accidental concurrent submission easy.

## Building

No LaTeX toolchain was available on the machine where these drafts were written,
so **none of them has been compiled**. The sources were validated structurally
instead (balanced environments, every `\cite` key resolving to an entry in
`references.bib`, no identity-revealing strings, no prohibited claims). Compile
before trusting anything about length:

```
pdflatex main
bibtex   main
pdflatex main
pdflatex main
```

Then **count the pages in the produced PDF**. The page-budget figures in
`submission_checklist.md` are estimates, not measurements.

## Evidence discipline

Every number in `main.tex` is traceable to a committed receipt through
`../common/evidence_ledger.md`, which records each artifact's SHA-256. Nothing
is recomputed, rounded into a new figure, or inferred.

`../common/claims_traceability.md` lists every claim this draft is permitted to
make, graded established / inconclusive / unavailable, and the statements that
are mechanically excluded.

`../common/figure_and_table_inventory.md` records why each table and figure
exists, where its values come from, and what was cut for this venue's page
limit.

## Before submitting — non-negotiable

1. **Verify every bibliography entry** against a DOI, publisher page or official
   proceedings page. The entries are real works, but author/title/venue/year
   were written from knowledge of the literature and DOIs are supplied only
   where confidence was high. Entries marked `% VERIFY-DOI` have no DOI on
   purpose rather than a guessed one. Do not submit this bibliography on trust.
2. **Re-check the CFP.** The venue facts in `submission_checklist.md` were read
   on 2026-09-18. Deadlines, page limits and policies change.
3. **Resolve or remove every `TODO`.** Unverified facts are marked in the
   evidence ledger and are absent from the paper; do not add them back without
   evidence.
4. **Do not submit this and a sibling draft concurrently.** See the warning at
   the end of `submission_checklist.md`.
