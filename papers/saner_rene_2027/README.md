# SANER 2027 — RENE Track draft

**Full paper. 10 pages + 2 reference-only pages. Double-anonymous.**

*A Development Gain That Did Not Survive Locking: Negative and Inconclusive
Results from an LLM-Based Mutation Test Generation Pipeline*

## Positioning

The RENE track exists for exactly this shape of result, so the paper leads with
the negative and inconclusive findings rather than burying them: a selection-panel
gain that did not establish reliable improvement under locking, and a candidate
reference-validity regression moving opposite to the headline metric. The
**methodological lessons are the central contribution** — Section VII states ten
of them as checks rather than principles, because principles are easy to agree
with and hard to fail.

The paper takes deliberate care over one distinction the track's reviewers will
look for: a non-significant paired difference is reported as *insufficient
evidence of an effect at a predeclared standard*, never as evidence of no
effect. Section IV says so explicitly and Section VIII repeats it under
statistical conclusion validity.

The sealed-final incident is included as a reproducibility lesson, with a causal
diagram (Figure 2), but no record, identifier or payload from the consumed split
appears anywhere.

## Contents

| Item | Location |
|---|---|
| Table I | main results, both panels, paired difference |
| Table II | candidate validity across four arms and both panels |
| Table III | claim status: established / inconclusive / unavailable |
| Figure 1 | split roles, with the consumed split carrying no number |
| Figure 2 | incident → permanent refusal → rehearsal safeguard |
| Section VII | ten actionable controls |

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
