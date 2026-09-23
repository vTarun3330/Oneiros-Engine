# ICST 2027 — Research Papers Track draft

**Full paper. 10 pages + 2 reference-only pages. Double-anonymous. US Letter.**

*Kill Rate Is Not Enough: Selection Inflation and Candidate-Validity Regression
in LLM-Based Mutation Test Generation*

## Positioning

The most ambitious of the four drafts, framed as empirical software testing
rather than as an incident report. The thesis is that mutation kill rate is
**alone can be misleading when candidate and reference validity are not
reported alongside it** — a test that fails against every implementation still
distinguishes a mutant from its reference, so a kill-rate-only evaluation cannot
separate a discriminating test from a broken one.

Per the venue guidance, **limitations are made extremely explicit**: Section
VIII covers construct, internal, external, statistical-conclusion and
reproducibility threats, including a criterion that failed through a defect of
ours, recorded as a failure rather than reinterpreted.

Section IX is a full replication-package section with receipted artifacts and
their digests, as ICST expects a submission to "include or point to an
anonymized replication package".

## Contents

| Item | Location |
|---|---|
| Figure 1 | full evaluation workflow and split roles (double column) |
| Figure 2 | development vs. locked Kill@8 with Wilson intervals |
| Table I–II | main results; candidate validity |
| Table III–IV | rehearsal outcome; receipted artifacts and digests |
| Section IX | replication package |

## Figure 2 warning

The interval coordinates are hand-mapped (`x = (Kill@8 − 0.50) × 25` cm). An
earlier draft of this figure carried an **incorrect Wilson offset** for one row;
it was caught and fixed before the file was finalised. Re-derive the coordinates
from `../common/data/kill_at_8.csv` before submitting rather than trusting them.

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
