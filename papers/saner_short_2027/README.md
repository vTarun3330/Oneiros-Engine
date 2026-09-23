# SANER 2027 — Short Papers and Posters draft

**Short paper. SIX pages INCLUDING references. Double-anonymous.**

*A Development Gain That Did Not Survive Locking in LLM-Based Mutation Test
Generation*

## Positioning

Reduced to the three most defensible contributions:

1. the development-versus-locked gap, with the predeclared rule that was not met;
2. the candidate reference-validity regression, isolated from parse and
   execution validity;
3. the full-scale permitted-split rehearsal that reproduced a prior value
   exactly.

Everything else from the full versions is cut. The paper is written to be
**self-contained**: a reader who has not seen the longer versions gets the
setting, the frozen decision rule, and the reason the locked result is not a
null result.

## The binding constraint

**References count against the six pages** on this track — there is no separate
reference allowance. That is the most likely reason this draft goes over length.
If it does, cut citations and prose; do not shrink margins, line spacing or font
size.

## Contents

| Item | Location |
|---|---|
| Table I | Kill@8 by panel, reduced to four rows plus the paired difference |
| Table II | claim status, three compact blocks |
| Figure 1 | development vs. locked, four rows, Wilson intervals |

Cut relative to the full versions: the candidate-validity table (stated inline
instead), the split diagram, the pipeline diagram, and the incident diagram.

## Same-conference warning

SANER RENE and SANER SP&P are **tracks of the same conference**. Submitting to
both is not a hedge, it is a policy violation. Choose one.

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
