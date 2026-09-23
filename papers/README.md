# Oneiros paper drafts

This directory contains four venue-specific, double-anonymous LaTeX drafts and
their reproducibility material.  The drafts are alternative submissions of the
same historical evidence snapshot; they are not four independent studies and
must not be submitted simultaneously where venue policies prohibit it.

The evidence ledger is frozen at commit
`4f465622adf273ffcae672c93dfe3cc82c7d8645`. Later experiments, including the
TAP capacity diagnostic, are intentionally absent until their claims and
receipts are incorporated through a new traceability revision.

## Validate

From the repository root:

```text
python papers/common/validate_papers.py
```

The validator checks LaTeX structure, citations, double-anonymity, prohibited
claims, numeric traceability, and the compiled PDFs. LaTeX intermediate files
are ignored because they include machine-specific paths; the `.tex`, `.bib`,
evidence files, and final PDFs are retained.

## Drafts

- `saner_rene_2027/`: SANER Research, Early Research Achievements track.
- `icst_2027/`: ICST research paper draft.
- `saner_short_2027/`: SANER short-paper draft.
- `cain_2027/`: CAIN paper draft.

Each venue directory contains its own README and submission checklist. Conference
deadlines and formatting rules are time-sensitive and must be re-verified from
the official call for papers before submission.
