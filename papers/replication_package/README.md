# Anonymized replication package — plan

**This is a plan. Nothing has been uploaded, archived or shared.**

## What this supports

Every numeric claim in the four accompanying paper drafts is traceable to a
receipt in this package by SHA-256. The receipts were written **before** the
measurements they govern: the generation protocol and evaluation scope before
the arms ran, and the promotion rule before either locked arm ran. That ordering
is what makes the reported decision interpretable rather than retrospective, and
it is checkable here — the rule is in a separate file from the result.

## Start here

| File | Purpose |
|---|---|
| `MANIFEST.md` | every file, its digest, and what it fixes |
| `provenance.json` | the same, machine-readable, with the source commit |
| `reproduction_commands.md` | exact commands to regenerate every table and figure |
| `exclusion_policy.md` | what is deliberately absent and why |

## What this package is not

It does **not** contain a final-test result. A one-time measurement on a
held-out split was authorized and attempted; it produced zero candidates and
zero reportable metrics, and that split may never be rerun. No number in the
papers derives from it, and no material from it is here, because none exists.

It does **not** contain a repository-level kill rate. Records of that execution
mode are excluded from every rate reported, since no validated native evaluator
for them exists.

It does **not** contain a comparison against any external baseline tool. None
was bundled or executed in any measurement reported.

It does **not** contain the evaluation pipeline source. The pipeline is pinned
by per-file digests inside the receipts, which is what the claims depend on. If
a venue requires the source, release it from the recorded commit as a separate,
anonymized artifact.

## Before archiving

1. Assemble the files listed in `MANIFEST.md` and generate `SHA256SUMS`.
2. Verify every digest against the full values in `provenance.json`.
3. Archive to an **anonymous-capable** service. A personal repository link would
   break double-anonymous review at all four target venues; ICST requires the
   replication package itself to be anonymized, and SANER's guidance says to
   avoid linking directly to code repositories that can reveal identity.
4. Confirm no file added during assembly carries an author name, host, account
   or remote URL.
5. Re-run `validate_papers.py` after assembly.

## A note on scope

The evidence snapshot is intentionally fixed at commit
`4f465622adf273ffcae672c93dfe3cc82c7d8645`. It predates the later TAP capacity
diagnostic (Gate 1) and therefore neither contains nor makes a claim from that
diagnostic. Updating a paper to use later evidence requires a new ledger,
provenance manifest, and review pass; the old snapshot must not be silently
relabelled as current.

This package is built for verification, not for re-execution. A reviewer can
confirm every reported number, re-derive every computed value, and check that
the decision rule predates the measurement. Re-running generation would need the
model and the corpus, which are not here. Each paper states that limit in its
threats-to-validity section rather than leaving it implicit.
