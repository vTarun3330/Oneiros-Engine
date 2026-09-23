# Replication package manifest

**Plan only. Nothing here has been uploaded, published or shared.**

This directory describes what an anonymized replication package would contain
and how to verify it. Assembling and archiving it is a separate, explicit step.

Source snapshot: commit `4f465622adf273ffcae672c93dfe3cc82c7d8645`, branch
`experiment/research-eval-ablations`. **Commit identifier only** — no repository
host, account, remote URL or author name appears anywhere in this package, in
keeping with double-anonymous review.

Machine-readable form with full digests: `provenance.json`.

## Contents

### Frozen receipts — the evidence every number rests on

| File | SHA-256 (prefix) | What it fixes |
|---|---|---|
| `v4_2_frozen_development_evaluation_receipt.json` | `062774028693cfcb` | development protocol and scope, frozen before the arms ran |
| `v4_2_development_selection_receipt.json` | `0188324fe9eb2a51` | four-arm metrics, predeclared primary comparison, selection decision |
| `v4_2_locked_validation_preflight.json` | `97be6855d41b1186` | **the promotion rule, committed before either locked arm ran** |
| `v4_2_locked_validation_result_receipt.json` | `f7cdf84df8e65c51` | locked metrics, paired comparison, criterion outcomes, decision |
| `v4_2_rehearsal_receipt.json` | `bae9510218641b67` | rehearsal scope and settings, frozen before the run |
| `v4_2_rehearsal_execution_receipt.json` | `06aca918d2821a1f` | **sanitised** rehearsal outcome: counts, integrity, failure taxonomy |

### Figure and table inputs

| File | SHA-256 (prefix) |
|---|---|
| `data/kill_at_8.csv` | `fe32f0205dbbfc66` |
| `data/validity.csv` | `d59e1724f3a57dc8` |
| `data/rehearsal_counts.csv` | `c0ccec8cab8249da` |

Each CSV carries a header comment naming its source receipt and that receipt's
digest, so a figure traces to an artifact without opening the paper.

### Traceability documents

| File | SHA-256 (prefix) |
|---|---|
| `evidence_ledger.md` | `0fe9232982bfd9bb` |
| `claims_traceability.md` | `ec87d1a321175e84` |
| `reference_verification.md` | `8fae02b7d26fb1ec` |
| `figure_and_table_inventory.md` | `54cb6646f28613f3` |

Digest prefixes are shown for readability; `provenance.json` carries the full
64-character values, which are what a verifier must check.

### Scripts

| File | Purpose |
|---|---|
| `validate_papers.py` | structural and claim validator for the drafts |
| `reproduction_commands.md` | exact commands to regenerate every table and figure |

## Verification

```
sha256sum -c SHA256SUMS
```

Every digest must match the full value in `provenance.json`, which also records
the expected file count.

## What is deliberately absent

See `exclusion_policy.md`. In summary: no raw model outputs, no prompts, no
candidate code, no material from the consumed final split — none exists — and no
link, host or identifier that could reveal an author.
