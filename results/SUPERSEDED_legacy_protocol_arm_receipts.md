# The first Arm A and Arm B receipts are SUPERSEDED — preserved, not deleted

Applies to:

- `results/v4_2_armA_clean_preflight.json` (Arm A, `ready: true`, 24/24 gates)
- `results/v4_2_armB_sidecar_preflight.json` (Arm B, `ready: true`)
- `results/v4_2_armA_frozen_preflight.json` (already quarantined separately for
  concurrent pytest; see its own `.DIAGNOSTIC_ONLY.md`)

They are kept as evidence. **Neither may authorize training.**

## Why Arm A is superseded

Its recorded `future_generation_contract` describes the **legacy** protocol:

| field | recorded | successor requires |
|---|---|---|
| `candidate_parse_mode` | `first_assertion` | `whole_output` |
| `retain_raw_output` | `false` | `true` |
| function generation completion limit | 128 | 1024 |

Those were the trainer defaults and no flags were passed. Absence of
`candidate_parse_mode` means legacy, so the contract was legacy by omission —
exactly the failure mode `harness/successor_protocol.py` now exists to prevent.

A run launched under this contract would produce numbers protocol-noncomparable
with every successor measurement, in the same way that first_assertion Kill@8
0.629133 and whole_output Kill@8 0.282216 are noncomparable: a flag, not a
model.

## Why Arm B is superseded

Two reasons, either sufficient.

1. **It predates the force-inclusion fix.** Arm B was produced before the
   trainer began resolving O1 records against the full train split. Its
   `contract_source_hashes` and source tree therefore do not describe the
   executable source that would run. It receipts code that no longer exists.

2. **It inherits Arm A's legacy contract**, and its held-constant block covered
   only model identity and SFT hyperparameters — not parser mode, raw-output
   policy, generation budgets, sampling settings, seed, or evaluator hashes. It
   could not have refused a legacy Arm A, because it never looked.

## What remains valid in them

Nothing here retracts the measurements themselves.

- Arm A's selection SHA `598eef61...` reproduced across two independent runs,
  and is re-derived independently by the successor Arm B rather than trusted.
- The coverage finding stands and was load-bearing: 836 of 1,305 sidecar rows
  name records outside the bounded selection, which is what exposed the
  force-inclusion bug before any GPU time was spent.
- The 1,305-row sidecar at 15.9985% is unaffected. It is data, not a receipt of
  code, and its hashes are re-verified rather than reused on trust.

## Superseded by

The successor-protocol Arm A and Arm B receipts, run against
`harness/successor_protocol.py` (`oneiros_successor_generation_protocol_v1`)
and the post-fix executable source.
