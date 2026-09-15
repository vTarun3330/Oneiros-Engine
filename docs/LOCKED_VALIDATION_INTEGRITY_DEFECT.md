# Locked validation, criterion 5: a defect preserved rather than repaired

**Status: historical. Not fixed in place, not re-scored, not grounds for rerunning anything.**

The locked-validation result receipt
`results/v4_2_locked_validation_result_receipt.json`
(sha256 `f7cdf84df8e65c51bd22410759c6fd54952ace3bfa80c6ab620674b53e288af9`,
commit `448ac38`) records criterion 5 as **FAIL**. This document explains why,
and why that FAIL stays.

## What failed

Criterion 5 of the frozen locked-validation promotion rule required, among
other things:

```
"both_arms_share_run_contract_sha256": true
```

The two arms did not share it. Base recorded
`320a87cc423b34ed8bb26502dbc503d163076b76edc4078a6d62089aa9d480e0`
in `locked_validation_binding.adapter_sha256`; Arm A@431 recorded
`e67dd599a37cbf2a738791c5cdf889cb4938a2ad53c991dca2fefae503b6f9e7`.

A field-by-field diff of the two run contracts showed the difference was
confined to **that single field and nothing else**.

## Why it failed

The cause is mine, and it is not a property of the data.

The final hardening step before locked validation placed a
`locked_validation_binding` block inside `run_contract`. That block carries
`adapter_sha256`. The criterion had been frozen earlier, when the adapter hash
sat *outside* `run_contract` — which is why all four arms of the development
experiment did share a contract hash.

Putting a per-arm fact inside a hash that was required to be shared made the
criterion **unsatisfiable by construction**: an arm that loads an adapter can
never share such a hash with an arm that does not. The criterion could not have
passed no matter what the models did.

## Why it was not repaired

Three reasons, in order of weight.

1. **A frozen criterion that gets reinterpreted once it becomes inconvenient
   was never frozen.** The whole value of freezing a rule before results exist
   is that the rule cannot move afterwards. Re-reading criterion 5 as "share
   everything except the adapter hash" after seeing it fail would have
   destroyed that value retroactively — including for the four criteria that
   were applied honestly.

2. **The decision does not depend on it.** Criterion 1 failed independently:
   Kill@8 +3.4346 points and net +26 functions both cleared their bars, but
   McNemar exact two-sided p = 0.0783 against a required p < 0.01. The
   decision — retain the base model — follows from criterion 1 alone. Nothing
   about the outcome hinges on criterion 5.

3. **Editing the receipt would destroy the evidence.** The receipt is the
   record of what was decided and on what basis. A record amended to look
   cleaner is not a record.

## What was actually intact

The substantive integrity properties criterion 5 was written to protect all
held:

| property | base | Arm A@431 |
|---|---|---|
| prompt-budget failures | 0 | 0 |
| raw-output present and hash-matching | 6,056 / 6,056 | 6,056 / 6,056 |
| `evaluation_scope_sha256` | `d37f75d6…`-era val scope, **identical across arms** | identical |
| evaluation-defining source blob identities | all 8 match the frozen receipt | all 8 match |

Both arms were verified by the post-run integrity verifier before the
comparison was computed. The arms were measured by the same evaluator over the
same panel under the same protocol. The comparison is sound; only the identity
*model* was wrong.

## What replaced it, going forward

`harness/comparison_contract.py` — version
`oneiros_comparison_contract_v2`, separately versioned and applied to future
measurements only. It splits the two kinds of fact that were wrongly merged:

- **shared** facts (corpus, split, scope, protocol, parse mode, candidate
  count, sampling, budgets, evaluator, policies) hash to `shared_sha256`, which
  **must** match across arms;
- **per-arm** facts (model, revision, adapter path, adapter hash, provenance,
  source run) hash to `arm_sha256`, which is **expected to differ** — and two
  arms sharing one is now itself refused, because that means the same weights
  were measured twice under two names.

`tests/test_comparison_contract.py` pins the regression directly: a
base-versus-adapter comparison must be valid *while* the arms' adapter hashes
differ.

That module does not read, re-hash, re-score or reinterpret any development or
locked-validation artifact, and a test asserts those receipts remain
byte-for-byte unchanged, with criterion 5 still recorded as `passed: false`.

## Standing instruction

This defect must not be used — now or later — to justify rerunning locked
validation, re-scoring its artifacts, selecting Arm A@431, or revisiting the
retain-base decision.
