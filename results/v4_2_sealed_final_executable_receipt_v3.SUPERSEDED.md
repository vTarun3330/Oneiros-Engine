# SUPERSEDED — the v3 executable receipt is NOT executable authorization

**Status: non-executable. Must never be used to authorize a sealed final run.**

`results/v4_2_sealed_final_executable_receipt_v3.json`
(sha256 `70821ef9504f76b7521ba236b5c92dc10275850c86a64341e07b9fbdbece7611`,
schema `oneiros_sealed_final_readiness_v3`, commit `03f1fc1`) is preserved
byte-for-byte as evidence. It is **not** deleted, rewritten or concealed. The
entrypoint now refuses it by schema version, alongside v1 and v2.

v3 did fix the nonexistent APIs and the legacy parse mode. An independent audit
then found five further defects, all of which would have degraded or invalidated
the one measurement that cannot be repeated.

## 1. Seed recorded but never applied

The bundle declared `generation_seed: 42`. Locked validation applied it —
`torch.manual_seed` and `torch.cuda.manual_seed_all` immediately before
generation — but that call was inline in the evaluator, so the sealed path
inherited the *claim* of a seed without the act of seeding. A seed recorded and
not applied is worse than none: the artifact asserts a reproducibility that does
not exist, and nothing in the artifact reveals it.

**Fixed:** `harness/generation_rng.py` is one shared initializer, called by both
the locked/development evaluator and the sealed path. The sealed run performs
the smoke, then **resets all generation RNGs to seed 42 immediately before real
sealed generation**, and records the reset — with RNG state fingerprints before
and after, so a reset that no-oped would be visible.

## 2. Batch shape differed from locked validation

Locked-validation artifacts record `batch_size: 2`. The v3 sealed path generated
one target per call. Batch composition changes left-padding, and padding changes
what the model samples — so the sealed number would not have been comparable
with the measurement it extends.

**Fixed:** `generation_batch_size` is frozen at 2, enforced by the settings
contract, and the evaluator generates in chunks through
`generation_adapter.generate_candidate_slots`, scoring each target individually.

## 3. Generation semantics read from mutable module globals

Several settings that decide the result were unbound: batch size,
`allow_test_function_candidates`, prompt information and output instruction
variants, prompt schema version, seed application method, attention backend,
tokenizer revision.

**Fixed:** all of them are fields on `GenerationSettings`, validated by
`problems()`, frozen into the bundle, and passed explicitly through the shared
adapter. The sealed path reads no module global.

## 4. A second model load after the token was spent

The v3 smoke loaded the model, then `sealed_generator(receipt)` **constructed
and loaded a second one** after authorization. A failure in that second load
would have wasted the single authorization on code that had never been proven —
the same class of defect as v1's ordering bug.

**Fixed:** the model is prepared once in phase 1, smoke-tested, and the **same
generator object** runs the sealed batch. The run artifact records
`same_object_as_smoke`.

## 5. Implied baseline comparison

The v3 bundle pinned val baseline artifacts — `v4_2_baseline_bundle_val.json`
and `v4_2_atheris_tasks_val.manifest.json` — inside a sealed-final freeze. Those
baselines were never run on sealed targets under the sealed budget, so their
presence implied a comparison the measurement cannot support.

**Fixed, by choosing option B explicitly:** the sealed final test is scoped to
the immutable-base Oneiros candidate alone. No baseline is bundled, none is
executed, and the receipt states that the result **cannot support comparative
baseline claims**. Option A — running frozen baseline runners on the sealed
targets — would require its own freeze, rehearsals and authorization.

## What replaced it

`results/v4_2_sealed_final_executable_receipt_v4.json`, schema
`oneiros_sealed_final_readiness_v4`, which binds by hash: evaluator, shared
generation adapter, RNG helper, smoke module, sealed loader, entrypoint, the
complete frozen generation settings, and the baseline scope.

The required pre-authorization smoke now exercises a **two-record synthetic
batch** and proves: two prompts padded and generated together, eight candidates
per target, raw-output retention and hashing for all sixteen slots,
`whole_output` parsing read back off the generator, and candidate scoring for
every slot.

## Evidence state at supersession

- `results/sealed_final_state.json` — **does not exist**. No authorization was
  ever granted and no token was ever spent.
- `results/sealed_final_audit.log` — preserved, all entries
  `sealed_access_refused`, zero granted.
- **No sealed payload, ids or records were accessed at any point.**

## Standing instruction

Do not present the v1, v2 or v3 receipt to the entrypoint; all three are refused
by schema version. Only the v4 executable receipt may later be approved.
