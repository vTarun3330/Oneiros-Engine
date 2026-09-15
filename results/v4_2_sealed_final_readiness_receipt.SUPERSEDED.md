# SUPERSEDED — `v4_2_sealed_final_readiness_receipt.json` is NOT executable authorization

**Status: pre-implementation readiness only. Must never be used to authorize a
sealed final run.**

The file beside this one,
`results/v4_2_sealed_final_readiness_receipt.json`
(sha256 `4ddad9834ed5e452e4b792842755f15967ab1dca9ece9ea3f659eb274e7f2b9a`,
schema `oneiros_sealed_final_readiness_v1`, commit `b237d65`), is preserved
byte-for-byte as evidence of what was prepared and when. It is **not** deleted,
hidden or edited. It is also not sufficient to open the sealed split.

## Why it was superseded

A critical ordering defect existed in the entrypoint it described.

`scripts/run_sealed_final_test.py`, as committed at `b237d65`, called
`guard.open_sealed_split(...)` — which **spends the one-time token** — and only
then reached this:

```python
# Reaching here means the guard granted the single authorized read and has
# already spent the token. The measurement itself is intentionally not
# implemented in this commit
```

So a valid authorization could have been consumed irreversibly to discover that
no measurement existed. The sealed split — the one thing in this project that
cannot be measured twice — would have been recorded as opened, with no result,
and no second chance.

The v1 receipt did say the measurement was unimplemented. That is not enough.
A receipt marked *ready for authorization* invites exactly the reading that
would have destroyed the measurement, and a comment inside a script is not a
safeguard.

## What replaced it

`results/v4_2_sealed_final_executable_receipt.json`, schema
`oneiros_sealed_final_readiness_v2`, which additionally declares:

- `final_evaluator_executable: true` — the entrypoint **refuses any receipt
  that does not**, so a v1 receipt cannot authorize a run even if presented;
- `final_evaluator_status` — in words, whether authorization must be refused;
- `final_evaluator_source` — version, canonical and raw hashes of
  `harness/sealed_final_evaluator.py`, its measurement-logic digest, and the
  canonical hash of `harness/sealed_final_loader.py`, so the code approved and
  the code that runs are demonstrably the same;
- `exact_command` — the command this receipt authorizes.

The entrypoint now checks **every** non-sealed prerequisite before presenting
the token: receipt hash, bundle freeze, evaluator version and source identity,
loader importability and identity, prior-run state, model files, model
revision, output path writability, free disk, and CUDA. A `--check-only` mode
runs all of it and presents no token at all.

## Evidence state at the time of supersession

- `results/sealed_final_state.json` — **does not exist**. No authorization was
  ever granted and no token was ever spent.
- `results/sealed_final_audit.log` — preserved. Four entries, all
  `sealed_access_refused`, zero `sealed_access_granted`. All four came from
  tests presenting an unknown token.
- **No sealed payload, ids or records were accessed at any point.**

## Standing instruction

Do not present this v1 receipt to the entrypoint. Do not treat its
`ready_for_authorization: true` as executable authorization. The only receipt
that may later be approved is the v2 executable receipt named above.
