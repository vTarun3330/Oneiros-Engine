# SUPERSEDED — the v5 executable receipt is NOT executable authorization

**Status: non-executable. Must never be used to authorize a sealed final run.**

`results/v4_2_sealed_final_executable_receipt_v5.json`
(sha256 `eff006ca98abe11dee583eca8a5e98593d2968aedc7f35dc1da09075f9ef4a48`,
schema `oneiros_sealed_final_readiness_v5`, commit `41aeb26`) is preserved
byte-for-byte as evidence. It is **not** deleted, rewritten or concealed. The
entrypoint now refuses it by schema version, alongside v1, v2, v3 and v4.

v5's corrections stand in full: the shared prompt factory, the sealed loader no
longer importing a trainer helper, variant validation derived from the prompt
engine's own constants, and the four prompt-defining sources bound by hash. None
of that is reopened here.

A read-only authorization audit of v5 found two defects. Neither was in the
measurement logic. Both were in the artifact an operator reads before spending
the one authorization that cannot be taken back.

## Defect 1 (blocking) — the receipt's own command named a receipt it refuses

`scripts/preflight_sealed_final.py` built every receipt's `exact_command` from a
hardcoded literal:

```python
"--executable-receipt", "results/v4_2_sealed_final_executable_receipt_v4.json",
```

The schema version moved to v5. That literal did not. So the v5 receipt
instructed its operator to present the **v4** file — which the entrypoint
refuses by schema version.

It failed closed: presenting v4 is refused before the token is checked, so
nothing would have been spent. But the one artifact whose entire purpose is to
be the authorization instruction carried the wrong path, and the only test
covering `exact_command` asserted index `1` (the script name) and never looked
at index `3`. The same hardcoded path rode through v3, v4 and v5 unexamined.

**Fixed:** `collect()` now takes the output path and `exact_command` is built
from it, so a receipt always names the file it is about to become. There is no
receipt-version literal left in the command builder.

## Defect 2 (semantic accuracy) — a Git commit labelled as a SHA-256

The frozen bundle carried:

```json
"adapter_source_tree_sha256": "67f5940d843f40738ea3fe32b3766fc19bb97bc4"
```

That value comes from `git rev-parse HEAD`. It is a 40-hex SHA-1 commit id, not
a SHA-256 digest of a source tree — `git cat-file -t` confirms it is a commit
object. The field name asserted both a hash function and an input the value
never had, inside the frozen contract that a one-time measurement rests on.

The selected final candidate is the immutable base model with **no adapter**, so
the "adapter" half of the name was wrong as well.

**Fixed:** the field is `candidate_source_tree_git_commit`, in
`REQUIRED_BUNDLE_FIELDS` and in the preflight. It remains required: renaming must
not quietly drop a field from the frozen contract.

## What replaced it

`results/v4_2_sealed_final_executable_receipt_v6.json`, schema
`oneiros_sealed_final_readiness_v6`.

The entrypoint additionally has **no default `--executable-receipt` for a real
run**. A receipt selected by omission is a receipt nobody chose, and the stale
file a version bump leaves behind is precisely the one a default would select.
`--check-only` may still fall back, because it authorizes nothing.

## Regressions pinned

- A receipt generated to an arbitrary path records **that** path in
  `exact_command` — the only check a hardcoded literal cannot pass by
  coincidence.
- No `exact_command` may contain a v3, v4 or v5 receipt path.
- **No field ending in `_sha256`, anywhere in the receipt, holds a 40-hex Git
  commit id** — the general form of defect 2, not just the one field that was
  wrong.
- Every field named `_sha256` is 64 hex characters.
- `candidate_source_tree_git_commit` equals the commit the preflight ran at.
- v1 through v5 are refused; v6 is required; an unknown future schema is also
  refused.
- The v5 file on disk, presented as-is with its correct hash, is refused.
- A real run without an explicit `--executable-receipt` exits 2.
- The receipt's own `exact_command`, with the placeholder replaced by v6's real
  SHA and the token dropped, passes `--check-only`.

## Scope B is unchanged

The sealed final test remains **Oneiros-base-only**. No baseline is bundled,
none is executed, and the result cannot support a comparison against Atheris or
any other baseline. No validation baseline file was reintroduced.

## Evidence state at supersession

- `results/sealed_final_state.json` — **does not exist**. No authorization was
  ever granted and no token was ever spent.
- `results/sealed_final_run_state.json` — does not exist.
- `results/sealed_final_audit.log` — preserved, all entries
  `sealed_access_refused`, zero granted.
- **No sealed payload, ids or records were accessed at any point.**

## Standing instruction

Do not present the v1, v2, v3, v4 or v5 receipt to the entrypoint; all five are
refused by schema version. Only the v6 executable receipt may later be approved.
