# SUPERSEDED — the v2 executable receipt is NOT executable authorization

**Status: non-executable. Must never be used to authorize a sealed final run.**

`results/v4_2_sealed_final_executable_receipt.json`
(sha256 `a1cafebac4f2de657f08b91d896492877b0902739597a4f7cab8600185a40353`,
schema `oneiros_sealed_final_readiness_v2`, commit `58cfa9a`) is preserved
byte-for-byte as evidence. It is **not** deleted, rewritten or concealed. It is
also not sufficient to open the sealed split, and the entrypoint now refuses it
by schema version.

## Why it was superseded

An independent audit found that `harness/sealed_final_loader.py`, which the v2
receipt declared executable, could not have worked:

1. **`build_test_generation_prompt` does not exist.** The real prompt functions
   are `build_unified_user_prompt` and `format_chat_prompt`.
2. **`Phi3Generator.generate_candidates` does not exist.** The real generation
   call is `generator.model.generate(...)` followed by
   `generator._parse_output(...)`.
3. **`parse_mode` was never set.** `Phi3Generator.parse_mode` defaults to
   `"first_assertion"`. The sealed run would therefore have been scored by the
   **legacy parser** while the receipt claimed the frozen successor protocol.
4. **No test executed the real path.** Every test injected a mock generator, so
   nothing above was reachable by the suite.

Defects 1 and 2 would have raised `ImportError`/`AttributeError` — but *after*
the token was spent, because the imports sat inside a function body while the
entrypoint's check only confirmed the module imported. That check was therefore
vacuous: it proved the file parsed, not that the path worked. This is the same
ordering defect the v2 work claimed to have fixed, one level deeper.

Defect 3 is worse, because it would not have raised at all. It would have
produced a plausible Kill@8 under the wrong parser, incomparable with locked
validation, on the one measurement that cannot be repeated.

## What replaced it

`results/v4_2_sealed_final_executable_receipt_v3.json`, schema
`oneiros_sealed_final_readiness_v3`.

- **One shared generation path.** `harness/generation_adapter.py` now holds the
  generation body. `scripts/train_on_dataset.py` delegates to it, so locked
  validation and the sealed run execute the same code. A second implementation
  is a second thing to be wrong, and its numbers would not be comparable with
  the first's even when both run.
- **No invented APIs.** The preflight resolves every symbol the path will call
  — `Phi3Generator._parse_output`, `compact_unified_user_prompt`,
  `build_unified_user_prompt`, `format_chat_prompt`, `build_pair_prompt`,
  `_record_to_pair`, `generate_candidate_slots` — and fails if
  `generate_candidates` ever reappears.
- **Parse mode is set explicitly, from settings, every time**, and the smoke
  test reads it back off the generator after generation rather than assuming it.
- **A required GPU smoke test** runs on a synthetic public record before any
  token can be presented: loads the exact Qwen snapshot, builds the real frozen
  prompt, generates exactly eight candidates, retains and hashes raw outputs,
  parses under `whole_output`, and scores against synthetic golden/mutant code.
  It verifies the sealed loader was not imported.
- **A non-sealed schema rehearsal.** Split parsing is a pure function
  (`select_split_records`) exercised against the permitted `ablation_dev` shard.

## A fourth defect the rehearsal caught

Writing that rehearsal exposed something the audit did not list: the loader
required fields named `golden_code` and `mutant_code`, which **no canonical
record has**. The corpus uses `reference_code` and `code_under_test`; the
adapted names are produced by `_record_to_pair`. The sealed run would have
loaded records the evaluator could not score. The loader now validates the
canonical field names and reuses `_record_to_pair` rather than reimplementing
the mapping.

## Evidence state at supersession

- `results/sealed_final_state.json` — **does not exist**. No authorization was
  ever granted and no token was ever spent.
- `results/sealed_final_audit.log` — preserved, all entries
  `sealed_access_refused`, zero granted.
- **No sealed payload, ids or records were accessed at any point.**

## Standing instruction

Do not present the v1 or v2 receipt to the entrypoint; both are refused by
schema version. Only the v3 executable receipt may later be approved.
