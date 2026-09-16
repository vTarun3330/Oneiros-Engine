# SUPERSEDED — the v4 executable receipt is NOT executable authorization

**Status: non-executable. Must never be used to authorize a sealed final run.**

`results/v4_2_sealed_final_executable_receipt_v4.json`
(sha256 `b5d35660bbc6a4a629bce199ff660966c5e1abf5f9d67c7fb468a16105335aeb`,
schema `oneiros_sealed_final_readiness_v4`, commit `09f4404`) is preserved
byte-for-byte as evidence. It is **not** deleted, rewritten or concealed. The
entrypoint now refuses it by schema version, alongside v1, v2 and v3.

v4's corrections stand: RNG seeding, batch size 2, generator reuse, the
no-baseline scope, and the synthetic two-record GPU smoke. One
reproducibility defect remained, in the prompt.

## The defect: the sealed prompt followed a mutable global

`harness/sealed_final_loader.py` imported
`scripts.train_on_dataset.build_pair_prompt`. That helper calls `build_prompt`,
which read two module globals **at call time**:

```python
information_variant=PROMPT_INFORMATION_VARIANT,
output_instruction_variant=OUTPUT_INSTRUCTION_VARIANT,
```

`GenerationSettings` recorded `prompt_information_variant`,
`output_instruction_variant` and `prompt_schema_version` — but those frozen
values were never passed to the builder. The sealed run would have rendered its
prompts under whatever the trainer's CLI state held, not under what its own
receipt froze.

It rendered the *correct* prompt only by coincidence: the trainer defaults are
`full` and `self_contained`, which are also the frozen values. Coincidence is
not a binding. Anyone changing a CLI default would have silently changed the
prompts of the one measurement that cannot be repeated, with nothing in the
artifact to show it.

## A second defect in the same area: validation was wrong in both directions

`GenerationSettings.problems()` validated prompt variants against hand-written
sets:

```python
if self.prompt_information_variant not in {"full", "minimal"}:
if self.output_instruction_variant not in {"self_contained", "bare"}:
```

The prompt engine's own constants are:

- `PROMPT_INFORMATION_VARIANTS = ("code_only", "code_specification", "full")`
- `OUTPUT_INSTRUCTION_VARIANTS = ("legacy_exactly_one", "self_contained", "metamorphic_allowed")`

So the hand-written sets **invented** `minimal` and `bare`, which
`build_unified_user_prompt` rejects outright, while **refusing four variants the
engine accepts**. A validator that accepts invalid values and rejects valid ones
is not a validator.

## A third gap: prompt-defining sources were not bound

The receipt bound the evaluator, adapter, RNG helper, smoke, loader and
entrypoint — but nothing that decides what the model is *shown*. A change to the
prompt engine, the compaction code, or the record adaptation would have changed
every rendered prompt without changing a single recorded setting or hash.

## What replaced it

`results/v4_2_sealed_final_executable_receipt_v5.json`, schema
`oneiros_sealed_final_readiness_v5`.

- **`harness/prompt_factory.py`** — one builder taking `PromptSettings` as an
  argument. `prompt_factory(settings)` returns a callable closed over an
  independent copy of the frozen settings, so no later change to any module
  global can reach it. Development code may read its CLI globals to *construct*
  the settings; the builder never reads one.
- **The sealed loader no longer imports any trainer prompt helper.** It passes
  `prompt_factory(settings.prompt_settings())` to the adapter.
- **Validation now derives from the engine's own constants.** All six canonical
  variants are accepted; `minimal` and `bare` are rejected. The frozen
  `full` / `self_contained` configuration remains valid.
- **Four prompt-defining sources are bound by hash** — the prompt factory, the
  prompt engine, the compaction code, and the record adaptation — and the
  runtime guard refuses execution if any differs from the approved receipt.

The regression that would have caught the original defect is pinned directly:
build the sealed factory, mutate `trainer.PROMPT_INFORMATION_VARIANT` and
`trainer.OUTPUT_INSTRUCTION_VARIANT`, and assert the rendered prompt does not
move.

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

Do not present the v1, v2, v3 or v4 receipt to the entrypoint; all four are
refused by schema version. Only the v5 executable receipt may later be approved.
