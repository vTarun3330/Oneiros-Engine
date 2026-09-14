# QUARANTINED — Step 1 run recorded an unpinned model revision

`base_validation_ablation-dev_parse-whole-output_completion1024_seed_42.json`
sha256 `1b53876a0104ea4e7c677ba2c465ead8ad023702a04ba1d11598e4612573b0a5`

**Preserved unchanged. Do not cite as an accepted control result.**

## Why

`base_model_revision` is recorded as `main` — a moving branch pointer — in all
three places the run describes itself:

- `generation_settings.base_model_revision`
- `run_contract.base_model_revision`
- `reproducibility.model_revision`

`main` is not an identity. The same string resolves to different weights and a
different tokenizer after any upstream push, so no later run can prove it used
the same model this one did.

## Cause

The revision pinning added in `b753ba1` fixed the shared
`resolved_base_model_identity()` and the preflight, but `train_on_dataset.py`
carried **three additional inline copies** of the same resolution — at the
resume-identity, adapter-validation and generation sites — each ending
`else "main"`. Those three are what write the run contract, the reproducibility
manifest and the generation settings.

The tests missed it because they exercised the resolver function rather than
the fields the artifacts actually record.

## What is and is not wrong with this artifact

The **execution was healthy**. Raw output is present and hash-matching for all
4,336 candidates, there were no prompt-budget failures, and the completion
limit was reached once (0.023%).

The **weights were almost certainly correct**: the local cache holds exactly
one Qwen snapshot, `2e1fd397ee46e1388853d2af2c993145b0f1098a`, and an offline
`main` resolves to it. But "almost certainly" is not provenance, and the
artifact cannot demonstrate it.

Its measurements are therefore usable as a diagnostic and are **not** an
accepted Step 1 control.

## Superseded by

The re-run Step 1 evaluation in a separate results directory, after the shared
immutable resolver lands.
