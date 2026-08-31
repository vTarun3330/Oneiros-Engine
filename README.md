# Oneiros Engine

Oneiros is a research pipeline for **execution-verified software test generation**. Its current Phase 3 task is deliberately scoped as:

> Given an already-localized defective Python source region, a behavioral specification, and legitimate buggy-repository execution context, generate one minimal, self-contained test case that passes on the reference behavior and fails on the supplied defective implementation.

Fault localization is outside the present test-generation scope. Repository records disclose this localized-region assumption; model-visible targets are selected from public metadata or buggy-side analysis, never by diffing against the reference implementation. Oneiros must not be described as an end-to-end fault-localization system.

## Active V4.1 pipeline

```text
verified training records
        ↓
gold-independent field construction + lineage audit
        ↓
V4.1 unified prompt + section-aware AST-unit budgeting
        ↓
execution-verified SFT
        ↓
training-only ablation_dev selection
        ↓
locked validation (seeds 42/43/44, 8 ordered candidates)
        ↓
DPO only if SFT reaches 58% on every completed locked seed
        ↓
one sealed final-test evaluation after all choices are frozen
```

The canonical corpus is `data/corpus/v4_1_research_hardened_candidate`. V4 remains frozen at commit `1d2cca8` and is identified by `research/baselines/V4_BASELINE_1d2cca8.json`; V4.1 never overwrites it.

## Visible and hidden information

Model-visible fields are limited to task mode, expected test format, sanitized public behavioral specification, buggy localized code, public/buggy-side target symbols, and non-gold buggy-environment support context. Every visible field has a `field_lineage` entry.

Reference code, gold patches, official test bodies, expected completions, oracle outcomes, mutation operators, fixed outputs, and hidden corrections are forbidden prompt lineage. Official tests remain usable only as hidden supervision/oracle evidence. The stronger audit is stored with the corpus as `leakage_audit.json`.

## Data and evaluation policy

- `train`: model fitting only.
- `ablation_dev`: fixed, semantic-group-disjoint subset removed from training and used for design choices.
- `val`: locked model selection and the unchanged 58% SFT gate.
- `test`: sealed until the final adapter, prompt, generation configuration, candidate count, and evaluator are frozen.
- Candidates remain in raw generation order. Reports separate requested, parse-valid, execution-valid, reference-valid, and killing candidates and include Kill@1/2/4/8.
- A partial seed is never a completed result.

The final test must not be inspected, debugged against, or rerun after retraining for the reported model.

## Training policy

SFT uses completion-only loss masking. Prompts are compacted by semantic section before chat rendering; complete target functions/classes are never token-spliced. The checkpoint monitor evaluates periodic checkpoints and the actual terminal optimizer step exactly once, including after resume.

DPO is not a rescue step for weak SFT. It starts only after the frozen SFT adapter passes the locked 58% per-seed gate, and SFT/DPO are then compared under the identical validation protocol.

## Prompt-budget prerequisite

The frozen 512-token function prompt budget is smaller than the prompt's own
required sections. Under fail-closed section-aware compaction this means most
records cannot be prompted at all: 388 of 542 `ablation_dev` and 583 of 757
`val` function records fail, and only about 1,930 of 5,595 synthetic training
records are usable. The preflight gate `evaluation_panel_fully_promptable`
blocks any run in this state.

The budget is not a free parameter to change quietly — it is part of the
evaluation scope hash. CPU checks reject 512 and 768; group J predeclares the
admissible 1024-vs-1280 comparison on `ablation_dev`. The integration queue
explicitly uses 1024 (with a separate `_p1024` run name and preflight artifact)
to test engineering readiness, not to select a research winner. All later
commands carry their budget explicitly and remain conditional on the group-J
decision. Runtime defaults remain unchanged; the failed 512-token evidence is
preserved. See `ABLATION_RESULTS.md` and `V4_1_NEXT_RUN.md`.

## Current repository limitation

Native generated-test execution against reconstructed BugsInPy and SWE-bench
buggy/fixed environments is implemented in `harness/native_repository_eval.py`
and driven by `scripts/evaluate_native_repository.py`. It is covered end to end
against a synthetic two-commit git repository, but it has **not** been validated
against real BugsInPy projects, which require provisioned checkouts and the
historical interpreters. Until such a run exists, repository examples provide
verified supervision and context coverage only and must not be reported as a
generated-test repository kill rate. A held-out realistic-mutation benchmark
(§37) is not implemented and remains the next scientific layer after that.

## Local verification

```powershell
py -3.12 scripts/v4_1_ready.py run-local --prompt-token-limit 1024
```

`run-local` performs the offline rebuild, lineage audit, complete test suite,
training-readiness audit, synthetic smoke and budget-specific 32-pair preflight;
it never launches a GPU. For just the preflight command, use
the integration stage in the generated queue or section 3 of the runbook.

The exact staged GPU commands and stop conditions are in `V4_1_NEXT_RUN.md`. Ablations are predeclared in `ABLATION_RESULTS.json`; unrun and negative results remain visible.

Generate the complete no-launch execution matrix and run its readiness doctor
with `py -3.12 scripts/v4_1_ready.py plan` and
`py -3.12 scripts/v4_1_ready.py doctor --check-modal`. The doctor checks Modal
authentication but deliberately does not treat authentication as proof of GPU
credit; the billing-aware failover dry run remains the spending gate.

Historical Phase 2 material is archived under `docs/archive/phase2/` and is not the active architecture.
