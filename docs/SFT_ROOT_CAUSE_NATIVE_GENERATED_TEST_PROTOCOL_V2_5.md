# Native generated-test protocol — v2.5 (contract-aligned, verifier-filtered intervention)

Protocols v2, v2.1, v2.2, v2.3 and v2.4 are unchanged. v2.4 is **completed and closed**:
- It ran at commit `965005d`.
- Its closure receipt is `results/sft_root_cause_native_v24_closure.json`.
- Its preflights and GPU authorisation are retired and non-executable.

This protocol was frozen on 2026-10-01, before any v2.5 model output exists.

## 0. Evidence that motivates v2.5 (development panel only)

These come from v2.4 diagnostics. The 23-target v2.4 panel is **diagnostic only**: it never
enters training, and it is never used for confirmation.

1. **Training/evaluation contract mismatch (confirmed).** I rebuilt the adapter's exact
   training data (receipt `fac747a1…`):
   - 6,889 effective examples;
   - 80% are bare single assertions;
   - 0% import pytest, and 0% import the target;
   - 8.8% are self-contained modules.
2. **Mechanical repair restores validity but not kills (exploratory).** Receipts:
   definition `09927f6a…`, results `a21a9cfe…`.

   | Measure | Base | SFT |
   |---|---|---|
   | Reached | 41 → 181 | 19 → 110 |
   | Fixed-valid | 18 → 67 | 3 → 20 |
   | Kills | 1 → 1 | 0 → 0 |

3. **Semantic oracle weakness** is the leading hypothesis that remains open.
4. **The Atheris v4 adapter is inadequate for a fair comparison:**
   - 3 targets are eligible;
   - 16 are unsupported by the adapter;
   - 5 fail on infrastructure: the runner uses the system Python's attrs 23.2.0.

**Every scientific claim boolean stays false** until the untouched v2.5 confirmation panel
(section F) is evaluated.

## A. Output contract `pytest_module_v1` (the only v2.5 output type)

1. **Required form.** One complete importable Python module that contains:
   - the exact public target import, as given in the prompt;
   - `import pytest` only when pytest is used;
   - one focused test function, or one tightly related parameterised test;
   - setup and several assertions only when they establish one defect.
2. **Forbidden content:**
   - Markdown fences or prose;
   - a hidden or reference implementation;
   - patches, fixed code, official tests, or expected values copied from protected data;
   - several independent test cases.
3. **Label.** The prompt label is `pytest_module_v1` everywhere; `pytest_fragment` and
   `assert_statement` are not used in v2.5.
4. **Prompt skeleton.** The prompt may show a structural skeleton: the import lines plus
   `def test_<name>():`. The skeleton never contains an oracle or a target-specific expected
   value.
5. **Normaliser.** The deterministic R4 normaliser (`repair_ceiling_transforms` v1) is applied
   identically to every candidate of every arm before execution. Normalised and raw outcomes
   are both reported.

## B. Supervision corpus (train split only)

1. **Stage 1, deterministic conversion (CPU).** Each verified example of the adapter's exact
   train-only corpus is converted to `pytest_module_v1`:
   - the verified oracle is preserved;
   - the permitted target import comes from train-only metadata;
   - pytest and standard-library imports are added only when statically justified;
   - the body is wrapped in one test function.

   **Synthetic targets** are imported from a stable train-only module interface; the
   reference implementation is never copied into the test. **Repository targets** run in
   their native prepared environment.
2. **Acceptance.** A converted positive is accepted only if it:
   - parses and is collected;
   - imports successfully and reaches the target;
   - passes on the fixed/reference revision;
   - fails on the buggy/mutant revision;
   - gives the same result on a rerun;
   - passes the candidate safety policy.

   Anything that cannot be derived safely is rejected, never guessed. The build runs twice
   and must be byte-identical.
3. **Integrity:**
   - zero exact duplicates; near duplicates are reported;
   - isolation by repository, function lineage and bug lineage;
   - v2.4 diagnostic targets excluded;
   - validation and sealed records never read;
   - failed outputs are never positives.
4. **Balance:**
   - unique-first selection;
   - synthetic families count as one broad source group;
   - no repetition to reach a percentage;
   - caps per repository, function and bug family;
   - simple/moderate/complex slices, defect families and oracle categories all reported.
5. **Stage 2, rejection sampling (GPU; conditional).** This stage runs only if stage 1 lacks
   diverse real-repository or complex supervision.
   - It starts with a bounded train-only pilot that reports yields and projected cost.
   - A full run proceeds only if the projection is adequate.
   - It needs a v2.5 preflight and explicit GPU authorisation.

## C. Atheris design v5

1. **Interpreter.** The probe, search, worker and replays all use each target's **prepared
   interpreter**.
2. **Atheris install.** Atheris 2.3.0 is exposed through a read-only overlay. Nothing is
   installed into a locked environment, and the environment hash is compared before and after
   every run.
3. **Receivers.** Instance receivers are built only from public constructors in the permitted
   context. Fixed code, patches and official tests are never used.
4. **Argument types.** Supported: bounded primitives, unions, `Any` (a fixed primitive
   menu), enums and literals. Everything else is explicitly `adapter_unsupported:<reason>`.
5. **Separate statuses:** eligible, adapter-unsupported, infrastructure failure, usable
   non-kill, confirmed kill.
6. **Unchanged from v4:** budgets, tolerance, corpus cap, modes, seeds and the kill rule.

## D. Confirmation panel (untouched; frozen before training)

The panel must be:
- repository-, function- and bug-lineage-disjoint from training;
- untouched by v2.4 development;
- selected by structural criteria alone;
- drawn from multiple repositories;
- stratified by complexity and covering several defect families.

The all-target model panel and a structurally selected Atheris-comparable subset are reported
separately. Identities, revisions, environments, prompts and eligibility criteria are hashed
and frozen. No generation or fuzzing happens during construction. The sealed final test stays
unopened.

## E. One controlled comparison

The comparison has three arms:
- **A:** base Qwen;
- **B:** the existing SFT adapter;
- **C:** one v2.5 SFT trained on verified `pytest_module_v1` supervision.

**C changes exactly one factor, the supervision data.** The base model and revision, learning
rate, epochs, optimiser, regularisation and seed match the baseline; any unavoidable
difference is documented. C also includes:
- balanced replay against forgetting;
- no DPO or RLVR;
- no validation or test data returned to training;
- retained checkpoints;
- durable execution.

## F. Promotion

**Promotion** requires improvement on the untouched confirmation panel in semantic kills and
in fixed validity, with no unacceptable regression in execution, diversity or
repository-level generalisation. There is no arbitrary kill-rate threshold.

## G. Freeze discipline

All v2.5 source, data, Atheris and panel work is completed **before** a single source freeze.
That freeze then requires, in order:
1. focused tests;
2. the full suite, run alone;
3. sandbox and Atheris canaries;
4. corpus and leakage gates;
5. model, tokenizer and adapter identities;
6. job, manifest, dataset and configuration receipts;
7. a green source-bound preflight;
8. a clean HEAD equal to the freshly fetched origin.

**Every GPU step and every real-panel step needs its own explicit authorisation.** That
covers the train-split pilot, training, confirmation generation and real Atheris.
