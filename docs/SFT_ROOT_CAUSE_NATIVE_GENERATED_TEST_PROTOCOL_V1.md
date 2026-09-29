# Native buggy/fixed execution of complete model-generated tests — frozen protocol v1

**Status:** frozen 2026-09-29, after the receiver-aware fixed-input pilot failed its
predeclared feasibility gate (Branch B;
`results/sft_root_cause_phase4_receiver_capture_receipt_v2.json`). Nothing here has been
launched. GPU generation needs a green preflight **and** explicit approval.

This path asks the original question directly: **does SFT change a model's ability to
write tests that kill real repository bugs?** Fixed-input value prediction was a
mediator study, not a prerequisite for this. The protocol specialises
`docs/REPOSITORY_NATIVE_EVALUATION_PROTOCOL.md` (§2 isolation, §4 qualification, §5
context policy, §7 Atheris, §9 metrics). Where this document is more specific, it
governs this experiment.

## Models compared

- **Base:** Qwen/Qwen2.5-Coder-1.5B-Instruct @ `2e1fd397ee46e1388853d2af2c993145b0f1098a`.
- **SFT:** the frozen arm A adapter `checkpoints/local_sft_armA_baseline_successor_s42/sft_adapter`,
  SHA-256 `e67dd599a37cbf2a738791c5cdf889cb4938a2ad53c991dca2fefae503b6f9e7`.
- **Not compared:** the Choice B adapters (internal, exploratory).

The same prompts, functions, seeds and generation settings are used for both models. Only
the adapter differs.

## Cohorts

- **Dress rehearsal:** the 24 natively qualified A-prime development targets in 8
  repositories. They exercise the pipeline only and are **never** confirmation or
  performance evidence.
- **Confirmation:** a **repository-disjoint** cohort, with no repository shared with the
  training corpus, the rehearsal pool or the A-prime development set.
  - Its bugs must be admitted under isolation v6 (§2) with authenticated provenance.
  - Qualification: fails 3/3 on buggy and passes 3/3 on fixed (§4).
  - Caps: at most 12 targets per repository and at most 25% per bug family (§6).
  - Size and power follow §6 and §9. Its acquisition is separately approved and is not
    authorised by this protocol.

## Generation (durable GPU, frozen)

- **Prompt:** built only from the buggy-side permitted view (§5): the target function,
  signature, docstring, imports and skeletons.
  - Never the fixed code, the diff, official tests, expected values, issue or PR text,
    or anything captured in the receiver pilot.
  - Every prompt is leakage-scanned against the fixed function, the patch and the
    official tests.
- **Output:** a **complete pytest test module** (imports, tests and asserts).
- **Settings:**
  - successor generation settings: 8 candidates, frozen temperature, top-p and seed,
    and parser;
  - generation seeds 42, 43 and 44, identical for both models;
  - identical per-target prompts and token budgets.
- **Execution:** runs through `scripts/gpu_run.py` with contracts, checkpointed progress
  and exact-contract resume, as in the Choice B v2 lifecycle.

## Execution (CPU/WSL, native)

- **Environment:** one per target, built from the fixed checkout.
  - It records the Python and uv versions, dependency-file hashes and a sanitised freeze.
  - It runs network-isolated in a temporary working directory.
- **Injection:** each candidate module is injected under an isolated file name into
  **both** revisions, with per-test timeouts. A discordant result is re-run once as a
  flakiness check.
- **Admission, in order:** syntax, collection, import, execution. Each stage failure is
  classified.
- **Kill criterion:** a candidate **fails on buggy and passes on fixed**.
  - Semantic kills (assertion) and crash kills (project-code exception) are reported
    separately.
  - A timeout on buggy is reported separately.
- **Separation of failures:** environment, dependency and harness failures are excluded
  and **never** counted as model failures or kills (§4 table).

## Atheris comparison

- **Tool:** actual `atheris` 2.3.0 on WSL Python 3.11 (present). Never the simulated
  fuzzer.
- **Callable:** the same target callable as the generated tests, reached through the
  §7 adapter. Ineligible targets are reported, never counted as zero.
- **Primary (equal budget):** per target and seed, Atheris gets the **same wall-clock
  seconds** that Oneiros spent on that target (generation plus native execution of 8
  candidates, measured). Its kill test is the same fails-on-buggy, passes-on-fixed
  replay.
- **Secondary (generous, labelled):** 600 CPU-seconds per target, mode and seed (§7).
- **Tertiary:** differential-oracle Atheris as an upper bound, reported separately.
- **Seeds:** 42, 43 and 44; identical functions for every system.

## Metrics and decisions

- **Primary:** SFT minus base Kill@8 on the confirmation cohort, with a paired
  repository-clustered bootstrap (10,000 resamples) and the decision regions of §9.
- **Secondary:** Kill@1 and Kill@4; semantic versus crash kills; admission-stage rates;
  and Oneiros versus Atheris on the jointly eligible set.
- **Dress-rehearsal gate (§8):** every stage has receipts, there are no unexplained
  failures, and at least 90% of qualified targets execute on both revisions.

## Provenance and boundaries

- **Hashes bound in the preflight:** exact source, model, adapter, dataset/cohort and
  environment hashes.
- **Access:** no locked-validation or sealed-final access, no Choice B gate reuse, and
  no training on anything produced here.
- **Claims:** `root_cause_established` and `generalization_established` stay false unless
  the confirmation cohort supports them under the frozen rules.
