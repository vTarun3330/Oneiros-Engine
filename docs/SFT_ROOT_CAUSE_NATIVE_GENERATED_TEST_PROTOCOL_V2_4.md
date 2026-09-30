# Native generated-test protocol — amendment v2.4 (additive; frozen before any model output)

Protocol v2 and amendments v2.1, v2.2 and v2.3 are unchanged. This amendment was frozen on
2026-09-30, after an independent audit of the green v2.3 preflight (commit d335b69,
SHA-256 `56f262bd…`) found fail-closed checks that the preflight did not enforce.

- **No model output exists; no generation, real execution or real Atheris run has
  occurred.**
- v2.4 supersedes v2.3 **only for new execution and analysis**. Every v2, v2.1, v2.2 and
  v2.3 artifact, including the quarantined first v2.3 Atheris-canary receipt and its
  successful retry, remains historical evidence and is never rewritten.
- Prompts, sampling, decoding, token budgets (2,048 / 1,024 / 3,072), seeds, candidate
  count and the 24 / 23 / 1 cohort are unchanged.

## 0. Interpretation (unchanged, restated)

- This is an **engineering dress rehearsal only**. It supports:
  - no causal root-cause claim;
  - no generalisation claim;
  - no SFT-benefit claim;
  - no Atheris-superiority claim;
  - no model-selection or promotion decision;
  - no confirmation statistics (intervals, p-values, non-inferiority);
  - no training use.
- Every scientific claim boolean stays false.

## A. Historical additivity

The historical scripts `scripts/native_generated_tests_preflight_v2.py` and
`scripts/native_generated_tests_preflight_v2_2.py` are restored byte-for-byte from commit
5ca22ee, the last pre-v2.3 commit. Successors are separate versioned files, never renames.
A regression test pins their bytes.

## B. Generation telemetry v2 (`oneiros_native_generation_telemetry_v2`; generator v4)

1. **Row identity:**
   - `seed` ∈ {42, 43, 44};
   - `key` = `<target_key>::<seed>`;
   - `target_seed` is an integer equal to `int(sha256(f"{seed}:{target_key}").hexdigest()[:8], 16)`.
2. **Prompt:** 0 < `prompt_tokens` ≤ 2,048, and `prompt_sha256` equals the job item's.
3. **Counts:** candidate and batch counts are exact (8 candidates, 4 batches).
4. **Memory:**
   - `peak_allocated_bytes` and `peak_reserved_bytes` are **per target-seed row**. CUDA
     peak statistics are reset (after a synchronise) immediately before the row, and read
     after a synchronise at its end.
   - Process-lifetime peaks are recorded separately as `process_peak_allocated_bytes` and
     `process_peak_reserved_bytes`.
5. **Enforcement:** these rules are checked when a row is generated, when a run resumes,
   and when generations are loaded downstream.

## C. Preparation binding

- The executor's `--prep` must resolve beneath the repository root. It must equal the
  manifest-declared requalification path
  (`results/sft_root_cause/native_v21_rehearsal/records.jsonl`) and hash to the declared
  SHA-256 (`13dcde39…`).
- **Records:**
  - exactly one record per qualified target, with no duplicates, missing or extra keys;
  - every record `requalified`;
  - views, view manifests, module hashes, environment, Python path, attestation and
    environment lock all present;
  - view hashes equal to the recorded view manifests.
- The same file and hash are bound into the preflight inputs and the Atheris contract.

## D. Auditable execution evidence

- **Each execution row retains, for buggy, fixed and kill reruns:**
  - the sanitised sandbox report: attestation and expected hash, collected nodes,
    collection errors, per-node phase outcomes, exception categories, reach flags,
    timeout, exit status, UID and wall time;
  - the SHA-256 of the raw report;
  - the static-admission result.
- The host-derived process tail is dropped; only sandbox-relative paths appear.
- A pure verifier recomputes every classification, and `fixed_valid`, from the retained
  evidence. A disagreeing row refuses the analysis.
- An environment-failure row retains the failed canary evidence.
- Rows keep the candidate generation metadata. The execution contract binds the
  generation files and contracts, so raw and extracted modules stay recoverable.

## E. Atheris (design v4)

1. **Ordinary mode:** a kill requires all of:
   - buggy outcome `raise` and fixed outcome `ok`;
   - both stable across confirmations;
   - no replay error.

   Two different exceptions are **not** a kill. The post-hoc and differential semantics are
   unchanged and remain separately labelled.
2. **One authoritative loader** validates contract and results together. For the
   rehearsal it requires:
   - 24 qualified targets × {ordinary, posthoc, differential} × {42, 43, 44} = **216**
     unique cells, with no missing, duplicate, extra, malformed or stale rows;
   - one contract;
   - the design version and the script, inner, preparation and manifest hashes;
   - a budget of 600 CPU-seconds, the 1.0 s + 2% tolerance and the 2,000 corpus cap.
3. **A usable row requires all of:**
   - eligible;
   - target reached and views unchanged;
   - within budget, with cleanup and replay cleanup both succeeding;
   - end reason `completed` or `cpu_budget_exhausted`;
   - zero replay errors;
   - a kill consistent with its confirmed replay evidence.
4. **Target handling:**
   - A target is jointly Atheris-eligible only if all nine of its rows are usable.
   - An Atheris-eligible target with any unusable row is an explicit, target-level
     **infrastructure exclusion**, never a non-kill.
   - The pre-generation-excluded target appears only in the Atheris-only section.

## F. Engineering-only analysis

- The rehearsal manifest declares `study_mode = engineering_dress_rehearsal`. The analysis
  CLI accepts only the manifest's own study mode.
- `confirmation` additionally requires a separately frozen confirmation manifest with an
  authorisation receipt. None exists, so a CLI flag alone can never enable confirmation.
- Engineering mode emits no interval, p-value, non-inferiority result, promotion,
  causal or model-selection conclusion.

## G. Inherited coverage gate

- **`engineering_gate_passed`** requires all of:
  - at least ⌈0.9 × qualified⌉ targets that are generated **and** infrastructure-eligible
    in both arms (22 of 24);
  - at least 5 represented repositories;
  - receipts for every stage;
  - no unexplained failure;
  - passing canaries.
- **Reported separately:**
  - 24 qualified;
  - 23 generated;
  - 1 pre-generation exclusion;
  - post-generation infrastructure exclusions;
  - the final eligible count;
  - the represented repositories.
- If the gate fails, all arm-comparison output is suppressed and the failures are listed.
  The threshold is never weakened.

## H. Duplication

Reported per arm, with explicit denominators:
- within-target-seed duplicates;
- per-target uniqueness across the 24 candidates of the three seeds;
- module hashes repeated across seeds;
- unique candidate count and rate.

Duplicates stay in the primary grid; nothing is reranked or deleted.

## I. Sequential launch

1. `scripts/gpu_run.py --exclusive-key native_v24_generation`:
   - A run with the same key blocks another launch while it is running, or while its
     state is unknown (fail-closed).
   - Completed runs, and runs whose recorded processes are all dead, do not block.
   - The key is independent of `--run-name` and of the training-artifact validator.
   - The frozen commands never use `--allow-concurrent`.
2. The SFT launch additionally requires the base arm to pass an exact single-arm verifier:
   - 69 rows and 552 candidates;
   - job, preflight, source, model and protocol identity;
   - full telemetry v2;
   - no missing, duplicate, partial, stale or malformed row;
   - base file and contract hashes recorded in the SFT identity.

   Arm files are never copied or merged.

## J. Reproducibility receipts

The v2.4 full-suite receipt and the three v2.4 canary receipts are tracked through narrow
`.gitignore` exceptions and are checked for secrets, user paths and protected data. The
v2.3 local receipts are preserved unchanged.

## K. Artifacts

- job v5 and manifest v7;
- preflight v2.4;
- output directories `results/sft_root_cause/native_v24_generations/{base,sft}`.

The preflight binds and rehashes every input, including the preparation file.
