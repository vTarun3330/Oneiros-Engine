# Native generated-test protocol — amendment v2.3 (additive; frozen before any model output)

Protocol v2 and amendments v2.1 and v2.2 are unchanged. This amendment was frozen on
2026-09-30, after the green v2.2b preflight (commit 5ca22ee, SHA-256 `bc6596df…`). An
independent review found downstream cohort, output-layout and generation-evidence defects
that the preflight did not test. **No model output exists; no generation has occurred.**

**What v2.3 does not change:**
- the prompts: the v2.2 job items and their prompt bytes and hashes are reused only after
  they are proved unchanged;
- the sampling distribution;
- the 2,048 / 1,024 / 3,072 token contract;
- the seeds;
- the candidate count.

Where v2, v2.1, v2.2 and v2.3 differ, **v2.3 governs**.

## 0. Historical evidence and a reporting correction

- **Preserved unchanged:**
  - every v2, v2.1 and v2.2 artifact;
  - the red v2.2 preflight and the green v2.2b preflight, which will not be used for launch;
  - job v3, manifest v5, the prompt records and all quarantines.
- **Correction:** the v2.2 report gave marshmallow@252090c as 1,766 prompt tokens. The
  committed prompt-record artifact records **1,748**, and the artifact is authoritative.
  1,766 came from a quarantined first build whose class headers still carried source
  comments. No receipt is edited to change prose.

## A. Cohorts

1. **Qualified cohort:** the 24 formally requalified, isolation-v6-admissible targets.
2. **Generation cohort:** exactly the 23 unique targets admitted by the frozen job. It is
   resolved only from the exact job artifact, never inferred from the kept or qualified
   targets.
3. **Pre-generation exclusion:** `cand:python-poetry/tomlkit@43668dd…`.
   - Its sequence overflow is recorded: prompt 2,385 tokens > 2,048; target source 1,540
     tokens.
   - It is never a missing generation, a model failure or an infrastructure exclusion.
   - It is never recovered.
4. **Expected sizes:**
   - per arm: 69 generation rows (23 targets × 3 seeds), 8 candidates per row, 552 candidates;
   - across both arms: **1,104** candidates and 1,104 execution rows.
5. **Base and SFT use identically:**
   - the ordered target set and prompts;
   - seeds 42, 43 and 44;
   - 8 candidates;
   - sampling (temperature 0.7, top-p 0.9, `do_sample`) and token budgets.

## B. Generation telemetry (schema `oneiros_native_generation_telemetry_v1`)

1. **Per target × seed row:**
   - the runtime chat-rendered prompt-token count;
   - the target seed;
   - total wall time;
   - wall time of each generation batch;
   - candidates requested and produced;
   - the arm;
   - the contract and identity hashes;
   - the model-load time of the producing process;
   - peak allocated and reserved CUDA memory, where available (null only for the mock
     backend).
2. **Per candidate:**
   - the raw output and its SHA-256;
   - the extracted module and its SHA-256;
   - `generated_tokens`;
   - `eos_reached`;
   - `finish_reason` ∈ {`eos`, `length`};
   - `hit_completion_limit`;
   - `fence_stripped`.
3. **Token-count convention:**
   - Counts use the exact generated token IDs, never text re-tokenisation.
   - The first generated token that is an EOS id (the model's generation-config EOS set,
     plus the tokenizer's EOS) ends the candidate. `generated_tokens` counts tokens up to
     **and including** that EOS.
   - Later tokens are batch padding and are excluded; this holds even when
     `pad_token_id` is an EOS id.
4. **Finish and limit:**
   - `finish_reason = eos` iff an EOS was generated.
   - `hit_completion_limit` is true only when no EOS was generated and exactly
     `max_new_tokens` (1,024) tokens were produced; `finish_reason` is then `length`.
   - Any other shape refuses the row.
5. **Timing:**
   - CUDA is synchronised before and after each timed section.
   - Batch time covers tokenisation plus `generate`; target-seed time is the sum over its
     batches.
   - Timing does not change sampling.
6. **Durability:**
   - A resumed row keeps its original telemetry.
   - A row with malformed or incomplete telemetry is refused and its file quarantined.

## C. Output layout

- Base and SFT write to **separate arm directories**:
  - `results/sft_root_cause/native_v23_generations/base/`
  - `results/sft_root_cause/native_v23_generations/sft/`
- Each directory holds `generations_primary_whole_module_<arm>.jsonl` and
  `contract_primary_whole_module_<arm>.json`.
- The executor accepts the generation root, or explicit per-arm paths.
- **It hashes every generation file and contract, and refuses:**
  - swapped arms;
  - a mismatched job, preflight, source identity, model, protocol or generation contract;
  - missing, duplicated, extra or stale rows.
- Generation files are never copied or merged by hand.

## D. Execution and analysis

1. **Execution:** only the 23 generation targets run, and exactly 1,104 rows are required.
   The execution contract binds:
   - the job file hash and `job_sha256`;
   - the successor manifest;
   - every generation file and contract hash;
   - the telemetry schema.

   Each execution row carries its candidate's generation telemetry.
2. **Analysis:**
   - It uses the exact 23-target job grid of 1,104 unique cells.
   - It reports separately:
     - qualified = 24;
     - generated/requested = 23;
     - pre-generation excluded = 1;
     - post-generation infrastructure exclusions.
   - It verifies the repository mapping of every generated target.
   - It binds the job and manifest hashes.
   - **Engineering telemetry report, for each arm:**
     - completion-limit hits (count and rate) and EOS rate;
     - generated tokens at p50, p90, p99 and maximum;
     - target-seed latency at p50, p90 and total;
     - duplicate candidates;
     - parsing, collection, execution, reach and fixed-valid rates;
     - Kill@1, Kill@4 and Kill@8.
   - A validity comparison is never reported without the completion-limit rates.
   - `engineering_dress_rehearsal` allows no inference.
3. **Atheris:**
   - It may run on all 24 qualified targets.
   - Joint Oneiros/Atheris comparisons use only generation targets ∩ Atheris-eligible
     targets ∩ infrastructure-eligible targets.
   - The excluded tomlkit target appears only in an Atheris-only section.
   - Every denominator is explicit.

## E. Launch gate (supplements v2.2 section D)

At launch, the gate:
- resolves every path in the preflight's `inputs` beneath the repository root, rejecting
  absolute and traversal paths;
- requires each input to exist, recomputes its SHA-256 and compares it with the immutable
  preflight.

A receipt-only descendant may change only an explicit allowlist: the preflight receipt and
the authorisation receipt. Any other committed change, under `results/` or elsewhere,
fails. Generation output in the ignored arm directories does not invalidate the second
arm's launch.

## F. Claims

- The engineering rehearsal cannot establish:
  - causal root cause;
  - generalisation;
  - an SFT benefit;
  - Atheris superiority.
- `root_cause_established` and `generalization_established` stay false.
