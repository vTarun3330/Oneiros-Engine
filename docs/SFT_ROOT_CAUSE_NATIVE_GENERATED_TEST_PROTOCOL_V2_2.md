# Native generated-test protocol — amendment v2.2 (additive; frozen before any model output)

Protocol v2 and amendment v2.1 are unchanged. This amendment was frozen on 2026-09-30,
after the v2.1 preflight returned `pipeline_ready=false` because job coverage failed
(19/24). **No model output exists; no generation has occurred.** This is a pre-generation
engineering correction. Where v2, v2.1 and v2.2 differ, **v2.2 governs**.

## 0. Historical evidence

- The following remain immutable historical evidence and are never rewritten:
  - the v2.1 job (`results/sft_root_cause_native_v21_rehearsal_job_v2.json`);
  - manifest v4;
  - isolation v6;
  - the v2.1 preflight (commit 29b880f);
  - the formal records;
  - the quarantine;
  - the Python 3.13 preliminary run.
- The red v2.1 preflight correctly caught a failed gate under the v2.1 implementation.
- **Correction of the v2.1 report:**
  - four prompts exceeded 2,048 tokens, not three: pyparsing 5,254, tomlkit 3,292,
    marshmallow 3,079 and sqlglot 2,615;
  - marshmallow@252090c was both leakage-refused and over the limit;
  - a scanner correction alone would therefore give 20/24, not 21/24.
- The coverage rule is **unchanged**: at least 20 targets and at least 90% of kept targets
  in the primary job, which is ≥ 22 of 24. There is no replacement, no cohort expansion and
  no weakening.

## A. Leakage scanner v2: exact-line provenance

1. Scanner version `oneiros_native_generated_test_leakage_v2`.
2. **The only change** is to the `issue_text` check.
   - An issue-text line is exempt only when its **entire** whitespace-normalised line
     exactly equals an **entire** whitespace-normalised line of the permitted buggy
     source.
   - A substring, a partial fragment, a line with any added prose or markup, or a
     semantically similar line is **not** exempt.
   - Non-exempt lines are split into sentences and checked exactly as in v1.
3. This is a provenance correction: the prompt may show buggy-side source, and an issue
   that quotes a buggy line reveals nothing beyond that line. It is not a relaxation of
   leakage detection.
   - All other checks are unchanged: fixed-only line, patch line, official-test line,
     expected literal and commit message.
   - Seal-mismatched or malformed input refuses.

## B. Multi-label refusal accounting

- Job construction evaluates **every** check for every kept target and records every
  applicable reason. It does not stop at the first reason.
- The reasons are: `prompt_refused`, `seal_mismatch`, `leakage`, `sequence_overflow`.
- It reports:
  - unique refused targets;
  - counts per reason;
  - the overlapping reason combinations;
  - admitted targets;
  - the kept denominator.
- Coverage counts **unique admitted targets**, never sums of reason counts.

## C. Prompt builder v2: deterministic buggy-source-only context

1. Builder version `oneiros_native_generated_test_prompt_v2`.
   - It is applied **uniformly to every target**, not only to over-limit ones.
   - Its inputs are only the sanitised target DTO and the permitted buggy source.
   - It has no target-specific exceptions and no model-result feedback.
2. **Retained:**
   - the full buggy target function or method, never truncated, with its decorators,
     signature, annotations, defaults and docstring;
   - for each enclosing class, outermost first: its decorators, its `class` line with
     bases, and the first line of its docstring;
   - constructor information for methods:
     - the `__init__` or `__new__` signature from the innermost class, or from the first
       same-module base class that defines one (breadth-first);
     - if no constructor is defined and the class is decorated, the class's annotated
       field declarations;
   - signatures (with the first docstring line) of helper methods the target references
     directly through `self.`/`cls.` (its first parameter), through an enclosing class
     name, or through `super()`, looked up in the enclosing class chain and then in
     same-module bases;
   - signatures of same-module functions, and headers of same-module classes, that the
     retained code references by name;
   - class attributes and module constants that the retained code references;
   - module imports, reduced to the bound names that the retained code references, plus
     every `from __future__` import;
   - the names used in retained annotations, decorators, defaults and base classes, all
     resolved the same way.

   Name resolution is a fixed point over the retained code.
3. **Omitted:**
   - unreferenced sibling members, imports and constants;
   - all helper bodies;
   - issue or PR text, fixed code, patches, official tests, expected outputs, execution
     outcomes and leakage verdicts.
4. A referenced constant or attribute whose source exceeds 400 characters is rendered as
   `NAME = ...` and recorded as `oversized_value_elided`. This never applies to the
   target source.
5. **Recorded per target:**
   - total chat-rendered prompt tokens, counted with the pinned tokenizer;
   - target-source tokens;
   - the retained components and the omitted categories;
   - the permitted-view and prompt hashes;
   - the builder and scanner versions.
6. **Limits unchanged** for this attempt:
   - prompt 2,048 tokens;
   - completion 1,024 tokens;
   - sequence 3,072 tokens;
   - no truncation.

   If fewer than 22 targets fit, the attempt stops. Any larger limit needs a further
   amendment and explicit approval.

## D. Pipeline preflight, authorisation and launch gate (supersedes v2.1 §F)

1. **Pipeline preflight v2.2** is immutable: the writer refuses to overwrite an existing
   receipt. It records:
   - `pipeline_ready`;
   - the canonical executable-source identity:
     - the commit;
     - a canonical hash of every tracked file under `engine/`, `harness/`, `scripts/`,
       `config/` and `tests/`, and of the protocol documents;
   - the fetched remote SHA and the fetch return code (a failed fetch is a blocker);
   - the protocol, job, manifest, model-snapshot, tokenizer, chat-template and adapter
     hashes;
   - the test and canary evidence.

   It does **not** evaluate authorisation.
2. **The GPU authorisation receipt** (`oneiros_native_gpu_authorization_v2`):
   - it is created only after explicit user approval;
   - it binds:
     - the exact preflight path and SHA-256;
     - the source identity;
     - the job, protocol, model and adapter hashes;
     - the allowed condition and arms;
     - the output directory for each arm;
     - its schema and the UTC timestamp.
3. **The launch gate is a separate, read-only check:**
   - it validates the preflight and the authorisation and writes neither;
   - it reports `pipeline_ready`, `gpu_authorized` and `launch_ready` as distinct states;
   - `gpu_authorized` requires a receipt whose content validates;
   - a file merely existing at the authorisation path never authorises.
4. **Source binding:**
   - the current executable-source hash must equal the preflight's;
   - HEAD must be the preflight's commit or a **receipt-only descendant**, whose diff from
     that commit touches only files under `results/`;
   - there must be no tracked modifications and no untracked files in the executable
     directories;
   - a fresh `git fetch` must succeed, and HEAD must equal the fetched remote SHA.
5. Generation with `--backend hf` runs the launch gate and refuses unless
   `launch_ready=true`. The same command with `--backend mock` and an authorisation runs
   the same gate.

## E. Atheris v3: aggregate CPU and fresh isolated revision views (supersedes v2.1 §D.1–2)

1. **Budget:** 600 CPU-seconds per target, per mode, per seed, over the **whole search
   process tree**, including the differential fixed-revision worker.
   - It is measured by a dedicated cgroup-v2 group (`cpu.stat`).
   - At the limit the whole group is killed with `cgroup.kill` and reaped.
   - Per-process RLIMIT_CPU and a wall backstop (budget + 300 s) remain as secondary
     guards.
   - Declared tolerance: overshoot ≤ 1.0 CPU-second + 2% of the budget.
   - Recorded separately:
     - main, worker, replay and aggregate CPU;
     - wall time;
     - the end reason: `completed`, `cpu_budget_exhausted`, `wall_timeout`, `crashed`
       or `infrastructure_failure`.
   - Replay failures are recorded separately.
2. **Isolation:**
   - each target, mode, seed and revision gets a **fresh copy** of the sanitised view;
   - the copy is mounted **read-only at `/target`** in a private mount, PID and network
     namespace;
   - the process runs as nobody with `no_new_privs` and no capabilities;
   - `PYTHONPATH` is `/target` plus the read-only Atheris site; no host checkout path is
     exposed.
   - The differential worker is a separate sandbox connected only by pipes.
   - Witnesses whose content does not match their name, because a write was cut off by a
     kill, are dropped and counted.
3. **Canaries required before any real run:**
   - aggregate budget, with two busy children;
   - write attempts to `/target`;
   - observed path equal across revisions;
   - no cross-seed state;
   - worker cleanup;
   - same-name and package-relative imports.
4. No real-target Atheris run is authorised by this amendment.

## F. Claims

- The engineering rehearsal cannot establish:
  - causal root cause;
  - generalisation;
  - an SFT benefit;
  - Atheris superiority.
- `root_cause_established` and `generalization_established` stay false.
