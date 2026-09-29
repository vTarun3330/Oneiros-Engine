# Native buggy/fixed execution of complete model-generated tests — protocol v2

**Status:** frozen 2026-09-29, before any component was implemented or any candidate was
generated.
- Supersedes protocol v1 (`SFT_ROOT_CAUSE_NATIVE_GENERATED_TEST_PROTOCOL_V1.md`, which
  is unchanged).
- It is **self-contained**. Where it differs from
  `REPOSITORY_NATIVE_EVALUATION_PROTOCOL.md`, this document governs, and each difference
  is named.
- Nothing runs on a GPU without a green, source-bound preflight v2 **and** explicit
  approval.

## 1. Claim scope

- **Primary question:** does the exact frozen arm A SFT adapter improve native
  real-repository **Kill@8** over its exact base model?
- **The 24-target cohort is an engineering dress rehearsal only.** Its results are not
  confirmation, generalisation or model-selection evidence.
- A future **repository-disjoint confirmation cohort** needs separate approval. It is not
  authorised here.
- Native base-versus-SFT evaluation measures a checkpoint's behaviour. **It cannot by
  itself establish the causal root cause** of failed SFT generalisation.
  `root_cause_established` stays false unless a separate predeclared causal intervention
  supports it.
- One trained SFT seed cannot show that SFT generally works or fails.
- Atheris results never support a claim that Oneiros generally beats Atheris.

## 2. Exact generation contract

| field | value |
|---|---|
| base model | `Qwen/Qwen2.5-Coder-1.5B-Instruct` @ `2e1fd397ee46e1388853d2af2c993145b0f1098a` |
| SFT adapter | `checkpoints/local_sft_armA_baseline_successor_s42/sft_adapter`: the complete directory manifest (every file with its SHA-256, including `adapter_config.json`) is bound in the preflight; `adapter_model.safetensors` = `e67dd599…f9e7` |
| tokenizer | the local snapshot of the base revision; `tokenizer.json`, `tokenizer_config.json` and the chat-template text are hashed in the preflight |
| generation seeds | 42, 43, 44, identical for both models; the RNG is re-seeded immediately before each target's batch |
| candidates | 8 per target per seed per model |
| sampling | temperature 0.7, top_p 0.9, `do_sample=True`, no top_k override |
| max_new_tokens | 1024 |
| prompt-token limit | 2048 (differs from the parent's 1024 for function targets, because repository targets carry buggy-side context) |
| model sequence limit | 3072 = prompt + max_new_tokens; a rendered prompt over 2048 tokens is **refused**, never truncated or compacted |
| batch size | 2 |
| extraction | whole output; if the whole output is exactly one fenced code block, the fence is removed; otherwise the output is used verbatim |
| raw outputs | every raw output is retained, with its SHA-256 |
| reranking | none |
| duplicates | kept; each candidate slot counts; byte-identical duplicates are reported |
| module limits | at most 25 test functions, 100 `assert` statements and 200 target-name call sites (static AST count); over-limit modules are candidate failures |
| execution limits | 10 s per test, 90 s wall per revision run, 60 CPU-seconds per run |

**Kill@8 estimand.**
- `kill(t, s, m)` = 1 if any of model m's 8 candidates for target t at seed s is a kill (§4).
- The target-level score is `K(t, m) = mean over s ∈ {42, 43, 44} of kill(t, s, m)`.
- The **primary estimand** is the mean over targets of `K(t, SFT) − K(t, base)`.
- Target × seed observations are **never** pooled as independent samples. Per-seed Kill@8
  is reported only as robustness evidence.

## 3. Conditions

- **Primary: whole module.** The model writes a complete pytest module. This measures
  end-to-end usability.
- **Secondary diagnostic: scaffolded.** A fixed, safe scaffold supplies the imports and
  one test function; the model writes only the body.
  - It separates collection and format ability from input selection and oracle
    correctness.
  - It is never used for checkpoint selection or promotion.

## 4. Kill and admission semantics

A candidate is a **kill** only if all of the following hold:
1. Identical candidate bytes run on the buggy and fixed revisions.
2. Syntax, import and collection succeed on both.
3. Both collect the **same node IDs**.
4. At least one real test executes. Skips, xfails and zero-test passes are not accepted.
5. The intended target is **dynamically reached** on both revisions.
6. The fixed revision genuinely passes.
7. The buggy revision fails with either:
   - an **assertion failure** after the target was reached (semantic kill), or
   - a **project-code exception** whose traceback includes the target package (crash
     kill).
8. A rerun of both revisions agrees.

**Outcome classes,** each counted separately:

| class | counted as |
|---|---|
| syntax failure | model failure |
| fabricated import / API | model failure |
| collection failure | model failure |
| no tests collected | model failure |
| skipped / xfail | model failure |
| target not reached | model failure |
| over limits | model failure |
| fixed-side failure | model failure (invalid test) |
| pass both | not a kill |
| buggy test-code error | not a kill |
| timeout | model failure |
| nondeterminism | not a kill |
| buggy assertion kill | kill (semantic) |
| buggy project-code crash kill | kill (crash) |
| harness failure | excluded; arm-independent only |
| dependency failure | excluded; arm-independent only |
| environment failure | excluded; arm-independent only |

- Model-caused failures stay in the **requested-candidate** denominator and count as
  zero.
- A failure is excluded as infrastructure only when it reproduces on a known-good
  canary module in the same environment, which proves it is arm-independent.

## 5. Generated-code sandbox (mandatory before any generated candidate executes)

- **Isolation:**
  - new mount, PID and network namespaces;
  - the process runs as **nobody** (uid/gid 65534) with `no_new_privs`;
  - the process sees tmpfs over `/mnt`, `/root` and `/home`. Only the needed read-only
    bind mounts are restored at their original paths: the uv Python runtime, the
    target's environment, and **one** revision's source. Test directories are masked,
    and the official regression tests are never copied in.
  - There is no Git history, no evidence store, no model cache and no main repository.
- **Environment:** a minimal `env -i`; bytecode writing disabled; stale editable-finder
  bytecode cleared; a writable, empty per-candidate work directory.
- **Limits (prlimit):** CPU, address space, process count, open files and file size, with
  `timeout -k` and process-group kill.
- **In-process hardening (defence in depth):** an audit hook refuses `subprocess`,
  `os.system`, `os.exec*`, `os.fork`, `socket` and `ctypes`.
- **Attestation:** the imported target module's real path must lie under the intended
  revision.
- **Refusal:** if the enforcement canaries fail, no generated code executes.

## 6. Leakage boundary

- **Prompt builder:** receives only a sanitised DTO (repository, buggy commit, target
  file and qualified name) and opens only the buggy-side file. It never sees the
  manifest, official test names, patch, fixed code, evidence sidecars or issue text.
- **Sealing:** the prompt is sealed (SHA-256) before scanning.
- **Scanner:** a separate component compares the sealed prompt with the fixed function
  body, the patch's added lines, the official test source, expected literals, issue/PR
  text and commit messages, when those are available to it. Every refusal is recorded
  with its reason.
- **Runtime:** generated code cannot read source outside the one revision under test, and
  cannot read tests, Git history, fixed code (during the buggy run) or evidence (§5).

## 7. Atheris comparison (three labelled modes; actual atheris 2.3.0, WSL Python 3.11)

1. **Ordinary:** buggy-side crash and declared-contract discovery. A finding is a
   bug-specific kill only if replaying it on the fixed revision does not fail.
2. **Post-hoc replay:** buggy-only coverage search. Its corpus is replayed afterwards on
   both revisions, and a behaviour difference is a kill. Fixed behaviour never guides the
   search.
3. **Online differential:** an explicitly labelled, oracle-assisted **upper bound**.

- **Budget:** the preregistered full comparison is **600 CPU-seconds per target, per
  mode, per seed** (seeds 42–44), as in the parent protocol. A latency-matched run
  (Atheris wall time equal to Oneiros's measured wall time) may be reported only as a
  secondary sensitivity analysis. It is labelled **latency-matched**, not equal compute.
  This resolves v1's conflicting "equal budget" rule.
- **Eligibility:** typed argument construction from annotations; a reachability proof;
  applicability and ineligibility reasons (for example, instance methods need a
  receiver, which makes them ineligible). Results use jointly eligible denominators.
- **Instrumentation canaries:** a synthetic must-kill and must-not-kill pair.
- **Scope:** ordinary Atheris is a crash/contract fuzzer without a semantic oracle for
  wrong-answer bugs; results are read with that in mind.

## 8. Statistics

- **Primary:** SFT − base Kill@8 (§2 estimand), with target-level pairing.
  - Inference: a two-sided 95% percentile interval from a **repository-clustered** paired
    bootstrap (10,000 resamples, seed 20260930).
  - Sensitivity: leave-one-repository-out.
- **Validity non-inferiority:** the fixed-valid rate per requested candidate (the
  candidate passes on fixed with at least one executed, target-reaching test). SFT is
  non-inferior if the **lower** 95% bound of SFT − base is at least −3 points. A point
  estimate below −3 with a bound that does not resolve it is an operational stop, not
  proof of harm.
- **Secondary and exploratory:** Kill@1 and Kill@4, semantic versus crash kills,
  admission-stage rates, unique bugs killed, the Atheris modes, and complexity and
  bug-family slices.
- **Power:** a confirmation cohort smaller than the parent's powered design (N = 400) is
  labelled **exploratory / underpowered**.
- **Dress rehearsal (engineering gate only):**
  - every stage has receipts;
  - there are no unexplained failures;
  - at least 90% of qualified targets execute on both revisions;
  - the sandbox, leakage and Atheris canaries pass;
  - no statistics are interpreted.

## 9. Cohort

- **Dress rehearsal:** the 24 natively qualified A-prime development targets in 8
  repositories.
  - Each must be **revalidated under isolation v6** and **re-qualified**: the official
    regression tests fail 3/3 on buggy and pass 3/3 on fixed, in a recorded environment.
  - Targets that fail are reported exactly and are never silently kept.
  - The rehearsal rule (20–30 targets, at least 5 repositories) is not weakened.
- **Confirmation:** repository-disjoint, isolation-v6-admitted and separately approved.

## 10. Boundaries

- No locked-validation or sealed-final access.
- No Choice B gate reuse.
- Nothing produced here is used for training.
