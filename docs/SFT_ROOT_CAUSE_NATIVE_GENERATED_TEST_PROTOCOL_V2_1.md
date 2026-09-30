# Native generated-test protocol — amendment v2.1 (additive; frozen before any model output)

Protocol v2 (`SFT_ROOT_CAUSE_NATIVE_GENERATED_TEST_PROTOCOL_V2.md`) is unchanged. This
amendment was frozen on 2026-09-30, after the engineering review of the first
implementation. No model output existed; nothing had been generated. Where v2 and v2.1
differ, **v2.1 governs**.

## A. Interpreters and runtime parity

1. The preliminary Python 3.13 re-qualification (23/24,
   `results/sft_root_cause/native_v2_rehearsal/`) is **engineering evidence only**. It is
   preserved and superseded.
2. **Approved interpreter: CPython 3.11** (the WSL system `/usr/bin/python3.11`). It is
   the same runtime as the Atheris 2.3.0 interpreter, so jointly compared targets have
   exact runtime parity. A target whose project does not support 3.11 is excluded as
   `no_approved_interpreter`. Any other interpreter needs a further amendment.
3. **One environment per target,** shared by the buggy and fixed revisions.
   - It is built from the fixed revision's declared test dependencies.
   - The project's own editable finder is removed; only its distribution metadata is
     kept.
   - It is recorded by a sanitised freeze and a site-packages manifest hash (together,
     the environment lock).
4. **Formal re-qualification:** **every** recorded difference-exposing official test
   fails 3/3 on buggy and passes 3/3 on fixed, under the approved interpreter and the
   shared environment.

## B. Candidate isolation (supersedes the v2 §5 view)

1. **Canonical views.** The trusted harness builds a sanitised copy of each revision's
   import root, and both are mounted at the **same canonical path `/target`**. The
   project is imported only through `PYTHONPATH=/target`; no editable finder and no host
   path exist.
   - The copy removes every test directory and test file at any depth (`test`, `tests`,
     `testing`; `test_*.py`, `*_test.py`, `conftest.py`), plus `.git` and all non-source
     metadata.
   - Build-generated files that exist only in the fixed tree are copied into both views.
2. The sandbox specification the candidate could see is **identical for both revisions**.
   Revision attestation (the hash of the imported target module) is checked by the
   trusted harness outside the sandbox.
3. **Candidate policy (static; refusal class `policy_refused`, a model failure).**
   - Refused names: `open`, `eval`, `exec`, `compile`, `__import__`, `globals`, `vars`,
     `breakpoint`.
   - Refused imports: `os`, `sys`, `pathlib`, `io`, `inspect`, `dis`, `importlib`,
     `pkgutil`, `subprocess`, `socket`, `ctypes`, `shutil`, `glob`, `tempfile`, `builtins`,
     `gc`, `marshal`, `pickle`, `linecache`, `traceback`, `code`, `runpy`, `signal`.
   - Refused attributes: `__file__`, `__code__`, `__dict__`, `__globals__`,
     `__builtins__`, `__loader__`, `__spec__`, `__closure__`, `__subclasses__`,
     `f_code`, `co_code`, `gi_code`.

   **Runtime source or revision introspection is forbidden.** The sandbox (v2 §5)
   remains the enforcement layer; the policy is a declared, best-effort front line.
4. **Whole-module rule (supersedes v2 §4, items 4–5).** Every collected node must exist
   on both revisions and execute on both, not skipped and not xfailed, and **every node
   must reach the target on both**. Otherwise the candidate is not admissible, and a
   module mixing an executed test with a skip or xfail cannot be a kill.

## C. Conditions

The scaffolded diagnostic (v2 §3) is **removed**. Its body-only instruction conflicted
with the production output instruction. Only the primary whole-module condition exists.

## D. Atheris

1. **Budget:** 600 **CPU**-seconds per target, per mode, per seed, enforced with
   RLIMIT_CPU. A separate wall-clock timeout (CPU budget + 300 s) is a backstop. CPU and
   wall time are both recorded.
2. **Isolation:** every revision runs in its own process, importing the package **by
   name** from its canonical view, so package-relative imports work and buggy and fixed
   never share a `sys.modules` identity. The differential mode calls a separate
   fixed-revision worker process.
3. **Replay and canonical results:**
   - arguments are rebuilt independently from the input bytes for every call;
   - **every** crash witness is replayed;
   - results are canonicalised (None, bool, int, float, str, bytes and containers of
     them), and opaque values refuse comparison;
   - alleged differences are re-confirmed in fresh single-input processes;
   - replay errors are reported separately and are never kills.
4. **Corpus cap:** 2,000 inputs, taken in sorted file-name order, with truncation
   recorded.
5. **Environment:** network disabled (`unshare -n`); the Python 3.11 parity of every
   target is checked; reachability is recorded.

## E. Analysis

1. **Infrastructure rows** (harness, dependency, environment) are never counted as
   parsed, collected, executed or reached.
   - A row is excludable only if the same-environment known-good canary failed, which
     proves the failure is arm-independent.
   - **Rule for asymmetric infrastructure:** a target with any infrastructure row in
     either arm is excluded from the paired comparison for **both** arms.
   - An infrastructure row whose canary passed refuses the analysis.
   - Requested and eligible denominators are reported separately.
2. **Fixed-valid** requires row evidence that the fixed run executed every node, all
   passed and all reached the target, and, for kill candidates, that the rerun agreed.
   `nondeterminism` is never fixed-valid.
3. **`study_mode`:**
   - `engineering_dress_rehearsal`: descriptive pipeline metrics only; no interpreted
     significance, non-inferiority or promotion decision.
   - `confirmation`: the frozen inferential analysis.

## F. GPU authorisation

- The Hugging Face generation path is committed source.
- It runs **only** with a GPU authorisation receipt that binds:
  - the green preflight hash;
  - the source commit;
  - the protocol and amendment hashes;
  - the job hash;
  - the allowed condition(s), arm(s) and output directory.
- No source edit is needed or allowed after preflight. The preflight reports
  `pipeline_ready`, `gpu_authorized` and `launch_ready` separately.

## G. Claims

The engineering rehearsal cannot establish causal root cause or generalisation.
`root_cause_established` and `generalization_established` stay false.
