# Local Windows GPU environment (2026-08-31)

This setup does not authorize training. No corpus, checkpoint, or sealed-test
payload was opened. Existing data, checkpoints, research settings, and the old
`venv` were preserved. Branch: `experiment/research-eval-ablations`.

## Interpreter and dependencies

Host: DESKTOP-A1EGVDN; user: Student2; GPU: NVIDIA RTX 4500 Ada Generation,
24 GB; driver: 595.95. The driver reports support through CUDA 13.2; the
installed PyTorch wheel carries CUDA 12.4. No driver/toolkit change was needed.

Use the explicit interpreter from the repository root:

```powershell
.\.venv-gpu\Scripts\python.exe --version
.\.venv-gpu\Scripts\python.exe scripts/check_local_gpu.py
.\.venv-gpu\Scripts\python.exe -m pip check
```

Python 3.12.14 is installed under
`.codex_tmp/python/cpython-3.12.14-windows-x86_64-none/`; do not delete that
folder as scratch, because `.venv-gpu` depends on it. The old `venv` still
references the missing `C:\Program Files\Python312` and is intentionally
untouched. Global PATH and launcher configuration were not changed: use the
explicit environment path, not `py -3.12` or the old demo activation scripts.

The signed Python.org 3.12.10 installer failed with Windows Installer error
1601. uv downloaded a standalone 3.12 runtime; its optional minor-version
junction failed with Windows error 448. Environment creation succeeded using
the actual versioned interpreter directly, without that junction.

`requirements-local-gpu.txt` matches the Modal image's pinned ML stack,
including torch 2.5.1, transformers 4.48.3, PEFT 0.14.0, TRL 0.15.2,
Accelerate 1.2.1 and bitsandbytes 0.45.0. It deliberately omits Modal and
mutmut; this is the training environment, not every optional project tool.
Original requirements files are unchanged. For recreation, install CUDA torch
first, then the overlay (or the resolved lock captured after verification):

```powershell
.\.venv-gpu\Scripts\python.exe -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
.\.venv-gpu\Scripts\python.exe -m pip install -r requirements-local-gpu.txt
```

Compatibility sources: [PyTorch wheels](https://pytorch.org/get-started/previous-versions/),
[bitsandbytes 0.45 Windows support](https://huggingface.co/docs/bitsandbytes/v0.45.0/installation),
[uv managed Python](https://docs.astral.sh/uv/concepts/python-versions/).
Legacy versions are retained for research compatibility; this is not a security
review or authorization to load untrusted models/checkpoints.

## Native GPU entry point

The native CLI now requires `--run-name`, isolates checkpoints/results, rejects
unsafe Windows names and escaped output paths, refuses `--fresh` for an existing
run, and requires CUDA for non-mock execution. Modal code remains available for
historical reproducibility but is not invoked by the local entry point.

Safe inspection only (this does not launch training or access corpus records):

```powershell
.\.venv-gpu\Scripts\python.exe scripts/train_on_dataset.py --run-name local_setup_dry_run --phase sft --evaluation-split ablation_dev --dry-run
```

Retain `--dry-run` until the blockers below are resolved. Its output is a plan,
not scientific preflight evidence or permission to use the sealed test.
For eventual local SFT/validation/DPO, use the native entry point and the same
run identity/options; final testing retains its existing explicit gate.
No final-test command is provided here.

Verified: Python CUDA access, synthetic CUDA matrix multiplication, BF16 support,
bitsandbytes NF4 quantize/dequantize, ML imports, and `pip check`. All 11 synthetic
output-isolation tests passed. An offline native CLI dry run passed with a Python
audit hook rejecting opens/listings under data, checkpoints and results.
`LOCAL_GPU_VERIFICATION.json` captures CUDA evidence; the lock file captures
all installed package versions. No model download, training or evaluation ran.

## What remains before local training

1. Use `scripts/train_on_dataset.py` as the local execution foundation, not
   `scripts/modal_train.py`. Running the latter with ordinary Python still
   dispatches Modal jobs. Do not run upload, failover, or Modal authentication
   commands for this local setup.
2. Use the new required `--run-name` consistently for a new run and every resume
   or evaluation of that run. Outputs are `checkpoints/<run-name>` and
   `results/<run-name>`. Do not rename or move existing research checkpoints.
   Resume without `--fresh`; a fresh run requires unused output directories.
3. Keep ordinary local data/checkpoint/result paths and a persistent local
   Hugging Face cache. Do not reuse Modal's `/root/oneiros` paths, symlink
   replacement helpers, volume upload/download, or periodic volume commits.
4. Resolve sealed-data access before invoking training or readiness commands.
   `harness/corpus.py:verify_corpus` hashes and deserializes the combined
   `records.json`, then checks all splits. `load_corpus_split` also loads the
   combined records before selecting IDs. A train/ablation_dev flag therefore
   does NOT prevent opening sealed-test records. Provision an independently
   verified development-only corpus view and split-aware verification, with
   immutable provenance retained, before local runs under a strict no-access
   rule. Do not weaken existing integrity gates or silently edit the corpus.
5. Adapt the readiness/queue workflow for a local backend: preserve source,
   corpus and scientific checks, but replace cloud authentication/billing
   checks with local CUDA, disk, dependency and output-isolation checks.
   Avoid current corpus-wide doctor/preflight commands under the sealed-data
   restriction.
6. Preserve model revision, NF4, compute dtype policy, attention implementation,
   completion-only loss, optimizer, accumulation, sequence limits, monitoring,
   seeds and evaluation protocol. Hardware availability alone does not unblock
   the documented frozen 512-token prompt-budget gate. Follow the declared
   group-J ablation process on ablation_dev; do not quietly raise the budget,
   drop failing examples, or weaken the 58% per-completed-seed DPO gate.
7. Validate Windows-specific execution separately before research claims.
   Function execution uses subprocess timeouts but POSIX resource limits are
   unavailable on Windows. Native BugsInPy/SWE-bench evaluation still requires
   historical interpreters and provisioned repositories, and may require a
   Linux/WSL environment. Do not conflate CUDA readiness with evaluator parity.

Only native CLI transport/output safety was changed; the model, optimizer,
prompt, sampling and evaluation settings were not changed. Source-bound readiness
evidence must be regenerated after code changes once sealed-data isolation permits
it. Existing Modal-generated queues still contain cloud commands and must not be
executed for a local run.

## Prompt-budget clarification

The current default function prompt budget remains 512 tokens; repository prompts
have a separate 1024-token budget. A GPU does not remove sequence/context bounds:
`engine/sft_trainer.py` currently caps the full SFT sequence at 2048 tokens.
The known 512-budget gate must be resolved before any run. The existing
`--sft-prompt-token-limit` allows a larger declared budget, including the planned
1024/1280 group-J comparisons. There is no safe unlimited-token setting. Do not
remove fail-closed compaction, completion reservation or overflow checks. Larger
budgets change the protocol identity and must not be mixed into old run resumes.
No prompt limit was silently changed during environment setup.
