# Oneiros repository guide

This repository contains the source, configuration, tests, manifests, and
documentation needed to inspect and reproduce the Oneiros Phase 3 pipeline.
Large datasets, repository checkouts, checkpoints, raw results, logs, caches,
and credentials are deliberately excluded.

## Current pipeline

```text
canonical V4 unified-prompt corpus
  -> manifest/hash/leakage verification
  -> deterministic train selection
  -> verified supervised fine-tuning (SFT)
  -> locked validation panel
  -> optional mutation-aware DPO
  -> validation-based model selection
  -> one sealed final-test evaluation
```

The canonical corpus identifier is centralized as
`config.CANONICAL_CORPUS_VERSION` and currently resolves to
`v4_unified_prompt_candidate` (8,237 records). Function-level candidates must be one
bounded assertion that calls the target entry point. Candidate execution uses
a restricted child process and a parent-enforced timeout.

All four source families use the same prompt headings and hidden-oracle
contract. Function records receive a 512-token prompt budget; repository
records receive 1,024 tokens because they also carry native execution context.

## Folder responsibilities

| Path | Responsibility |
|---|---|
| `baseline/` | Random, static, grammar, coverage-guided, and human-test benchmark paths. |
| `config/` | Canonical corpus, model, dataset, and training configuration. |
| `engine/` | Generator, SFT, DPO, prompt budgeting, oracle, memory, and bug discovery. |
| `harness/` | Corpus validation, dataset adapters, mutation handling, candidate policy, and safe execution. |
| `metrics/` | Benchmark aggregation and scoring. |
| `scripts/` | Corpus construction/auditing, local preflight, Modal ingestion, training, calibration, and evaluation. |
| `tests/` | Unit and regression tests for the hardened critical paths. |
| `utils/` | Logging, embeddings, and reproducibility fingerprints. |
| `data/` | Local corpora and repository checkouts; only the canonical manifest and exclusion ledger are tracked. |
| `checkpoints/` | Local model artifacts; only its policy README is tracked. |
| `results/` | Local experiment outputs; only its reporting-policy README is tracked. |

## Local setup and verification

Python 3.12 is the supported local version for the current Modal image.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

The automated suite is GPU-free. The full SFT/DPO pipeline requires a CUDA GPU
or the configured Modal environment.

## Data preparation

The repository does not download or publish the canonical corpus
automatically. Obtain the authorized `v4_unified_prompt_candidate` corpus, place it at
`data/corpus/v4_unified_prompt_candidate/`, and verify that its file hashes match the
tracked manifest. For deliberate corpus reconstruction, follow
`dataset_documentation.md`; never edit a released corpus in place.

Run the exact GPU-free readiness preflight before a performance experiment.
The 800-record scope is large enough to exercise the checkpoint-monitoring
schedule; a 32-record selection is only an integration smoke and is not
expected to satisfy the performance-readiness gates.

```powershell
python scripts/preflight_sft_run.py --corpus-version v4_unified_prompt_candidate --max-pairs 800 --epochs 1 --batch-size 1 --learning-rate 0.00005 --lr-scheduler-type constant_with_warmup --repository-prompt-token-limit 1024 --repository-completion-token-limit 1024 --min-function-kill-rate 0.58
```

## Training and evaluation order

1. Re-evaluate the historical smoke adapter on validation seeds 42, 43, and
   44 under the hardened evaluator.
2. Run a 32-record integration SFT smoke and verify all fingerprints and
   adapter checksums.
3. Reproduce the 800-record smoke, then evaluate the selected adapter on the
   complete function-level validation split.
4. Train the full SFT candidate only after the bounded smoke is stable.
5. Run DPO only from the verified frozen SFT reference, and keep it only if it
   improves the locked validation result without unacceptable regressions.
6. Freeze every decision, then explicitly unlock the sealed test split for one
   final evaluation.

Exact commands and gates are in
`V4_UNIFIED_PROMPT_NEXT_RUN_2026-08-27.md`. The earlier
`HARDENING_AND_NEXT_STEPS_2026-08-27.md` remains the historical V3 audit.

## Artifact policy

Do not commit raw data, external repository checkouts, model weights, run logs,
secrets, API tokens, or local virtual environments. A result is not
reproducible unless it records the source tree, dependency set, model revision,
corpus hashes, adapter checksums, panel hash, seed, and full evaluation output.
The optional Modal failover launcher takes local profile names through
`--profiles` or `ONEIROS_MODAL_PROFILES`; no personal profile name or token is
hard-coded in the repository.
