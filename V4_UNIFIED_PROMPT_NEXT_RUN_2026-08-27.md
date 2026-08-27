# V4 unified-prompt next run

## Current status

The local V4 data and SFT pipeline are ready for a fresh integration run on
the `experiment/research-eval-ablations` branch. No V4 model has been trained
yet, so this document records data/pipeline readiness—not a new kill-rate
result.

The canonical model-visible schema is identical across HumanEval, MBPP,
BugsInPy, and SWE-bench:

1. task mode;
2. expected test format;
3. target symbols;
4. behavioral specification;
5. available execution context;
6. code under test;
7. task instructions; and
8. output constraints.

Reference code, gold patches, mutation details, oracle labels, dataset names,
and official test outcomes are never passed to the model. Function prompts
have a 512-token budget. Repository prompts use a 1,024-token budget because
they include verified native test-module context; the schema and ordering do
not change.

## Repairs included

- Restored behavioral MBPP specifications from the cached upstream records.
- Sanitized patch/diff leakage from repository problem statements.
- Added explicit target symbols and first-class support context.
- Derived 37 incorrect MBPP entry points from the outer public call in the
  official tests instead of the first helper definition.
- Re-executed every retained function assertion against both implementations
  and stored per-test reference/target outcomes.
- Applied the live candidate policy during corpus construction.
- Excluded 150 V3 records that no longer had a current-policy,
  reference-valid, mutation-killing assertion.
- Preserved group-disjoint splits and semantic deduplication after entry-point
  repair.
- Unified SFT, DPO, and live-generation chat formatting.
- Increased the repository prompt budget to 1,024 tokens and the shared SFT
  sequence window to 2,048 tokens; function completion remains capped at 128
  tokens and repository completion at 1,024 tokens.

## Recorded verification

- V4 records: 8,237 (7,644 function, 593 repository)
- Splits: 6,646 train / 781 validation / 810 sealed test
- Records SHA-256: `c7ba54830005b19f7cf2511f0a7ebe0a43c61e1fceb2e5bd3222f32d8abe6ec1`
- Splits SHA-256: `319470878c352c3587fc3294df2d3f47e6a75e1761a7227cc8f76c1b547e29f3`
- Full train readiness: `ready: true`; zero prompt leaks, zero function records
  without a verified winner, and zero repository records without official
  evidence.
- 800-pair preflight: `ready: true`; 2,280 effective SFT examples, 20% real
  repository supervision, zero sequence overflows, and validation checkpoints
  planned at optimizer steps 50, 100, and 143.

Local generated reports (not committed):

- `results/v4_unified_prompt_candidate_train_readiness.json`
- `results/v4_unified_prompt_candidate_sft_preflight_800.json`

## Run order

First run a fresh 32-pair integration smoke under a new V4 checkpoint name.
This checks Modal upload, model loading, tokenization, checkpoint writing, and
artifact synchronization; it is too small to support a performance claim.

```powershell
py -3.12 scripts/modal_train.py --fresh --phase sft --corpus-version v4_unified_prompt_candidate --run-name v4_unified_prompt_integration_32_seed42 --seed 42 --max-pairs 32 --sft-epochs 1 --sft-learning-rate 0.00005 --sft-lr-scheduler-type constant_with_warmup --sft-batch-size 1 --sft-repository-completion-token-limit 1024 --no-sft-monitor-kill-rate
```

If the integration artifacts and fingerprints are valid, run the separately
named 800-pair monitored smoke. Do not resume or overwrite a V3 checkpoint.

```powershell
py -3.12 scripts/modal_train.py --fresh --phase sft --corpus-version v4_unified_prompt_candidate --run-name v4_unified_prompt_smoke_800_seed42 --seed 42 --max-pairs 800 --sft-epochs 1 --sft-learning-rate 0.00005 --sft-lr-scheduler-type constant_with_warmup --sft-batch-size 1 --sft-repository-completion-token-limit 1024 --sft-monitor-validation-functions 100 --sft-monitor-patience 3 --sft-monitor-min-function-kill-rate 0.58
```

Keep the adapter only if its locked validation curve is stable and the selected
checkpoint satisfies the declared function kill-rate gate. Then evaluate the
same selected checkpoint on seeds 42, 43, and 44 before considering a larger
SFT run. DPO remains optional and must begin from the checksum-verified frozen
SFT adapter; it is accepted only if repeated locked validation improves without
unacceptable function-level regressions.

Do not merge this branch into `main` solely because local preflight passed.
Merge only after the fresh V4 integration run is satisfactory and its
fingerprints/checkpoint metadata have been reviewed.

## Known remaining boundary

Repository examples have official native fixed-pass/buggy-fail evidence and
now include the context needed to ask for a test fragment. The engine still
does not execute newly generated repository fragments inside reconstructed
BugsInPy/SWE-bench environments. Therefore, repository supervision must not be
reported as a measured generated-test repository kill rate until that native
execution stage is implemented and validated.
