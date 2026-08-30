# Oneiros V4.1 next-run procedure

Run every command from the repository root on `experiment/research-eval-ablations`. Do not merge to `main`. Do not access `test` during development.

## 0. Generate and audit the launch queue

The queue command materializes every CPU, integration, ablation, selected-model,
and conditional DPO stage without submitting a GPU job. The doctor fails when
the branch, corpus, source-bound tests, leakage/readiness evidence, 32-pair
preflight, or frozen evaluation protocol is stale.

```powershell
py -3.12 scripts/v4_1_ready.py plan --output research/v4_1/GPU_READY_QUEUE.json
py -3.12 scripts/v4_1_ready.py doctor --check-modal --output results/v4_1_doctor.json
py -3.12 scripts/v4_1_ready.py show-integration
```

`doctor --check-modal` proves that a Modal profile is authenticated; it does not
invent or assume the account's spending allowance. Before a paid launch, use the
actual account limit with the existing billing-aware dry run:

```powershell
py -3.12 scripts/modal_train_failover.py --profiles <profile> --credit-limit <actual-limit> --reserve 0.25 --estimated-cost <estimate> --dry-run -- --phase sft --run-name <run-name> --corpus-version v4_1_research_hardened_candidate --evaluation-split ablation_dev
```

Do not launch from the generated queue unless the doctor is green and this
billing check leaves the declared reserve.

## 1. Rebuild and verify locally

The first online build populates immutable buggy-revision context caches. Subsequent rebuilds are offline and deterministic.

```powershell
py -3.12 scripts/build_corpus_v4_1.py --workers 16
py -3.12 scripts/build_corpus_v4_1.py --workers 16 --offline
py -3.12 -c "from pathlib import Path; from harness.corpus import verify_corpus; print(verify_corpus(Path('data/corpus/v4_1_research_hardened_candidate'))['corpus_id'])"
py -3.12 scripts/audit_prompt_lineage.py --corpus-dir data/corpus/v4_1_research_hardened_candidate
py -3.12 scripts/verify_v4_1_local.py
py -3.12 scripts/audit_sft_readiness.py --corpus-version v4_1_research_hardened_candidate --split train --output results/v4_1_research_hardened_candidate_train_readiness.json
```

Do not continue unless tests have zero failures; all corpus hashes verify; split overlaps, schema failures, prohibited lineage, verbatim reference leaks, pending manual reviews, and sequence overflows are zero.

## 2. Freeze/inspect the ablation plan

```powershell
py -3.12 scripts/research_ablations.py plan --run-name v4_1_ablation --corpus-version v4_1_research_hardened_candidate --output research/v4_1/ablation_plan.json
py -3.12 scripts/research_ablations.py smoke --output results/v4_1_research_metrics_local_smoke.json
```

All design experiments must pass `--evaluation-split ablation_dev`. Record their JSON results in `ABLATION_RESULTS.json` without deleting negative runs. Only the accepted configuration may proceed to locked validation.

The generated plan now contains executable, distinct run names for:

- A0/A1/A2: code only, code + specification, and full legitimate context;
- C0/C1: legacy exactly-one wording and V4.1 self-contained-test wording;
- F: 0%, 10%, 20%, and 30% repository-supervision targets;
- G: proportional, project-balanced, and project + synthetic-family-balanced sampling;
- I: 800, 2,000, 4,000, and full eligible training scales.

B, D, E, and H remain explicitly fail-closed in the plan where a clean paired
control is not yet available. The queue emits no misleading GPU command for a
leaky legacy prompt, malformed head/tail prompt, unmatched oracle-localization
panel, or confounded old-supervision corpus.

## 3. Preflight the 32-pair integration

```powershell
py -3.12 scripts/preflight_sft_run.py --corpus-version v4_1_research_hardened_candidate --max-pairs 32 --epochs 1 --batch-size 1 --learning-rate 0.00005 --lr-scheduler-type constant_with_warmup --real-target-fraction 0.20 --repository-prompt-token-limit 1024 --repository-completion-token-limit 1024 --minimum-monitor-checkpoints 1 --min-function-kill-rate 0.58 --output results/v4_1_integration_32_preflight.json
```

If `ready` is false, stop. Do not weaken a gate. This declared integration
requires exactly one terminal monitor evaluation because it has only six
planned optimizer steps. Production and research runs retain the default
two-checkpoint minimum.

## 4. Run the 32-pair GPU integration

```powershell
py -3.12 scripts/modal_train.py --fresh --phase sft --corpus-version v4_1_research_hardened_candidate --run-name v4_1_integration_32_seed42 --seed 42 --max-pairs 32 --evaluation-split ablation_dev --sft-epochs 1 --sft-learning-rate 0.00005 --sft-lr-scheduler-type constant_with_warmup --sft-batch-size 1 --sft-repository-completion-token-limit 1024 --sft-min-monitor-checkpoints 1 --sft-monitor-validation-functions 32 --sft-monitor-patience 1 --sft-monitor-min-function-kill-rate 0.58
```

Resume an interrupted integration with the identical command without `--fresh`.
The run is integration evidence only. Confirm finite loss, dataset load, section
budgets, unique V4.1 namespace, save/resume, generation/evaluation artifacts,
and a monitor-history entry for terminal step 6. A sub-58% integration metric
is not a research conclusion and must not promote the adapter or be reported as
a validation result; functional integration succeeds only if the terminal
evaluation and artifacts complete. Do not call it a research result.

## 5. Preflight and run 800 only after integration succeeds

```powershell
py -3.12 scripts/preflight_sft_run.py --corpus-version v4_1_research_hardened_candidate --max-pairs 800 --epochs 1 --batch-size 1 --learning-rate 0.00005 --lr-scheduler-type constant_with_warmup --real-target-fraction 0.20 --repository-prompt-token-limit 1024 --repository-completion-token-limit 1024 --min-function-kill-rate 0.58 --output results/v4_1_smoke_800_preflight.json
py -3.12 scripts/modal_train.py --fresh --phase sft --corpus-version v4_1_research_hardened_candidate --run-name v4_1_smoke_800_seed42 --seed 42 --max-pairs 800 --evaluation-split ablation_dev --sft-epochs 1 --sft-learning-rate 0.00005 --sft-lr-scheduler-type constant_with_warmup --sft-batch-size 1 --sft-repository-completion-token-limit 1024 --sft-monitor-validation-functions 100 --sft-monitor-patience 3 --sft-monitor-min-function-kill-rate 0.58
```

Resume by repeating the second command without `--fresh`. The 800 run remains an intermediate baseline.

## 6. Larger SFT learning curve

After ablation winners are frozen, repeat preflight and SFT for 2,000, 4,000, and the full eligible training set by changing only `--max-pairs` (omit it for full). Use distinct run names. Keep the selected optimizer, prompt, sampling, seeds, and evaluator fixed.

Train the accepted combined configuration under the reserved selection name (the command below represents the current full-scale candidate defaults; change it only if a recorded ablation winner justifies the exact changed flag):

```powershell
py -3.12 scripts/modal_train.py --fresh --phase sft --corpus-version v4_1_research_hardened_candidate --run-name v4_1_selected_candidate --seed 42 --evaluation-split ablation_dev --sft-epochs 1 --sft-learning-rate 0.00005 --sft-lr-scheduler-type constant_with_warmup --sft-batch-size 1 --sft-repository-completion-token-limit 1024 --sft-monitor-validation-functions 100 --sft-monitor-patience 3 --sft-monitor-min-function-kill-rate 0.58
```

## 7. Locked validation and Kill@k

Run only after ablation decisions are frozen:

```powershell
py -3.12 scripts/modal_train.py --phase sft_eval --corpus-version v4_1_research_hardened_candidate --run-name v4_1_selected_candidate --seed 42 --evaluation-split val
py -3.12 scripts/modal_train.py --phase sft_eval --corpus-version v4_1_research_hardened_candidate --run-name v4_1_selected_candidate --seed 43 --evaluation-split val
py -3.12 scripts/modal_train.py --phase sft_eval --corpus-version v4_1_research_hardened_candidate --run-name v4_1_selected_candidate --seed 44 --evaluation-split val
```

Train the accepted combined configuration under the reserved `v4_1_selected_candidate` run name before these commands. Do not point that name at a different adapter. Each result already contains ordered Kill@1/2/4/8, candidate-quality layers, Wilson intervals, and dataset/family slices. A partial seed remains partial and must be resumed under the same identity.

## 8. DPO gate

If any completed SFT seed is below 58%, do not start DPO. If all predeclared SFT conditions pass:

```powershell
py -3.12 scripts/modal_train.py --phase dpo --corpus-version v4_1_research_hardened_candidate --run-name v4_1_selected_candidate --seed 42 --evaluation-split val
```

Resume with the identical command. DPO must beat the frozen SFT under the unchanged protocol; otherwise retain SFT.

## 9. Native repository and final test stop points

The generated-test native BugsInPy/SWE-bench harness is not implemented yet. Do not run a substitute command or report official stored-test execution as generated-test kill rate. Implement and locally validate native injection into both buggy and fixed revisions before adding a run command.

The final test remains sealed because the current CLI exposes the explicit final DPO measurement only after model selection. Do not run `--phase dpo_eval --confirm-final-test` during development. Add the final command to the signed experiment record only after the selected adapter, evaluator, candidate count, generation configuration, and one-time-test policy are frozen.
