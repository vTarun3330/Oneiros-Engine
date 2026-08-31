# Oneiros V4.1 next-run procedure

Run every command from the repository root on `experiment/research-eval-ablations`. Do not merge to `main`. Do not access `test` during development.

## 0. Generate and audit the launch queue

The queue command materializes every CPU, integration, ablation, selected-model,
and conditional DPO stage without submitting a GPU job. The doctor fails when
the branch, corpus, source-bound tests, leakage/readiness evidence, 32-pair
preflight, or frozen evaluation protocol is stale.

```powershell
py -3.12 scripts/v4_1_ready.py plan --prompt-token-limit 1024 --output research/v4_1/GPU_READY_QUEUE.json
py -3.12 scripts/v4_1_ready.py doctor --prompt-token-limit 1024 --check-modal --output results/v4_1_doctor.json
py -3.12 scripts/v4_1_ready.py show-integration --prompt-token-limit 1024
```

The default **integration candidate** is explicitly 1024 tokens, the smallest
tested admissible budget. This is not a selected research winner. Runtime
defaults and the frozen evaluator are unchanged. The queue records this
distinction, gives runs a `_p1024` namespace, and puts group J before other
ablation groups. Every generated GPU command includes its prompt budget;
none falls back silently to 512. After a group-J decision, regenerate the
queue and plan with the recorded budget (1024 or 1280). Other groups and
selected-model commands remain conditional templates until those decisions exist.

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
py -3.12 scripts/research_ablations.py plan --run-name v4_1_ablation --corpus-version v4_1_research_hardened_candidate --prompt-token-limit 1024 --output research/v4_1/ablation_plan.json
py -3.12 scripts/research_ablations.py smoke --output results/v4_1_research_metrics_local_smoke.json
```

All design experiments must pass `--evaluation-split ablation_dev`. Record their JSON results in `ABLATION_RESULTS.json` without deleting negative runs. Only the accepted configuration may proceed to locked validation.

The generated plan now contains executable, distinct run names for:

- A0/A1/A2: code only, code + specification, and full legitimate context;
- C0/C1: legacy exactly-one wording and V4.1 self-contained-test wording;
- F: 0%, 10%, 20%, and 30% repository-supervision targets;
- G: proportional, dataset-balanced, and dataset + synthetic-family-balanced sampling;
- I: 800, 2,000, 4,000, and full eligible training scales;
- J: 1024 and 1280 function prompt budgets.

**Run group J first.** It is a prerequisite, not a preference: at the frozen 512
budget the pipeline cannot prompt most of the evaluation panel, so every other
ablation would be measured through a broken prompt. Once a budget is selected,
regenerate the plan and queue with `--prompt-token-limit <accepted-budget>` and
re-run each treatment's preflight. This propagates the budget into both training
and evaluation commands and their namespaces without hand-editing each command.

B, D, E, and H remain explicitly fail-closed in the plan where a clean paired
control is not yet available. J's 512 and 768 rows are recorded as gate
failures with no GPU command. The queue emits no misleading GPU command for a
leaky legacy prompt, malformed head/tail prompt, unmatched oracle-localization
panel, or confounded old-supervision corpus.

## 3. Preflight the 32-pair integration

```powershell
py -3.12 scripts/preflight_sft_run.py --corpus-version v4_1_research_hardened_candidate --max-pairs 32 --epochs 1 --batch-size 1 --learning-rate 0.00005 --lr-scheduler-type constant_with_warmup --real-target-fraction 0.20 --repository-prompt-token-limit 1024 --repository-completion-token-limit 1024 --minimum-monitor-checkpoints 1 --min-function-kill-rate 0.58 --evaluation-split ablation_dev --prompt-token-limit 1024 --output results/v4_1_integration_32_p1024_preflight.json
```

The preflight now also renders a generation prompt for every function record on
the declared evaluation panel. Training selection only ever inspects the records
it chose to train on, but the generation phase must prompt the whole panel, and
section-aware compaction is fail-closed. `evaluation_panel_fully_promptable`
therefore blocks any run whose panel cannot be prompted under the declared
budget.

**The historical 512-token preflight remains a recorded failure** in
`results/v4_1_integration_32_preflight.json`; do not overwrite it with a new-budget
success. Current readiness is checked against the separate `_p1024_preflight`
artifact, including its exact budget, split, corpus/source hashes, optimizer
settings, sampling and prompt variants. The panels
were swept exhaustively on CPU:

| Panel | Records | 512 | 768 | 1024 |
| --- | ---: | ---: | ---: | ---: |
| `ablation_dev` | 542 | 388 unpromptable | 28 | 0 |
| `val` | 757 | 583 unpromptable | 0 | 0 |

Their required sections — system prompt, headings, specification, and the
complete target function — exceed 512 tokens before any support context is
added, so those records produce zero candidates. Usable synthetic training
records rise from 1,930 to 5,396 of 5,595 between 512 and 768.

**1024 is the smallest budget that clears both panels**, and 1024 + 128 fits the
2,048 sequence budget comfortably. Ablation group J therefore emits GPU commands
only for 1024 and 1280; 512 and 768 are recorded as gate failures with no
command, because buying GPU time for a budget that cannot prompt the panel buys
a foregone conclusion. Do not disable this gate, and do not drop the failing
records to make a smaller budget pass — that silently changes the panel.

If `ready` is false, stop. Do not weaken a gate. This declared integration
requires exactly one terminal monitor evaluation. Read its optimizer-step
count from the current budget-specific preflight; changing prompt eligibility
can change the number of selected verified examples and steps. Production and research runs retain the default
two-checkpoint minimum.

**Local evidence, 2026-08-31:** the explicit 1024-token integration preflight
passes after 171 automated tests. All 542 `ablation_dev` function records are
promptable, with zero sequence overflows. The 32 selected records produce 88
effective examples after filtering/balancing (70 synthetic, 18 repeated
repository examples; 20.45% repository share), padded to 96 optimizer samples
and six steps with terminal monitoring at step 6. No GPU model was loaded;
this is not a measured model kill rate or a completed GPU integration.

## 4. Run the 32-pair GPU integration

```powershell
py -3.12 scripts/modal_train.py --fresh --phase sft --corpus-version v4_1_research_hardened_candidate --run-name v4_1_integration_32_seed42_p1024 --seed 42 --max-pairs 32 --evaluation-split ablation_dev --sft-prompt-token-limit 1024 --sft-real-target-fraction 0.20 --sft-epochs 1 --sft-learning-rate 0.00005 --sft-lr-scheduler-type constant_with_warmup --sft-batch-size 1 --sft-repository-completion-token-limit 1024 --sft-min-monitor-checkpoints 1 --sft-monitor-validation-functions 32 --sft-monitor-patience 1 --sft-monitor-min-function-kill-rate 0.58
```

Resume an interrupted integration with the identical command without `--fresh`.
The run is integration evidence only. Confirm finite loss, dataset load, section
budgets, unique V4.1 namespace, save/resume, generation/evaluation artifacts,
and a monitor-history entry for the terminal step declared by that preflight. A sub-58% integration metric
is not a research conclusion and must not promote the adapter or be reported as
a validation result; functional integration succeeds only if the terminal
evaluation and artifacts complete. Do not call it a research result.

## 5. Preflight and run 800 only after integration succeeds

Use `J_prompt_budget_1024` and `J_prompt_budget_1280` in the generated ablation
plan. Each row has its own exact `preflight_command`, fresh 800-record training
command and three evaluation commands. Require the matching preflight to pass
before running that row; the 32-pair doctor's success does not certify 800 pairs.
Resume training using the same row's training command without `--fresh`.
Record both outcomes, including failures. Compare only on `ablation_dev`.

Do not run A-I until the group-J budget decision is recorded. Regenerate their
candidate commands with that exact budget before continuing. Neither CPU
promptability nor the 32-pair GPU integration result selects the winning budget.

## 6. Larger SFT learning curve

Use the I-group commands and distinct run names in the regenerated plan for
2,000 and 4,000 pairs. Keep optimizer, prompt, sampling, seeds and evaluator
fixed. The full-scale row remains a template: its full eligible-count preflight
is not yet materialized, so do not launch it just because bounded preflights pass.

After recording individual ablation decisions, train the accepted combined
configuration as `v4_1_selected_candidate_p<accepted-budget>` using only the
recorded winning flags and confirm it on `ablation_dev` before locked validation.
No selected configuration currently exists; do not alias an unrelated adapter
to this reserved run name.

## 7. Locked validation and Kill@k

Only after ablation decisions and the selected adapter are frozen, use
`selected_model.commands.locked_validation` in the regenerated queue. These
three explicit-budget templates use seeds 42, 43 and 44 and the reserved
budget-specific selection name. Check all other accepted prompt/sampling flags
against the frozen adapter manifest too; these templates are not a selection
receipt. Each result contains ordered Kill@1/2/4/8, candidate-quality layers,
Wilson intervals, and dataset/family slices. A partial seed remains partial and
must be resumed under the same identity.

## 7b. Failure taxonomy

Build the §43 diagnostic from any legitimately produced `ablation_dev` or locked
validation artifact. The script refuses an artifact whose `final_test_measurement`
is true or whose split is `test`.

```powershell
py -3.12 scripts/failure_taxonomy.py results/<run>/sft_validation_hardened_results_seed_42.json --output results/<run>/failure_taxonomy_seed_42.json --markdown results/<run>/FAILURE_TAXONOMY.md
```

Use the aggregate categories to decide what training data to improve. Do not
hand-author training examples from individual locked validation failures.

## 8. DPO gate

If any completed SFT seed is below 58%, do not start DPO. Only when all
predeclared SFT conditions pass may the budget-specific conditional DPO template
in `selected_model.commands` be used. A Codex reset, CPU doctor success, or
integration smoke success does not satisfy this gate.

Resume with the identical command. DPO must beat the frozen SFT under the unchanged protocol; otherwise retain SFT.

## 9. Native repository evaluation

The generated-test native harness now exists. It creates worktrees of the buggy
and fixed revisions, injects one generated test beside the project's official
tests so `conftest` fixtures resolve, runs it in both revisions, and reports a
candidate as `difference_exposing` only when the buggy revision fails and the
fixed revision passes.

```powershell
py -3.12 scripts/evaluate_native_repository.py --generated results/<run>/repository_generations.json --bugsinpy-root data/bugsinpy_v2_ingestion/BugsInPy --repository-cache data/bugsinpy_v2_ingestion/repositories --output results/<run>/native_repository_eval.json
```

`--generated` is a JSON object mapping a repository record id to its ordered
list of generated test sources. No model is loaded; this re-runs text that was
already generated.

**Validation status.** The harness is covered end to end by
`tests/test_native_repository_eval.py`, which builds a real two-commit git
repository and proves that a discriminating test, a non-discriminating test, a
reference-invalid test, and a syntactically invalid test are each classified
correctly, and that an unavailable environment is never scored as a success. It
has **not** yet been validated against real BugsInPy projects, which need
provisioned checkouts and the historical interpreters from
`scripts/provision_bugsinpy_runtimes.py`. Until a real run exists, do not quote
a real-repository kill rate; report `executed_candidates` alongside
`inconclusive_candidates` so the coverage is visible.

## 9b. Final test stop point

The final test remains sealed because the current CLI exposes the explicit final DPO measurement only after model selection. Do not run `--phase dpo_eval --confirm-final-test` during development. Add the final command to the signed experiment record only after the selected adapter, evaluator, candidate count, generation configuration, and one-time-test policy are frozen.

## 10. Remaining CPU preparation

A green 32-pair doctor is not completion of all 56 brief sections. Before the
later research stages can be called fully prepared:

- Materialize an exact full-eligible-training preflight; the I-full command is
  still a conditional template without one.
- Audit the G-group dataset labels end to end: `source_name` currently uses
  `source.name`, which may be an ingestion-source name rather than the upstream
  HumanEval/MBPP dataset. Do not claim dataset balancing until the groups are
  checked and their achieved weights reported.
- Finish clean B/D/E/H controls or retain their INCONCLUSIVE/blocked status.
- Add result-ingestion and frozen-selection receipts so later stages can verify
  decisions mechanically, including every accepted flag, not only the budget.
- Validate the native repository harness on provisioned real-project
  environments and integrate a licensed held-out realistic-mutation benchmark.

No GPU ablation result or selected configuration has been created by local
preparation. Preserve unrun, negative and inconclusive evidence throughout.
