# Oneiros Research Evaluation and Modal Runbook

Status date: 27 August 2026

Branch: `experiment/research-eval-ablations`

Final test split: sealed and not used by this work

## What is implemented

The validation pipeline now retains every raw generation slot in rank order.
Parse failures are not silently dropped. Each slot records parse validity,
reference validity, mutation kill status, failure mode, and the hash of the raw
model output. This enables the following comparable metrics:

- function kill rate with Wilson 95% interval;
- candidate kill rate and end-to-end candidate kill rate;
- Kill@1, Kill@2, Kill@4, and Kill@8;
- Pass@1, Pass@2, Pass@4, and Pass@8, defined as at least one
  reference-valid test in the first `k` raw generation slots;
- parse, reference-valid, and execution-invalid rates;
- exact, AST-shape, input-shape, and outcome-mode diversity;
- redundant killing candidates versus unique functions killed;
- source/benchmark and mutation-family breakdowns.

The following ablation mechanisms are also implemented:

1. `base_eval` evaluates the pinned base Phi-3 revision on exactly the same
   validation scope as the adapter.
2. Reference-free execution feedback can divide the fixed eight-candidate
   budget over one or two repair rounds. Only results from executing tests on
   the visible code under test enter the next prompt. The fixed implementation
   and oracle classification never enter the prompt.
3. AST or input-shape diversity prioritisation reorders the same generated
   candidates; it does not receive free oversampling or a larger candidate
   budget.
4. `--holdout-bug-family` removes a family from SFT training, removes the same
   family from checkpoint selection, and evaluates the frozen adapter only on
   that held-out family.
5. Compatible seed results can be aggregated with mean, sample standard
   deviation, standard error, range, and a Student-t 95% interval over the seed
   mean.
6. Paired policy comparison reports newly killed functions, regressions,
   retained kills, net unique coverage, candidate redundancy, and diversity
   deltas. This is the required SFT-versus-DPO diagnostic.
7. Candidate JSONL exported from another LLM can be evaluated with the same
   local oracle and metric schema using `scripts/evaluate_external_generations.py`.

## Local verification completed

The CPU-only smoke does not download or claim to evaluate Phi-3. It runs the
real candidate policy and isolated execution harness on three deterministic
synthetic mutants, then verifies metric invariants and ablation analysis.

```powershell
py -3.12 scripts/research_ablations.py smoke
py -3.12 -m pytest -q
```

Current result: local research smoke passed; sealed test access was `false`;
the complete repository suite passed 90 tests.

## Modal smoke order

Run the following after Modal access is restored. These use only 32 validation
functions and are pipeline checks, not reportable experimental results.

```powershell
py -3.12 scripts/modal_train.py --phase base_eval --run-name v3_repo1024_aligned_constant_lr_smoke_800_20260821_1 --corpus-version v3_final_candidate --seed 42 --max-validation-functions 32

py -3.12 scripts/modal_train.py --phase sft_eval --run-name v3_repo1024_aligned_constant_lr_smoke_800_20260821_1 --corpus-version v3_final_candidate --seed 42 --max-validation-functions 32

py -3.12 scripts/modal_train.py --phase sft_eval --run-name v3_repo1024_aligned_constant_lr_smoke_800_20260821_1 --corpus-version v3_final_candidate --seed 42 --max-validation-functions 32 --eval-feedback-rounds 1

py -3.12 scripts/modal_train.py --phase sft_eval --run-name v3_repo1024_aligned_constant_lr_smoke_800_20260821_1 --corpus-version v3_final_candidate --seed 42 --max-validation-functions 32 --eval-diversity-mode ast
```

Acceptance requirements for these smokes:

- command completes without an incompatible-resume or corpus error;
- exactly 32 function results and 256 raw candidate slots are present;
- Kill@k and Pass@k are monotonic in `k`;
- no output reports `final_test_measurement: true`;
- feedback results retain a total budget of eight candidates per function;
- the base and SFT smoke have identical evaluation-scope hashes.

Do not compare the percentages from a 32-function smoke as research results.

## Full validation experiment

The predeclared plan is stored in
`reports/modal_research_ablation_plan_2026-08-27.json`. It contains exact
commands for seeds 42, 43, 44, 45, and 46 for:

- base Phi-3;
- frozen SFT;
- one and two execution-feedback rounds;
- AST and input-shape diversity prioritisation;
- combined feedback plus AST prioritisation;
- six leave-one-mutation-family-out training/evaluation runs.

Generate the plan again without editing commands by hand:

```powershell
py -3.12 scripts/research_ablations.py plan `
  --run-name v3_repo1024_aligned_constant_lr_smoke_800_20260821_1 `
  --output reports/modal_research_ablation_plan_2026-08-27.json
```

The ordered-metric evaluator intentionally uses new filenames such as
`sft_validation_standard_seed_42.json`. It will not overwrite or silently reuse
the older aggregate-only `sft_validation_hardened_results_seed_42.json` files.

Aggregate five compatible results:

```powershell
py -3.12 scripts/research_ablations.py aggregate `
  results/<run>/sft_validation_standard_seed_42.json `
  results/<run>/sft_validation_standard_seed_43.json `
  results/<run>/sft_validation_standard_seed_44.json `
  results/<run>/sft_validation_standard_seed_45.json `
  results/<run>/sft_validation_standard_seed_46.json `
  --output reports/sft_five_seed_aggregate.json
```

Compare SFT and DPO only after DPO has passed its validation gate:

```powershell
py -3.12 scripts/research_ablations.py compare `
  results/<run>/sft_validation_standard_seed_42.json `
  results/<run>/dpo_validation_checkpoint_<trained-pairs>.json `
  --output reports/sft_vs_dpo_seed42.json
```

## External modern-LLM baseline format

Create one JSONL row per canonical validation function:

```json
{"record_id":"mutation::...","model":"provider/model-version","seed":42,"candidates":["assert target(...) == ..."]}
```

Then evaluate it locally:

```powershell
py -3.12 scripts/evaluate_external_generations.py `
  --corpus-dir data/corpus/v3_final_candidate `
  --input external_candidates.jsonl `
  --output results/external_model_seed42.json
```

This scorer supports only `val`; there is no option that can open `test`.

## Corpus quality and equivalent-mutant treatment

Run:

```powershell
py -3.12 scripts/report_corpus_quality.py `
  --corpus-dir data/corpus/v3_final_candidate `
  --output reports/corpus_quality_and_equivalence_2026-08-27.json
```

All 8,387 canonical records have a retained behavioral witness. Eight semantic
duplicate records were removed and 36 retained repository records are
explicitly excluded from training because of the context gate. A confirmed
equivalent mutant cannot be admitted because admission requires at least one
test that passes the fixed implementation and fails the code under test. This
does not prove semantic equivalence over every possible input for candidates
that were never admitted, so the paper must state that limitation.

## Still dependent on external systems

The following are not claimed as completed results:

- native BugsInPy/SWE-bench generated-test execution in isolated repository
  environments;
- scores for modern external LLMs, until their validation candidate JSONL is
  generated;
- any five-seed Phi-3/SFT/feedback/diversity result, until Modal runs complete;
- DPO contribution analysis, until a new SFT adapter passes the full repeated
  validation gate and DPO is legitimately trained;
- final-test numbers, deployment claims, or a final product readiness claim.

The corpus already keeps 225 BugsInPy tasks locked for external evaluation and
contains verified real-repository training records, but the function-level
headline metric must remain separate from native repository evaluation.

## Merge rule

Keep this branch separate from `main` through local verification and Modal
smokes. Merge only after the Modal smoke acceptance requirements pass. Full
research results should be committed as curated reports; raw checkpoints and
candidate outputs remain outside Git.
