# Local Research Pipeline Smoke — 27 August 2026

## Decision

The research evaluation and ablation orchestration is locally ready for Modal
smokes. This is a software/pipeline decision, not evidence that a model improved.

The final test split was not accessed.

## Checks completed

| Check | Result |
|---|---|
| Python compilation of changed pipeline modules | Passed |
| Complete pytest suite | 94 passed |
| Isolated candidate execution | Passed |
| Ordered invalid/valid/killing slot accounting | Passed |
| Kill@1/2/4/8 monotonicity | Passed |
| Pass@1/2/4/8 monotonicity | Passed |
| Wilson intervals | Passed |
| Exact/AST/input-shape/outcome diversity | Passed |
| Fixed-budget feedback orchestration | Passed: 4 initial + 4 feedback = 8 |
| AST/input-shape equal-budget prioritisation | Passed |
| Paired policy redundancy/coverage comparison | Passed |
| Multi-seed mean/std/Student-t interval | Passed |
| External model JSONL scorer | Passed with local fixture |
| Modal CLI discovery of new arguments | Passed |
| Git whitespace/error check | Passed |

## Deterministic synthetic smoke

The local smoke evaluated three small synthetic functions with eight raw slots
per function. The purpose was to exercise the real candidate policy and
subprocess execution harness with known outcomes.

| Metric | Smoke result |
|---|---:|
| Functions killed | 2/3 |
| Function kill rate | 66.67% |
| Kill@1 | 0.00% |
| Kill@2 | 0.00% |
| Kill@4 | 66.67% |
| Kill@8 | 66.67% |
| Pass@1 | 66.67% |
| Pass@2/4/8 | 100.00% |

These values were deliberately constructed and must never appear in the paper
or slides as model results.

## Corpus audit

The full canonical corpus hash/quality audit also completed:

- 8,387/8,387 records have behavioral verification and retained oracle witness
  tests;
- eight semantic duplicates were removed during V3 construction;
- 36 retained repository records are explicitly excluded from training by the
  context gate;
- the external inventory contains 501 tasks: 269 materialized for training,
  seven repository-evaluation-only tasks, and 225 locked untouched tasks.

## Next execution

When Modal access is restored, run the four 32-function commands in
`RESEARCH_EVALUATION_AND_MODAL_RUNBOOK.md` in order. Keep this branch unmerged
until all four complete with the same expected scopes and no final-test access.
