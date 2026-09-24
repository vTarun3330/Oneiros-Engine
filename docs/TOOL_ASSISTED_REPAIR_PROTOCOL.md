# Execution-feedback repair at inference — frozen design

Frozen 2026-09-24. CPU audit, implementation, tests and preflight only. **No
generation has been run. GPU generation needs explicit approval.** The
execution-supervision SFT line is closed and is not reopened here: this is a
different hypothesis, it writes no weights, and it trains nothing.

## Question

Can execution feedback, used at inference time, improve test generation beyond
what the **same inference compute** achieves by plain resampling?

The model is never asked to simulate the program. The loop does the following,
using only what a developer could see:
1. executes each candidate in the sandbox against the code under test;
2. classifies failures into a closed taxonomy;
3. returns bounded, structured feedback for demonstrably invalid artifacts;
4. lets the model make one repair per slot.

## Arms

| arm | generation | feedback | role |
|---|---|---|---|
| A | unchanged successor path: 8 samples, batch 2, one seed | none | canonical historical control (secondary) |
| B | shared round 1 (8 samples) + 8 single-sequence canonical resamples | none | **compute-matched control** |
| C | shared round 1 (8 samples) + 8 single-sequence slots; slot *i* is a repair of round-1 candidate *i* only if that candidate is demonstrably invalid, otherwise it is the identical canonical sample B receives | structured, non-leaking | **treatment** |

**The comparison of interest is C versus B.**

**Why the pairing isolates the feedback.** B and C use the *same* round-1
generations. Each round-2 slot is seeded per slot. So C differs from B only in
slots where feedback was given. Both arms spend exactly 16 sequences, 9 model
calls and the same 1,024-token cap per target. Actual input and output tokens
and generation wall-clock are recorded for every candidate. A is not
compute-matched; its comparisons are labelled secondary.

**Held constant across arms:**
- model `Qwen/Qwen2.5-Coder-1.5B-Instruct` @ `2e1fd397…` and the frozen control adapter `c1fe9e52…`;
- the same targets and the canonical prompt information;
- seed 42, temperature 0.7, top-p 0.9;
- the whole-output parser and the candidate policy;
- the restricted executor with its 0.5 s timeout;
- the unchanged rehearsal evaluator;
- eight final slots per target.

## Information boundary

**What the loop receives.** The loop sees a *permitted view* of each record:
- the specification;
- the code under test and its permitted context;
- the model's own candidates;
- parser and sandbox diagnostics;
- behaviour on the code under test.

**What it never receives.** The fixed implementation and the gold tests are
removed before the loop runs. The evaluator uses them only after the eight
final slots are frozen.

The preflight proves that the canonical prompt built from the view is
byte-identical to the prompt built from the full record, for all 264 panel
records.

**Where each run happens.** Execution for feedback uses
`harness/buggy_side_execution.py`. Its signature has no parameter through which
a reference could arrive. It runs each candidate in a fresh isolated process
and reports **which frame raised**.

## Feedback taxonomy (`oneiros_execution_feedback_v1`, closed)

**Repairable** (the model gets one fixed-template sentence and may repair):
- unparsable output;
- invalid Python syntax;
- prohibited construct (import, disallowed name, attribute or call);
- missing call to the function under test;
- invalid candidate shape;
- an attribute or a name the candidate invented;
- a malformed invocation (TypeError at the call site);
- a timeout inside the candidate's own code;
- duplicate of an earlier candidate.

**Retained silently** (kept exactly, and the model is told nothing):
- executes without error;
- **observed assertion failure**;
- exception raised **inside the code under test**;
- timeout **inside the code under test**;
- infrastructure failure;
- a repair identical to its parent.

**Why the retained cases stay silent:**
- An assertion failure may be a legitimate kill, so it is never repaired toward passing.
- Passing implies that no kill happened, so saying so would reveal kill status.
- An exception raised by the target may be the defect itself.

Every feedback object is canonical JSON with a SHA-256. It is scanned for
evaluator vocabulary and for any line of the fixed implementation or gold
tests. A candidate-derived detail that trips the scan is dropped.

## Final-slot policy (frozen, identical for B and C)

Candidates are interleaved: round-1 slot 1, round-2 slot 1, round-1 slot 2, and
so on. Well-formed non-duplicates come first in that order, and the first eight
are kept. "Well-formed" treats passing, failing and raising in the target
identically, so the policy is blind to every assertion outcome, to kill status
and to the fixed implementation. There is no reranking.

## Panel (qualified exploratory pilot)

`results/v4_3_tool_assisted_panel.json` holds 264 train-shard function records
from 110 lineages (MBPP; 186 simple, 78 moderate). They are disjoint from all of
the following:
- every training arm of the frozen control and its successors;
- the 97-item mechanism panel;
- the 613-record retention panel;
- the confirmation lineages;
- every protected split.

**It is not untouched.** Every train lineage, these included, appears in the O1
oracle-dataset artifact, which holds labelled base-model generations. No untouched
train function lineage exists. There are also no complex-tier, HumanEval or
repository records available: all 13 train repository projects are in the
control arm. Any result is an **exploratory pilot**, never confirmatory.

## Metrics and gates (frozen before any generation)

**Primary:** C − B, paired by record, with 90% lineage-cluster bootstrap intervals (10,000 replicates, seed 20260925):
- Kill@1, Kill@4 and Kill@8;
- reference-valid candidates per requested candidate;
- functions with at least one reference-valid candidate;
- parse success and execution success;
- unique defects killed;
- model calls, generated tokens and wall-clock time.

**Secondary:**
- duplicates in the final slots;
- candidate diversity;
- failure-category transitions for repaired slots;
- A − B and A − C.

| outcome | condition |
|---|---|
| **pass** | Kill@8 C − B ≥ +5 pp (the project's existing minimum practically important gain) with lower bound > 0; reference-valid-per-requested lower bound ≥ −3 pp (the project's existing noninferiority margin); exact-unique ratio loss ≤ 0.05; C wall ≤ 1.5× B; compute matched |
| **fail** | Kill@8 upper bound < +5 pp, or reference-validity upper bound < −3 pp |
| **inconclusive** | anything else; does not pass |

A syntax-only improvement cannot pass: the gate is on Kill@8, and parse and
execution rates never gate. If compute is not matched, the analysis refuses and
the comparison is not called controlled.

## Stopping, rollback and confirmation

**Stopping rules:**
- Stages run strictly one after another: `generate-a`, `generate-bc`, `score-b`, `score-c`, then the frozen analysis on the CPU.
- The analysis refuses partial results.
- A crash resumes from the append-only journal without repeating a completed model call.
- There is no rerun with changed prompts, budgets, taxonomy or thresholds.

**Rollback:** no weights are written and nothing is promoted. If an integrity
check fails, the artifacts are kept, the result is declared invalid, and the
frozen control remains the reference.

**Confirmation:** only a PASS permits a *request* for authorization to design a
confirmation on untouched data. The 100 unopened confirmation lineages were
reserved for the execution-supervision line and are not opened by this
experiment.

## Actual Atheris and native-repository readiness

**Atheris:**
- **What is installed:** the real `atheris` 2.3.0 package on Python 3.11.15 under WSL2 Ubuntu 24.04, not the simulated `baseline.coverage_fuzzer`.
- **Instrumentation smoke:** instrumentation is active. On a synthetic reachable branch, coverage rose from 2 to 3 and a semantic kill was found in 11 runs, recorded separately from crash kills.
- **Defects:**
  - fixed input-adapter ranges: integers are limited to [−1000, 1000], so reachability must be reported per target;
  - the earlier validation-panel artifacts predate the instrumentation fix;
  - no permitted panel exists.

**Native repository:**
- **Official-test pilot:** 3 of 5 official BugsInPy tests reproduce buggy-fail / fixed-pass (thefuck, sanic, tornado). Infrastructure failure is 40%.
- **Generated tests:** none has been run natively.
- **Evidential status:** all train repository projects are in the control arm, so a native run is an engineering smoke only.

**Smoke plan (3–5 cases, not yet run):**
- the three reproduced tasks;
- a passes-both negative control;
- a forced environment failure, which must be classed `environment_unavailable`.

It needs about 15–45 CPU minutes in WSL and about 1 GPU minute. The full
readiness record is in `results/v4_3_tool_assisted_audit.json`.

## Runtime and storage estimate (GPU, strictly sequential)

- **Arm A:** about 6 min.
- **Arms B and C generation:** one shared pass, roughly 45–130 min; each record needs 8 single-sequence calls plus repairs.
- **Scoring B and C:** about 3 CPU minutes.
- **Storage:** about 40 MB in total, covering the lineage files, the journal and three evaluation artifacts.

## Artifacts

**Tracked:**
- `results/v4_3_tool_assisted_audit.json`
- `results/v4_3_tool_assisted_panel.json`
- `results/v4_3_tool_assisted_design_receipt.json`

**Written when run (ignored):** `results/v4_3_tool_assisted_v1/`.
