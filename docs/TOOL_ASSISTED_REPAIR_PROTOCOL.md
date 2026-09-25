# Execution-feedback repair at inference — frozen design

First frozen 2026-09-24 and re-frozen the same day to add a sham-feedback
control. Both freezes happened before any generation. This document covers the
CPU audit, implementation, tests and preflight only. **No generation has been
run. GPU generation needs explicit approval.**

The execution-supervision SFT line is closed and is not reopened here. This is a
different hypothesis: it writes no weights and trains nothing.

## Question

> Does structured execution feedback improve test generation beyond sham
> self-conditioning under matched calls, sequences, input tokens and output-token
> caps?

The model is never asked to simulate the program. The loop does the following,
using only what a developer could see:
1. executes each candidate in the sandbox against the code under test;
2. classifies failures into a closed taxonomy;
3. returns bounded, structured feedback for demonstrably invalid artifacts;
4. lets the model make one repair per slot.

## Arms

| arm | round 1 | round 2 (8 single-sequence calls, one per slot) | role |
|---|---|---|---|
| A | unchanged successor path: 8 samples, batch 2, one seed | — | canonical historical control (secondary; not matched) |
| B | **shared** 8 samples | on every slot where C is repaired: a **sham** prompt; elsewhere the identical canonical sample C gets | **sham-feedback control** |
| C | **shared** 8 samples | slot *i*: a repair of a demonstrably invalid round-1 candidate when a matched sham exists; otherwise the identical canonical sample B gets | **treatment** |

**The comparison of interest is C versus B.**

### The sham

On each repaired slot, the B sham contains four parts:
1. the identical canonical prompt;
2. the identical echoed round-1 candidate, with the same parent;
3. the neutral sentence **"No diagnostic information is provided."**;
4. the identical final request **"Write one test. Output only the test."**

B is never told whether its previous candidate parsed, passed, failed, raised,
timed out, duplicated another or violated policy. The sham sentence is shorter
than every feedback template.

**Padding to C's exact length.** The sham is padded with a frozen neutral marker,
`" ."`. This marker is punctuation, not a word, and each repetition adds exactly
one token on the Qwen chat template. The unit was measured before freezing: it
never merges with its neighbours, whereas whitespace runs do. Padding continues
until the sham's rendered input-token count equals C's, rendering both through
the adapter's own prompt path and chat template.

**When a match is impossible.** Nothing is ever truncated. If either prompt would
need compaction, or an exact count cannot be reached, **neither** arm is repaired:
both take the identical canonical sample, and the slot is recorded as
`repairs_skipped_match_infeasible`.

### What is and is not matched

This is a **call-, sequence-, cap- and rendered-input-token-matched control**
(request-budget-matched). It is not "exact compute matched".

| quantity | status |
|---|---|
| model calls per target | equal by construction |
| sequences per target | equal by construction (16) |
| max-new-token cap | equal (1,024) |
| rendered input tokens, every compared repair/sham slot | exactly equal |
| actual output tokens | measured outcome; may differ (early EOS) |
| wall-clock time | measured outcome; may differ |

Before launch, the sham was measured on the panel (264 records × 10 categories):
- **Typical candidate echo:** a sham could be matched in every case (2,640 of 2,640), with at most 24 pad units.
- **Echo headroom:** after the canonical prompt and the longest feedback, 514–604 tokens remain for the echo (median 571).
- **Long echoes:** a realistic 1,500-character test-function echo can be matched in only 5 of 2,640 cases, and a token-dense one in none. Such repairs are skipped in both arms and recorded.

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
removed before the loop runs. The evaluator uses them only after the eight final
slots are frozen.

The preflight proves that the canonical prompt built from the view is
byte-identical to the one built from the full record, for all 264 records.
Buggy-side execution uses `harness/buggy_side_execution.py`. Its signature has
no channel for a reference; it runs each candidate in a fresh isolated process
and reports which frame raised.

## Feedback taxonomy (`oneiros_execution_feedback_v1`, closed)

**Repairable** (C receives one fixed-template sentence):
- unparsable output;
- invalid Python syntax;
- a prohibited construct;
- a missing call to the function under test;
- an invalid candidate shape;
- an invented attribute or name;
- a malformed invocation at the call site;
- a timeout inside the candidate's own code;
- a duplicate candidate.

**Retained silently** (kept exactly, and the model is told nothing):
- executes without error;
- **observed assertion failure**;
- an exception or timeout **inside the code under test**;
- infrastructure failure;
- a repair identical to its parent.

Every feedback object is canonical JSON with a SHA-256. It is scanned for
evaluator vocabulary and for any line of hidden material.

## Final-slot policy (frozen, identical for B and C, text-only)

1. Interleave the candidates: round-1 slot 1, round-2 slot 1, round-1 slot 2, and so on.
2. Put parser- and policy-valid, non-duplicate candidates first, in that order.
3. Keep the first eight.

The policy reads only the slot, the round, parse validity and the candidate
text. It never reads any execution, assertion, kill or fixed-side outcome, so a
clean pass, an assertion failure and a target exception are treated identically.
An adversarial test that injects fake kill, validity, assertion, execution and
category fields shows that the selection does not change.

## Panel (unchanged; qualified exploratory pilot)

`results/v4_3_tool_assisted_panel.json` holds 264 MBPP train-shard function
records from 110 lineages (186 simple, 78 moderate). They are disjoint from all
of the following:
- every training arm;
- the 97-item mechanism panel;
- the 613-record retention panel;
- the confirmation lineages;
- every protected split.

It is **not untouched**: every train lineage appears in the O1 oracle-dataset
artifact. It also has no complex-tier, HumanEval or repository records. Any result
is exploratory, never confirmatory.

## Metrics and gates (unchanged thresholds)

**Primary:** C − B, paired by record, with 90% lineage-cluster bootstrap intervals (10,000 replicates, seed 20260925):
- Kill@1, Kill@4 and Kill@8;
- reference-valid candidates per requested candidate;
- functions with at least one reference-valid candidate;
- parse success and execution success;
- unique defects killed.

**Reported separately:**
- input tokens and output tokens for B and C, and the output-token ratio;
- generation wall time and the wall-time ratio;
- repairs attempted, delivered, and skipped because matching was infeasible;
- failure-category transitions;
- A − B and A − C (secondary).

| outcome | condition |
|---|---|
| **pass** | Kill@8 C − B ≥ +5 pp with lower bound > 0; reference-valid-per-requested lower bound ≥ −3 pp; exact-unique ratio loss ≤ 0.05; C wall ≤ 1.5× B; request-budget-matched pairing on every target |
| **fail** | Kill@8 upper bound < +5 pp, or reference-validity upper bound < −3 pp |
| **inconclusive** | anything else; does not pass |

**The analyzer refuses** the whole comparison unless every target satisfies all
of the following:
- equal B/C calls, sequences and caps, and a single cap across targets;
- a shared round 1;
- a B sham for every C repair, echoing the same parent;
- identical rendered input tokens;
- an exactly neutral sham, with no diagnostic text;
- identical shared samples on non-repair slots;
- delivered and skipped slots that are consistent.

The output-token difference is reported as a limitation and never called equal
compute.

## Stopping, rollback and confirmation (unchanged)

**Stopping rules:**
- Stages run strictly in order: `generate-a`, `generate-bc`, `score-b`, `score-c`, then the frozen analysis.
- The analysis refuses partial results.
- A crash resumes from the append-only journal. Repair and sham are journalled separately, so neither member of a pair is repeated.
- There is no rerun with changed prompts, budgets, taxonomy or thresholds.

**Rollback:** no weights are written and nothing is promoted. If an integrity
check fails, the artifacts are kept, the result is declared invalid, and the
frozen control remains the reference.

**Confirmation:** only a PASS permits a *request* to design a confirmation on
untouched data. The 100 unopened confirmation lineages are not opened by this
experiment.

## Actual Atheris and native-repository readiness

These are unchanged; see `results/v4_3_tool_assisted_audit.json`.

**Atheris:** the installed package is the real `atheris` 2.3.0 on Python 3.11.15
under WSL2, and its instrumentation is active: a semantic kill was found in 11
runs, with coverage rising from 2 to 3.

**Native repository:** not ready. 3 of 5 official tests reproduce, and no
generated test has been run natively.

## Runtime and storage estimate (GPU, strictly sequential)

- **Arm A:** about 6 min.
- **Arms B and C generation:** one shared pass of about 45–130 min. For *k* delivered repairs, each record needs one 8-sequence call plus 8 + *k* single-sequence calls. That count is unchanged from the previous design, because the sham replaces B's canonical call on repaired slots. Matching adds CPU tokenisation only.
- **Scoring B and C:** about 3 CPU minutes.
- **Storage:** about 40 MB.

## Artifacts

**Tracked:**
- `results/v4_3_tool_assisted_audit.json`
- `results/v4_3_tool_assisted_panel.json`
- `results/v4_3_tool_assisted_design_receipt.json`

**Written when run (ignored):** `results/v4_3_tool_assisted_v1/`.

## Execution record (appended after the run; the protocol above is unchanged)

GPU generation was authorized for `5e8692b` (design receipt built at `cab29d3`).
The pilot ran strictly sequentially on 2026-09-25:

| run | stage | duration |
|---|---|---:|
| `20260925-103511-toolassist-generate-a-qwen15b-s42` | generate A | 298.5 s |
| `20260925-104038-toolassist-generate-bc-qwen15b-s42` | generate B/C | 3,168.2 s |
| `20260925-113357-toolassist-score-b-qwen15b-s42` | score B | 42.1 s |
| `20260925-113501-toolassist-score-c-qwen15b-s42` | score C | 44.1 s |

Every run exited with code 0.

**Integrity.** B/C pairing passed on all 264 targets, and the journal holds 2,825
entries. No call was replayed, because there was no disconnect.

**Repairs.** 450 were attempted, 449 delivered, and 1 skipped because matching
was infeasible. Of the delivered repairs, 419 were duplicate-candidate feedback.

**Result.** Kill@8 C − B was +0.38 pp, with a 90% interval of [−1.07, +1.77].
The frozen verdict is **FAIL**, and the line is stopped.

The full numbers are in `results/v4_3_tool_assisted_analysis.json` and
`results/v4_3_tool_assisted_decision_receipt.json`. The latter also binds the
journal, the lineage-file manifest, every run, the adapter and the source hashes.
