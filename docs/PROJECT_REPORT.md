# Oneiros: a chronological record

**Branch:** `experiment/research-eval-ablations`
**Panels:** train / ablation_dev (selection) / val (locked) / test (sealed, never opened)
**Model:** Qwen2.5-Coder-1.5B-Instruct @ `2e1fd397`, QLoRA NF4 4-bit, LoRA r=16 α=32
**Task:** given a specification and a mutated function, generate a test that
distinguishes the mutant from the correct implementation. Scored as **kill@8**.

This is the running record, not a summary of the wins. On this project the
negative results and the defects we found in our own measurement are the
substance: three of the most consequential findings are places where we were
measuring wrong in a way that flattered us.

Every figure here is rebuilt from committed artifacts. Where the record is
thin, the document says so rather than filling the gap with a plausible story.

---

## 1. Why the model changed from Phi-3 to Qwen

**2026-09-01.** A zero-shot screen ran both candidate backends through the
production generation and verification path, on the exact 32-function
ablation_dev monitor panel, seed and candidate budget already used for the
Phi-3 integration run — so the comparison was under the frozen protocol rather
than a convenient one. It trained nothing and touched neither val nor the
sealed test.

| backend | kill | parse rate | reference validity | seconds/function |
|---|---|---|---|---|
| Phi-3-mini (eager) | 19/32 | 0.508 | 0.270 | 6.53 |
| Qwen2.5-Coder-1.5B | 23/32 | **0.980** | **0.566** | **1.28** |

The harness reproduced Phi-3's recorded 19/32 baseline exactly, which is what
licensed trusting the second row.

**The decision did not rest on the kill rate.** Wilson intervals overlap at
n=32. It rested on candidate validity — parse 0.508 → 0.980 and reference
validity 0.270 → 0.566, measured over 256 candidates per arm — and on a 5×
speed difference that made every later experiment affordable.

Recorded honestly at the time: Phi-3 + SDPA is *infeasible*, not omitted.
transformers 4.48.3 refuses SDPA for `Phi3ForCausalLM` and flash-attention does
not cover its sliding-window attention, so eager was the only admissible Phi-3
backend. Phi-3 was therefore measured on its best available configuration, not
handicapped.

---

## 2. The corpus and the supervision

- **V4.1** — the immutable research corpus. 8,237 records across train,
  ablation_dev, val and a sealed test split.
- **V4.2** — a versioned successor adding 26 verified HumanEval records, with
  its own hashes and manifest. V4.1 was never overwritten.
- **Multi-mutant supervision** — one broad test per lineage, built by
  deterministic weighted greedy set cover, killing a **mean of 8.43 sibling
  mutants**; 64.6% kill ≥4 siblings and 58.7% kill ≥5.

That last point answers a review question directly: the request was one test
covering four or five defects. The dataset does that and then some. Whether the
*model* does it is a separate question, and section 6 is where it gets
answered — the answer turned out to be uncomfortable.

---

## 3. Every arm that was trained, and what it measured

All figures: kill@8 on the locked validation panel, seed 42, 757 synthetic
targets. Rebuilt by `scripts/build_results_table.py`.

| arm | val@8 | vs base | train@8 | gap |
|---|---|---|---|---|
| base (untrained) | 0.6209 | — | 0.6789 | 5.80 |
| **relearning** | **0.6592** | **+0.0383** | 0.8267 | 16.75 |
| v4.2 corpus | 0.6407 | +0.0198 | 0.8389 | 19.82 |
| full_density | 0.6354 | +0.0145 | 0.8289 | 19.35 |
| regularised dropout 0.10 | 0.6196 | −0.0013 | 0.8300 | 21.04 |
| long (2 epochs) | 0.6182 | −0.0026 | 0.8267 | 20.84 |
| relearning checkpoint-100 | 0.6116 | −0.0092 | 0.8122 | 20.06 |

**What this says.** Every arm fits the training panel hard (~0.83) and moves
validation barely. The untrained base already shows a 5.8-point train/val gap
purely from panel difficulty, so SFT is adding 11–15 points of gap on top. Two
of six arms are *negative*. Regularisation did not reduce training fit — the
dropout arm has the highest of all.

**Overfitting is front-loaded.** Between step 0 and 100 the model gained +13.3
on train and *lost* 0.9 on validation. Two epochs made it worse, not better.

---

## 4. The one arm that works, and why it still isn't significant

Relearning — retraining on verified corrections for hard training-split
failures — is the only intervention with a consistent direction.

Twelve paired seeds on locked validation:

```
+0.0383 +0.0449 +0.0159 +0.0542 +0.0423 +0.0132 +0.0066 −0.0291
+0.0343 +0.0145 +0.0013 +0.0185
```

**11 positive, 1 negative, mean +0.0212.** Exact two-sided sign test over seed
signs: raw **p = 0.0063**.

**Holm-adjusted within the pre-registered family of thirteen: p = 0.0825.**
That is the number that counts. An unadjusted p below 0.05 that does not
survive adjustment is not significant; it is not significant *with a caveat*.

Two honest notes. The extension from 8 to 12 seeds was decided *after* seeing
p = 0.0703 — optional stopping — and was declared in writing before the seeds
ran (`docs/RELEARNING_SEED_EXTENSION_DECLARATION.md`). And the effect estimate
**shrank** as seeds accumulated, from +0.0383 at the first seed to +0.0212
across twelve. The direction held up better than the magnitude.

---

## 5. Where the model is strong, and why that was misleading

Per-benchmark, base → relearning on locked validation:

| benchmark | n | base | SFT | gain |
|---|---|---|---|---|
| humaneval | 56 | 0.8393 | 0.9464 | **+10.7** |
| mbpp | 701 | 0.6034 | 0.6362 | +3.3 |

SFT helps where the model is already strong. Since the panel is **92.6% mbpp**,
that is close to useless for the headline number.

### 5.1 Why MBPP is hard: the specification gap

| split | benchmark | with a worked example | median spec |
|---|---|---|---|
| train | humaneval | 82.5% | 346 chars |
| train | **mbpp** | **0.0%** | **80 chars** |
| val | mbpp | **0.0%** | 76 chars |

Every MBPP specification is one sentence with no input/output example —
*"Write a function to find the demlo number for the given number."* The model
cannot determine what the function should return.

The dominant failure is `wrong_expected_value`: the candidate asserts something
the **correct** reference does not produce. Conditional on being valid on
correct code, SFT improved test *design* on both benchmarks (+8.1 humaneval,
+11.9 mbpp) — but MBPP validity *fell* from 43.7% to 32.5%, cancelling the
gain. Training the model to assert more sharply made it commit to more wrong
values.

### 5.2 The headroom is real

Re-running each failed candidate's own call against reference and mutant:
**76% of failed MBPP candidates probe an input where the bug actually shows.**
The model picks killing inputs and mispredicts the outputs. For the relearning
arm, 88.5% of the functions it fails already contain such a candidate.

Upper bound under perfect value prediction: 0.7156 → 0.9674. That is a *bound*,
not a forecast.

**Correction — those two figures are `ablation_dev`, not locked validation.**
Found while acting on external review (§11). This section otherwise reads as
locked-panel, and ablation_dev overstates every arm on this project by ~3.2×.
Recomputed on the **locked** panel, and reported per benchmark rather than
pooled (`scripts/build_oracle_vs_input_taxonomy.py`):

| arm | panel | MBPP oracle share | MBPP functions one value from a kill | upper bound |
|---|---|---|---|---|
| base | ablation_dev | 0.7560 | — | 0.8951 |
| relearning | ablation_dev | 0.7619 | — | 0.9674 |
| base | **locked val** | **0.7074** | 207 / 701 | 0.8983 |
| relearning | **locked val** | **0.7310** | 216 / 701 | 0.9458 |

The finding survives the correction — it is the *magnitude* that moves, by 3–5
points. The locked-panel statement is: **216 of 701 MBPP validation functions
(30.8%) already contain a candidate that probes a distinguishing input and
fails only on its asserted value.**

---

## 6. Findings that changed the plan

### 6.1 HumanEval prompts contain the answers

46% of ablation_dev HumanEval records state, in their own prompt, an input and
the correct output for it that the mutant does not produce. Asserting that
stated pair kills the mutant with **no reasoning at all**.

Verified by hand: `HumanEval_103`'s prompt states `rounded_avg(1, 5) -> "0b11"`
while its mutants return `'0b100'`, `-1`, `'0b1001011'`.

And the shortcut is taken. Counting only exact matches, so these are floors:

| panel | base | relearning |
|---|---|---|
| ablation_dev | 44.0% of kills copied | 45.7% |
| locked val | 12.8% | **24.5%** |

**SFT roughly doubled the copying rate.** Part of the +10.7-point HumanEval
gain is the model learning to exploit prompt-borne answers.

**Correction to an earlier claim of mine.** I first reported that removing the
giveaway made HumanEval and MBPP equally hard. That was wrong by ~16 points —
my stratification swept 18 records whose examples cannot be verified into the
"clean" bucket. Correctly split, the giveaway is worth **+3.11 points** (base)
and +6.50 (relearning). Clean HumanEval still beats MBPP by 16.2 points.

**This retired a planned intervention.** Synthesising worked examples for MBPP
was next, and technically feasible at 95% lineage coverage — but 56% of the
generated examples failed on their mutant, so the arm would have raised the
MBPP score by importing exactly this artifact. It was not run.

### 6.2 The evaluator could only ever score one assertion

The generation parser scanned each model output for the first line beginning
`assert ` and discarded the rest. The multi-mutant capability was therefore
demonstrated in the **dataset** and never measurable in the **model**.

A successor protocol was declared (`oneiros_whole_output_successor`), with
source hashes, budgets, timeout policy, a no-reranking rule, and an explicit
statement that legacy results remain historical. Both interpretations were then
computed from the **same retained generations**, so no generation randomness
separates them.

**The model does write multi-assertion tests.** 3,729 of 4,336 relearning
outputs carry more than one assertion; the mode is three. The base model emits
a single bare assertion 98.6% of the time.

**And scoring them whole makes it worse:**

| | legacy (first assertion) | successor (whole output) |
|---|---|---|
| kill@8 | **0.7159** | 0.5480 |
| reference validity | 0.4880 | 0.2749 |

120 targets die under legacy that survive the successor; 29 the other way.

**My own hypothesis was wrong here.** I attributed the gap to truncation at the
128-token completion limit. Raising the budget to 1024 removed 90% of the
truncation — incomplete ASTs fell 481 → 49 — and moved the gap by **less than a
point**. The mechanism is **conjunction**: a test function is valid only if
*every* assertion holds, and at 3.65 assertions with roughly even odds each,
validity collapses. The frozen parser had been acting as a repair step.

### 6.3 The Atheris baseline was never coverage-guided

The harness set `enable_python_coverage=True`, so it looked instrumented. But
libFuzzer's counters come from Atheris *bytecode* instrumentation, which a
target loaded by a bare `exec()` never receives.

The evidence was in the logs the whole time: **740 of 757** warned *"no
interesting inputs were found. Is the code instrumented for coverage?"* and
**697 finished with the corpus still at one input of one byte.** That is random
search reported as coverage-guided fuzzing.

A second defect: reference and mutant were called with the **same argument
objects**, so a target that mutates its input made two identical
implementations disagree — recorded as a kill.

A third: 41.7% of surviving targets survived because our adapter could not
construct the input that kills them. `_consume` fell back to an integer for any
shape it did not recognise, so `list[list]` (109 occurrences) and `list[tuple]`
(24) were handed integers. Five-argument targets had a kill rate of **exactly
zero**.

After fixing all three, on the same frozen panel:

| seed | blind | instrumented | + widened adapter |
|---|---|---|---|
| 42 | 360 | 371 | **451** |
| 43 | 362 | 371 | **429** |
| 44 | 365 | 438 | **496** |

**Mean kill rate 0.4786 → 0.6059.**

```
Atheris, corrected     0.6059
Oneiros base           0.6209
Oneiros relearning     0.6592
```

The relearning arm still leads — **by ~5.3 points, not the ~18 the old figures
implied.** Three defects in *our* harness had been understating the baseline,
and with 23% of survivors still unreachable, 0.6059 is a floor.

Also measured, and worth stating: no equivalent mutants were found in the
sample. Every survivor is demonstrably killable.

### 6.4 The relearning queue was correcting the wrong thing

`classify_loser` read per-candidate detail from `candidates` and
`candidate_results`. **No artifact writes either name** — they write
`candidate_outcomes`. The loop never executed once.

| | before | after |
|---|---|---|
| losers labelled `no_kill` | **153/153** | 10 |
| losers labelled `wrong_oracle` | 0 | **140** |

**89% of relearning's inputs are wrong-oracle cases**, and relearning had been
generating corrections for a category describing 6% of them. A plausible reason
the arm is positive in direction but small in size.

---

## 7. The rebuild (P1–P3)

**Decision settled by measurement: fewer, surer assertions.** Per-assertion
*scoring* was rejected — it raises the reported number without the model
improving.

- **P2** — 663/663 lineages yielded one verified killing assertion, each
  executed against reference and mutant first; anything failing on the
  reference discarded rather than taught. Mean 2.96 → 1.00 assertions.
- **P1** — difficulty tiered from six *training-only* signals. Validation is
  refused by construction: passing a val artifact to the loader raises. Four
  progressive mixed blocks, hard share 0.15 → 0.65, replay anchors retained,
  realised mixture recorded rather than assumed.
- **P3** — 614 examples, fully deduplicated, every block mixed, no validation
  signal.

**Cost of the decision, measured:** single assertions retain **83.3%** of
sibling-kill breadth (5297 → 4414; mean 8.63 → 7.19 siblings).

**An honest negative from P3:** mbpp holds **83.6%** of the regularised view —
*worse* than its 54.7% corpus share — and the source cap **never fired**,
because it caps repetition and deduplication had already removed every repeat.
A control that is configured but not binding is worse than none. The remedy
consistent with the standing instruction is more non-mbpp lineages, not fewer
mbpp ones.

### 7.1 The result: the rebuild made it worse

The arm combining all three (`local_sft_curriculum_s42`, seed 42) was trained
and evaluated on both panels.

| arm | ablation_dev | locked val | vs base (val) |
|---|---|---|---|
| base (untrained) | 0.5959 | 0.6209 | — |
| **curriculum (P1+P2+P3)** | 0.6550 | **0.5984** | **−0.0225** |
| relearning | 0.7177 | 0.6592 | +0.0383 |

**It is worse than the untrained base on locked validation**, and well below
the relearning arm. Per benchmark: humaneval 0.9286, mbpp **0.5720** — the mbpp
figure is below base's 0.6034, and mbpp is 92.6% of the panel.

It also reproduces the selection-panel trap exactly: **+5.9 points on
ablation_dev, −2.3 on locked val.** Judged on the panel it was built against,
this arm looks like the second-best result in the project. Judged on the locked
panel, it is the worst trained arm measured.

**The prediction behind the decision did not hold.** Fewer, surer assertions
was supposed to raise reference validity by removing the conjunction penalty.
It did not: 0.3661 here against relearning's 0.3590 on the same panel — within
noise of each other, and both far below the untrained base's 0.4536. Cutting
assertions from 2.96 to 1.00 bought essentially nothing in validity and cost
kill@8.

**Confounds, stated plainly.** This is one seed, and three changes were made at
once — single-assertion supervision, curriculum ordering, and deduplication —
so no individual contribution can be attributed. More seriously, **I dropped
`--relearning-dataset` from this arm's command**, so it lacks the relearning
corrections entirely. The like-for-like comparison is therefore against
`full_density` (multi-assertion supervision, also no relearning): 0.5984 against
0.6354. Still worse, but by 3.7 points rather than 6.1.

The supervision was also narrower: 4,414 keyed completions against
multi_mutant_v1's 5,588.

**What it does support.** The single-assertion decision, as implemented, is not
an improvement, and section 6.2's finding should not be read as implying it
would be. That the frozen parser scores multi-assertion output better than the
successor does *not* mean training on single assertions produces a better model
— those are different claims, and only the first is supported.

---

## 8. The 80% target

**Not reached, and not close.** The best arm is 0.6592 on locked validation.

The arithmetic: 757 targets = 56 humaneval + 701 mbpp. Reaching 80% needs 606
killed, 121 more than the best arm. Even with *perfect* humaneval, mbpp would
need 0.7846 against its current 0.6191 — **+16.6 points**, where the best any
arm has produced is +3.3.

On the evidence, 80% is not reachable on this panel at 1.5B without
contaminating it. The one intervention that would have moved the number —
synthesising MBPP examples — was retired precisely because it would have
manufactured the threshold rather than reached it.

---

## 9. Threats to validity

- **All reported kill rates are over synthetic targets.** The 24 repository
  targets in val are not evaluated by any number here
  (`REPOSITORY_EVALUATION_STATUS = not_implemented_requires_native_project_environment`).
- **ablation_dev overstates arms by 3.2×** (+10.5 pts vs +3.3 on locked val,
  three seeds, non-overlapping ranges). Any ablation_dev figure is not a
  generalisation estimate.
- **HumanEval numbers carry the prompt-copying caveat** throughout.
- **Atheris comparisons must use the corrected numbers.** Every pre-correction
  Atheris figure is invalid.
- **Monitor kill rates are not generalisation.** The 2-epoch arm reached 81% on
  the monitor and came in below the untrained base on locked validation.
- **The sealed test split has never been opened.**

---

## 10. Honest summary

**What works:** a reproducible, heavily instrumented mutation-testing pipeline;
a verified multi-mutant corpus that genuinely produces one test covering many
defects; relearning as the one directionally positive intervention.

**What does not:** seven of eight interventions were flat or negative,
including the P1-P3 rebuild built specifically to address the diagnosis. The single
positive one is **not statistically significant** after the correction it was
pre-registered under. The 80% target is out of reach on this panel.

**What we got wrong ourselves**, and this is the part worth carrying forward:
the evaluator could only score one assertion; the Atheris baseline was never
coverage-guided, was inflated by argument aliasing, and was crippled by our own
input adapter; the relearning queue mislabelled 100% of its inputs; and I
personally published two claims — that truncation explained the parser gap, and
that removing prompt giveaways equalised the benchmarks — that further
measurement contradicted.

The most defensible contribution here is not the kill rate. It is that
**HumanEval mutation scores are substantially a reading-comprehension
measurement** — 46% of records contain the answer, and ~45% of kills take it —
which we found by measuring rather than assuming, and which nothing in the
surrounding literature appears to report.

---

## 11. Response to external review

A review of §§1–10 proposed a diagnosis (an *oracle-and-distribution* problem
rather than undertraining), five process criticisms, five next steps, and a
ranking of alternatives to SFT. Each was checked against the repository rather
than accepted, because two of them turned out to describe something this
project already does, and one of them is contradicted by measurement.

The headline conclusion of the review is **correct and is now better
supported**: the bottleneck is oracle prediction, not training volume.

### 11.1 The five process criticisms, checked

| # | Criticism | Verdict |
|---|---|---|
| 1 | All kill failures treated as one learning problem | **Half right — now fixed** |
| 2 | Optimising a misleading headline | **Already done, extended** |
| 3 | Several interventions changed at once | **Correct, and already stated (§7)** |
| 4 | Assuming hard-first prevents forgetting | **Mis-aimed — we never did this** |
| 5 | Dropout treated as the whole overfitting solution | **Mis-aimed — it was one arm, reported flat** |

**(1) is the substantive one.** §6.4 fixed a classifier that labelled every
loser `no_kill`; it did not go far enough. `wrong_expected_value` was still one
bucket holding two opposite failures: a candidate that probes an input where
reference and mutant *differ* (one correct value from a kill), and one that
probes an input where they *agree* (no oracle could ever have saved it).
`harness/oracle_diagnosis.py` now separates them by re-executing the
candidate's own call, and the split is measured in §5.2 above.

**(4) and (5) are mis-aimed, and saying so matters** because acting on them
would have replaced working code. The shipped schedule opens at 40% easy / 35%
moderate / 25% hard and rises to 65% hard, with replay anchors in every block —
easy-first and mixed, which is what the review recommends. It was already so
before the review; §7 records "Do not use a hard-first/easy-last sequence" as
the governing rule. What *was* missing is that the property lived in a comment,
so it is now an executable guard
(`assert_schedule_is_progressive_and_mixed`) that refuses a hard-first or
single-tier schedule at import, with tests. The opening block moved from
50/35/15 to 40/35/25 and the curriculum was rebuilt (`curriculum_v2`); realised
mixtures are recorded per block, and every block clears a 20% replay floor.
Dropout was one arm among eight, reported at −0.0013 and explicitly called out
in §3 as *not* having reduced training fit.

### 11.2 New measurements taken in response

**Oracle errors and input errors are now separated** (locked validation, §5.2):

| benchmark | arm | oracle share of decided failures |
|---|---|---|
| mbpp | base | 0.7074 |
| mbpp | relearning | 0.7310 |
| humaneval | base | **0.1556** |
| humaneval | relearning | 0.2000 |

The two benchmarks fail in opposite ways. MBPP failures are ~73% *wrong value
on a bug-revealing input*. HumanEval failures are ~84% *wrong input entirely* —
consistent with §6.1, since a prompt that hands over a worked example makes the
value easy and leaves only input choice to get wrong. Nothing in this project
had reported that inversion.

**Cross-split near-duplication does not explain the train/val gap.**
`scripts/audit_cross_split_near_duplicates.py` compares normalised references
(AST-unparsed, docstrings stripped) by exact Jaccard over 5-token shingles. The
sealed split is never read.

| comparison | max similarity | at ≥0.80 | at ≥0.75 |
|---|---|---|---|
| train vs **locked val** | **0.7971** | 0 records / 0 functions | 45 records / **3 functions** |
| train vs ablation_dev | 0.9111 | 14 records / **2 functions** | 28 records / 4 functions |

Exact collisions: 0. Shared `group_id` lineages: 0. Specification
near-duplicates: 0.

This **refutes** the review's implied contamination route for the locked panel.
The gap between 0.83 train and 0.66 validation is not memorised duplicates.

> A note on how this was reported. The first run printed "0 near-duplicate
> pairs in val" — true, and misleading, because the closest pair sat at 0.7971,
> three thousandths under the cut. The script now reports a threshold sweep, a
> max, and record counts separately from function counts (45 records were 3
> functions). This is the same failure mode as the retracted claims in §6: a
> threshold chosen after seeing the data flatters whoever chose it.

**Verifier-guided repair is measurably NOT the highest-value next step here** —
the review's strongest recommendation, and the one that does not survive
contact with the panel. `harness/verifier_guided_repair.py` implements the loop
under a strict boundary: it may execute the candidate against *the code the
model was shown*, and may never reveal the reference, the mutation diff,
sibling mutants, gold tests, or kill status — kill status is computed from the
hidden reference, so it is evaluator information. `observe()` takes no
reference parameter, which is the structural reason it cannot leak.

That boundary is also its ceiling, and the ceiling was measured before spending
any GPU (`scripts/measure_repair_loop_headroom.py`, ablation_dev):

| arm | benchmark | repairable by execution feedback | blind spot |
|---|---|---|---|
| base | humaneval | 83.4% | 16.6% |
| base | **mbpp** | 57.9% | 42.1% |
| relearning | humaneval | 92.3% | 7.7% |
| relearning | **mbpp** | **28.3%** | **71.7%** |

The blind spot is the candidate that *already fails* on the shown code — the
loop's only honest feedback is "keep that input, check your value", and whether
the value is right requires the reference. On MBPP after relearning, **49 of
122 unkilled functions have every failed candidate in the blind spot**: the
loop has nothing to say about 40% of them.

**And this is the sharpest new result in the project:**

> **SFT converts input-selection failures into oracle failures.** The base
> model's MBPP failures are 57.9% addressable by execution feedback; after
> relearning SFT only 28.3% are. Training made the model better at finding a
> bug-revealing input and no better at saying what the function should return
> instead — the same conclusion §5.1 reached from validity rates, arrived at
> independently.

A repair loop would therefore help the **untrained base model more than the
trained one**. That inversion is worth an experiment; it is not worth a
full-panel run at the priority the review assigned it.

**Gains are not uniform across bug families** (`scripts/slice_kill_rate.py`,
relearning vs base, locked val, paired on record id):

| bug family | n | kill@8 | delta |
|---|---|---|---|
| boundary | 339 | 0.6077 | +0.0236 |
| arithmetic | 201 | 0.6617 | +0.0249 |
| comparison | 87 | 0.8276 | +0.0805 |
| off_by_one | 56 | 0.6071 | +0.0714 |
| index | 22 | 0.6818 | +0.0909 |

The +0.0383 headline is carried by the smaller families; the two that make up
71% of the panel move by ~2.4 points. Slices under 20 targets are marked
uninformative rather than quoted.

**A gap this exposed:** `complexity_tier` is `unknown` for all 757 validation
records, so the per-complexity slice the review asked for **cannot be produced**
from current artifacts. Stated rather than substituted.

### 11.3 What was not done, and why

**DPO remains out of scope.** The review ranks it as a method to compare. The
standing instruction on this project is explicit — *"This scope explicitly
excludes DPO. Do not prepare, launch, debug, evaluate, or spend GPU time on
DPO."* That instruction is not overridden by a review recommending it, so no
DPO work was prepared. Lifting it is a decision for the project owner, and the
one thing worth recording is that the review's own caveat is correct and now
has a number behind it: DPO pairs need a *precise* loser reason, and until
§11.2 the dominant loser reason was a bucket holding two opposite failures.
Building DPO pairs before that split existed would have trained on noise.

**RLVR was not attempted.** The review places it last and conditions it on a
frozen verifier. The verifier is not frozen: §6.2's successor protocol is
declared but legacy results remain the committed baseline.

**No new training run was launched.** Every measurement above is CPU-only and
runs on retained generations. The single-factor ablations that §7's confound
calls for — curriculum, deduplication and single-assertion supervision varied
one at a time, with `--relearning-dataset` present in all of them — are the
correct next GPU spend, and they are a decision about hours on the RTX 4500,
not a change to make silently.

### 11.4 Where this leaves the diagnosis

The review's framing is right and is now measured rather than argued:

1. The failure is **oracle prediction**, not training volume. 73.1% of MBPP's
   dominant failure is a wrong value on an input that already reveals the bug;
   216 of 701 locked-validation functions are one correct value from a kill.
2. It is **not** contamination. Locked validation has no near-duplicate of any
   training function at any threshold the data supports.
3. It is **not** fixable by execution feedback alone. The one signal available
   without the reference — does this test fail on the code I was shown — is
   already satisfied by 72% of the failures.
4. SFT **actively shifts** failures toward the oracle bucket while leaving the
   total roughly flat.

What follows from 1–4 is that the next intervention has to put *behavioural
evidence about the correct implementation* in front of the model, which is
exactly what MBPP's one-sentence specification withholds (§5.1) and what §6.1
retired the synthetic route to, on the grounds that it would have manufactured
the score rather than earned it. That tension is the real open problem, and
naming it precisely is what this round of work bought.
