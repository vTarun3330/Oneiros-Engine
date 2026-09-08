# Prompt-borne answers in HumanEval mutation testing

**Panel:** `ablation_dev` (112 HumanEval, 429 MBPP mutation targets) and the locked
`val` panel (56 HumanEval, 701 MBPP). Seed 42. Corpus
`v4_1_research_hardened_candidate`. The sealed test split was not opened; every
script here refuses an artifact marked `final_test_measurement` or split `test`.

**Rebuild:** `scripts/audit_native_example_leakage.py`,
`scripts/measure_example_copying.py`, `scripts/measure_prompt_giveaway_effect.py`.
No figure below is typed by hand.

---

## 1. The observation

HumanEval specifications are docstrings, and docstrings routinely contain worked
examples. MBPP specifications are a single sentence. Measured across all three
non-sealed splits:

| split | benchmark | n | with a worked example | median spec |
|---|---|---:|---:|---:|
| train | humaneval | 964 | 82.5% | 346 chars |
| train | mbpp | 4650 | **0.0%** | 80 chars |
| ablation_dev | humaneval | 112 | 87.5% | 338 chars |
| ablation_dev | mbpp | 429 | **0.0%** | 77 chars |
| val | humaneval | 56 | 60.7% | 497 chars |
| val | mbpp | 701 | **0.0%** | 76 chars |

A worked example states an input and its correct output. In mutation testing that
is not neutral context: if the mutant produces something else for that input, the
prompt contains a killing assertion.

## 2. How often the prompt contains a killing assertion

Examples were parsed out of each record's own specification and executed against
both the reference and the mutant that record actually contains.

`ablation_dev`, HumanEval, 112 records:

- 76 records state at least one parseable example
- 282 examples parsed; 108 could not be verified against the reference
- **98 examples (34.8%) state a value the mutant does not produce**
- **52 records (46% of all 112) hand the model a killing assertion**

Verified by hand rather than trusted. `HumanEval_103`'s prompt states
`rounded_avg(1, 5) -> "0b11"`; its mutants return `'0b100'`, `-1`, and
`'0b1001011'` respectively. `assert rounded_avg(1, 5) == '0b11'` kills all three
and requires no reasoning about the code.

## 3. The shortcut is taken

A kill counts as *copied* when a killing candidate calls the function with the
same arguments as a stated example and asserts that example's stated output.
Only exact `assert call == value` matches count, so paraphrases, reordered
arguments and equivalent literals are missed. **These are floors.**

| panel | arm | kills | prompt had an example | copied | of all kills | where available |
|---|---|---:|---:|---:|---:|---:|
| ablation_dev | base | 75 | 46 | 33 | **44.0%** | 71.7% |
| ablation_dev | relearning | 81 | 54 | 37 | **45.7%** | 68.5% |
| val | base | 47 | 17 | 6 | 12.8% | 35.3% |
| val | relearning | 53 | 17 | 13 | 24.5% | **76.5%** |
| val | v4.2 corpus | 51 | 16 | 14 | 27.5% | **87.5%** |

On the locked panel, where an example was available, the base model copies it in
35.3% of its kills and the relearning arm in 76.5%. **SFT roughly doubled the
copying rate.** Only 17 val records carry an example, so the val rates are
small-sample; `ablation_dev` is the larger and more reliable measurement.

## 4. What it is worth — the correction

The obvious next claim is that HumanEval's advantage over MBPP is this leakage.
**That claim is wrong, and the first version of this analysis made it.**

A quick stratification sorted records into "states a killing value" and
"everything else". The second bucket silently absorbed 18 records whose stated
examples cannot be verified against the reference at all. Those records kill at
0.2222. Pooling them moved the clean stratum from 0.7381 to 0.5833 — which sits
almost exactly on MBPP's 0.5758 and produced the false headline that the two
benchmarks are equally hard once the giveaway is removed.

The correct three-way split, same evaluation, no re-run:

| stratum | n | base | relearning |
|---|---:|---:|---:|
| prompt states a killing value | 52 | 0.7692 | 0.8269 |
| no giveaway, examples verified | 42 | **0.7381** | **0.7619** |
| examples unverifiable on the reference | 18 | 0.2222 | 0.3333 |
| mbpp (no examples exist) | 429 | 0.5758 | 0.7156 |

**The giveaway is worth +3.11 points to the base model and +6.50 to the
relearning arm.** Clean HumanEval still beats MBPP by 16.2 points for the base
model. HumanEval is genuinely easier for this model, not merely leakier.

`tests/test_prompt_giveaway_effect.py` reconstructs the pooling bug so it cannot
return, and asserts the advantage stays below 10 points — a guard against the
overclaim, not only against the undercount.

## 5. Reconciling 44% of kills with +3 points

These are not in tension. Each target gets 8 candidates and is killed if any one
of them kills. Copying is the *cheapest* route to a kill, not the only one: on a
record whose prompt gives the answer away, the model would frequently have found
some other killing assertion among its eight attempts.

So prompt-borne answers inflate **candidate-level efficiency** far more than
**function-level kill@8**. `kill@k` is comparatively robust to the shortcut, and
a candidate-level metric on this benchmark would not be.

## 6. What this changes

1. **HumanEval mutation scores are partly a reading-comprehension measurement.**
   46% of records contain the answer and ~45% of kills on `ablation_dev` take it.
   Any HumanEval mutation-testing number should report this, and none we found in
   the surrounding literature does.
2. **SFT learns the shortcut.** Copying roughly doubled on the locked panel after
   training. Some of the reported +10.7-point HumanEval gain is that, not improved
   test design.
3. **It retired a planned intervention.** MBPP has no worked examples, and
   synthesising them was the next arm — 95% lineage coverage was achievable. But
   56% of the synthesised examples failed on their mutant, so the arm would have
   raised the MBPP score by importing exactly this artifact. It was not run. The
   script is kept, unused, with its disclosure report, because the measurement it
   produced is the reason not to use it.
4. **It does not explain the MBPP ceiling.** Clean HumanEval is still 16 points
   above MBPP. The MBPP gap is a real difficulty gap, addressed separately in the
   value-prediction analysis.

## 7. Threats to validity

- **The copy count is a floor.** Exact matches only.
- **`ablation_dev` is the selection panel** and overstates arms by ~3.2x. It is
  used here for the *leakage* measurement, which concerns prompts and mutants
  rather than an arm's gain, so selection bias does not apply to sections 2-4 —
  but the arm kill rates in the stratification table are still selection-panel
  numbers.
- **Example parsing is heuristic** — doctest form and inline arrows. 108 of 282
  parsed examples could not be verified against the reference; they are reported
  as their own stratum rather than assumed either way.
- **Single seed** for the stratification.
- **Synthetic mutation targets only.** The 24 repository targets in `val` are not
  evaluated by any number in this document.
