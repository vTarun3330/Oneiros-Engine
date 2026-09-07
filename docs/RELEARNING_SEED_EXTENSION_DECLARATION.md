# Declaration: extending the relearning arm from 8 to 12 paired seeds

Written 2026-09-08, **before** any of seeds 50-53 had produced a result.

## What was already known when this was decided

The relearning arm had been measured against the base model on 8 paired seeds
(42-49) of the locked validation split. Per-seed deltas:

    +0.0383  +0.0449  +0.0159  +0.0542  +0.0423  +0.0132  +0.0066  -0.0291

Seven of eight positive. Exact two-sided sign test: **p = 0.0703**. Not
significant at 0.05, uncorrected, and the pre-declared Holm-Bonferroni family
makes it further from significance rather than closer.

## What is being done

Four more paired seeds (50, 51, 52, 53), base and relearning, on the same
locked validation panel with the same adapter, prompts, timeouts, candidate
count and selection rules. Nothing else changes.

## Why this is optional stopping, and must be reported as such

The decision to add seeds was made **after** observing p = 0.0703. That is
optional stopping: had the 8-seed result been p = 0.01, no extension would
have been run. A p-value from the pooled 12 seeds therefore does not have its
nominal false-positive rate, and reporting it as though the sample size had
been fixed in advance would overstate the evidence.

Accordingly:

1. The 8-seed result (p = 0.0703) stays in the record as the pre-specified
   analysis and is reported first.
2. The 12-seed result is reported as an **exploratory extension**, labelled as
   optional stopping, never as the primary test.
3. Both are reported whatever they show. If the four new seeds are negative
   and the pooled result weakens, that is the result.
4. No further extension will be run on the strength of a near-miss. Twelve is
   the stopping point regardless of outcome, and this sentence is what makes
   that a commitment rather than an intention.

## What this cannot fix

At n = 12 the smallest reachable two-sided sign-test p is 2/2^12 = 0.00049,
so significance is reachable in principle. It is not reachable for the
*primary* hypothesis under the declared Holm family, which is a separate
pre-existing defect already recorded in the seed-power analysis. This
extension does not repair that, and no claim here should be read as doing so.

## Isolation

Validation only. The sealed final test is not opened, and no failure from
validation enters relearning.
