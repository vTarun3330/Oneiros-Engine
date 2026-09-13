# O1 sidecar mixing ratio: fixed at 16.00%, decided before emission

This receipt is committed **before** the sidecar is emitted, so the ratio is a
protocol decision rather than a number chosen after seeing what the data
happened to allow.

## The decision

**1,305 unique O1 rows**, against Arm A's 6,852 effective baseline examples.

```
1305 / (6852 + 1305) = 1305 / 8157 = 15.998529%
```

Eleven of the 1,316 verified O1 positives are **deliberately left unused**. No
row is duplicated, and the 0.70 MBPP source cap is not relaxed.

## Why 16.00% and not the 16.11% maximum

The whole supply is 1,316 rows, which would give
`1316 / (6852 + 1316) = 16.1117%`. That number is not a chosen ratio - it is
whatever was left after the O1 build's caps finished binding. Reporting it as
the mixing ratio would mean the experiment's headline parameter was set by
supply, and a later rebuild of O1 yielding a different count would silently
change the design of the comparison.

16.00% is a round number fixed in advance, reachable from this supply and from
any larger one. If O1 is rebuilt and yields 1,400 or 2,000 positives, the arm B
protocol is unchanged and the comparison stays comparable. The eleven unused
rows are the price of that, and they are cheap.

## Why 20% was not achievable

The original target was 20% of Arm B's effective examples. Arm A's baseline is
**6,852 effective training examples**, not the 2,400 selected pairs - each pair
yields up to three verified completions, and repository examples are
project-balanced on top. A true 20% share therefore needs

```
0.20 * 6852 / 0.80 = 1713 rows
```

and O1 has 1,316. The gap could only be closed two ways, both refused:

- **Relaxing the 0.70 MBPP cap** to admit more positives. The cap exists to stop
  MBPP dominating; raising it to grow a dataset is precisely the failure it was
  put there to prevent, and it is explicitly out of scope.
- **Shrinking Arm A's baseline** so 1,316 reaches 20%. That would change the
  frozen corpus, and Arm A would no longer be the baseline it is supposed to be.
  The arms must differ by the sidecar and nothing else.

So the ratio was lowered rather than the data bent to reach it.

## Token budget, settled before emission

Arm A's function-mode generation completion budget is **128 tokens**. Measured
across all 1,316 positives with the pinned tokenizer
(`2e1fd397ee46e1388853d2af2c993145b0f1098a`): median 18, p95 44, max 141.
**Exactly one row exceeds 128.**

That one row is dropped **before** the proportional subsample rather than during
the Arm B preflight. Dropping it afterwards would deliver 1,304 rows if the
subsample happened to include it and 1,305 if it did not - a ratio that depends
on which rows were drawn. Filtering first makes 1,305 exact and deterministic.

No completion is ever truncated to fit. A truncated assertion is not the
verified assertion, and its verification would no longer describe it.

## What this ratio does not license

The sidecar is a bounded auxiliary component. 16.00% is a mixing ratio, not
evidence: it says how much O1 is present in Arm B, and nothing about whether O1
helps. That question is what the controlled comparison exists to answer, and it
has not been run.
