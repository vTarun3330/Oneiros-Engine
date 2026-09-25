# Sampling-budget study — optional protocol draft

**Status: optional secondary draft. Not run.** This is an exploratory,
train-derived study that supports no generalisation claim. Structured content is
in `results/v4_3_next_direction_design.json`, under `sampling_budget_protocol`.

## Why

In the closed tool-assisted pilot, the unmatched secondary comparison B − A came
to Kill@8 +6.44 pp, with a 90% interval of [+2.46, +10.59]. Arm B used 16
generated sequences and text-only de-duplicating selection; arm A was the
canonical 8-sample protocol. That was observed post hoc, on a panel whose
outcomes are now known. This study asks the question cleanly and predeclared:

> Does generating sixteen candidates and keeping eight by text-only selection
> improve Kill@8 over the first eight, at twice the generation cost?

## Panel

**Recommended:** the 613-record retention panel (100 pilot-development
lineages), with a **fresh seed family (43)**. Its canonical seed-42 Kill@8 for
the frozen control is already known (0.633), and this is disclosed.

**Rejected:** the 264-record tool-assisted panel. Its B − A result has already
been observed, so any study there would be post hoc.

## Arms (paired by construction)

| arm | definition |
|---|---|
| S8 | the first eight of sixteen samples drawn in one call |
| S16→8 | all sixteen from the same call, then the frozen text-only policy: parser/policy-valid, first-occurrence non-duplicates in generation order, first eight |

**Held constant:**
- model and revision, and the frozen control adapter `c1fe9e52…`;
- the prompt and seed family;
- the whole-output parser and the candidate policy;
- the evaluator and scoring.

**Forbidden:**
- execution-based reranking;
- any fixed-side information in selection;
- any generalisation claim.

## Cost accounting

S16→8 costs twice the generated sequences of S8. Generated tokens, GPU seconds
and wall time are reported for both arms. The result is framed as "more samples
plus selection", never as a free gain.

## Gate (frozen before generation)

| outcome | condition |
|---|---|
| pass | S16→8 − S8 Kill@8 ≥ +5 pp, with the 90% lineage-cluster bootstrap lower bound > 0, and reference validity within −3 pp |
| fail | the Kill@8 upper bound is below +5 pp |
| inconclusive | anything else; does not pass |

Every outcome remains exploratory.

## Estimate

- **GPU:** about 20 minutes (613 records × one 16-sample call).
- **CPU:** about 6 minutes of scoring.
- **Storage:** about 15 MB.
