# Sampling-budget study — optional protocol draft (revised)

**Status: optional secondary draft. Not run.** This is an exploratory,
train-derived study that supports no generalisation claim. Structured content is
in `results/v4_3_next_direction_design.json`, under `sampling_budget_protocol`.

## Question

> Does generating sixteen candidates and keeping eight by frozen text-only
> selection improve Kill@8 over the nested first eight, at twice the generation
> cost?

The closed tool-assisted pilot observed B − A = +6.44 pp Kill@8 post hoc,
unmatched, on a panel whose outcomes are now known. That result motivates this
question; it is not evidence for the answer.

## Panel

**Recommended:** the 613-record retention panel (100 pilot-development lineages)
with a **fresh seed family (43)**. The control's canonical seed-42 Kill@8 on it
is already known (0.633), and this is disclosed.

**Rejected:** the 264-record tool-assisted panel, because its 16-versus-8 result
has already been observed.

## Frozen generation and selection

| item | frozen rule |
|---|---|
| generation | one `generate` call per record with `num_return_sequences=16`. The seed `derived_seed(43, record_id, "s16")` is applied immediately before the call. The unchanged successor settings, frozen control adapter `c1fe9e52…`, prompt, parser and evaluator are used. |
| candidate ordering | the 16 sequences in the order the call returns them (index 0–15) |
| S8 | sequences 0–7 exactly, in order. The analysis checks that the raw outputs and hashes are byte-identical to sequences 0–7 of the same record's S16 set, which proves the arms are nested. |
| validity | a candidate is valid if the frozen whole-output parser accepts it (parse plus candidate policy). **Nothing is executed for selection.** |
| duplicate | whitespace-collapsed normalised code (`harness.execution_feedback.normalised_code`) equal to that of an *earlier valid* candidate |
| S16→8 selection | walk indices 0–15 and keep valid first occurrences, in order, until eight are kept |
| fewer than eight kept | fill the remaining slots with the not-kept candidates (invalid or duplicate) in ascending index order. There are always exactly eight slots; the fill is deterministic. |
| implementation | `harness.tool_assisted_generation.select_final`, with `slot=index` and `round=1`. Its policy is exactly this rule, and its tests cover adversarial fields. |
| bound sources | `engine/generator.py`, `harness/generation_adapter.py`, `harness/tool_assisted_generation.py`, `harness/execution_feedback.py`, `metrics/research_evaluation.py`, `harness/sealed_final_evaluator.py` |

**Forbidden:**
- execution-based reranking;
- any fixed-side information;
- any generalisation claim.

**Not the canonical protocol.** S8 is the first eight of a 16-sample call, not
the canonical batch-of-2, 8-sample protocol. S8 and S16→8 are compared only with
each other.

## Cost accounting

S16→8 costs twice the generated sequences of S8. Generated tokens, GPU seconds
and wall time are reported for both arms, and the result is framed as "more
samples plus selection".

## Gate (frozen)

| outcome | condition |
|---|---|
| pass | S16→8 − S8 Kill@8 ≥ +5 pp, with the 90% lineage-cluster bootstrap lower bound > 0, and reference validity within −3 pp |
| fail | the Kill@8 upper bound is below +5 pp |
| inconclusive | anything else; does not pass |

Every outcome remains exploratory.

## Estimate

- **GPU:** about 20 minutes.
- **CPU:** about 6 minutes of scoring.
- **Storage:** about 15 MB.
