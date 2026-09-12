# Evaluation protocol freeze

**Declared:** 2026-09-12, before any further training run.
**Enforced by:** `harness/evaluation_protocol.py`, `tests/test_evaluation_protocol.py`.

## Why this exists

Section 6.2 of the project report established that the legacy parser and the
successor protocol are not two measurements of one quantity. On the *same
retained generations*:

| interpretation | kill@8 | reference validity |
|---|---|---|
| legacy — first assertion only | 0.7159 | 0.4880 |
| successor — whole output, conjunctive | 0.5480 | 0.2749 |

The legacy parser scanned each output for the first line beginning `assert `
and discarded the rest, so it was acting as a repair step. The seventeen-point
difference between the two rows measures **the parser**, not the model.

The trap is that legacy artifacts predate the `candidate_parse_mode` field
entirely. They record *nothing*, and an absent field reads as "no reason for
concern" rather than as "this is the legacy protocol". Subtracting a successor
number from a legacy one therefore produces a plausible-looking delta that is
pure artefact.

## What is frozen

**The protocol of record for the committed baseline is `first_assertion`
(legacy).** Every result in the report's arm table, the twelve relearning
seeds, and the Atheris comparison was computed under it. Those numbers are not
being restated, and nothing about them changes.

**The protocol of record for new work is `whole_output` (successor).** It
scores every assertion the model emitted rather than the one the parser
rescued, so it describes what the model actually produces.

Both claims stand. The only thing forbidden is mixing them in one comparison.

## What is enforced

1. `protocol_of(payload)` resolves an **absent** `candidate_parse_mode` to
   `first_assertion`. The absence is evidence, not a gap.
2. An **unrecognised** parse mode raises rather than defaulting to legacy, so a
   future third protocol cannot silently inherit this one's identity.
3. `assert_comparable(...)` refuses any comparison spanning protocols and names
   which side is which.
4. The guard is wired into `scripts/slice_kill_rate.py` (which computes
   per-slice deltas) and `scripts/build_results_table.py` (which now records
   `evaluation_protocol` on every arm). A test asserts the wiring, not just the
   availability, because a guard nobody calls is a comment.
5. A test asserts the committed baseline artifacts really are legacy. If that
   test fails, the arm table is mixing protocols.

## What this does not do

It does not re-measure the existing arms under the successor. Doing so is GPU
work — twelve seeds plus seven arms — and it is not required in order to make
future numbers interpretable, which is the purpose of this freeze. Until that
re-measurement happens, a successor-protocol arm cannot be compared to the
committed table, and `assert_comparable` will say so rather than let it
through.
