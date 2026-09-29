# Choice A: receiver-aware replay pilot (CPU protocol, predeclared; NOT run)

Status: prepared 2026-09-29 after the Choice B v2 screen was frozen, analysed and
committed. Running it requires explicit approval. No mass acquisition.

## Why

The runtime argument-capture pilot (`results/sft_root_cause_phase4_argument_capture_receipt_v1.json`)
found usable literal fixed calls for 3/24 development targets in 1/8 repositories, all
python-humanize. The dominant loss was instance-method receivers (131 of 177 rejections):
the receiver is object state, so the call cannot be replayed from literals alone.

## Scope

- **Targets:** only the existing 24 natively qualified development targets (the A-prime
  retrospective), with the same manifest, worktrees and environments. Dependency receipts
  (Python and uv versions, install command, dependency-file hashes, sanitised freeze) are
  recorded per target.
- **Nothing is added:** no new repositories, candidates or acquisition.

## Method

1. **Recipe capture (fixed revision, difference-exposing official tests only).**
   - When a target instance method is called, look up how that receiver was constructed:
     the class's `__init__` (or an allowlisted classmethod constructor) is also profiled.
   - A **constructor recipe** is recorded only if the construction call's arguments are
     literal-only, under the same bounds as v1: depth 4, 64 items, 500 characters, exact
     types, and `literal_eval` round-trip.
2. **Receiver identity.** The recipe is linked to the target call by `id()` within the
   same test. It is rejected if the receiver was mutated between construction and the
   target call; this is detected by a second literal snapshot of any `__dict__` that is
   literal-serialisable, and is otherwise rejected as opaque state.
3. **Allowlist.** Recipes are allowed only for classes defined in the target package.
   No pickle, no `copy.deepcopy` of foreign objects, and no object deserialisation of any
   kind.
4. **Replay.**
   - One fresh isolated process per revision (buggy and fixed), with a per-call timeout
     of 5 s.
   - The receiver is rebuilt from the recipe independently in each process, then the
     target is called with independently reconstructed literal arguments.
5. **Oracle.** A call is kept only if:
   - the fixed revision returns a short round-trippable literal (at most 120 characters);
   - the buggy revision differs (a different value or an exception);
   - the result is deterministic across two fixed-revision replays.

Gold tests, fixed code and official asserted values stay **verifier-only**. They never
enter a model prompt or a training target. A kept oracle is `recipe + call + value`, and
the value comes from executing the fixed revision, not from the test's assertion text.

## Predeclared thresholds (decided before running)

- **Feasible,** continuing to an isolation-v6 revalidation of the usable targets, only if
  all three hold:
  - at least **8/24** targets have at least one usable fixed call;
  - at least **4/8** repositories have at least one usable target;
  - no single repository contributes more than **40%** of the usable targets.
- **Otherwise infeasible.** Stop fixed-input mining for Choice A and pivot the
  real-repository evidence to **native execution of complete model-generated tests**
  (bug-killing on the buggy versus fixed revision). Fixed-input value prediction is a
  mediator study, not a prerequisite for repository bug-killing evaluation.
- **Reporting:** target-level and repository-level yield, failure categories, Wilson
  intervals labelled descriptive (the targets are clustered and purposive), and timings.
- **Admission:** mass acquisition stays refused unless the pilot is feasible **and** the
  usable targets pass isolation-v6 revalidation **and** a separate approval is given.
