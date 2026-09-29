# Choice A receiver-aware replay pilot — protocol v2 (hardened before implementation)

This hardens `SFT_ROOT_CAUSE_CHOICE_A_RECEIVER_REPLAY_PROTOCOL.md` (v1, unchanged). It was
written before any receiver-aware capture was implemented or run.

**The feasibility gate is unchanged:**
- at least **8/24** targets have at least one usable fixed call;
- at least **4/8** repositories are represented among usable targets;
- no single repository contributes more than **40%** of usable targets.

## Nature of the study

A **retrospective feasibility study** on the same 24 natively qualified A-prime
development targets (`results/sft_root_cause_phase4_argument_capture_manifest_v1.json`).
It is not an efficacy or generalisation evaluation. Its artifacts are feasibility evidence
only and are never automatically admitted into SFT or relearning.

## Capture (fixed revision; difference-exposing official tests only)

1. **Strong identity.** While a wanted test runs, every candidate receiver created by an
   allowlisted construction is held in a per-test registry that keeps a **strong
   reference** to the object. Its `id()` therefore cannot be reused while the entry is
   live, and a lookup also checks `entry.obj is receiver`. The registry is cleared at the
   end of each test.
   - Limitation: the registry extends object lifetimes during a wanted test.
2. **Construction captured only when:**
   - the class is defined in the target's top-level package;
   - `type(type(obj)) is type` (no custom metaclass);
   - the class does not override `__new__`;
   - the outermost `__init__` frame for the object is the class's own `__init__` (after
     unwrapping), i.e. ordinary `cls(...)` construction;
   - every positional and keyword argument passes the v1 literal bounds (exact types,
     depth 4, 64 items, 500 characters, `literal_eval` round-trip).
3. **Classmethod factories.** Allowed only when:
   - the factory is a Python classmethod defined in the target package;
   - it returns an instance of **exactly** its class;
   - the returned object was not seen before in the test (so cached or singleton factories
     are rejected);
   - its arguments are literal.

   Everything else is rejected with a named reason: ambiguous `__new__`, metaclasses,
   descriptors, dependency injection, other factories, and `__init__` invoked on an
   uninitialised object.
4. **Snapshots.** Receiver state is snapshotted immediately after construction (the
   `__init__` or factory return) and immediately before the target call.
   - `__dict__` is supported only if every value passes literal serialisation.
   - With `__slots__`, every declared slot in the MRO is snapshotted under the same rules
     (an unset slot is recorded as unset). Anything else is rejected as opaque.
   - A receiver whose two snapshots differ is rejected as **mutated**.
5. **No values from tests.** No official assertion value is extracted. Records hold only
   the recipe, the call and results produced by executing the code.

## Replay

1. **Isolation:** a fresh process for each revision and each repetition.
   - Every argument and the recipe are reconstructed from literal text in each process,
     so no state or alias is shared.
   - The working directory is a new temporary directory.
   - Network is disabled: on WSL through a new network namespace (`unshare -n`),
     backed by an in-process socket guard.
2. **Reconstruction:** only through the allowlisted constructor (`cls(*args, **kwargs)`)
   or factory (`cls.factory(*args, **kwargs)`), never by calling `__init__` on a raw
   object. The reconstructed post-construction snapshot must equal the captured one.
3. **Budget:** a 5-second per-call timeout, with a process-level timeout as a backstop.
   Timeouts, crashes, environment failures and semantic outcomes are recorded separately.
4. **Determinism:** each call runs **twice on the fixed revision and twice on the buggy
   revision**. Both fixed outcomes must be identical, both buggy outcomes must be
   identical, and fixed must differ from buggy (a different value or an exception).
5. **Usable oracle:** the fixed result is a short round-trippable literal of at most 120
   characters.

## Boundaries

- Gold tests and fixed code are verifier-only.
- Fixed code, gold-test text, official expected values, captured values and recipes never
  enter a model prompt or training corpus during this pilot.
- No protected split is read. No repository or target is added or replaced.

## Outcome branches (frozen)

- **A — all three thresholds hold.**
  - Run isolation-v6 revalidation on the qualifying targets only, and re-verify every
    call on both revisions.
  - Remove any target failing lineage, near-duplicate, environment, determinism or
    provenance checks, then report the post-v6 yield.
  - Admit nothing to training. Prepare, but do not launch, a fresh evaluation design.
- **B — any threshold fails.**
  - Declare receiver-aware fixed-input mining infeasible for this cohort, and stop
    fixed-input mining and mass acquisition.
  - Prepare and preflight a frozen protocol for native buggy/fixed execution of complete
    model-generated tests. Do not launch its GPU generation.
