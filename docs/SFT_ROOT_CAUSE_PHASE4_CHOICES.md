# Phase 4: Choice A or Choice B (decision required; nothing frozen or launched)

Power figures come from `results/sft_root_cause_phase4_power_analysis_v1.json`, a
simulation from the train-side Phase 3A outcomes with semantic-group resampling and both
schemas, 2,000 simulations per cell. The simulated SE matches the v3 cluster bootstrap
(2.075 vs 2.073 points under prefill). **Training-seed variance is not modelled.** No
permitted evidence exists for it, so real power is lower than shown.

## What the five-point target actually requires (80% power, both schemas)

| True effect (points) | Groups to show a **positive** effect (lower bound > 0) | Groups to **confirm ≥ 5** (lower bound > +5) |
|---|---|---|
| 3 | ≈ 700 | impossible |
| 5 | ≈ 300 (item-level effect) to 352 (group-level) | impossible: a true 5-point effect clears +5 with probability ≈ 0 |
| 7 | ≈ 150–200 | ≈ 2,000–3,000 |
| 10 | ≈ 100–150 | ≈ 352–500 |

The superseded V1 gate (point estimate ≥ 5 and lower bound > 0) never establishes an
effect of at least 5. For a true 5-point effect it "passes" only about 25–30% of the time
at any size.

**Current pools:**

- **Strict untouched pool:** 33 feasible groups. Even a true 10-point effect is shown
  positive only 13% of the time. Unusable as a gate.
- **Arm-A-exposed remainder:** 352 feasible groups, *all* of them used as the gate.
  - Power to show a 5-point effect is positive: 0.98 (item-level) or 0.91 (group-level).
  - Power to confirm ≥ 5: ≈ 0 unless the true effect is about 10 (0.86 / 0.59).
  - Using all 352 leaves none of the remainder for training or for a second
    development panel.
  - This pool is internal, used by arm A, and not repository-disjoint.

## Choice A: recommended path to final evidence (acquire before training)

**Target.** At least 350 new, unique, lineage-disjoint functions for the mediator gate.
That is enough to show a 5-point effect is positive and to confirm ≥ 5 if the true effect
is about 10. A separate repository-disjoint Phase 6 cohort of similar size is needed too.

**Steps**

1. **Mining:** continue A′ repository-native mining over new repositories under the frozen
   policy (licence screen, same-organisation exclusion, ≤ 12 targets per repository, at
   least 34 repositories).
2. **Native qualification:**
   - install each environment (uv, Python 3.10/3.13, WSL);
   - run the official tests on both revisions;
   - retain only functions whose buggy/fixed behaviour differs by native execution;
   - keep only unique executable examples, never duplicating MBPP/HumanEval rows.
3. **Freezing:** freeze train, gate and confirmation lineages *by repository* before any
   model run. The gate and Phase 6 must be repository-disjoint from each other and from
   training.
4. **Integrity:** no validation, test, sealed-final or reserved data is touched, with the
   protected-location audit and hashed receipts as before.

**Expected work** (from the A′ fresh pilot: 5 of 120 admitted, 4.17%, Wilson [1.79%,
9.38%], before native-qualification losses):

- **Candidates:** 350 admitted targets need ≈ 8,400 mined candidates (range ≈ 3,700–
  19,600), and ≈ 2× that for gate plus Phase 6.
- **API and time:** ≈ 0.64 REST calls per candidate, so ≈ 5,400 calls per 8,400
  candidates.
  - Unauthenticated (60/hour): ≈ 90 hours of rate limit.
  - With an existing `GITHUB_TOKEN` (never printed or stored): ≈ 1–2 hours of API time,
    plus git fetching.
- **Native qualification:** CPU-bound environment builds and test runs per repository.
  Never measured, because native qualification has never run. A bounded 20–30 target
  rehearsal would measure it first.
- **GPU:** none for acquisition. The gate evaluation later needs two arms × about 350
  groups × two inputs × two schemas of greedy generation (minutes), plus production
  Kill@8 generation.

**Stop conditions**

- the admitted yield in the first 1,000 candidates is below 2%;
- the qualified-to-admitted ratio is below 50% in the rehearsal;
- any protected-data access event;
- a repository cap or diversity gate fails.

**Blocker resolved by A:** it builds a genuine repository-disjoint Phase 6 cohort.

## Choice B: bounded exploratory internal pilot

**Scope**

- Use a permitted subset of the 352 feasible arm-A-exposed-remainder groups as the gate,
  and train C and T on disjoint arm A groups.
- Label everything **internal and exploratory**:
  - the old arm A is never compared on this panel;
  - the panel is never called untouched;
  - no cross-dataset or repository generalisation is claimed;
  - it is never treated as Phase 6.

**Outcomes:** exactly three, as defined in the power receipt:

- `promising`: lower bound > 0 under both schemas, no flip, guardrails pass;
- `harm`: an upper bound < 0, or a guardrail fails;
- `inconclusive_power`.

An inconclusive result never rejects the intervention.

**Power, with a 200-group gate** (the other ≈ 150 feasible groups are kept for training
and development):

- a true 7-point effect shows positive 0.97 (item-level) or 0.87 (group-level);
- a true 5-point effect 0.55 or 0.37;
- confirmation of ≥ 5 is effectively impossible (0.00 at 7 points; 0.33 or 0.18 even at
  10 points).

**Before any promotion or paper claim:** later independent confirmation (Choice A's
cohorts) is required.

## Recommendation

**Choice A** is the only path that can reach terminal rule A, because it resolves the
repository-disjoint Phase 6 blocker.

If you want an early signal first, Choice B is a reasonable, cheap and clearly labelled
precursor. Its only possible positive output is "promising", and it cannot replace A.

Either way, the next GPU step is a bounded **timing run** (arm A's training time was
never recorded). It needs separate authorisation.

## Remaining blockers (both choices)

- There is no repository-disjoint Phase 6 cohort. An internal split of arm A's source
  distribution cannot satisfy it.
- Training-seed variance is unknown. One seed is a feasibility pilot only.
- H1 cannot be tested without A1/A2 public-context variants, which do not exist for the
  synthetic MBPP records.
