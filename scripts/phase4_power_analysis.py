"""Phase D: simulation-based power for the Phase 4 mediator gate, and the A/B choice.

CPU only.  Reads only the permitted train-side Phase 3A retained outcomes (re-scored
exactly as the v3 receipt, whose canonical row hash must match) and the committed
Phase 4 census.  Freezes no cohort.  Output is deterministic (no timestamps or
durations), so a rerun is byte-identical.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness.atomic_publish import publish_file_atomically
from harness.fixed_input_rescoring import load_inputs, row_identity, score_function, sha_file
from harness.parallel_execution import map_jobs
from harness.power_simulation import SCHEMAS, empirical_groups, required_groups, simulate

OUTPUT = "results/sft_root_cause_phase4_power_analysis_v1.json"
V3_3A = "results/sft_root_cause_phase3a_result_receipt_v3.json"
CENSUS = "results/sft_root_cause_phase4_cohort_census.json"
GRID = (33, 50, 75, 100, 150, 200, 250, 300, 352, 400, 500, 700, 1000, 1500, 2000, 3000)
EFFECTS = (3, 5, 7, 10)
SCENARIOS = ("item", "group")
SIMS = 2000
BASE_SEED = 20260928
RULES = ("positive_both", "confirmatory_5", "old_rule")


def cell_seed(groups: int, delta: float, scenario: str) -> int:
    digest = hashlib.sha256(f"{BASE_SEED}:{groups}:{delta}:{scenario}".encode()).hexdigest()
    return int(digest[:8], 16)


def sandwich_se(c: np.ndarray, t: np.ndarray) -> list[float]:
    mask = ~np.isnan(c)
    d = np.where(mask, t - c, 0.0)
    n_g, d_g = mask.sum(axis=1), d.sum(axis=1)
    N = n_g.sum(axis=0)
    D = d_g.sum(axis=0) / N
    G = c.shape[0]
    se = np.sqrt(G / (G - 1) * ((d_g - n_g * D) ** 2).sum(axis=0)) / N
    return [round(float(x) * 100, 3) for x in se]


def main() -> int:
    inputs = load_inputs(("base", "arm_a_431"))
    jobs = [{"function": f, "record": inputs["records"][f["record_id"]],
             "generations": inputs["generations"]} for f in inputs["cohort"]["functions"]]
    rows = [r for chunk in map_jobs(score_function, jobs) for r in chunk]
    v3 = json.loads((ROOT / V3_3A).read_text(encoding="utf-8"))
    identity = row_identity(rows)
    if identity["canonical_scored_rows_sha256"] != v3["rescoring"]["canonical_scored_rows_sha256"]:
        raise SystemExit("REFUSED: re-scored rows differ from the v3 receipt")
    c, t, groups = empirical_groups(rows, "base", "arm_a_431")

    # Calibration: sandwich SE on the empirical data vs the v3 paired cluster bootstrap.
    boot_se = [round((v3["strata"]["all"][part]["ci95_points"][1]
                      - v3["strata"]["all"][part]["ci95_points"][0]) / 3.92, 3)
               for part in ("prefill", "answer")]
    calibration = {"sandwich_se_points": dict(zip(SCHEMAS, sandwich_se(c, t))),
                   "bootstrap_se_points_from_v3_ci": dict(zip(SCHEMAS, boot_se)),
                   "note": "the simulation's interval uses the cluster-robust sandwich SE; "
                           "these two columns show it matches the v3 cluster bootstrap"}

    results = []
    for scenario in SCENARIOS:
        for delta in EFFECTS:
            for g in GRID:
                results.append(simulate(c, t, g, delta, SIMS, cell_seed(g, delta, scenario),
                                        scenario))
                print(f"{scenario:5s} delta={delta:2d} G={g:5d} "
                      + " ".join(f"{r}={results[-1][r]['power']:.3f}" for r in RULES),
                      flush=True)
    required = {scenario: {rule: {str(delta): required_groups(
        [r for r in results if r["scenario"] == scenario and r["delta_points"] == delta], rule)
        for delta in EFFECTS} for rule in RULES} for scenario in SCENARIOS}

    census = json.loads((ROOT / CENSUS).read_text(encoding="utf-8"))
    pools = census["pools"]

    def at(g, scenario="item"):
        return {str(d): {rule: next(r[rule]["power"] for r in results if r["groups"] == g
                                    and r["delta_points"] == d and r["scenario"] == scenario)
                         for rule in ("positive_both", "confirmatory_5")} for d in EFFECTS}

    receipt = {
        "schema_version": "oneiros_sft_root_cause_phase4_power_v1",
        "model_calls": 0, "gpu_used": False, "training": False, "cohort_frozen": False,
        "restricted_splits_accessed": "none (train-side Phase 3A retained outcomes and the "
                                      "committed Phase 4 census only)",
        "inputs": {V3_3A: sha_file(V3_3A), CENSUS: sha_file(CENSUS),
                   "harness/power_simulation.py": sha_file("harness/power_simulation.py"),
                   "scripts/phase4_power_analysis.py": sha_file("scripts/phase4_power_analysis.py"),
                   "canonical_scored_rows_sha256": identity["canonical_scored_rows_sha256"]},
        "empirical_basis": {
            "groups": len(groups), "items_per_group": int(c.shape[1]), "schemas": list(SCHEMAS),
            "control": "base 1.5B outcomes", "treatment_template": "arm A 431 outcomes on the "
            "same items (a realistic second model), shifted to each target effect",
            "caveat": ("the gate cohort would be drawn from different train groups; Phase 3A's "
                       "MBPP simple/moderate outcomes are the closest permitted evidence")},
        "method": {
            "simulations_per_cell": SIMS, "grid_groups": list(GRID), "effects_points": EFFECTS,
            "scenarios": {"item": "effect injected by independent item flips (low intragroup "
                                  "correlation of the effect)",
                          "group": "effect injected by whole-group flips (high intragroup "
                                   "correlation of the effect)"},
            "interval": "two-sided 95%: difference +/- 1.96 x cluster-robust SE over groups",
            "alpha_and_power": "alpha = 0.05 two-sided (the protocol's 95% intervals); target "
                               "power 0.80",
            "seeds": f"per cell sha256({BASE_SEED}:G:delta:scenario)",
            "not_modelled": ["training-seed variance: no permitted evidence (every retained "
                             "multi-seed run was evaluated on restricted splits), so it is "
                             "NOT estimated and real power is lower than shown",
                             "the answer-rate gate and validity/diversity guardrails",
                             "more than two fixed inputs per function"]},
        "calibration": calibration,
        "rules": {
            "positive_both": "lower 95% bound > 0 under both schemas, no sign flip",
            "confirmatory_5": ("CORRECTED confirmatory rule: lower 95% bound > +5 under both "
                               "schemas, no sign flip (plus answer-rate and guardrail gates, "
                               "not simulated)"),
            "old_rule": ("SUPERSEDED V1-draft gate (point >= 5 and lower bound > 0): shows a "
                         "positive effect with a point estimate >= 5, not an effect >= 5; "
                         "reported for comparison only")},
        "results": results,
        "required_groups_for_80pct_power": required,
        "required_groups_note": ("None means not reached within the grid (up to 3000 groups). "
                                 "A true effect of exactly 5 points can essentially never clear a "
                                 "lower bound of +5; the confirmatory five-point rule needs true "
                                 "effects well above 5"),
        "pools": {
            "strict_untouched": {
                "groups": pools["strict"]["groups"],
                "feasible": pools["strict"]["groups_with_a_feasible_fixed_input_function"],
                "label": ("not used by arm A training and in neither Phase 3 cohort: genuinely "
                          "untouched under the current definition; underpowered"),
                "power_at_feasible_count": at(pools["strict"]["groups_with_a_feasible_fixed_input_function"])},
            "arm_a_exposed_remainder": {
                "groups": pools["arm_a_exposed_remainder"]["groups"],
                "feasible": pools["arm_a_exposed_remainder"][
                    "groups_with_a_feasible_fixed_input_function"],
                "label": ("USED in arm A training, not in Phase 3; can be held out from new C/T "
                          "training; an internal within-corpus development pool only; NOT "
                          "untouched by the research process; NOT repository-disjoint; never "
                          "independent final confirmation"),
                "power_at_150": at(150), "power_at_all_352": at(352)},
            "phase3_unexposed_cohort": "development evidence only (inspected; 120 groups)"},
        "gate_definitions": {
            "confirmatory_five_point_pass": [
                "effect directionally correct under both schemas", "no schema sign flip",
                "answer-rate gate passes",
                "declared lower 95% bound > +5 under both schemas",
                "validity and diversity guardrails pass"],
            "exploratory_pilot_outcomes": {
                "promising": "lower bound > 0 under both schemas, no flip, guardrails pass",
                "harm": "upper bound < 0 under either schema, or a guardrail fails",
                "inconclusive_power": "anything else; an underpowered non-pass never "
                                      "automatically rejects the intervention"}},
        "old_150_150_rest_proposal": "NOT frozen: its 150-group gate is underpowered for a "
                                     "five-point target",
        "repository_disjoint_phase6": ("UNRESOLVED BLOCKER: no repository-disjoint cohort exists; "
                                       "an internal split of arm A's source distribution cannot "
                                       "satisfy it; terminal rule A and real-world "
                                       "generalisation claims remain blocked"),
    }
    publish_file_atomically(ROOT / OUTPUT, (json.dumps(receipt, indent=1, sort_keys=True) + "\n")
                            .encode("utf-8"))
    print(json.dumps({"calibration": calibration, "required": required}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
