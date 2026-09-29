"""Phase 4 power SENSITIVITY (successor to results/sft_root_cause_phase4_power_analysis_v1.json,
which is preserved unchanged).  CPU only; deterministic; no cohort frozen.

* broad grid with the FAST method (cluster-sandwich normal interval);
* EXACT planned gate (paired cluster-bootstrap percentile interval) at selected sizes, on
  the same simulated datasets, with per-dataset agreement and discrepancies;
* cross-schema coupling (shared / partial / independent) x effect level (item / group);
* ranges across scenarios rather than single numbers.
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
from harness.power_sensitivity import (
    bootstrap_decisions, rules, sandwich_decisions, simulate_datasets, summarise,
)
from harness.power_simulation import empirical_groups

OUTPUT = "results/sft_root_cause_phase4_power_sensitivity_v1.json"
PREDECESSOR = "results/sft_root_cause_phase4_power_analysis_v1.json"
V3_3A = "results/sft_root_cause_phase3a_result_receipt_v3.json"
GRID = (100, 150, 200, 250, 300, 352, 400, 500, 700, 1000, 1500, 2000, 3000)
EFFECTS = (3, 5, 7, 10)
LEVELS = ("item", "group")
COUPLINGS = {"shared": 1.0, "partial": 0.5, "independent": 0.0}
FAST_SIMS = 1000
EXACT = {"groups": (150, 352), "effects": (5, 7, 10), "levels": LEVELS,
         "couplings": ("shared", "independent"), "sims": 200, "bootstrap_resamples": 2000}
BASE_SEED = 20260929


def seed(*parts) -> int:
    return int(hashlib.sha256(":".join(map(str, (BASE_SEED, *parts))).encode()).hexdigest()[:8],
               16)


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

    fast = []
    for level in LEVELS:
        for cname, coupling in COUPLINGS.items():
            for delta in EFFECTS:
                for g in GRID:
                    cs, ts, mask = simulate_datasets(c, t, g, delta, FAST_SIMS,
                                                     seed("fast", level, cname, delta, g),
                                                     level, coupling)
                    dec = sandwich_decisions(cs, ts, mask)
                    fast.append({"method": "fast_sandwich_normal", "level": level,
                                 "coupling": cname, "delta_points": delta, "groups": g,
                                 "sims": FAST_SIMS,
                                 "mean_effect_points": [round(float(x), 3)
                                                        for x in dec["point"].mean(axis=0)],
                                 **summarise(rules(dec))})
            print(f"fast grid done: {level}/{cname}", flush=True)

    exact = []
    for level in EXACT["levels"]:
        for cname in EXACT["couplings"]:
            for delta in EXACT["effects"]:
                for g in EXACT["groups"]:
                    cs, ts, mask = simulate_datasets(c, t, g, delta, EXACT["sims"],
                                                     seed("exact", level, cname, delta, g),
                                                     level, COUPLINGS[cname])
                    fast_hits = rules(sandwich_decisions(cs, ts, mask))
                    exact_hits = rules(bootstrap_decisions(
                        cs, ts, mask, EXACT["bootstrap_resamples"],
                        seed("boot", level, cname, delta, g)))
                    exact.append({
                        "level": level, "coupling": cname, "delta_points": delta, "groups": g,
                        "sims": EXACT["sims"],
                        "exact_bootstrap_gate": summarise(exact_hits),
                        "fast_on_same_datasets": summarise(fast_hits),
                        "per_dataset_agreement": {
                            rule: round(float((fast_hits[rule] == exact_hits[rule]).mean()), 4)
                            for rule in ("positive_both", "confirmatory_5")},
                        "probability_discrepancy_exact_minus_fast": {
                            rule: round(float(exact_hits[rule].mean() - fast_hits[rule].mean()),
                                        4) for rule in ("positive_both", "confirmatory_5")}})
                    print(f"exact {level}/{cname} d={delta} G={g} "
                          f"exact={exact[-1]['exact_bootstrap_gate']['positive_both']['probability']}"
                          f" fast={exact[-1]['fast_on_same_datasets']['positive_both']['probability']}",
                          flush=True)

    def ranges(rule: str) -> dict:
        out = {}
        for delta in EFFECTS:
            for g in GRID:
                probs = [r[rule]["probability"] for r in fast
                         if r["delta_points"] == delta and r["groups"] == g]
                out[f"delta={delta},G={g}"] = [min(probs), max(probs)]
        return out

    def required(rule: str, target=0.8) -> dict:
        out = {}
        for delta in EFFECTS:
            per_scenario = []
            for level in LEVELS:
                for cname in COUPLINGS:
                    hits = sorted(r["groups"] for r in fast if r["level"] == level
                                  and r["coupling"] == cname and r["delta_points"] == delta
                                  and r[rule]["probability"] >= target)
                    per_scenario.append(hits[0] if hits else None)
            reached = [x for x in per_scenario if x is not None]
            out[str(delta)] = {
                "range_across_scenarios": ([min(reached), max(reached)] if reached else None),
                "scenarios_not_reaching_80pct_within_3000_groups":
                    sum(x is None for x in per_scenario)}
        return out

    discrepancies = [e for e in exact
                     if abs(e["probability_discrepancy_exact_minus_fast"]["positive_both"]) > 0.05
                     or abs(e["probability_discrepancy_exact_minus_fast"]["confirmatory_5"]) > 0.05]
    receipt = {
        "schema_version": "oneiros_sft_root_cause_phase4_power_sensitivity_v1",
        "status": "PLANNING EVIDENCE, NOT A POWER GUARANTEE",
        "successor_of": {PREDECESSOR: sha_file(PREDECESSOR)},
        "predecessor_modified": False,
        "model_calls": 0, "gpu_used": False, "training": False, "cohort_frozen": False,
        "restricted_splits_accessed": "none (train-side Phase 3A retained outcomes only)",
        "inputs": {V3_3A: sha_file(V3_3A),
                   "harness/power_sensitivity.py": sha_file("harness/power_sensitivity.py"),
                   "scripts/phase4_power_sensitivity.py":
                       sha_file("scripts/phase4_power_sensitivity.py"),
                   "canonical_scored_rows_sha256": identity["canonical_scored_rows_sha256"]},
        "assumptions": {
            "outcome_source": "synthetic train-side MBPP simple/moderate Phase 3A outcomes "
                              f"({len(groups)} semantic groups, 2 items each), NOT "
                              "repository-native outcomes",
            "control": "base 1.5B outcomes", "treatment_template": "arm A 431 outcomes on the "
            "same items, shifted to each target effect by flipping outcomes",
            "effect": "equal target effect in both schemas",
            "equal_effect_limitation": ("because the true effect is equal in both schemas, "
                                        "simulated sign flips are near zero; Phase 3A OBSERVED "
                                        "sign flips between schemas (schema-dependent effects), "
                                        "which this simulation does not model and which would "
                                        "lower power further"),
            "effect_levels": {"item": "independent draws per item",
                              "group": "one draw per semantic group"},
            "cross_schema_coupling": {k: f"probability {v} that a flip draw is shared between "
                                         "schemas" for k, v in COUPLINGS.items()},
            "fast_method": "cluster-sandwich normal 95% interval",
            "exact_method": (f"paired cluster-bootstrap percentile 95% interval, "
                             f"{EXACT['bootstrap_resamples']} resamples (the planned gate uses "
                             "10,000; percentile resolution differs slightly)"),
            "rules": {"positive_both": "lower 95% bound > 0 under both schemas, no sign flip",
                      "confirmatory_5": "lower 95% bound > +5 under both schemas, no sign flip",
                      "sign_flip": "point estimates of opposite sign across the two schemas"},
            "not_modelled": ["training-seed variance (no permitted evidence; NOT estimated)",
                             "answer-rate gate", "validity guardrail", "diversity guardrail",
                             "more than two fixed inputs per function"],
            "interpretation": ("group counts are planning ranges for this synthetic "
                               "distribution, not guaranteed repository sample sizes; real power "
                               "is expected to be lower")},
        "fast_grid": fast,
        "exact_gate_checks": exact,
        "exact_vs_fast_discrepancies_over_5_points": discrepancies,
        "ranges_across_scenarios": {"positive_both": ranges("positive_both"),
                                    "confirmatory_5": ranges("confirmatory_5"),
                                    "sign_flip": ranges("sign_flip")},
        "required_groups_for_80pct_power_ranges": {"positive_both": required("positive_both"),
                                                   "confirmatory_5": required("confirmatory_5")},
    }
    publish_file_atomically(ROOT / OUTPUT, (json.dumps(receipt, indent=1, sort_keys=True)
                                            + "\n").encode("utf-8"))
    print(json.dumps({"required": receipt["required_groups_for_80pct_power_ranges"],
                      "discrepancies": len(discrepancies)}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
