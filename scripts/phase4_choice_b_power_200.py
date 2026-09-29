"""Exact paired cluster-bootstrap power at the planned 200-group Choice B gate.

The 200-group figures in power v2 came from the fast cluster-sandwich method; the exact
bootstrap gate was checked only at 150 and 352 groups, where one cell differed by 7.5
points. This runs the exact gate itself at 200 groups (every level x coupling x effect),
using the corrected v2 simulation and the same seeds scheme. PLANNING EVIDENCE ONLY:
synthetic train-side Phase 3A outcomes; training-seed and repository-native variance are
unmodelled; the answer-rate, validity and diversity guardrails are outside the
calculation. Frozen before any Choice B outcome exists. CPU only; deterministic.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness.atomic_publish import publish_file_atomically
from harness.fixed_input_rescoring import load_inputs, row_identity, score_function, sha_file
from harness.parallel_execution import map_jobs
from harness.power_sensitivity_v2 import (
    bootstrap_decisions, rules, sandwich_decisions, simulate_datasets, summarise,
)
from harness.power_simulation import empirical_groups

OUTPUT = "results/sft_root_cause_phase4_choice_b_power_200_v1.json"
V3_3A = "results/sft_root_cause_phase3a_result_receipt_v3.json"
POWER_V2 = "results/sft_root_cause_phase4_power_sensitivity_v2.json"
GROUPS = 200
EFFECTS = (3, 5, 7, 10)
LEVELS = ("item", "group")
COUPLINGS = {"shared": 1.0, "partial": 0.5, "independent": 0.0}
SIMS = 400
RESAMPLES = 2000
BASE_SEED = 20260930
RULES = ("positive_both", "confirmatory_5", "sign_flip")


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
    c, t, _ = empirical_groups(rows, "base", "arm_a_431")
    fast_v2 = {(r["level"], r["coupling"], r["delta_points"]): r for r in json.loads(
        (ROOT / POWER_V2).read_text(encoding="utf-8"))["fast_grid"] if r["groups"] == GROUPS}
    cells = []
    for level in LEVELS:
        for cname, coupling in COUPLINGS.items():
            for delta in EFFECTS:
                cs, ts, mask = simulate_datasets(c, t, GROUPS, delta, SIMS,
                                                 seed("sim", level, cname, delta), level, coupling)
                exact = rules(bootstrap_decisions(cs, ts, mask, RESAMPLES,
                                                  seed("boot", level, cname, delta)))
                fast = rules(sandwich_decisions(cs, ts, mask))
                old = fast_v2[(level, cname, delta)]
                cells.append({
                    "level": level, "coupling": cname, "delta_points": delta, "groups": GROUPS,
                    "sims": SIMS, "bootstrap_resamples": RESAMPLES,
                    "exact_bootstrap_gate": summarise(exact),
                    "fast_on_same_datasets": summarise(fast),
                    "power_v2_fast_grid_probability": {r: old[r]["probability"] for r in RULES},
                    "exact_minus_power_v2_fast": {
                        r: round(float(exact[r].mean()) - old[r]["probability"], 4)
                        for r in RULES}})
            print(f"done {level}/{cname}", flush=True)

    def band(rule, delta):
        probs = [c["exact_bootstrap_gate"][rule]["probability"] for c in cells
                 if c["delta_points"] == delta]
        return [min(probs), max(probs)]
    worst = max(abs(c["exact_minus_power_v2_fast"][r]) for c in cells for r in RULES[:2])
    receipt = {
        "schema_version": "oneiros_phase4_choice_b_power_200_v1",
        "status": "PLANNING EVIDENCE, NOT A POWER GUARANTEE",
        "frozen_before_any_choice_b_outcome": True,
        "inputs": {V3_3A: sha_file(V3_3A), POWER_V2: sha_file(POWER_V2),
                   "harness/power_sensitivity_v2.py": sha_file("harness/power_sensitivity_v2.py"),
                   "scripts/phase4_choice_b_power_200.py":
                       sha_file("scripts/phase4_choice_b_power_200.py")},
        "method": ("exact paired semantic-group cluster-bootstrap percentile 95% gate on "
                   f"simulated datasets ({SIMS} per cell, {RESAMPLES} resamples; the frozen "
                   "gate uses 10,000), corrected v2 coupling"),
        "cells": cells,
        "exact_positive_effect_probability_by_effect": {str(d): band("positive_both", d)
                                                        for d in EFFECTS},
        "exact_exceeds_five_probability_by_effect": {str(d): band("confirmatory_5", d)
                                                     for d in EFFECTS},
        "max_abs_difference_from_power_v2_fast_200": round(worst, 4),
        "reading": ("a five-point true effect has LOW useful detection power at 200 groups "
                    "(see the positive-effect band for 5 points); detection is not "
                    "impossible. The exceeds-five gate is not a gate of this screen."),
        "not_modelled": ["training-seed variance", "repository-native variance",
                         "answer-rate, validity and diversity guardrails",
                         "schema-dependent (unequal or opposite-signed) true effects"],
    }
    publish_file_atomically(ROOT / OUTPUT, (json.dumps(receipt, indent=1, sort_keys=True)
                                            + "\n").encode("utf-8"))
    print(json.dumps({k: receipt[k] for k in ("exact_positive_effect_probability_by_effect",
                                              "exact_exceeds_five_probability_by_effect",
                                              "max_abs_difference_from_power_v2_fast_200")},
                     indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
