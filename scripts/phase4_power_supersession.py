"""Supersession addendum: power sensitivity v1 -> v2 (partial-coupling defect).

v1 is preserved byte-for-byte as historical. This addendum states what was wrong, what
changed and what did not, reading every figure from the two receipts. Deterministic.
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

OUTPUT = "results/sft_root_cause_phase4_power_sensitivity_v1_supersession.json"
V1 = "results/sft_root_cause_phase4_power_sensitivity_v1.json"
V2 = "results/sft_root_cause_phase4_power_sensitivity_v2.json"


def sha(rel):
    return hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()


def main() -> int:
    v1 = json.loads((ROOT / V1).read_text(encoding="utf-8"))
    v2 = json.loads((ROOT / V2).read_text(encoding="utf-8"))
    v1_exact = [abs(e["probability_discrepancy_exact_minus_fast"][r]) for e in
                v1["exact_gate_checks"] for r in ("positive_both", "confirmatory_5")]
    receipt = {
        "schema_version": "oneiros_power_sensitivity_supersession_v1",
        "superseded": {"path": V1, "sha256": sha(V1), "modified": False,
                       "status": "HISTORICAL; superseded for planning by v2"},
        "successor": {"path": V2, "sha256": sha(V2)},
        "defect": ("v1 harness/power_sensitivity._draws drew an independent shared/not-shared "
                   "selector for EACH schema, so the probability that both schemas used the "
                   "common draw was coupling**2 (0.25 at the declared partial 0.5), contrary to "
                   "the receipt's description of coupling as the shared-draw probability"),
        "fix": v2["supersedes"]["reason"],
        "unchanged": {
            "shared_and_independent_cells": ("identical to v1 cell-for-cell (same seeds, same "
                                             "draw order; the selector is constant at 0 and 1)"),
            "required_groups_for_80pct_power_ranges": v2[
                "required_group_ranges_changed_from_v1"] == {"positive_both": {},
                                                             "confirmatory_5": {}}},
        "changed": {
            "partial_coupling_cells_max_abs_probability_change":
                v2["checks"]["max_abs_partial_probability_change_v2_minus_v1"],
            "exact_gate_checks_now_include_partial_coupling": True,
            "exact_vs_fast_v1": {"cells": len(v1["exact_gate_checks"]),
                                 "max_abs_probability_discrepancy": round(max(v1_exact), 4),
                                 "cells_over_5_points": len(
                                     v1["exact_vs_fast_discrepancies_over_5_points"])},
            "exact_vs_fast_v2": v2["exact_vs_fast"]},
        "conclusion_changes": [
            "no required-group range changes",
            ("the exact-versus-fast statement weakens: v2 finds "
             f"{v2['exact_vs_fast']['cells_with_discrepancy_over_5_points']} cell(s) where the "
             "exact bootstrap gate is more than 5 points below the fast method (max "
             f"{v2['exact_vs_fast']['max_abs_probability_discrepancy']}); v1's 'none above 5 "
             "points' no longer holds, so fast-method probabilities near the gate size should "
             "be read as slightly optimistic"),
            ("the scientific-gate wording is corrected: around 200-350 groups the design is an "
             "exploratory positive-effect screen, not confirmation of a five-point effect")],
        "still_true": ["power is planning evidence only", "training-seed variance is unmodelled",
                       "repository-native variance is unavailable",
                       "answer-rate, validity and diversity gates are outside the power "
                       "calculation"],
    }
    publish_file_atomically(ROOT / OUTPUT, (json.dumps(receipt, indent=1, sort_keys=True)
                                            + "\n").encode("utf-8"))
    print(json.dumps(receipt["conclusion_changes"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
