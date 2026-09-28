"""Phase 3C analysis v3 (CPU only; v1 and v2 receipts are preserved unchanged).

Repairs relative to v2:
* the frozen answer-rate gate is RECOMPUTED from the rescored rows (v2 copied v1's) and
  enforced before any interpretation (``capacity_interpretation``);
* exact item pairing in every contrast;
* Monte Carlo p-values are worded exactly and never presented as bounds or exact values;
* evidence against a zero effect (sign-flip + Holm across C1/C2) is reported separately
  from the frozen meaningful-effect rule (per-contrast, NOT multiplicity-adjusted 95%
  interval lower bound >= 5); a Bonferroni-width sensitivity is labelled post hoc;
* every frozen secondary analysis is restored (7B - arm A at A0/A4, 7B A4 - A0,
  7B A3 - A0, 7B - 1.5B under the ANSWER schema, C1 by cohort, C2 by tier) and labelled
  exploratory, with denominators and clustered intervals and no p-values.

Retained generations only; no model call, no GPU.
"""
from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness.atomic_publish import publish_file_atomically
from harness.fixed_input_probe import CONTROL_LEVEL, CORRECTED_LEVEL_LABELS, LEVELS
from harness.fixed_input_rescoring import (
    ALL_LEVELS, DESIGN_3A, load_inputs, row_identity, score_function, sha_file,
)
from harness.parallel_execution import map_jobs
from harness.probe_statistics import (
    RESAMPLES, SEED, answer_rate_gate, bootstrap_contrast, capacity_interpretation, holm,
    sign_flip_test,
)

DESIGN = "results/sft_root_cause_phase3c_design_receipt.json"
V1 = "results/sft_root_cause_phase3c_result_receipt.json"
V2 = "results/sft_root_cause_phase3c_result_receipt_v2.json"
RECEIPT = "results/sft_root_cause_phase3c_result_receipt_v3.json"
ARMS = ("base", "arm_a_431", "qwen7b_base")
EXPECTED_ROWS = 240 * 2 * len(ARMS) * len(ALL_LEVELS)
GATE_MARGIN, STRICT_MIN = 0.02, 0.95     # frozen in the 3C design receipt
SOURCES = ["scripts/analyze_capacity_probe_v3.py", "harness/probe_statistics.py",
           "harness/fixed_input_rescoring.py", "harness/fixed_input_probe.py",
           "harness/safe_execution.py", "engine/test_generation_prompt.py"]
PRIMARY = {"C1_A4": (("base", "A4"), ("qwen7b_base", "A4")),
           "C2_A0": (("base", "A0"), ("qwen7b_base", "A0"))}


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def everything(r):
    return True


def main() -> int:
    started = time.time()
    inputs = load_inputs(ARMS)
    jobs = [{"function": f, "record": inputs["records"][f["record_id"]],
             "generations": inputs["generations"]} for f in inputs["cohort"]["functions"]]
    rows = [row for chunk in map_jobs(score_function, jobs) for row in chunk]
    identity = row_identity(rows)
    if identity["rows"] != EXPECTED_ROWS or identity["duplicates"]:
        raise SystemExit(f"REFUSED: unexpected row identity {identity}")

    cells = {}
    for arm in ARMS:
        for level in ALL_LEVELS:
            sel = [r for r in rows if r["arm"] == arm and r["level"] == level]
            cells[f"{arm}/{level}"] = {
                "items": len(sel), "correct": sum(r["correct"] for r in sel),
                "answer_rate": round(sum(1 for r in sel if r["answer"]) / len(sel), 4),
                "accuracy": round(sum(r["correct"] for r in sel) / len(sel), 4),
                "methods": dict(Counter(r["method"] for r in sel))}

    gate = answer_rate_gate(rows, "qwen7b_base", "base", LEVELS, CONTROL_LEVEL,
                            GATE_MARGIN, STRICT_MIN)
    primary = {}
    for name, (a, b) in PRIMARY.items():
        primary[name] = {**bootstrap_contrast(rows, a, b, extra_levels=(97.5,)),
                         "sign_flip": sign_flip_test(rows, a, b)}
    adjusted = holm({n: p["sign_flip"]["monte_carlo_p"] for n, p in primary.items()})
    for name, p in primary.items():
        p["zero_effect"] = {
            "holm_adjusted_monte_carlo_p": round(adjusted[name], 5),
            "family": "C1 and C2 (frozen)",
            "statement": ("evidence against a zero effect only; it does not establish an "
                          "effect of at least 5 points"),
            "resolution_note": ("adjusted from Monte Carlo p-values at the minimum reportable "
                                "resolution (0/10000 as extreme); it is a Monte Carlo value, "
                                "not an exact p" if p["sign_flip"]["extreme"] == 0 else None)}
        low = p["ci95_points"][0]
        p["meaningful_effect_frozen_rule"] = {
            "rule": "per-contrast 95% cluster-bootstrap lower bound >= +5 points (frozen)",
            "multiplicity_adjusted": False,
            "lower_bound_points": low,
            "clears_5_points_per_contrast": low >= 5,
            "familywise_claim": "none: the frozen intervals are ordinary per-contrast intervals"}
        p["post_hoc_sensitivity_bonferroni_97_5"] = {
            "label": "POST HOC sensitivity, not part of the frozen analysis",
            "interval_points": p["ci97.5_points"],
            "lower_bound_clears_5": p["ci97.5_points"][0] >= 5}

    def by(field, value):
        return lambda r: r[field] == value

    secondary = {
        "7b_minus_arm_a_A0": bootstrap_contrast(rows, ("arm_a_431", "A0"), ("qwen7b_base", "A0")),
        "7b_minus_arm_a_A4": bootstrap_contrast(rows, ("arm_a_431", "A4"), ("qwen7b_base", "A4")),
        "7b_A4_minus_A0": bootstrap_contrast(rows, ("qwen7b_base", "A0"), ("qwen7b_base", "A4")),
        "7b_A3_minus_A0": bootstrap_contrast(rows, ("qwen7b_base", "A0"), ("qwen7b_base", "A3")),
        "7b_minus_1_5b_answer_schema": bootstrap_contrast(
            rows, ("base", CONTROL_LEVEL), ("qwen7b_base", CONTROL_LEVEL)),
        "C1_by_cohort": {c: bootstrap_contrast(rows, ("base", "A4"), ("qwen7b_base", "A4"),
                                               by("cohort", c)) for c in ("exposed", "unexposed")},
        "C2_by_tier": {t: bootstrap_contrast(rows, ("base", "A0"), ("qwen7b_base", "A0"),
                                             by("complexity_tier", t))
                       for t in ("simple", "moderate")},
    }

    # --- reconciliation with v1 and v2 --------------------------------------------------------
    v1 = json.loads((ROOT / V1).read_text(encoding="utf-8"))
    v2 = json.loads((ROOT / V2).read_text(encoding="utf-8"))
    mismatches, compared = [], 0
    for key, old in v1["cells"].items():
        new = cells[key]
        for field, new_field in (("items", "items"), ("answer_rate", "answer_rate"),
                                 ("accuracy", "accuracy"), ("methods", "methods")):
            compared += 1
            if old[field] != new[new_field]:
                mismatches.append(("v1", key, field, old[field], new[new_field]))
        implied = old["accuracy"] * old["items"]
        if math.isclose(implied, round(implied), abs_tol=0.02):
            compared += 1
            if round(implied) != new["correct"]:
                mismatches.append(("v1", key, "correct", round(implied), new["correct"]))
    v1_gate = v1["gate"]
    for level, block in gate["per_level"].items():
        compared += 2
        if (v1_gate["per_level"][level]["qwen7b"], v1_gate["per_level"][level]["base_1_5b"]) \
                != (block["large"], block["small"]):
            mismatches.append(("v1_gate", level, v1_gate["per_level"][level], block))
    compared += 2
    if v1_gate["control_strict_rate"] != gate["control_strict_rate"] or \
            v1_gate["passed"] != gate["passed"]:
        mismatches.append(("v1_gate", "control/passed", v1_gate, gate))
    for name, new in primary.items():
        old = v2["primary"][name]
        compared += 3
        for field in ("difference_points", "ci95_points"):
            if old[field] != new[field]:
                mismatches.append(("v2", name, field, old[field], new[field]))
        if old["sign_flip"]["extreme"] != new["sign_flip"]["extreme"]:
            mismatches.append(("v2", name, "sign_flip_extreme", old["sign_flip"]["extreme"],
                               new["sign_flip"]["extreme"]))
    for name, old in v2["secondary"].items():
        compared += 2
        for field in ("difference_points", "ci95_points"):
            if old[field] != secondary[name][field]:
                mismatches.append(("v2_secondary", name, field, old[field],
                                   secondary[name][field]))
    # v1 bootstrapped subgroups by resampling ALL 240 groups (non-members contribute zero,
    # so subgroup size is random); v2/v3 resample only the subgroup's own groups.  Point
    # estimates must agree exactly; the intervals legitimately differ and both are kept.
    subgroup_interval_method_change = {}
    for group, members in (("C1_by_cohort", v1["secondary"]["C1_by_cohort"]),
                           ("C2_by_tier", v1["secondary"]["C2_by_tier"])):
        for member, old in members.items():
            compared += 1
            new = secondary[group][member]
            if old["difference_points"] != new["difference_points"]:
                mismatches.append(("v1_secondary", group, member, "difference_points",
                                   old["difference_points"], new["difference_points"]))
            subgroup_interval_method_change[f"{group}/{member}"] = {
                "v1_ci95_resampling_all_240_groups": old["ci95_points"],
                "v3_ci95_resampling_subgroup_groups_only": new["ci95_points"]}
    if mismatches:
        raise SystemExit(f"REFUSED: v3 does not reconcile: {mismatches[:5]}")

    interpretation = capacity_interpretation(gate, primary)
    receipt = {
        "schema_version": "oneiros_sft_root_cause_phase3c_result_v3",
        "phase": "3C", "version": 3, "command": "python scripts/analyze_capacity_probe_v3.py",
        "branch": git("branch", "--show-current"), "starting_commit": git("rev-parse", "HEAD"),
        "model_calls": 0, "gpu_used": False, "training": False,
        "restricted_splits_accessed": "none (frozen cohort, train shard and retained Phase 3 "
                                      "generations only)",
        "provenance": {
            "supersedes": {V1: sha_file(V1), V2: sha_file(V2)}, "v1_v2_modified": False,
            "design_receipts": {DESIGN: sha_file(DESIGN), DESIGN_3A: sha_file(DESIGN_3A)},
            "cohort": {inputs["design"]["cohort"]["path"]:
                       sha_file(inputs["design"]["cohort"]["path"])},
            "record_shard": {inputs["design"]["record_shard"]["path"]:
                             sha_file(inputs["design"]["record_shard"]["path"])},
            "generations": inputs["completions"],
            "sources": {p: sha_file(p) for p in SOURCES}},
        "rescoring": {**identity, "expected_rows": EXPECTED_ROWS,
                      "reproduction": "all retained v1 aggregate statistics and the v1 gate "
                                      "reproduced; all v2 primary and secondary contrasts, "
                                      "intervals and permutation counts reproduced; exact "
                                      "row-level reproduction of v1 is not claimed (v1 kept no "
                                      "row hash)",
                      "reconciliation": {"fields_compared": compared, "mismatches": mismatches},
                      "numerical_changes": {
                          "exploratory_subgroup_intervals": subgroup_interval_method_change,
                          "why": ("v1's subgroup bootstrap resampled all 240 semantic groups "
                                  "(non-members contribute zero, so the subgroup's size was "
                                  "random); v3 resamples the subgroup's own groups, as v2 and "
                                  "every other v3 contrast do. Point estimates are identical; "
                                  "only these exploratory intervals move, by at most ~0.3 "
                                  "points"),
                          "everything_else": "unchanged"}},
        "level_labels": CORRECTED_LEVEL_LABELS,
        "cells": cells,
        "gate": {**gate, "recomputed_from_rows": True, "frozen_in": DESIGN},
        "inference": {"intervals": (f"paired cluster bootstrap over semantic group_id on exactly "
                                    f"paired items, {RESAMPLES} resamples, seed {SEED}"),
                      "tests": ("paired semantic-group sign-flip, Monte Carlo p = (extreme + 1) "
                                "/ (resamples + 1), Holm across the frozen C1/C2 family")},
        "primary": primary if gate["passed"] else None,
        "primary_suppressed_because_gate_failed": not gate["passed"],
        "secondary_exploratory": {"label": "EXPLORATORY: clustered intervals and denominators; "
                                           "no confirmatory p-value claims",
                                  "contrasts": secondary},
        "interpretation": {
            **interpretation,
            "within_7b_A4_minus_A0": ("+19.38 points is a within-7B contrast (fixed "
                                      "implementation vs production information); it is not "
                                      "the +24.79 7B-minus-1.5B A4 contrast and does not show "
                                      "that specification became the binding limit at 7B"),
            "A3": "this particular oracle-derived type/length hint did not measurably improve "
                  "accuracy at 7B"},
        "duration_seconds": round(time.time() - started, 1),
    }
    publish_file_atomically(ROOT / RECEIPT, (json.dumps(receipt, indent=1, sort_keys=True)
                                             + "\n").encode("utf-8"))
    print(json.dumps({"gate": gate, "rows": identity, "fields_compared": compared}, indent=1))
    for name, p in primary.items():
        print(name, p["difference_points"], p["ci95_points"], p["sign_flip"]["report"],
              "holm", p["zero_effect"]["holm_adjusted_monte_carlo_p"],
              "post-hoc 97.5%", p["ci97.5_points"])
    for name, s in secondary.items():
        if "difference_points" in s:
            print(f"  {name}: {s['difference_points']} {s['ci95_points']} "
                  f"({s['a']['items']} paired items)")
        else:
            for member, c in s.items():
                print(f"  {name}/{member}: {c['difference_points']} {c['ci95_points']} "
                      f"({c['a']['items']} items)")
    print(json.dumps(interpretation, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
