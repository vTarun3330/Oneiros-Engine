"""Decision receipt v2 (2026-09-29): Choice A vs Choice B after the corrections.

Successor to results/sft_root_cause_decision_receipt_2026-09-29_v1.json (unchanged).
What changed: power v2 (corrected partial coupling, reconciled gates), native
denominators and readiness flags (v2), portable evidence bundles, the execution-dose
provenance receipt, and a frozen interpretation of the two-arm objective contrast as
a COMPOSITE intervention. Deterministic; every figure is read from a committed
receipt; launches nothing.
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

OUTPUT = "results/sft_root_cause_decision_receipt_2026-09-29_v2.json"
PREDECESSOR = "results/sft_root_cause_decision_receipt_2026-09-29_v1.json"
SOURCES = {
    "decision_v1": PREDECESSOR,
    "power_sensitivity_v2": "results/sft_root_cause_phase4_power_sensitivity_v2.json",
    "power_supersession": "results/sft_root_cause_phase4_power_sensitivity_v1_supersession.json",
    "native_rehearsal_v2": "results/sft_root_cause_phase4_native_rehearsal_receipt_v2.json",
    "native_evidence": "results/sft_root_cause_phase4_native_evidence_v1.json",
    "smoke_evidence": "results/sft_root_cause_phase4_smoke_evidence_v1.json",
    "objective_smoke": "results/sft_root_cause_phase4_objective_smoke_v1.json",
    "execution_dose_provenance": "results/v4_3_execution_dose_source_provenance_v1.json",
    "phase4_census": "results/sft_root_cause_phase4_cohort_census.json",
}
GATE_GROUPS = 200


def load(rel):
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))


def pct(bounds) -> str:
    low, high = bounds
    return f"{round(low * 100)}-{round(high * 100)}%"


def main() -> int:
    v1 = load(PREDECESSOR)
    power = load(SOURCES["power_sensitivity_v2"])
    native = load(SOURCES["native_rehearsal_v2"])
    smoke = load(SOURCES["smoke_evidence"])["facts"]["arms"]
    at_gate = power["power_at_planned_gate_sizes"][str(GATE_GROUPS)]
    control_tokens = smoke["control"]["dataset"]["supervised_tokens"]
    treatment_tokens = smoke["treatment"]["dataset"]["supervised_tokens"]
    next_experiment = dict(v1["recommendation"]["next_authorized_gpu_experiment"])
    next_experiment.update({
        "estimand": ("COMPOSITE: value-only masking plus reduced supervised-token mass "
                     "(interpretation A, frozen)"),
        "gate": (f"~{GATE_GROUPS} Arm-A-exposed remainder groups frozen before training; "
                 "fixed-input exact-output accuracy under both answer schemas; paired "
                 "semantic-group bootstrap; answer-rate non-inferiority; original validity "
                 "guardrail; Kill@8 and diversity reported; final checkpoint only; exactly one "
                 "look; outcomes promising/harm/inconclusive_power"),
        "power_at_gate_size_planning_only": {
            "positive_effect_gate_probability_range_by_true_effect_points":
                at_gate["positive_both"],
            "exceeds_five_gate_probability_range_by_true_effect_points":
                at_gate["confirmatory_5"],
            "reading": ("at ~200 groups the screen reliably detects a positive effect only if "
                        "the true effect is about 7 points or more ({p7}); a 5-point effect is "
                        "detected in {p5} of simulated datasets, so inconclusive_power is a "
                        "likely outcome for small effects; the gate cannot confirm that any "
                        "effect exceeds 5 points unless it is about 10 points or more, and "
                        "even then only {c10} of the time").format(
                            p7=pct(at_gate["positive_both"]["7"]),
                            p5=pct(at_gate["positive_both"]["5"]),
                            c10=pct(at_gate["confirmatory_5"]["10"]))},
        "preflight": "scripts/phase4_choice_b.py preflight (must be green before any launch)",
        "requires_user_authorisation": True})
    receipt = {
        "schema_version": "oneiros_sft_root_cause_decision_v2", "date": "2026-09-29",
        "supersedes": {"path": PREDECESSOR, "modified": False},
        "launches_nothing": True,
        "sources": {name: {"path": rel, "sha256": hashlib.sha256((ROOT / rel).read_bytes())
                           .hexdigest()} for name, rel in SOURCES.items()},
        "state_of_evidence": {**v1["state_of_evidence"], "root_cause_established": False},
        "what_changed_from_v1": [
            "power v2 replaces power v1 for planning (partial coupling corrected; required-group "
            "ranges unchanged; one exact-vs-fast cell over 5 points)",
            "the two scientific gates are reconciled: lower bound > 0 is a positive-effect "
            "screen; lower bound > +5 is exceeds-five; ~200-350 groups is an exploratory "
            "positive-effect screen",
            "native denominators are separated (25/26 environment, 24/25 semantic, 24/26 "
            "operational) and readiness is explicit (choice_A_ready=false)",
            "the two-arm objective contrast is frozen as a composite intervention",
            "the historical execution-dose artifacts are recorded as not current-run-ready"],
        "power": {"status": power["status"],
                  "required_groups_80pct": {rule: {d: v["range_across_scenarios"]
                                                   for d, v in ranges.items()}
                                            for rule, ranges in
                                            power["required_groups_for_80pct_power_ranges"]
                                            .items()},
                  "exact_vs_fast": power["exact_vs_fast"],
                  "scientific_gates": power["scientific_gates"],
                  "not_modelled": power["assumptions"]["not_modelled"]},
        "native_rehearsal": {"denominators": {k: {x: v[x] for x in ("k", "n", "point",
                                                                    "wilson_95")}
                                              for k, v in native["denominators"].items()},
                             "readiness": native["readiness"],
                             "infrastructure_failures": native["infrastructure_failures"]},
        "objective_smoke_interpretation": {
            "plumbing_evidence": ["identical input sequences and attention masks per batch",
                                  "labels differ", "both arms completed 2 optimiser steps",
                                  "both applied LoRA updates",
                                  "about 3.2 GiB peak GPU allocation",
                                  "no evaluation and no promotion"],
            "supervised_tokens": {"control_full_completion": control_tokens,
                                  "treatment_value_only": treatment_tokens},
            "consequence": ("the arms differ in BOTH the positions and the quantity of "
                            "supervised tokens; the two-arm contrast estimates the composite "
                            "intervention 'value-only masking plus reduced supervised-token "
                            "mass' and cannot by itself attribute an effect to value-token "
                            "specificity rather than to reduced token mass"),
            "frozen_interpretation": "A",
            "options": {
                "A": "keep the two-arm screen and call it a composite intervention (FROZEN)",
                "B": ("add a token-count-matched masking arm supervising the same number of "
                      "tokens per example at deterministic non-value positions, so value-only "
                      "vs matched isolates value specificity; NOT added; needs separate "
                      "authorisation and its own preflight")},
            "forbidden_claim": "pure value-token specificity"},
        "choice_A": {**v1["choice_A"],
                     "readiness": {"native_environment_feasible": True,
                                   "fixed_input_ready": False, "choice_A_ready": False}},
        "choice_B": {**v1["choice_B"],
                     "labels": ["internal", "exploratory", "Arm-A-exposed", "not untouched",
                                "not Phase 6", "not cross-dataset or repository generalisation",
                                "single seed cannot establish efficacy"],
                     "old_arm_A_compared_on_panel": False},
        "recommendation": {
            "decision": ("RUN CHOICE B FIRST ONLY AS AN INEXPENSIVE COMPOSITE-INTERVENTION "
                         "SCREEN, after a green source-bound preflight and explicit user "
                         "authorisation, while Choice A's CPU blocker (fixed-input argument "
                         "capture) is worked separately; Choice A stays required for any "
                         "generalisation claim"),
            "why": v1["recommendation"]["why"] + [
                "the screen is cheap enough to run even though it is underpowered for "
                "effects near 5 points; an inconclusive_power result is informative about "
                "the scale needed and does not justify a larger claim"],
            "next_authorized_gpu_experiment": next_experiment},
        "not_launched": ["full Choice A acquisition", "Choice A training", "Choice B training",
                         "token-count-matched third arm"],
    }
    publish_file_atomically(ROOT / OUTPUT, (json.dumps(receipt, indent=1, sort_keys=True)
                                            + "\n").encode("utf-8"))
    print(json.dumps(receipt["recommendation"]["decision"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
