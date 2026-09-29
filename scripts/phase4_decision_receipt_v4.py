"""Decision receipt v4 (2026-09-29): Choice B v2 screen interpreted under the frozen rules.

Additive successor to v3 (unchanged). Reads the committed Choice B v2 analysis and records
the single justified interpretation. Deterministic; launches nothing.
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

OUTPUT = "results/sft_root_cause_decision_receipt_2026-09-29_v4.json"
SOURCES = {
    "decision_v3": "results/sft_root_cause_decision_receipt_2026-09-29_v3.json",
    "choice_b_v2_preflight": "results/sft_root_cause_phase4_choice_b_v2_preflight.json",
    "choice_b_v2_analysis": "results/sft_root_cause_phase4_choice_b_v2_analysis.json",
    "choice_b_evaluation_spec_v2": "results/sft_root_cause_phase4_choice_b_evaluation_spec_v2.json",
    "choice_b_power_200": "results/sft_root_cause_phase4_choice_b_power_200_v1.json",
    "choice_a_receiver_protocol": "docs/SFT_ROOT_CAUSE_CHOICE_A_RECEIVER_REPLAY_PROTOCOL.md",
}


def sha(rel):
    return hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()


def main() -> int:
    analysis = json.loads((ROOT / SOURCES["choice_b_v2_analysis"]).read_text(encoding="utf-8"))
    decision = analysis["decision"]
    if decision["outcome"] != "inconclusive_power":
        raise SystemExit("REFUSED: this receipt records interpretation C only")
    primary = {s: v["accuracy"]["difference_points"]
               for s, v in analysis["primary_fixed_input"].items()}
    receipt = {
        "schema_version": "oneiros_sft_root_cause_decision_v4", "date": "2026-09-29",
        "supersedes": {"path": SOURCES["decision_v3"], "modified": False},
        "launches_nothing": True,
        "sources": {k: {"path": v, "sha256": sha(v)} for k, v in SOURCES.items()},
        "choice_b_v2": {
            "frozen_outcome": decision["outcome"], "reason_codes": decision["reason_codes"],
            "interpretation": "C: inconclusive power",
            "primary_difference_points": {s: {k: p[k] for k in ("point", "low", "high")}
                                          for s, p in primary.items()},
            "statement": ("The screen did not resolve the hypothesis. It is NOT evidence of no "
                          "effect. Descriptively, both 95% intervals are narrow and their upper "
                          "bounds are +3.0 (answer schema) and +1.25 (prefilled assertion) "
                          "points, so on this internal Arm-A-exposed gate, for this single seed, "
                          "the data do not support a composite improvement of 5 points or more. "
                          "The paired intervals were narrower than the planning simulation "
                          "assumed, because control and treatment answer the same items far "
                          "more alike than base and arm A did."),
            "guardrails": {n: {"operational_pass": g["operational_pass"],
                               "adverse_supported": g["adverse_supported"]}
                           for n, g in analysis["guardrails"].items()},
            "not_claimed": ["no effect", "value-location causality", "generalisation",
                            "root cause", "efficacy"],
            "not_done": ["additional seeds", "reuse of the exposed gate", "prompt changes",
                         "checkpoint selection"]},
        "state_of_evidence": {"root_cause_established": False,
                              "generalization_established": False},
        "next": {
            "options": {
                "fresh_cohort": ("a fresh preregistered cohort on untouched groups with several "
                                 "seeds; the narrow single-seed intervals suggest only small "
                                 "composite effects remain plausible, so this is costly for "
                                 "little expected information"),
                "choice_a_pilot": ("the bounded CPU receiver-aware replay pilot on the existing "
                                   "24 development targets, with predeclared feasibility and "
                                   "diversity thresholds (" + SOURCES["choice_a_receiver_protocol"]
                                   + "); if infeasible, pivot to native execution of complete "
                                   "model-generated tests")},
            "recommended": "choice_a_pilot",
            "requires_approval": True},
    }
    publish_file_atomically(ROOT / OUTPUT, (json.dumps(receipt, indent=1, sort_keys=True)
                                            + "\n").encode("utf-8"))
    print(json.dumps(receipt["choice_b_v2"]["primary_difference_points"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
