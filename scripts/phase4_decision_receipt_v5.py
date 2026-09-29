"""Decision receipt v5 (2026-09-29): additive interpretation corrections for Choice B v2.

v4, the Choice B v2 analysis and every Choice B artifact are unchanged; this binds their
hashes and corrects the reporting. The frozen outcome stays ``inconclusive_power``.
Deterministic; launches nothing.
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

OUTPUT = "results/sft_root_cause_decision_receipt_2026-09-29_v5.json"
SOURCES = {
    "decision_v4": "results/sft_root_cause_decision_receipt_2026-09-29_v4.json",
    "choice_b_v2_analysis": "results/sft_root_cause_phase4_choice_b_v2_analysis.json",
    "choice_b_v2_training_control": "results/sft_root_cause_phase4_choice_b_v2_training_control.json",
    "choice_b_v2_training_treatment":
        "results/sft_root_cause_phase4_choice_b_v2_training_treatment.json",
    "choice_b_evaluation_spec_v2": "results/sft_root_cause_phase4_choice_b_evaluation_spec_v2.json",
    "choice_b_power_200": "results/sft_root_cause_phase4_choice_b_power_200_v1.json",
}
TELEMETRY_RUNS = {"control": "runs/20260929-161756-choice_b_v2_train_control/telemetry.jsonl",
                  "treatment": "runs/20260929-165315-choice_b_v2_train_treatment/telemetry.jsonl"}


def sha(rel: str) -> str:
    return hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()


def load(rel: str):
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))


def interval(metric: dict) -> dict:
    d = metric["difference"]
    return {"point": d["point"], "low": d["low"], "high": d["high"]}


def device_peak(rel: str):
    path = ROOT / rel
    if not path.exists():
        return None
    values = [json.loads(l).get("vram_used_mib") for l in path.read_text(encoding="utf-8")
              .splitlines() if l.strip()]
    return max(v for v in values if v is not None)


def main() -> int:
    analysis = load(SOURCES["choice_b_v2_analysis"])
    down = analysis["downstream_kill_at_8"]
    parse = interval(down["parse_success"])
    if not (parse["high"] < 0):
        raise SystemExit("REFUSED: parse-success interval no longer excludes zero")
    power = load(SOURCES["choice_b_power_200"])
    cell = power["cells"][0]
    receipt = {
        "schema_version": "oneiros_sft_root_cause_decision_v5", "date": "2026-09-29",
        "supersedes": {"path": SOURCES["decision_v4"], "modified": False,
                       "nature": "additive reporting corrections; no outcome changed"},
        "launches_nothing": True,
        "sources": {k: {"path": v, "sha256": sha(v)} for k, v in SOURCES.items()},
        "choice_b_v2": {
            "frozen_outcome": analysis["decision"]["outcome"],
            "frozen_outcome_unchanged": True,
            "predeclared_stopping_criterion_met": False,
            "parse_success_regression": {
                "difference_points": parse,
                "label": ("statistically distinguishable ADVERSE EXPLORATORY SECONDARY signal "
                          "under the nominal paired 95% interval"),
                "qualifications": ["not a predeclared stopping guardrail",
                                   "one of several reported secondary metrics",
                                   "not adjusted for multiple comparisons",
                                   "insufficient to change the frozen outcome"]},
            "related_descriptive_downward_trends": {
                "execution_success": interval(down["execution_success"]),
                "exact_equality_validity_per_requested":
                    interval(down["exact_equality_validity_per_requested"]),
                "reference_validity_per_requested":
                    interval(down["reference_validity_per_requested"])},
            "mechanistic_hypothesis": {
                "statement": ("value-only masking may remove useful structural/syntactic "
                              "supervision while offering at most a small oracle-accuracy "
                              "benefit"),
                "status": "PLAUSIBLE, UNPROVEN; not causal, not established"},
            "kill_at_8": {**interval(down["kill_at_8"]),
                          "reading": "interval crosses zero; not an improvement"},
        },
        "seed_variance_correction": {
            "training_seed_variance": "UNMEASURED",
            "statement": ("the reported intervals describe within-seed paired gate uncertainty "
                          "only; training-seed variance was not measured, so narrow within-seed "
                          "intervals cannot establish that a multi-seed experiment would add "
                          "little information"),
            "why_choice_a_next": ("cheaper, CPU-only, and addresses real-repository "
                                  "feasibility; NOT because seed variance is known to be "
                                  "negligible"),
            "corrects": "decision receipt v4 next.options.fresh_cohort wording"},
        "memory_terminology": {
            "pytorch_peak_allocated_mib": {
                arm: load(SOURCES[f"choice_b_v2_training_{arm}"])["peak_allocated_mib"]
                for arm in ("control", "treatment")},
            "device_telemetry_peak_used_mib": {arm: device_peak(rel)
                                               for arm, rel in TELEMETRY_RUNS.items()},
            "rule": ("PyTorch peak allocated memory and device-level used memory are different "
                     "quantities; neither is reported as the other")},
        "power_monte_carlo_note": {
            "planning_simulations_per_cell": cell["sims"],
            "planning_bootstrap_resamples": cell["bootstrap_resamples"],
            "final_analysis_resamples": load(SOURCES["choice_b_evaluation_spec_v2"])[
                "primary"]["inference"]["resamples"],
            "monte_carlo_wilson_intervals": "preserved in the power receipt"},
        "path_policy": ("future receipts store repository-relative paths; the historical "
                        "absolute checkpoint path already committed in the Choice B training "
                        "result is not rewritten"),
        "state_of_evidence": {"root_cause_established": False,
                              "generalization_established": False},
        "not_claimed": ["no effect", "root cause", "generalisation",
                        "causal parse-regression mechanism", "seed stability"],
        "next": {"recommended": "bounded CPU receiver-aware Choice A replay pilot",
                 "gpu_training_justified": False},
    }
    publish_file_atomically(ROOT / OUTPUT, (json.dumps(receipt, indent=1, sort_keys=True)
                                            + "\n").encode("utf-8"))
    print(json.dumps({"parse": parse, "memory": receipt["memory_terminology"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
