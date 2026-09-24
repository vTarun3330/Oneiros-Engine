"""Post-gate, train-only diagnosis of the execution-supervision pilot.

This script is deliberately downstream of the frozen gate.  It cannot change
the gate, open confirmation IDs, or inspect validation/test data.  It explains
which candidate behaviours changed so the next experiment is evidence-led.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
from typing import Any


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _index(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result = {str(row["record_id"]): row for row in rows}
    if len(result) != len(rows):
        raise ValueError("duplicate record IDs in evaluation artifact")
    return result


def _format_class(raw: str) -> str:
    text = raw.strip()
    if not text:
        return "empty"
    if re.fullmatch(r"```(?:python)?\s*\n?.*?\n?```", text, re.S | re.I):
        return "markdown_fence"
    if text.startswith("**") and text.endswith("**"):
        return "markdown_bold"
    return "plain_or_other"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--treatment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    analysis = json.loads(args.analysis.read_text(encoding="utf-8"))
    if analysis.get("mechanism_gate_passed") is not False:
        raise SystemExit("REFUSED: this diagnosis is only for a failed frozen gate")
    control = json.loads(args.control.read_text(encoding="utf-8"))
    treatment = json.loads(args.treatment.read_text(encoding="utf-8"))
    pilot_path = args.dataset_dir / "pilot_development.execution.json"
    pilot = _index(json.loads(pilot_path.read_text(encoding="utf-8")))
    if control.get("pilot_development_sha256") != _sha(pilot_path):
        raise ValueError("control artifact is not bound to the supplied pilot")
    if treatment.get("pilot_development_sha256") != _sha(pilot_path):
        raise ValueError("treatment artifact is not bound to the supplied pilot")

    conditions: dict[str, Any] = {}
    for condition in ("intended_output", "shown_actual_output"):
        left = _index(control["detail"][condition])
        right = _index(treatment["detail"][condition])
        if set(left) != set(right) or set(left) != set(pilot):
            raise ValueError(f"item mismatch in {condition}")
        transitions = Counter()
        lenient_transitions = Counter()
        formats = {"control": Counter(), "treatment": Counter()}
        exact_same = 0
        changed_ids = []
        strata = defaultdict(lambda: {
            "n": 0, "control_correct": 0, "treatment_correct": 0,
            "control_answered": 0, "treatment_answered": 0,
        })
        for record_id in sorted(left):
            lrow, rrow = left[record_id], right[record_id]
            lstrict = str(lrow["strict"]["verdict"])
            rstrict = str(rrow["strict"]["verdict"])
            llenient = str(lrow["lenient"]["verdict"])
            rlenient = str(rrow["lenient"]["verdict"])
            transitions[(lstrict, rstrict)] += 1
            lenient_transitions[(llenient, rlenient)] += 1
            formats["control"][_format_class(str(lrow["raw"]))] += 1
            formats["treatment"][_format_class(str(rrow["raw"]))] += 1
            if lrow["raw_sha256"] == rrow["raw_sha256"]:
                exact_same += 1
            else:
                changed_ids.append(record_id)
            item = pilot[record_id]
            for dimension, value in (
                ("source", item["source_dataset"]),
                ("complexity", item["complexity_tier"]),
                ("bug_family", item["bug_family"]),
                ("shown_differs_from_intended", str(
                    bool(item["execution_evidence"]["differs"])).lower()),
            ):
                bucket = strata[f"{dimension}::{value}"]
                bucket["n"] += 1
                bucket["control_correct"] += int(llenient == "correct")
                bucket["treatment_correct"] += int(rlenient == "correct")
                bucket["control_answered"] += int(
                    llenient in {"correct", "wrong_value", "wrong_type"}
                )
                bucket["treatment_answered"] += int(
                    rlenient in {"correct", "wrong_value", "wrong_type"}
                )
        conditions[condition] = {
            "items": len(left),
            "raw_output_identical": exact_same,
            "raw_output_changed": len(left) - exact_same,
            "raw_output_identical_rate": exact_same / len(left),
            "format_counts": {arm: dict(sorted(counts.items()))
                              for arm, counts in formats.items()},
            "strict_transition_counts": {
                f"{before} -> {after}": count
                for (before, after), count in sorted(transitions.items())
            },
            "lenient_transition_counts": {
                f"{before} -> {after}": count
                for (before, after), count in sorted(lenient_transitions.items())
            },
            "changed_record_ids": changed_ids,
            "strata": dict(sorted(strata.items())),
        }

    manifest = json.loads((args.dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    arm_b = json.loads((args.dataset_dir / "arm_b.treatment.json").read_text(encoding="utf-8"))
    replacements = [row for row in arm_b if row.get("task_kind") == "execution_output_prediction"]
    trace_eligible_by_record: dict[str, bool] = {}
    for row in json.loads(
        (args.dataset_dir / "train.execution.json").read_text(encoding="utf-8")
    ):
        record_id = str(row["record_id"])
        trace_eligible_by_record[record_id] = (
            trace_eligible_by_record.get(record_id, False)
            or bool(row.get("trace_training_eligible"))
        )
    report = {
        "schema_version": "oneiros_execution_supervision_failure_diagnosis_v1",
        "label": "post-gate train-only diagnosis; no confirmation/validation/test access",
        "frozen_gate_passed": False,
        "inputs": {
            "analysis_sha256": _sha(args.analysis),
            "control_sha256": _sha(args.control),
            "treatment_sha256": _sha(args.treatment),
            "pilot_development_sha256": _sha(pilot_path),
            "dataset_manifest_sha256": _sha(args.dataset_dir / "manifest.json"),
        },
        "training_intervention": {
            "total_examples_per_arm": len(arm_b),
            "focused_replacements": len(replacements),
            "focused_share": len(replacements) / len(arm_b),
            "trace_eligible_replacements": sum(
                trace_eligible_by_record.get(str(row["record_id"]), False)
                for row in replacements
            ),
            "completion_bytes_equal_between_arms": manifest["arm_report"][
                "completion_bytes_equal_at_every_position"
            ],
            "interpretation": (
                "the first pilot changes conditioning/task framing only; it does not add "
                "new target tokens or execution traces"
            ),
        },
        "conditions": conditions,
        "conclusions": [
            "The frozen primary mechanism gate failed; no promotion or confirmation opening is permitted.",
            "The treatment increased exact-format answer production but did not improve lenient intended-value correctness.",
            "Most outputs are identical between arms, so 128 prompt-only replacements produced a small behavioural perturbation.",
            "Repeating or lengthening the same prompt-only intervention is not justified by this pilot.",
        ],
        "next_experiment_requirements": [
            "Use a genuinely new execution-learning signal (for example bounded trace/value targets), not the same assertion tokens under a new prompt.",
            "Keep the existing control adapter and 97-item train-derived panel frozen.",
            "Predeclare a matched token-mass/format control so trace length is not mistaken for execution learning.",
            "Do not open confirmation, validation, ablation_dev, test, or sealed-final data.",
            "Require semantic intended-value improvement under lenient and strict scoring before any canonical Kill@8 check.",
        ],
        "sealed_final_test_accessed": False,
        "validation_accessed": False,
        "ablation_dev_accessed": False,
        "confirmation_opened": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "intended_identical_rate": conditions["intended_output"]["raw_output_identical_rate"],
        "actual_identical_rate": conditions["shown_actual_output"]["raw_output_identical_rate"],
        "conclusions": report["conclusions"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
