"""Verified access to the evaluation evidence of the two closed pilots.

Every evaluation is checked twice over: the envelope file hash against the
decision receipt, and the raw rehearsal-result file hash against both the
decision receipt and the envelope.  Envelope leakage flags must be false.
Paths are repository-relative POSIX paths.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

TOOL_ASSISTED_RECEIPT = "results/v4_3_tool_assisted_decision_receipt.json"
EXECUTION_DOSE_RECEIPT = "results/v4_3_execution_dose_decision_receipt.json"
LEAKAGE_FLAGS = ("validation_accessed", "ablation_dev_accessed", "test_accessed",
                 "sealed_final_test_accessed", "confirmation_opened")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_json(root: Path, relative: str) -> dict[str, Any]:
    return json.loads((root / relative).read_text(encoding="utf-8"))


def evaluation_entries(root: Path) -> dict[str, dict[str, Any]]:
    """{"tool_assisted/A": receipt entry, ..., "execution_dose/base": ...}."""
    tool = load_json(root, TOOL_ASSISTED_RECEIPT)
    dose = load_json(root, EXECUTION_DOSE_RECEIPT)
    entries = {f"tool_assisted/{arm}": entry for arm, entry in tool["evaluations"].items()}
    entries.update({f"execution_dose/{arm}": entry for arm, entry in dose["retention"].items()})
    return entries


def verify_evaluation(root: Path, entry: dict[str, Any]) -> list[str]:
    """Problems with one evaluation (empty = envelope and raw result both verify)."""
    problems: list[str] = []
    envelope_ref, result_ref = entry["envelope"], entry["rehearsal_result"]
    for label, ref in (("envelope", envelope_ref), ("rehearsal_result", result_ref)):
        path = root / ref["path"]
        if Path(ref["path"]).is_absolute() or "\\" in ref["path"]:
            problems.append(f"{label} path is not repository-relative POSIX: {ref['path']}")
        elif not path.is_file():
            problems.append(f"{label} missing: {ref['path']}")
        elif sha256_file(path) != ref["sha256"]:
            problems.append(f"{label} hash differs from the decision receipt: {ref['path']}")
    if problems:
        return problems
    envelope = json.loads((root / envelope_ref["path"]).read_text(encoding="utf-8"))
    bound = envelope.get("rehearsal_result") or {}
    if bound.get("path") != result_ref["path"] or bound.get("sha256") != result_ref["sha256"]:
        problems.append(f"envelope does not bind the raw result: {envelope_ref['path']}")
    if envelope.get("status") != "complete":
        problems.append(f"envelope is not complete: {envelope_ref['path']}")
    if any(envelope.get(flag) for flag in LEAKAGE_FLAGS):
        problems.append(f"envelope reports protected access: {envelope_ref['path']}")
    return problems


def verify_closed_pilots(root: Path) -> dict[str, Any]:
    tool = load_json(root, TOOL_ASSISTED_RECEIPT)
    dose = load_json(root, EXECUTION_DOSE_RECEIPT)
    artifacts = {key: sha256_file(root / tool[key]["path"]) == tool[key]["sha256"]
                 for key in ("analysis", "lineage_manifest", "design_receipt", "panel", "journal")}
    evaluations = {name: verify_evaluation(root, entry)
                   for name, entry in evaluation_entries(root).items()}
    return {
        "tool_assisted": {"verdict": tool["decision"]["outcome"],
                          "artifact_hashes_verify": all(artifacts.values()),
                          "protected_flags_false": not any(tool["leakage"].values())},
        "execution_dose": {"outcome": dose["decision"]["outcome"],
                           "protected_flags_false": not any(
                               value for value in dose["leakage"].values()
                               if isinstance(value, bool))},
        "evaluations": {name: {"envelope_and_raw_result_verify": not problems,
                               "problems": problems}
                        for name, problems in evaluations.items()},
        "all_evaluations_verify": not any(evaluations.values()),
    }
