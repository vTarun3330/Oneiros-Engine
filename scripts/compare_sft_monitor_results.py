"""Compare SFT policies on one identical locked validation panel."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_on_dataset import _paired_function_diagnostics


def compare_monitor_results(result_paths: Iterable[Path]) -> Dict:
    paths = [Path(path) for path in result_paths]
    if len(paths) < 2:
        raise ValueError("At least two monitor result files are required")
    results: List[Dict] = [
        json.loads(path.read_text(encoding="utf-8")) for path in paths
    ]
    identity_fields = ("selection_sha256", "seed", "tests_per_function")
    reference = results[0]
    reference_ids = [
        item["record_id"] for item in reference.get("function_results", [])
    ]
    if not reference_ids:
        raise ValueError("Reference result lacks per-function outcomes")
    for path, result in zip(paths[1:], results[1:]):
        if any(result.get(field) != reference.get(field) for field in identity_fields):
            raise ValueError(f"Result does not use the locked common panel: {path}")
        if [item["record_id"] for item in result.get("function_results", [])] != reference_ids:
            raise ValueError(f"Per-function outcome order differs: {path}")

    comparisons = []
    for path, result in zip(paths, results):
        comparisons.append({
            "path": str(path.resolve()),
            "checkpoint_step": result.get("checkpoint_step"),
            "function_kill_rate": result["function_kill_rate"],
            "function_validation_killed": result["function_validation_killed"],
            "candidate_kill_rate": result["candidate_kill_rate"],
            "end_to_end_candidate_kill_rate": result.get(
                "end_to_end_candidate_kill_rate"
            ),
            "parse_success_rate": result.get("parse_success_rate"),
            "delta_function_kill_rate_from_reference": round(
                float(result["function_kill_rate"])
                - float(reference["function_kill_rate"]),
                6,
            ),
            "paired_vs_reference": _paired_function_diagnostics(reference, result),
        })
    return {
        "mode": "locked_sft_common_panel_comparison",
        "selection_sha256": reference["selection_sha256"],
        "seed": reference["seed"],
        "tests_per_function": reference["tests_per_function"],
        "function_count": len(reference_ids),
        "comparisons": comparisons,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    comparison = compare_monitor_results(arguments.results)
    rendered = json.dumps(comparison, indent=2) + "\n"
    if arguments.output:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
