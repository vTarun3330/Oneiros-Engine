"""Fair comparison of actual Atheris against Oneiros on one shared panel.

Both systems are scored on the SAME targets, with the same eligibility rule and
the same oracle question: does this system produce something that distinguishes
the buggy implementation from the correct one?

Two asymmetries are deliberate and are reported rather than hidden:

* **Atheris is given the reference implementation.** Its harness runs a
  differential oracle, because most of these defects return a wrong value
  instead of crashing and a crash-only score would understate it badly. Oneiros
  never sees the reference. The comparison is therefore generous to Atheris.

* **Atheris is given far more executions.** Oneiros emits 8 candidates per
  target - roughly 8 executions. Atheris is run at up to 20,000. Both budgets
  are reported so a reader can see the result at matched and at generous cost.

The simulated coverage fuzzer in ``baseline.coverage_fuzzer`` is a different
system and is never reported under the Atheris name.
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness.corpus import write_json
from metrics.research_evaluation import wilson_interval
from utils.reproducibility import source_tree_sha256


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    if resolved.is_relative_to(ROOT):
        return str(resolved.relative_to(ROOT)).replace("\\", "/")
    return str(resolved)


def load_atheris(
    directory: Path,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    results: dict[str, dict[str, Any]] = {}
    configurations: dict[str, dict[str, Any]] = {}
    for path in sorted(directory.glob("task_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        configuration = {
            "harness_version": payload.get("harness_version"),
            "python_version": payload.get("python_version"),
            "max_runs": payload.get("max_runs"),
            "time_budget_seconds": payload.get("time_budget_seconds"),
            "seed": payload.get("seed"),
            "runner_wall_limit": payload.get("runner_wall_limit"),
        }
        key = json.dumps(configuration, sort_keys=True)
        configurations[key] = configuration
        for row in payload.get("results", []):
            task_id = row.get("task_id")
            if task_id:
                results[str(task_id)] = row
    return results, list(configurations.values())


def load_oneiros_panel(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("final_test_measurement"):
        raise SystemExit(f"Refusing {path}: sealed final-test measurement")
    return payload


def compare(atheris_dir: Path, panels: dict[str, Path]) -> dict[str, Any]:
    atheris, run_configurations = load_atheris(atheris_dir)
    if not atheris:
        raise SystemExit(f"no atheris results under {atheris_dir}")

    killed = [row for row in atheris.values() if row.get("outcome") == "killed"]
    unsupported = [
        row for row in atheris.values() if row.get("outcome") == "unsupported_target"
    ]
    survived = [row for row in atheris.values() if row.get("outcome") == "survived"]
    exhausted = [
        row for row in atheris.values() if row.get("outcome") == "time_budget_exhausted"
    ]
    outer_timeouts = [
        row for row in atheris.values() if row.get("outcome") == "outer_wall_timeout"
    ]
    incomplete = [
        row for row in atheris.values() if row.get("outcome") == "incomplete"
    ]
    harness_failures = [
        row for row in atheris.values() if row.get("outcome") == "harness_failure"
    ]
    unit_timeouts = [
        row for row in atheris.values() if row.get("outcome") == "unit_timeout"
    ]
    supported = len(atheris) - len(unsupported)
    completed_processes = (
        len(killed) + len(survived) + len(exhausted) + len(unit_timeouts)
    )
    kill_kinds = collections.Counter(
        row.get("kill_kind") for row in killed if row.get("kill_kind")
    )
    atheris_budgets = sorted({
        int(item["max_runs"]) for item in run_configurations
        if item.get("max_runs") is not None
    })

    report: dict[str, Any] = {
        "schema_version": "oneiros_atheris_comparison_v1",
        "source_tree_sha256": source_tree_sha256(ROOT),
        "sealed_final_test_accessed": False,
        "atheris": {
            "package": "atheris==2.3.0",
            "runtime": "WSL2 Ubuntu 24.04, Python 3.11",
            "is_actual_atheris": True,
            "oracle": "differential against the reference implementation",
            "run_configurations": run_configurations,
            "targets": len(atheris),
            "supported_targets": supported,
            "unsupported_targets": len(unsupported),
            "killed": len(killed),
            "survived": len(survived),
            "time_budget_exhausted": len(exhausted),
            "outer_wall_timeouts": len(outer_timeouts),
            "incomplete_results": len(incomplete),
            "harness_failures": len(harness_failures),
            "unit_timeouts": len(unit_timeouts),
            "completed_supported_processes": completed_processes,
            "comparison_complete": (
                completed_processes == supported
                and not outer_timeouts
                and not incomplete
                and not harness_failures
            ),
            "kill_rate_over_all_targets": round(len(killed) / max(1, len(atheris)), 6),
            "kill_rate_over_supported_targets": round(
                len(killed) / max(1, supported), 6
            ),
            "kill_rate_wilson_95": wilson_interval(len(killed), len(atheris)),
            "kill_kinds": dict(kill_kinds),
            "median_runs_to_kill": (
                sorted(row.get("runs", 0) for row in killed)[len(killed) // 2]
                if killed else None
            ),
            "total_recorded_fuzz_callback_runs": sum(
                int(row.get("runs") or 0) for row in atheris.values()
            ),
            "unsupported_reasons": collections.Counter(
                str(row.get("harness_error", ""))[:80] for row in unsupported
            ),
            "unit_timeout_note": (
                "A per-input timeout is not counted as a kill because the "
                "harness cannot attribute the stall to the buggy side without "
                "also timing the reference side in isolation."
            ),
        },
        "oneiros": {},
        "fairness": {
            "shared_panel_with_oneiros_evidence": bool(panels),
            "atheris_sees_the_reference_implementation": True,
            "oneiros_sees_the_reference_implementation": False,
            "atheris_execution_budget": atheris_budgets,
            "oneiros_execution_budget": "8 generated candidates per target",
            "execution_count_matched": atheris_budgets == [8],
            "note": (
                "The differential oracle favours Atheris because Oneiros never "
                "sees the reference implementation. Runs above eight inputs "
                "also favour Atheris by execution count; the matched-eight arm "
                "does not have that second asymmetry."
            ),
        },
    }

    for name, path in panels.items():
        payload = load_oneiros_panel(path)
        results = payload.get("function_results") or []
        panel_ids = {str(row.get("record_id")) for row in results}
        shared = panel_ids & set(atheris)
        panel_killed = sum(
            1 for row in results if row.get("killed") or row.get("function_killed")
        )
        report["oneiros"][name] = {
            "artifact": _display_path(path),
            "checkpoint_step": payload.get("checkpoint_step"),
            "seed": payload.get("seed"),
            "function_kill_rate": payload.get("function_kill_rate"),
            "functions": len(results),
            "functions_killed_recomputed": panel_killed or None,
            "shared_targets_with_atheris": len(shared),
            "panel_matches_atheris_targets": panel_ids == set(atheris),
        }
        if shared:
            atheris_on_shared = sum(
                1 for task_id in shared
                if atheris[task_id].get("outcome") == "killed"
            )
            report["oneiros"][name]["atheris_kill_rate_on_shared"] = round(
                atheris_on_shared / len(shared), 6
            )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atheris-dir", type=Path, required=True)
    parser.add_argument(
        "--panel", action="append", default=[], metavar="NAME=PATH",
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "results" / "v4_2_atheris_vs_oneiros.json",
    )
    arguments = parser.parse_args()
    panels = {}
    for item in arguments.panel:
        name, _, path = item.partition("=")
        panels[name] = Path(path)
    report = compare(arguments.atheris_dir, panels)
    write_json(arguments.output, report)
    print(json.dumps(report, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
