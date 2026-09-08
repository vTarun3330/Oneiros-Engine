"""Why does Atheris fail on the targets it fails on?

"Atheris kills 49% at 20000 runs" is a number, not a diagnosis. It cannot say
whether the survivors are hard for a fuzzer, unreachable by the input adapter,
or equivalent mutants that nothing could kill. Those demand different
responses, and only the last would mean the baseline is being measured fairly.

This cuts every target by the properties that plausibly drive the outcome -
execution budget, argument arity, inferred parameter type, benchmark, mutation
family, and semantic versus crash kill - and reports the outcome distribution
in each cell.

It reads finished run artifacts. It never fuzzes, never regenerates and never
touches the sealed split.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness.corpus import write_json

OUTCOMES = ("killed", "survived", "unit_timeout", "time_budget_exhausted",
            "incomplete", "harness_error")


def _load_tasks(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    tasks = payload["tasks"] if isinstance(payload, dict) and "tasks" in payload else payload
    return {str(task["task_id"]): task for task in tasks}


def _rows(directory: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(glob.glob(str(directory / "*.json"))):
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        for row in (payload.get("results") or [payload]):
            row = dict(row)
            row["_max_runs"] = payload.get("max_runs")
            row["_log"] = Path(path).with_suffix(".log")
            rows.append(row)
    return rows


def _coverage_reached(log: Path) -> bool | None:
    """Did libFuzzer report finding new coverage beyond the initial input?

    A survivor that never grew its corpus was never really searching, which is
    a different failure from one that explored and still could not distinguish
    the mutant.
    """
    if not log.exists():
        return None
    text = log.read_text(encoding="utf-8", errors="ignore")
    if "cov:" not in text:
        return None
    return "NEW " in text or "REDUCE" in text


def _bucket(value: Any) -> str:
    return str(value) if value not in (None, "") else "unknown"


def build(directories: list[Path], tasks_path: Path) -> dict[str, Any]:
    tasks = _load_tasks(tasks_path)
    by_dimension: dict[str, dict[str, Counter]] = defaultdict(
        lambda: defaultdict(Counter))
    totals: Counter = Counter()
    kill_kinds: Counter = Counter()
    coverage_without_kill = Counter()
    per_set: dict[str, Counter] = {}

    for directory in directories:
        rows = _rows(directory)
        set_counts: Counter = Counter()
        for row in rows:
            task = tasks.get(str(row.get("task_id"))) or {}
            outcome = _bucket(row.get("outcome"))
            totals[outcome] += 1
            set_counts[outcome] += 1
            if row.get("kill_kind"):
                kill_kinds[row["kill_kind"]] += 1

            kinds = row.get("parameter_kinds") or []
            dimensions = {
                "execution_budget": _bucket(row.get("_max_runs")),
                "argument_arity": str(len(kinds)),
                "parameter_kinds": ",".join(kinds) if kinds else "none",
                "benchmark": _bucket(task.get("source_dataset")),
                "mutation_family": _bucket(task.get("bug_family")),
                "instrumented": _bucket(row.get("coverage_instrumented")),
            }
            for dimension, value in dimensions.items():
                by_dimension[dimension][value][outcome] += 1

            if outcome == "survived":
                reached = _coverage_reached(row["_log"])
                coverage_without_kill[
                    "explored_and_survived" if reached
                    else "never_found_new_coverage" if reached is False
                    else "coverage_unknown"
                ] += 1
        per_set[directory.name] = set_counts

    def summarise(counter: Counter) -> dict[str, Any]:
        total = sum(counter.values())
        return {
            "targets": total,
            "counts": {k: counter.get(k, 0) for k in OUTCOMES if counter.get(k)},
            "kill_rate": round(counter.get("killed", 0) / total, 4) if total else None,
        }

    return {
        "schema_version": "oneiros_atheris_failure_matrix_v1",
        "sealed_final_test_accessed": False,
        "sets": [d.name for d in directories],
        "per_set": {name: summarise(counts) for name, counts in per_set.items()},
        "overall": summarise(totals),
        "kill_kinds": dict(kill_kinds.most_common()),
        "by_dimension": {
            dimension: {
                value: summarise(counts)
                for value, counts in sorted(
                    values.items(),
                    key=lambda item: -sum(item[1].values()))
            }
            for dimension, values in by_dimension.items()
        },
        "survivors_by_search_progress": dict(coverage_without_kill),
        "interpretation": (
            "a survivor that never found new coverage was not searching, which "
            "is an adapter or harness problem. A survivor that explored and "
            "still could not distinguish the mutant is either a genuinely hard "
            "target or an equivalent mutant, and only sampling can tell those "
            "apart."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", nargs="+", type=Path)
    parser.add_argument("--tasks", type=Path,
                        default=ROOT / "results" / "v4_2_atheris_tasks_val.json")
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    report = build(arguments.directories, arguments.tasks)
    write_json(arguments.output, report)
    print(json.dumps({k: v for k, v in report.items() if k != "by_dimension"},
                     indent=2)[:1200])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
