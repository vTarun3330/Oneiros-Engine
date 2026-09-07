"""Cut the failure taxonomy by benchmark, because the panel is not one panel.

The locked validation panel is 757 synthetic targets, and 701 of them are mbpp
against 56 humaneval. Every arm so far has moved humaneval a lot and mbpp
barely, so a pooled taxonomy averages the two into a number that describes
neither and hides where the ceiling actually is.

This re-uses the shipped classifier rather than reimplementing it, so the
categories cannot drift from the pooled report, and it never re-executes
anything: it reads recorded outcomes only. Sealed final-test artifacts are
refused the same way the pooled taxonomy refuses them.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness.corpus import write_json
from metrics.research_evaluation import classify_candidate_failure


def _refuse_sealed(payload: dict[str, Any], path: Path) -> None:
    if payload.get("final_test_measurement") or payload.get("evaluation_split") == "test":
        raise SystemExit(f"{path} is a sealed final-test artifact; refusing to read it")


def taxonomy(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    _refuse_sealed(payload, path)

    per_benchmark: dict[str, Counter] = {}
    functions: dict[str, dict[str, int]] = {}
    for record in payload.get("function_results") or []:
        benchmark = str(record.get("dataset_name") or record.get("source_name") or "unknown")
        counts = per_benchmark.setdefault(benchmark, Counter())
        stats = functions.setdefault(benchmark, {"functions": 0, "killed": 0})
        stats["functions"] += 1
        stats["killed"] += 1 if record.get("killed") else 0
        family = str(record.get("bug_family") or "unknown")
        for outcome in record.get("candidate_outcomes") or []:
            counts[classify_candidate_failure(outcome, family)] += 1

    report: dict[str, Any] = {}
    for benchmark, counts in sorted(per_benchmark.items()):
        total = sum(counts.values())
        stats = functions[benchmark]
        report[benchmark] = {
            "functions": stats["functions"],
            "functions_killed": stats["killed"],
            "function_kill_rate": round(stats["killed"] / stats["functions"], 6)
            if stats["functions"] else None,
            "candidates": total,
            "counts": dict(counts.most_common()),
            "rates": {k: round(v / total, 6) for k, v in counts.most_common()}
            if total else {},
        }
    return {
        "schema_version": "oneiros_failure_taxonomy_by_benchmark_v1",
        "artifact": path.as_posix().split("results/", 1)[-1],
        "evaluation_split": payload.get("evaluation_split"),
        "sealed_final_test_accessed": False,
        "adapter": payload.get("adapter"),
        "by_benchmark": report,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    reports = {path.as_posix().split("results/", 1)[-1]: taxonomy(path)
               for path in arguments.artifacts}
    write_json(arguments.output, {
        "schema_version": "oneiros_failure_taxonomy_by_benchmark_set_v1",
        "sealed_final_test_accessed": False,
        "artifacts": reports,
    })
    print(json.dumps(reports, indent=2)[:400])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
