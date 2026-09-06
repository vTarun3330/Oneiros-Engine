"""Aggregate actual-Atheris runs across seeds into one comparable artifact.

Every Atheris figure reported before this was seed 42 alone, and it sits in the
headline claim that Oneiros beats actual Atheris at both budgets. A single draw
from a stochastic system is not a measurement of that system, and this project
already has a demonstration of why: the simulated coverage fuzzer spans
0.378-0.514 across its three seeds, a range wider than the entire SFT effect
under study.

The aggregate reports mean, range and per-seed rates so the reader can see the
spread rather than take a mean on trust, and separates semantic kills (wrong
answer against the reference) from crash kills, because two runs can reach the
same total by different routes - seeds 42 and 43 both kill 116 targets with
different semantic/crash splits, which is what proves they are independent runs
rather than a duplicated artifact.
"""
from __future__ import annotations

import argparse
import collections
import glob
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


def score_directory(directory: Path) -> dict[str, Any]:
    """Score one seed's per-target result files.

    Every target writes its own file because ``atheris.Setup()`` may be called
    only once per process and libFuzzer ends the process itself. A missing or
    unreadable file is counted as incomplete rather than skipped silently: a
    target that never ran is not a target that survived, and conflating them
    would flatter Atheris's denominator.
    """
    killed = total = incomplete = 0
    kinds: collections.Counter[str] = collections.Counter()
    seeds: set[int] = set()
    max_runs: set[int] = set()

    for path in sorted(glob.glob(str(directory / "task_*.json"))):
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
        except Exception:
            incomplete += 1
            continue
        if payload.get("seed") is not None:
            seeds.add(int(payload["seed"]))
        if payload.get("max_runs") is not None:
            max_runs.add(int(payload["max_runs"]))
        for row in payload.get("results", []):
            total += 1
            if row.get("outcome") == "killed":
                killed += 1
                kinds[str(row.get("kill_kind"))] += 1

    low, high = wilson_interval(killed, max(1, total))
    return {
        "directory": directory.as_posix(),
        "seed": sorted(seeds)[0] if len(seeds) == 1 else sorted(seeds),
        "max_runs": sorted(max_runs)[0] if len(max_runs) == 1 else sorted(max_runs),
        "targets": total,
        "killed": killed,
        "kill_rate": round(killed / max(1, total), 6),
        "kill_rate_wilson_95": [low, high],
        "kill_kinds": dict(kinds),
        "incomplete_task_files": incomplete,
    }


def aggregate(arms: dict[str, list[Path]]) -> dict[str, Any]:
    report: dict[str, Any] = {
        "schema_version": "oneiros_atheris_seed_aggregate_v1",
        "source_tree_sha256": source_tree_sha256(ROOT),
        "sealed_final_test_accessed": False,
        "system": "actual Atheris, not the simulated coverage fuzzer",
        "arms": {},
    }
    for name, directories in arms.items():
        seeds = [score_directory(directory) for directory in directories]
        rates = [row["kill_rate"] for row in seeds]
        report["arms"][name] = {
            "by_seed": seeds,
            "seeds": [row["seed"] for row in seeds],
            "mean_kill_rate": round(sum(rates) / len(rates), 6) if rates else None,
            "range_across_seeds": (
                round(max(rates) - min(rates), 6) if len(rates) > 1 else None
            ),
            "min_kill_rate": min(rates) if rates else None,
            "max_kill_rate": max(rates) if rates else None,
            "total_incomplete_task_files": sum(
                row["incomplete_task_files"] for row in seeds
            ),
        }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arm", action="append", required=True, metavar="NAME=DIR[,DIR...]",
        help="an arm and the per-seed result directories that make it up",
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "results" / "v4_2_atheris_seed_aggregate.json",
    )
    arguments = parser.parse_args()

    arms: dict[str, list[Path]] = {}
    for item in arguments.arm:
        name, _, paths = item.partition("=")
        arms[name] = [Path(p) for p in paths.split(",") if p]

    report = aggregate(arms)
    write_json(arguments.output, report)
    print(json.dumps({
        name: {
            "per_seed": [
                {"seed": row["seed"], "kill_rate": row["kill_rate"],
                 "killed": row["killed"], "targets": row["targets"]}
                for row in arm["by_seed"]
            ],
            "mean_kill_rate": arm["mean_kill_rate"],
            "range_across_seeds": arm["range_across_seeds"],
        }
        for name, arm in report["arms"].items()
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
