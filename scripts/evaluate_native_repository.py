"""Evaluate generated repository tests inside real project checkouts (§36).

This is the harness the research plan requires before any real-repository kill
rate may be reported.  It takes tests the model generated for repository
records, injects each one into both the buggy and the fixed revision of the
project, and reports how they behaved.

    py -3.12 scripts/evaluate_native_repository.py \
        --generated results/<run>/repository_generations.json \
        --bugsinpy-root data/bugsinpy_v2_ingestion/BugsInPy \
        --repository-cache data/bugsinpy_v2_ingestion/repositories \
        --output results/<run>/native_repository_eval.json

``--generated`` is a JSON object mapping a repository record id to the ordered
list of generated test sources::

    {"bugsinpy::youtube-dl::12": ["def test_x():\\n    ...", "..."]}

Nothing here loads a model or touches a GPU; it re-runs already generated text.
A record whose project environment cannot be built is reported as
``environment_unavailable`` and is excluded from every success rate rather than
counted as a failure to discriminate.  Do not report a real-repository kill rate
from a run whose inconclusive share is large -- say how many records actually
executed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness.bugsinpy_v2 import BugsInPyTask, discover_tasks
from harness.native_repository_eval import (
    evaluate_generated_repository_tests,
    summarise_native_outcomes,
)
from utils.reproducibility import build_reproducibility_manifest


def load_generated(path: Path) -> Dict[str, List[str]]:
    """Load the record -> ordered generated tests mapping, preserving order."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("final_test_measurement") or payload.get("evaluation_split") == "test":
        raise SystemExit(f"Refusing {path}: sealed final-test material")
    generated = payload.get("generated_repository_tests", payload)
    if not isinstance(generated, Mapping):
        raise SystemExit(f"Refusing {path}: expected a record -> tests mapping")
    result: Dict[str, List[str]] = {}
    for record_id, tests in generated.items():
        if not isinstance(tests, list) or not all(isinstance(t, str) for t in tests):
            raise SystemExit(f"Record {record_id!r} must map to a list of test sources")
        result[str(record_id)] = list(tests)
    return result


def index_tasks(bugsinpy_root: Path) -> Dict[str, BugsInPyTask]:
    if not bugsinpy_root.exists():
        raise SystemExit(
            f"BugsInPy checkout not found at {bugsinpy_root}. "
            "Native evaluation cannot be simulated; provision it first."
        )
    return {task.id: task for task in discover_tasks(bugsinpy_root)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generated", type=Path, required=True)
    parser.add_argument(
        "--bugsinpy-root", type=Path,
        default=ROOT / "data" / "bugsinpy_v2_ingestion" / "BugsInPy",
    )
    parser.add_argument(
        "--repository-cache", type=Path,
        default=ROOT / "data" / "bugsinpy_v2_ingestion" / "repositories",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--runner-python", default=sys.executable)
    parser.add_argument(
        "--no-prepare-environment", action="store_true",
        help="Use the current interpreter instead of an isolated project environment",
    )
    parser.add_argument(
        "--max-records", type=int, default=0,
        help="Evaluate at most this many records (0 evaluates every supplied record)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    generated = load_generated(args.generated)
    tasks = index_tasks(args.bugsinpy_root)
    args.repository_cache.mkdir(parents=True, exist_ok=True)

    record_ids = list(generated)
    if args.max_records:
        record_ids = record_ids[: args.max_records]

    outcomes = []
    unknown_records: List[str] = []
    for record_id in record_ids:
        task = tasks.get(record_id)
        if task is None:
            unknown_records.append(record_id)
            continue
        outcomes.extend(evaluate_generated_repository_tests(
            task,
            args.repository_cache / task.project,
            generated[record_id],
            timeout=args.timeout,
            runner_python=args.runner_python,
            prepare_environment=not args.no_prepare_environment,
        ))

    summary: Dict[str, Any] = summarise_native_outcomes(outcomes)
    summary["requested_records"] = len(record_ids)
    summary["unresolved_records"] = unknown_records
    summary["generated_source"] = str(args.generated).replace("\\", "/")
    summary["sealed_final_test_accessed"] = False
    summary["reproducibility"] = build_reproducibility_manifest(ROOT, "", "")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    printable = {key: value for key, value in summary.items() if key != "outcomes"}
    print(json.dumps(printable, indent=2))
    if unknown_records:
        print(
            f"WARNING: {len(unknown_records)} record ids had no BugsInPy task and "
            "were not evaluated.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
