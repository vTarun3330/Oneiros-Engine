"""Build the hybrid multi-mutant supervised dataset for the training split.

For every synthetic function lineage in the development view this produces:

* one BROAD example - a single test function whose assertions were selected by
  a verified kill matrix over all sibling mutants of that lineage;
* zero or more TARGETED examples - one per sibling the broad example does not
  distinguish, so rare and hard mutations keep narrow supervision.

Every emitted completion is re-executed against the reference implementation
and against every sibling before it is written.  Nothing unverified is stored.

The output is a versioned, hashed artifact under ``data/training_views``.  The
immutable V4.1 corpus is read only.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness.corpus import write_json
from harness.multi_mutant_examples import (
    MULTI_MUTANT_BUILDER_VERSION,
    build_kill_matrix,
    build_multi_mutant_example,
)
from harness.training_data import extract_dataset_assertions
from utils.reproducibility import source_tree_sha256


DEFAULT_VIEW = (
    ROOT / "data" / "corpus" / "v4_1_research_hardened_candidate" / "development_view"
)
DEFAULT_OUTPUT = ROOT / "data" / "training_views" / "multi_mutant_v1"
UNIFIED_DATASET = ROOT / "data" / "unified_dataset.json"
SYNTHETIC_UPSTREAMS = {"humaneval", "mbpp"}

_MUTANT_SUFFIX = re.compile(r"_mut_\d+$")
_UNIFIED_BY_ID: dict[str, dict[str, Any]] = {}


def _load_unified() -> dict[str, dict[str, Any]]:
    global _UNIFIED_BY_ID
    if not _UNIFIED_BY_ID and UNIFIED_DATASET.exists():
        rows = json.loads(UNIFIED_DATASET.read_text(encoding="utf-8"))
        _UNIFIED_BY_ID = {str(row["id"]): row for row in rows}
    return _UNIFIED_BY_ID


def upstream_assertions(record: dict[str, Any]) -> list[str]:
    """The benchmark's own reference suite for this record's function.

    These are public test inputs shipped with HumanEval/MBPP.  They are the
    same class of information the specification already exposes, and they carry
    no mutation, patch, or reference-implementation content.
    """
    unified = _load_unified()
    upstream_id = str((record.get("provenance") or {}).get("upstream_record_id") or "")
    base = _MUTANT_SUFFIX.sub("", upstream_id)
    source = unified.get(base)
    if not source:
        return []
    return extract_dataset_assertions(
        source.get("test_cases") or [], str(record.get("entry_point") or ""),
    )


#: Targeted examples re-verify against every sibling, so generating one per
#: survivor is quadratic in lineage size - a 63-mutant lineage would cost
#: thousands of subprocess executions on its own. Rare-mutation coverage does
#: not need every survivor: it needs a bounded, deterministic sample of them.
MAX_TARGETED_EXAMPLES_PER_LINEAGE = 3


def _process_lineage(
    payload: tuple[str, list[dict[str, Any]], int, int, float, int],
) -> dict[str, Any]:
    lineage, records, max_assertions, min_assertions, timeout, max_targeted = payload
    started = time.perf_counter()
    extras = upstream_assertions(records[0])
    try:
        matrix = build_kill_matrix(records, timeout, extra_assertions=extras)
    except ValueError as exc:
        return {
            "lineage": lineage, "status": "skipped", "reason": str(exc),
            "broad": None, "targeted": [], "seconds": time.perf_counter() - started,
        }

    broad = build_multi_mutant_example(
        records, 0, max_assertions, timeout, matrix=matrix,
        min_assertions=min_assertions,
    )
    targeted: list[dict[str, Any]] = []
    survivors_total = 0
    if broad is not None and broad.verified:
        survivors = set(broad.surviving_mutant_ids)
        survivor_indexes = [
            index for index, mutant_id in enumerate(matrix.mutant_ids)
            if mutant_id in survivors
        ]
        survivors_total = len(survivor_indexes)
        # Deterministic bounded sample: prefer survivors whose mutation family
        # the broad example did not already cover, so the targeted budget buys
        # new defect kinds rather than more of the same one.
        covered_families = set(broad.covered_mutation_families)
        survivor_indexes.sort(
            key=lambda index: (
                matrix.mutant_families[index] in covered_families,
                matrix.mutant_families[index],
                matrix.mutant_ids[index],
            )
        )
        for index in survivor_indexes[:max_targeted]:
            # A sibling the broad test misses is exactly the rare or hard
            # mutation the plan says to keep narrow supervision for.
            example = build_multi_mutant_example(
                records, index, max_assertions, timeout, matrix=matrix,
                min_assertions=min_assertions,
            )
            if example is not None and example.verified:
                targeted.append(example.to_dict())
    return {
        "lineage": lineage,
        "status": "built" if broad is not None else "no_example",
        "reason": "" if broad is not None else "no_target_killing_assertion",
        "broad": broad.to_dict() if broad is not None else None,
        "targeted": targeted,
        "survivors_total": survivors_total,
        "targeted_cap": max_targeted,
        "candidate_assertions": len(matrix.assertions),
        "upstream_assertions": len(extras),
        "rejected_assertions": len(matrix.rejected),
        "siblings": len(records),
        "seconds": round(time.perf_counter() - started, 3),
    }


def build(
    view_dir: Path, output_dir: Path, split: str, max_assertions: int,
    min_assertions: int, timeout: float, workers: int, limit: int | None,
    max_targeted: int = MAX_TARGETED_EXAMPLES_PER_LINEAGE,
) -> dict[str, Any]:
    records = json.loads((view_dir / f"{split}.records.json").read_text(encoding="utf-8"))
    synthetic = [
        record for record in records
        if str((record.get("source") or {}).get("upstream")) in SYNTHETIC_UPSTREAMS
        and record.get("entry_point")
    ]
    lineages: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for record in synthetic:
        lineages[str(record["group_id"])].append(record)
    # Deterministic order: lineage key, then record id inside it.
    ordered = sorted(lineages)
    if limit:
        ordered = ordered[:limit]
    tasks = [
        (
            lineage,
            sorted(lineages[lineage], key=lambda item: str(item["id"])),
            max_assertions, min_assertions, timeout, max_targeted,
        )
        for lineage in ordered
    ]

    started = time.time()
    results: list[dict[str, Any]] = []
    if workers <= 1:
        for task in tasks:
            results.append(_process_lineage(task))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_process_lineage, task): task[0] for task in tasks}
            done = 0
            for future in as_completed(futures):
                results.append(future.result())
                done += 1
                if done % 10 == 0 or done == len(tasks):
                    print(
                        f"  {done}/{len(tasks)} lineages "
                        f"({time.time() - started:.0f}s)",
                        file=sys.stderr, flush=True,
                    )
    results.sort(key=lambda item: item["lineage"])

    broad = [item["broad"] for item in results if item.get("broad")]
    verified_broad = [item for item in broad if item["verified"]]
    targeted = [example for item in results for example in item["targeted"]]
    examples = verified_broad + targeted
    assertion_counts = [item["assertion_count"] for item in examples]
    kills = [item["mutants_killed"] for item in examples]

    summary = {
        "schema_version": "oneiros_multi_mutant_dataset_v1",
        "builder_version": MULTI_MUTANT_BUILDER_VERSION,
        "source_view": str(view_dir.relative_to(ROOT)).replace("\\", "/"),
        "split": split,
        "source_tree_sha256": source_tree_sha256(ROOT),
        "policy": {
            "max_assertions": max_assertions,
            "min_assertions": min_assertions,
            "execution_timeout_seconds": timeout,
            "one_broad_example_per_lineage": True,
            "targeted_examples_for_uncovered_siblings": True,
            "max_targeted_examples_per_lineage": max_targeted,
            "completion_reverified_against_reference_and_all_siblings": True,
            "prompt_shows_single_target_only": True,
        },
        "counts": {
            "synthetic_records": len(synthetic),
            "lineages": len(ordered),
            "lineages_built": sum(1 for item in results if item["status"] == "built"),
            "lineages_without_example": sum(
                1 for item in results if item["status"] != "built"
            ),
            "broad_examples": len(broad),
            "broad_examples_verified": len(verified_broad),
            "targeted_examples": len(targeted),
            "uncovered_siblings_total": sum(
                int(item.get("survivors_total") or 0) for item in results
            ),
            "total_examples": len(examples),
        },
        "quality": {
            "mean_assertions_per_example": round(
                sum(assertion_counts) / max(1, len(assertion_counts)), 3
            ),
            "mean_mutants_killed_per_example": round(
                sum(kills) / max(1, len(kills)), 3
            ),
            "examples_killing_at_least_four_siblings": sum(
                1 for value in kills if value >= 4
            ),
            "percent_killing_at_least_four_siblings": round(
                100.0 * sum(1 for value in kills if value >= 4) / max(1, len(kills)), 3
            ),
            "assertion_count_histogram": dict(sorted(
                collections.Counter(assertion_counts).items()
            )),
        },
        "verification_status_counts": dict(sorted(collections.Counter(
            item["verification_status"] for item in broad
        ).items())),
        "elapsed_seconds": round(time.time() - started, 1),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / f"{split}.examples.json", examples)
    write_json(output_dir / f"{split}.lineage_report.json", results)
    summary["examples_sha256"] = __import__("harness.corpus", fromlist=["sha256_file"]).sha256_file(
        output_dir / f"{split}.examples.json"
    )
    write_json(output_dir / f"{split}.manifest.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--view-dir", type=Path, default=DEFAULT_VIEW)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-assertions", type=int, default=8)
    parser.add_argument("--min-assertions", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--max-targeted", type=int, default=MAX_TARGETED_EXAMPLES_PER_LINEAGE,
    )
    arguments = parser.parse_args()

    summary = build(
        arguments.view_dir, arguments.output_dir, arguments.split,
        arguments.max_assertions, arguments.min_assertions,
        arguments.timeout, arguments.workers, arguments.limit,
        arguments.max_targeted,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
