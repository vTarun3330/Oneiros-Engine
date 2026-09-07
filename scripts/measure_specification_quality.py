"""Measure how well each benchmark's targets are specified, per split.

The evaluation panel is 92.6% mbpp, and every arm has gained ~+10 points on
humaneval against ~+3 on mbpp. The per-benchmark taxonomy says why: the
dominant mbpp failure is wrong_expected_value - a candidate that asserts
behaviour the CORRECT reference does not exhibit. That is not a test-design
failure. It means the model could not determine what the function should
return.

This measures the input side of that. A worked example in the specification
tells the model an exact input/output pair; without one it must infer the
intended behaviour from prose alone. If the two benchmarks differ sharply
here, the mbpp ceiling is a property of the specifications rather than
something more mbpp supervision can move.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness.corpus import write_json

#: A worked example states an input and its expected output. The doctest form,
#: the arrow forms HumanEval uses in prose, and an explicit "For example:"
#: lead-in all qualify; a bare mention of the word "example" does not.
WORKED_EXAMPLE = re.compile(r">>>|==>|=>|\bfor example\b|\bExample[s]?:", re.IGNORECASE)


def _benchmark(record_id: str) -> str | None:
    lowered = record_id.lower()
    for name in ("mbpp", "humaneval"):
        if name in lowered:
            return name
    return None


def measure(corpus_dir: Path, splits_wanted: tuple[str, ...]) -> dict[str, Any]:
    records = json.loads((corpus_dir / "records.json").read_text(encoding="utf-8"))
    splits = json.loads((corpus_dir / "splits.json").read_text(encoding="utf-8"))
    by_id = {str(record["id"]): record for record in records}

    report: dict[str, Any] = {}
    for split in splits_wanted:
        buckets: dict[str, list[tuple[bool, int, int]]] = {}
        for record_id in splits.get(split) or []:
            record = by_id.get(str(record_id))
            if record is None:
                continue
            benchmark = _benchmark(str(record_id))
            if benchmark is None:
                continue
            specification = str(record.get("specification") or "")
            buckets.setdefault(benchmark, []).append((
                bool(WORKED_EXAMPLE.search(specification)),
                len(specification),
                specification.count("\n") + 1,
            ))
        report[split] = {
            benchmark: {
                "records": len(rows),
                "with_worked_example": sum(1 for has, _, _ in rows if has),
                "worked_example_rate": round(
                    sum(1 for has, _, _ in rows if has) / len(rows), 4),
                "median_specification_chars": int(
                    statistics.median(chars for _, chars, _ in rows)),
                "median_specification_lines": int(
                    statistics.median(lines for _, _, lines in rows)),
            }
            for benchmark, rows in sorted(buckets.items()) if rows
        }
    return {
        "schema_version": "oneiros_specification_quality_v1",
        "corpus_dir": corpus_dir.name,
        "sealed_final_test_accessed": False,
        "measures": "presence of an input/output example in the specification text",
        "by_split": report,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus", type=Path,
        default=ROOT / "data" / "corpus" / "v4_2_balanced_expansion_candidate")
    parser.add_argument(
        "--splits", nargs="+", default=["train", "ablation_dev", "val"],
        help="the sealed test split is deliberately not a default")
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "results" / "v4_2_specification_quality.json")
    arguments = parser.parse_args()

    if "test" in arguments.splits:
        raise SystemExit("refusing to profile the sealed test split")

    report = measure(arguments.corpus, tuple(arguments.splits))
    write_json(arguments.output, report)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
