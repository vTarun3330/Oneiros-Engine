"""Kill@k cut by bug family, complexity tier and benchmark - not one headline.

A single pooled kill@8 on a panel that is 92.6% mbpp is the mbpp number wearing
a disguise, and the same is true one level down: a gain concentrated in one bug
family or one complexity tier is not the general improvement the headline
implies.

This re-executes nothing. It reads recorded per-function outcomes and regroups
them, so every slice reconciles with the pooled figure by construction - the
report prints the pooled number back out as a check rather than asking the
reader to assume it.

Paired against a baseline artifact, it also reports the per-slice DELTA, with
the slice size beside it, because a +20-point gain on a slice of five targets
is noise wearing a result's clothing.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness.corpus import write_json

DIMENSIONS = ("dataset_name", "bug_family", "complexity_tier")

#: Slices smaller than this are reported but marked, because a delta on a
#: handful of targets is not evidence of anything.
MIN_INFORMATIVE_SLICE = 20


def _refuse_sealed(payload: dict[str, Any], path: Path) -> None:
    if payload.get("final_test_measurement") or payload.get("evaluation_split") == "test":
        raise SystemExit(str(path) + " is a sealed final-test artifact; refusing")


def _rows(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    _refuse_sealed(payload, path)
    return list(payload.get("function_results") or []), payload


def _slice(rows: list[dict[str, Any]], dimension: str) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(dimension) or "unknown")].append(row)
    out: dict[str, dict[str, Any]] = {}
    for key, group in grouped.items():
        killed = sum(1 for row in group if row.get("killed"))
        out[key] = {
            "n": len(group),
            "killed": killed,
            "kill_rate": round(killed / len(group), 6),
        }
    return dict(sorted(out.items(), key=lambda item: -item[1]["n"]))


def build(artifact: Path, baseline: Path | None) -> dict[str, Any]:
    rows, payload = _rows(artifact)
    base_rows: list[dict[str, Any]] = []
    if baseline is not None:
        base_rows, base_payload = _rows(baseline)
        if base_payload.get("evaluation_split") != payload.get("evaluation_split"):
            raise SystemExit(
                "baseline is split " + str(base_payload.get("evaluation_split"))
                + " but artifact is split " + str(payload.get("evaluation_split"))
                + "; a cross-panel delta is not a delta")

    by_id = {str(row.get("record_id")): row for row in base_rows}
    killed = sum(1 for row in rows if row.get("killed"))

    slices: dict[str, Any] = {}
    for dimension in DIMENSIONS:
        current = _slice(rows, dimension)
        if base_rows:
            # Restrict the baseline to the SAME record ids, so a slice delta is
            # paired rather than two independent panels compared by name.
            paired = [by_id[str(row.get("record_id"))] for row in rows
                      if str(row.get("record_id")) in by_id]
            previous = _slice(paired, dimension)
            for key, block in current.items():
                before = previous.get(key)
                block["baseline_kill_rate"] = (
                    before["kill_rate"] if before else None)
                block["delta"] = (
                    round(block["kill_rate"] - before["kill_rate"], 6)
                    if before else None)
                block["informative"] = block["n"] >= MIN_INFORMATIVE_SLICE
        slices[dimension] = current

    return {
        "schema_version": "oneiros_kill_rate_slices_v1",
        "artifact": artifact.as_posix().split("results/", 1)[-1],
        "baseline": (baseline.as_posix().split("results/", 1)[-1]
                     if baseline else None),
        "evaluation_split": payload.get("evaluation_split"),
        "sealed_final_test_accessed": False,
        "functions": len(rows),
        "pooled_kill_rate": round(killed / len(rows), 6) if rows else None,
        "min_informative_slice": MIN_INFORMATIVE_SLICE,
        "slices": slices,
        "reading": (
            "a delta on a slice below min_informative_slice is marked "
            "informative=false. Slices are paired on record id, so a delta "
            "compares the same targets rather than two similarly named groups."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--baseline", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    report = build(arguments.artifact, arguments.baseline)
    write_json(arguments.output, report)
    print("pooled kill rate: " + str(report["pooled_kill_rate"])
          + "  (n=" + str(report["functions"]) + ", split="
          + str(report["evaluation_split"]) + ")")
    for dimension, block in report["slices"].items():
        print("")
        print("== " + dimension)
        for key, slice_block in block.items():
            line = ("  " + key.ljust(28) + " n=" + str(slice_block["n"]).rjust(4)
                    + "  kill=" + f"{slice_block['kill_rate']:.4f}")
            if slice_block.get("delta") is not None:
                line += "  delta=" + f"{slice_block['delta']:+.4f}"
                if not slice_block.get("informative"):
                    line += "  (small slice)"
            print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
