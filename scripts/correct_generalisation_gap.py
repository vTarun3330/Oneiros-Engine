"""Recompute the train/val gap on panels that share a benchmark mixture.

The reported "generalisation gap" was measured between a train panel of 893
humaneval and 7 curated records - zero mbpp - and a validation panel that is
92.6% mbpp. The two sides were never the same distribution, so their difference
was partly benchmark composition rather than generalisation.

This recomputes it from the full train-split generation, which covers 4650 mbpp
records, and reports three numbers instead of one:

* whole-panel, which is still mixture-dependent but now over comparable
  mixtures;
* mbpp like-for-like, which is the honest measure for a validation panel that
  is almost entirely mbpp;
* humaneval like-for-like, reported even though its validation side is only 56
  records, because omitting the inconvenient slice is how the original figure
  went wrong.

Legacy protocol on both sides, deliberately: the committed arm table was
measured under the first-assertion parser, so the correction has to be too.
Mixing protocols here would replace one composition error with a worse one.
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
from harness.evaluation_protocol import assert_comparable, protocol_of


def _panel(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("final_test_measurement") or payload.get("evaluation_split") == "test":
        raise SystemExit(str(path) + " is sealed final-test data; refusing")
    return list(payload.get("function_results") or []), payload


def _rate(rows: list[dict[str, Any]], benchmark: str | None = None):
    if benchmark is not None:
        rows = [r for r in rows if str(r.get("dataset_name")) == benchmark]
    if not rows:
        return None, 0
    killed = sum(1 for r in rows if r.get("killed"))
    return round(killed / len(rows), 6), len(rows)


def build(train: Path, val: Path, legacy_train: Path | None) -> dict[str, Any]:
    train_rows, train_payload = _panel(train)
    val_rows, val_payload = _panel(val)
    assert_comparable([train_payload, val_payload], ["train", "val"])

    benchmarks = sorted(
        {str(r.get("dataset_name")) for r in train_rows}
        & {str(r.get("dataset_name")) for r in val_rows})

    per_benchmark = {}
    for benchmark in benchmarks:
        train_rate, train_n = _rate(train_rows, benchmark)
        val_rate, val_n = _rate(val_rows, benchmark)
        per_benchmark[benchmark] = {
            "train_kill_rate": train_rate, "train_n": train_n,
            "val_kill_rate": val_rate, "val_n": val_n,
            "gap_points": (round((train_rate - val_rate) * 100, 2)
                           if train_rate is not None and val_rate is not None
                           else None),
        }

    train_pooled, train_total = _rate(train_rows)
    val_pooled, val_total = _rate(val_rows)

    superseded = None
    if legacy_train is not None and legacy_train.exists():
        old_rows, old_payload = _panel(legacy_train)
        old_rate, old_n = _rate(old_rows)
        superseded = {
            "artifact": legacy_train.as_posix().split("results/", 1)[-1],
            "train_kill_rate": old_rate,
            "train_n": old_n,
            "composition": dict(Counter(
                str(r.get("dataset_name")) for r in old_rows)),
            "reported_gap_points": (round((old_rate - val_pooled) * 100, 2)
                                    if old_rate is not None else None),
            "defect": (
                "the panel contains no mbpp at all, while validation is 92.6% "
                "mbpp, so this gap is a difference of mixtures"),
        }

    return {
        "schema_version": "oneiros_generalisation_gap_correction_v1",
        "sealed_final_test_accessed": False,
        "evaluation_protocol": protocol_of(train_payload),
        "protocol_note": (
            "legacy on both sides on purpose: the committed arm table was "
            "measured under the first-assertion parser"),
        "train_artifact": train.as_posix().split("results/", 1)[-1],
        "val_artifact": val.as_posix().split("results/", 1)[-1],
        "train_composition": dict(Counter(
            str(r.get("dataset_name")) for r in train_rows)),
        "val_composition": dict(Counter(
            str(r.get("dataset_name")) for r in val_rows)),
        "whole_panel": {
            "train_kill_rate": train_pooled, "train_n": train_total,
            "val_kill_rate": val_pooled, "val_n": val_total,
            "gap_points": round((train_pooled - val_pooled) * 100, 2),
        },
        "like_for_like": per_benchmark,
        "superseded_measurement": superseded,
        "reading": (
            "the whole-panel gap still depends on mixture; the mbpp row is the "
            "honest figure for a validation panel that is 92.6% mbpp. This "
            "corrects the BASE model only - every SFT arm's train@8 was also "
            "measured on the humaneval-only panel, so those gaps remain "
            "uncorrected and require a full-train evaluation per arm."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path,
                        default=ROOT / "results" / "local_base_qwen_train_full_s42"
                        / "base_validation_train_seed_42.json")
    parser.add_argument("--val", type=Path,
                        default=ROOT / "results" / "local_base_qwen_val_seed42"
                        / "base_validation_standard_seed_42.json")
    parser.add_argument("--legacy-train", type=Path,
                        default=ROOT / "results" / "local_base_qwen_train_seed42"
                        / "base_validation_train_smoke900_seed_42.json")
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    report = build(arguments.train, arguments.val, arguments.legacy_train)
    write_json(arguments.output, report)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
