"""One results table, rebuilt from the evaluation artifacts every time.

Every figure in the writeup comes from here. Nothing is retyped, because the
one report on this project that WAS typed by hand ended up quoting a base
control from a different run than the arms it was comparing.

What it covers, per arm: kill@8 with its Wilson interval on the locked
validation panel and on the selection panel, the per-benchmark split, and the
train-split fit where one was measured. Arms missing an evaluation are named
in `arms_missing` rather than dropped, so a gap in the table is visible rather
than silent.

It refuses to read anything marked as sealed final-test data.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness.corpus import write_json

RESULTS = ROOT / "results"

#: arm -> (run directory, human description). The base model is separated out
#: because every gain is quoted against it.
BASE_RUNS = {
    "val": "local_base_qwen_val_seed{seed}/base_validation_standard_seed_{seed}.json",
    "ablation_dev": "local_base_qwen_adev_seed{seed}/base_validation_ablation-dev_seed_{seed}.json",
    "train": "local_base_qwen_train_seed42/base_validation_train_smoke900_seed_42.json",
}

ARMS: dict[str, dict[str, str]] = {
    "full_density": {
        "run": "local_sft_mm_full_v2_seed42",
        "description": "multi-mutant supervision at full density",
    },
    "relearning": {
        "run": "local_sft_relearn_v2_seed42",
        "description": "hard-example relearning on train-split failures",
    },
    "regularised_dropout_010_unpromoted": {
        "run": "local_sft_reg_mild_s42",
        "description": "mild regularisation; failed the promotion gate, staged to be measurable",
    },
    "long_two_epoch": {
        "run": "local_sft_long_s42",
        "description": "two epochs instead of one",
    },
    "relearning_checkpoint_100": {
        "run": "local_sft_relearn_ckpt100",
        "description": "intermediate checkpoint of the relearning arm",
    },
    "v4_2_corpus": {
        "run": "local_sft_v42_s42",
        "description": "corpus grown with 26 verified humaneval records",
    },
}

ARM_FILES = {
    "val": "{run}/sft_validation_standard_seed_{seed}.json",
    "ablation_dev": "{run}/sft_validation_ablation-dev_seed_{seed}.json",
    "train": "{run}/sft_validation_train_smoke900_seed_42.json",
}


def _read(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("final_test_measurement") or payload.get("evaluation_split") == "test":
        raise SystemExit(f"{path} is sealed final-test data; refusing to read it")
    return payload


def _summary(payload: dict[str, Any]) -> dict[str, Any]:
    entry = payload["kill_at_k"]["8"]
    by_benchmark: dict[str, dict[str, Any]] = {}
    for result in payload.get("function_results") or []:
        benchmark = str(result.get("dataset_name") or "unknown")
        bucket = by_benchmark.setdefault(benchmark, {"functions": 0, "killed": 0})
        bucket["functions"] += 1
        bucket["killed"] += 1 if result.get("killed") else 0
    for bucket in by_benchmark.values():
        bucket["kill_rate"] = round(bucket["killed"] / bucket["functions"], 6)
    return {
        "kill_at_8": round(float(entry["rate"]), 6),
        "wilson_95": [round(v, 6) for v in entry.get("wilson_95", [])],
        "targets": int(payload["evaluation_split_records"]),
        "by_benchmark": dict(sorted(by_benchmark.items())),
    }


def build(seed: int, results: Path = RESULTS) -> dict[str, Any]:
    base: dict[str, Any] = {}
    for panel, layout in BASE_RUNS.items():
        payload = _read(results / layout.format(seed=seed))
        if payload is not None:
            base[panel] = _summary(payload)

    arms: dict[str, Any] = {}
    missing: dict[str, list[str]] = {}
    for name, meta in ARMS.items():
        entry: dict[str, Any] = {"description": meta["description"], "run": meta["run"]}
        absent = []
        for panel, layout in ARM_FILES.items():
            payload = _read(results / layout.format(run=meta["run"], seed=seed))
            if payload is None:
                absent.append(panel)
                continue
            summary = _summary(payload)
            if panel in base:
                summary["gain_over_base"] = round(
                    summary["kill_at_8"] - base[panel]["kill_at_8"], 6)
            entry[panel] = summary
        if "val" in entry and "train" in entry:
            entry["generalisation_gap_points"] = round(
                (entry["train"]["kill_at_8"] - entry["val"]["kill_at_8"]) * 100, 2)
        arms[name] = entry
        if absent:
            missing[name] = absent

    base_gap = None
    if "train" in base and "val" in base:
        base_gap = round(
            (base["train"]["kill_at_8"] - base["val"]["kill_at_8"]) * 100, 2)

    return {
        "schema_version": "oneiros_results_table_v1",
        "seed": seed,
        "sealed_final_test_accessed": False,
        "metric": "kill@8 over synthetic mutation targets",
        "panels": {
            "val": "locked; never selected on; the number that counts",
            "ablation_dev": "selection panel; overstates arms, measured at 3.2x",
            "train": "first 900 train records; fit, not generalisation",
        },
        "base": base,
        "base_generalisation_gap_points": base_gap,
        "arms": arms,
        "arms_missing": missing,
        "repository_targets_evaluated": 0,
        "caveats": [
            "every kill@8 here is over SYNTHETIC targets; the 24 val "
            "repository targets are not evaluated and are excluded from all "
            "reported rates",
            "humaneval figures carry the prompt-copying caveat: 46% of "
            "ablation_dev humaneval records state a value their mutant does "
            "not produce",
            "ablation_dev gains are inflated ~3.2x relative to locked val",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path,
                        default=RESULTS / "v4_2_results_table.json")
    arguments = parser.parse_args()

    table = build(arguments.seed)
    write_json(arguments.output, table)

    print(f"{'arm':<36} {'val@8':>8} {'gain':>8} {'adev@8':>8} {'train@8':>8} {'gap':>7}")
    base = table["base"]
    print(f"{'base (untrained)':<36} {base['val']['kill_at_8']:>8.4f} "
          f"{'-':>8} {base['ablation_dev']['kill_at_8']:>8.4f} "
          f"{base['train']['kill_at_8']:>8.4f} "
          f"{table['base_generalisation_gap_points']:>7.2f}")
    for name, arm in table["arms"].items():
        val = arm.get("val", {})
        adev = arm.get("ablation_dev", {})
        train = arm.get("train", {})
        print(f"{name:<36} "
              f"{val.get('kill_at_8', float('nan')):>8.4f} "
              f"{val.get('gain_over_base', float('nan')):>+8.4f} "
              f"{adev.get('kill_at_8', float('nan')):>8.4f} "
              f"{train.get('kill_at_8', float('nan')):>8.4f} "
              f"{arm.get('generalisation_gap_points', float('nan')):>7.2f}")
    if table["arms_missing"]:
        print()
        for name, panels in table["arms_missing"].items():
            print(f"missing: {name} has no {', '.join(panels)} evaluation")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
