"""How much does the selection panel overstate an arm?

ablation_dev is where checkpoints and settings were chosen, so it is
optimistically biased by construction. That is not a defect - it is why the
validation split is locked and why nothing selects on it. But every number
quoted from ablation_dev has to carry the size of that bias, and until it is
measured the size is a guess.

This pairs the two panels seed by seed. The arm is the same adapter and the
same prompts on both, so the difference between the gains is the selection
effect and nothing else.
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

PANELS = {
    "ablation_dev": {
        "base": "local_base_qwen_adev_seed{seed}/base_validation_ablation-dev_seed_{seed}.json",
        "arm": "{arm_run}/sft_validation_ablation-dev_seed_{seed}.json",
        "role": "selection panel; checkpoints and settings were chosen on it",
    },
    "val": {
        "base": "local_base_qwen_val_seed{seed}/base_validation_standard_seed_{seed}.json",
        "arm": "{arm_run}/sft_validation_standard_seed_{seed}.json",
        "role": "locked; never selected on",
    },
}


def _kill_at_8(path: Path) -> tuple[float, int] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return float(payload["kill_at_k"]["8"]["rate"]), int(
        payload["evaluation_split_records"])


def compare(arm_run: str, arm_name: str, seeds: list[int],
            results: Path = RESULTS) -> dict[str, Any]:
    panels: dict[str, Any] = {}
    for panel, layout in PANELS.items():
        rows = []
        for seed in seeds:
            base = _kill_at_8(results / layout["base"].format(seed=seed))
            arm = _kill_at_8(
                results / layout["arm"].format(seed=seed, arm_run=arm_run))
            if base is None or arm is None:
                continue
            rows.append({
                "seed": seed,
                "base_kill_at_8": round(base[0], 6),
                "arm_kill_at_8": round(arm[0], 6),
                "gain": round(arm[0] - base[0], 6),
                "targets": base[1],
            })
        panels[panel] = {
            "role": layout["role"],
            "seeds": rows,
            "mean_gain": round(sum(r["gain"] for r in rows) / len(rows), 6)
            if rows else None,
            "min_gain": round(min((r["gain"] for r in rows), default=0), 6)
            if rows else None,
            "max_gain": round(max((r["gain"] for r in rows), default=0), 6)
            if rows else None,
        }

    selection, locked = panels.get("ablation_dev"), panels.get("val")
    inflation = None
    disjoint = None
    if selection and locked and selection["mean_gain"] and locked["mean_gain"]:
        inflation = round(selection["mean_gain"] / locked["mean_gain"], 3)
        disjoint = selection["min_gain"] > locked["max_gain"]

    return {
        "schema_version": "oneiros_panel_selection_effect_v1",
        "arm": arm_name,
        "arm_run": arm_run,
        "sealed_final_test_accessed": False,
        "seeds": seeds,
        "panels": panels,
        "selection_panel_inflation_factor": inflation,
        "per_seed_ranges_disjoint": disjoint,
        "interpretation": (
            "the same adapter and prompts on both panels, so the difference "
            "between the gains is the selection effect. A figure quoted from "
            "ablation_dev is not a generalisation estimate and must not be "
            "reported as one."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm-run", default="local_sft_relearn_v2_seed42")
    parser.add_argument("--arm-name", default="relearning")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--output", type=Path,
                        default=RESULTS / "v4_2_panel_selection_effect.json")
    arguments = parser.parse_args()

    report = compare(arguments.arm_run, arguments.arm_name, arguments.seeds)
    write_json(arguments.output, report)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
