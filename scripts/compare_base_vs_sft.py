"""Paired base-model versus SFT comparison on the locked validation split.

The 100-function ablation_dev monitor panel and the 757-function validation
split answer different questions, and conflating them has been the single
biggest source of over-claiming in this project.  This script reports only the
paired validation comparison: for each evaluation seed, the same split, the
same protocol, the same candidate budget, base model against adapter.

Reporting rules enforced here:

* Wilson intervals are attached to per-seed binomial proportions only.  The
  across-seed summary is a mean of proportions and is reported with its range,
  never as a binomial sample.
* A paired within-seed delta cancels seed choice, so it is judged against
  same-seed run-to-run variability, not against the across-seed range.
* Reference validity, parse, and execution are reported beside the kill rate.
  An adapter that kills marginally more while producing more invalid tests has
  not straightforwardly improved.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness.corpus import write_json
from metrics.research_evaluation import wilson_interval
from utils.reproducibility import source_tree_sha256


#: Same-seed run-to-run variability measured earlier on the 100-function panel,
#: expressed as a fraction of the 757-function validation split.
PAIRED_NOISE_FLOOR_FUNCTIONS = 2


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("final_test_measurement"):
        raise SystemExit(f"Refusing {path}: sealed final-test measurement")
    if str(payload.get("evaluation_split", "")) == "test":
        raise SystemExit(f"Refusing {path}: evaluates the sealed test split")
    return payload


def _arm(run: str, pattern: str) -> dict[int, dict[str, Any]]:
    """Collect one arm's per-seed artifacts.

    ``run`` may name a single results directory or a glob across several: each
    base-model seed was launched as its own run, so its artifacts live in
    sibling directories rather than one.
    """
    found: dict[int, dict[str, Any]] = {}
    directories = sorted((ROOT / "results").glob(run)) if any(
        character in run for character in "*?["
    ) else [ROOT / "results" / run]
    paths = [
        path for directory in directories if directory.exists()
        for path in sorted(directory.glob(pattern))
    ]
    for path in paths:
        if ".progress." in path.name:
            continue
        payload = _load(path)
        seed = payload.get("seed")
        if seed is None:
            stem = path.stem.rsplit("_", 1)[-1]
            seed = int(stem) if stem.isdigit() else None
        if seed is None:
            continue
        found[int(seed)] = payload
    return found


def _summary(payload: dict[str, Any]) -> dict[str, Any]:
    killed = payload.get("functions_killed")
    evaluated = payload.get("functions_evaluated")
    rate = float(payload["function_kill_rate"])
    if killed is None or evaluated is None:
        # Recover the counts the rate was computed from so the interval is a
        # real binomial interval rather than a rate reported with no n.
        evaluated = int(payload.get("validation_functions") or 0)
        killed = round(rate * evaluated) if evaluated else None
    return {
        "function_kill_rate": round(rate, 6),
        "functions_killed": killed,
        "functions_evaluated": evaluated,
        "wilson_95": payload.get("function_kill_rate_wilson_95")
        or (wilson_interval(killed, evaluated) if evaluated else None),
        "reference_valid_rate": payload.get("reference_valid_rate"),
        "parse_success_rate": payload.get("parse_success_rate"),
        "execution_valid_rate": payload.get("execution_valid_rate"),
        "candidate_kill_rate": payload.get("candidate_kill_rate"),
        "end_to_end_candidate_kill_rate": payload.get("end_to_end_candidate_kill_rate"),
    }


def compare(base_run: str, arms: dict[str, str]) -> dict[str, Any]:
    base = _arm(base_run, "base_validation_*.json")
    if not base:
        raise SystemExit(f"no base validation artifacts under results/{base_run}")

    report: dict[str, Any] = {
        "schema_version": "oneiros_base_vs_sft_comparison_v1",
        "source_tree_sha256": source_tree_sha256(ROOT),
        "evaluation_split": "val",
        "sealed_final_test_accessed": False,
        "design": "paired within evaluation seed; same split, protocol, and budget",
        "base_run": base_run,
        "base_per_seed": {
            str(seed): _summary(payload) for seed, payload in sorted(base.items())
        },
        "arms": {},
    }
    base_rates = [float(payload["function_kill_rate"]) for _, payload in sorted(base.items())]
    report["base_across_seed"] = {
        "seeds": sorted(base),
        "mean": round(statistics.fmean(base_rates), 6),
        "range": round(max(base_rates) - min(base_rates), 6),
        "stdev": round(statistics.stdev(base_rates), 6) if len(base_rates) > 1 else None,
        "note": "mean of proportions; reported with range, never as a binomial sample",
    }

    for name, run in arms.items():
        arm = _arm(run, "sft_validation_*.json")
        if not arm:
            report["arms"][name] = {"run": run, "status": "no artifacts found"}
            continue
        shared = sorted(set(arm) & set(base))
        deltas = [
            float(arm[seed]["function_kill_rate"]) - float(base[seed]["function_kill_rate"])
            for seed in shared
        ]
        reference_deltas = [
            float(arm[seed].get("reference_valid_rate") or 0)
            - float(base[seed].get("reference_valid_rate") or 0)
            for seed in shared
        ]
        evaluated = int(
            _summary(arm[shared[0]])["functions_evaluated"] or 0
        ) if shared else 0
        floor = (
            PAIRED_NOISE_FLOOR_FUNCTIONS / evaluated if evaluated else None
        )
        report["arms"][name] = {
            "run": run,
            "per_seed": {
                str(seed): _summary(payload) for seed, payload in sorted(arm.items())
            },
            "paired_seeds": shared,
            "paired_kill_rate_delta": {
                str(seed): round(delta, 6) for seed, delta in zip(shared, deltas)
            },
            "paired_kill_rate_delta_mean": round(statistics.fmean(deltas), 6) if deltas else None,
            "positive_in_every_paired_seed": all(delta > 0 for delta in deltas) if deltas else None,
            "paired_noise_floor_rate": round(floor, 6) if floor else None,
            "exceeds_paired_noise_floor_in_every_seed": (
                all(delta > floor for delta in deltas) if deltas and floor else None
            ),
            "paired_reference_validity_delta": {
                str(seed): round(delta, 6)
                for seed, delta in zip(shared, reference_deltas)
            },
            "reference_validity_regressed_in_every_seed": (
                all(delta < 0 for delta in reference_deltas)
                if reference_deltas else None
            ),
            "across_seed_mean": round(statistics.fmean(
                [float(payload["function_kill_rate"]) for payload in arm.values()]
            ), 6),
        }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-run", default="local_base_qwen_val_seed42")
    parser.add_argument(
        "--arm", action="append", default=[],
        metavar="NAME=RUN", help="an SFT arm to compare, e.g. final=local_sft800_qwen_final_seed42",
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "results" / "v4_2_base_vs_sft_validation.json",
    )
    arguments = parser.parse_args()

    arms: dict[str, str] = {}
    for item in arguments.arm:
        name, _, run = item.partition("=")
        arms[name] = run
    report = compare(arguments.base_run, arms)
    write_json(arguments.output, report)
    print(json.dumps({
        "base_across_seed": report["base_across_seed"],
        "arms": {
            name: {
                key: value for key, value in arm.items()
                if key in {
                    "paired_kill_rate_delta", "paired_kill_rate_delta_mean",
                    "positive_in_every_paired_seed",
                    "reference_validity_regressed_in_every_seed",
                    "across_seed_mean", "status",
                }
            }
            for name, arm in report["arms"].items()
        },
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
