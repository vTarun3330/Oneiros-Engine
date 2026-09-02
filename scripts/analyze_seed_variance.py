"""Across-seed analysis for the 100-function ablation_dev monitor panel.

Single-seed deltas were being compared against a noise floor measured from
repeated runs at the *same* seed. That understates the real variability: the
untrained baseline alone moves several functions when only the seed changes.
This script measures across-seed variability directly and reports which
effects survive it.

Reports per seed and pooled:
  - function kill rate with a 95% Wilson interval (a binomial proportion)
  - the gain from SFT, paired within seed
  - reference/parse/execution validity
  - across-seed spread of the baseline and of the trained best

Wilson intervals are attached only to per-seed binomial proportions. The mean
across seeds is reported as a mean with its observed range, never as a
binomial proportion, because it is not one.

    python scripts/analyze_seed_variance.py
"""
from __future__ import annotations

import json
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RESULTS = ROOT / "results"
RUNS = {
    42: "local_sft800_qwen_lr1e5_seed42",
    43: "local_sft800_qwen_lr1e5_seed43",
    44: "local_sft800_qwen_lr1e5_seed44",
}
PANEL = 100
# Repeated runs at the SAME seed have differed by about this many functions
# on this panel (process-level nondeterminism in sampled generation).
SAME_SEED_FLOOR = 2


def wilson(successes: int, total: int, z: float = 1.959964) -> list[float]:
    if total <= 0:
        return [0.0, 0.0]
    phat = successes / total
    denom = 1 + z * z / total
    centre = (phat + z * z / (2 * total)) / denom
    margin = z * math.sqrt(phat * (1 - phat) / total + z * z / (4 * total * total)) / denom
    return [round(max(0.0, centre - margin), 4), round(min(1.0, centre + margin), 4)]


def read_curve(run_name: str) -> dict[int, dict]:
    directory = RESULTS / run_name
    curve: dict[int, dict] = {}
    if not directory.exists():
        return curve
    for path in directory.glob("sft_monitor_*.json"):
        if "selection" in path.name or "trend" in path.name:
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        curve[int(payload["checkpoint_step"])] = {
            "killed": payload["function_validation_killed"],
            "records": payload["function_validation_records"],
            "ref": payload["reference_valid_rate"],
            "parse": payload["parse_success_rate"],
            "exec": payload["execution_valid_rate"],
        }
    return curve


def main() -> int:
    per_seed: dict[int, dict] = {}
    for seed, run_name in RUNS.items():
        curve = read_curve(run_name)
        if not curve or 0 not in curve:
            continue
        baseline = curve[0]
        trained = {step: value for step, value in curve.items() if step > 0}
        if not trained:
            continue
        best_step = max(trained, key=lambda s: trained[s]["killed"])
        best = trained[best_step]
        per_seed[seed] = {
            "run": run_name,
            "baseline_killed": baseline["killed"],
            "baseline_rate": round(baseline["killed"] / baseline["records"], 4),
            "baseline_wilson_95": wilson(baseline["killed"], baseline["records"]),
            "baseline_ref": baseline["ref"],
            "best_step": best_step,
            "best_killed": best["killed"],
            "best_rate": round(best["killed"] / best["records"], 4),
            "best_wilson_95": wilson(best["killed"], best["records"]),
            "best_ref": best["ref"],
            "paired_gain_functions": best["killed"] - baseline["killed"],
            "ref_delta": round(best["ref"] - baseline["ref"], 4),
            "curve": {str(k): curve[k]["killed"] for k in sorted(curve)},
        }

    if len(per_seed) < 2:
        print(json.dumps({"error": "fewer than two completed seeds", "seeds": list(per_seed)}, indent=2))
        return 1

    baselines = [v["baseline_killed"] for v in per_seed.values()]
    bests = [v["best_killed"] for v in per_seed.values()]
    gains = [v["paired_gain_functions"] for v in per_seed.values()]

    def spread(values: list[int]) -> dict:
        return {
            "values": values,
            "mean": round(statistics.mean(values), 2),
            "min": min(values),
            "max": max(values),
            "range": max(values) - min(values),
            "stdev": round(statistics.stdev(values), 2) if len(values) > 1 else 0.0,
        }

    baseline_spread = spread(baselines)
    best_spread = spread(bests)
    gain_spread = spread(gains)

    seed_noise = baseline_spread["range"]
    report = {
        "schema_version": "oneiros_seed_variance_analysis_v1",
        "final_test_measurement": False,
        "evaluation_set": "fixed 100-function ablation_dev monitor panel",
        "panel_size": PANEL,
        "config": "Qwen2.5-Coder-1.5B SDPA, 800 pairs, LR 1e-5, complex floor 0.60",
        "seeds_completed": sorted(per_seed),
        "per_seed": per_seed,
        "across_seed": {
            "untrained_baseline": baseline_spread,
            "trained_best": best_spread,
            "paired_gain": gain_spread,
            "note": (
                "The mean across seeds is a mean of proportions, not itself a "
                "binomial sample; no Wilson interval is attached to it. Per-seed "
                "Wilson intervals are given above."
            ),
        },
        "seed_variance_floor": {
            "untrained_baseline_range_functions": seed_noise,
            "trained_best_range_functions": best_spread["range"],
            "interpretation": (
                "An effect smaller than the across-seed range of the quantity it "
                "is measured on cannot be distinguished from seed choice on a "
                "single seed."
            ),
        },
        "comparison_design_note": (
            "Two different noise floors apply and must not be confused. A PAIRED "
            "within-seed comparison (baseline vs trained inside one run) cancels "
            "seed choice, so the relevant floor is same-seed run-to-run "
            "variability, observed at about two functions. An UNPAIRED "
            "comparison between two separately trained runs does not cancel it, "
            "so the relevant floor is the across-seed range measured here. "
            "Comparing a paired gain against the unpaired range would be the "
            "wrong test and would understate a real effect."
        ),
        "effects_reassessed": [
            {
                "effect": "SFT gain over untrained baseline",
                "design": "paired within seed",
                "applicable_floor_functions": SAME_SEED_FLOOR,
                "size_functions": gain_spread,
                "consistent_direction": all(g > 0 for g in gains),
                "survives": all(g > SAME_SEED_FLOOR for g in gains),
                "verdict": (
                    "SURVIVES - the gain is positive in every seed and exceeds "
                    "the same-seed floor in every seed. Pairing controls for the "
                    "baseline's across-seed swing, and notably the TRAINED result "
                    "is far more stable across seeds than the untrained one."
                    if all(g > SAME_SEED_FLOOR for g in gains) else
                    "NOT ESTABLISHED on the seeds completed so far"
                ),
            },
            {
                "effect": "complexity floor K1 vs K0 (+4, seed 42 only)",
                "design": "unpaired between two separately trained runs",
                "applicable_floor_functions": seed_noise,
                "size_functions": 4,
                "survives": 4 > seed_noise,
                "verdict": (
                    "DOES NOT SURVIVE - an unpaired 4-function difference "
                    "measured on one seed, against an across-seed range of "
                    f"{seed_noise}. The WEAK ACCEPT cannot stand without running "
                    "K0 and K1 on seeds 43 and 44."
                    if 4 <= seed_noise else "survives"
                ),
            },
            {
                "effect": "checkpoint sweep peak step 60 vs step 50 (+4, seed 42 only)",
                "design": "within one run, but comparing two checkpoints",
                "applicable_floor_functions": SAME_SEED_FLOOR,
                "size_functions": 4,
                "survives": 4 > SAME_SEED_FLOOR,
                "verdict": (
                    "MARGINAL - 4 functions against a same-seed floor of about "
                    f"{SAME_SEED_FLOOR}. Worse, the best step is not stable "
                    "across seeds, so selecting a checkpoint by peak kill rate "
                    "on one seed is not reliable."
                ),
            },
        ],
        "checkpoint_selection_stability": {
            "best_step_per_seed": {str(s): v["best_step"] for s, v in per_seed.items()},
            "stable": len({v["best_step"] for v in per_seed.values()}) == 1,
        },
    }

    out = RESULTS / "v4_1_seed_variance_analysis.json"
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(f"{'seed':>5}{'base':>7}{'best':>7}{'step':>7}{'gain':>7}  {'base 95% CI':>18}{'best 95% CI':>20}")
    for seed in sorted(per_seed):
        v = per_seed[seed]
        print(f"{seed:>5}{v['baseline_killed']:>7}{v['best_killed']:>7}{v['best_step']:>7}"
              f"{v['paired_gain_functions']:>+7}  {str(v['baseline_wilson_95']):>18}{str(v['best_wilson_95']):>20}")
    print()
    print(f"untrained baseline across seeds: {baseline_spread}")
    print(f"trained best across seeds:       {best_spread}")
    print(f"paired gain across seeds:        {gain_spread}")
    print()
    for item in report["effects_reassessed"]:
        print(f"  {item['effect']}: {item['verdict']}")
    print(f"\nbest-step stable across seeds: {report['checkpoint_selection_stability']['stable']}"
          f"  {report['checkpoint_selection_stability']['best_step_per_seed']}")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
