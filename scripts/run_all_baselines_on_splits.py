"""
Run ALL baselines on the Train/Val/Test splits.

Outputs:
    results/all_baselines_splits.json
"""
import json
import time
import sys
import gc
import random
import warnings
from pathlib import Path
from collections import Counter

warnings.filterwarnings("ignore", category=SyntaxWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, str(Path(__file__).parent.parent))

from baseline.benchmark_runner import (
    TestCasesBaseline, RandomBaseline, StaticBaseline,
    GrammarBaseline, CoverageBaseline, safe_exec, kills_mutant, log
)

DATA_DIR    = Path(__file__).parent.parent / "data"
SPLITS_DIR  = DATA_DIR / "splits"
RESULTS_DIR = Path(__file__).parent.parent / "results"

TESTS_PER_MUTANT = 8
SEED = 42

def evaluate_baseline_on_split(split_name: str, pairs: list, baseline_name: str, baseline_obj) -> dict:
    log(f"  --> {baseline_name} Baseline  ({len(pairs):,} pairs)")
    t0 = time.time()
    random.seed(SEED)

    killed = 0
    survived = 0
    total_tests = 0

    for i, pair in enumerate(pairs):
        golden = pair["golden_code"]
        mutant = pair["mutant_code"]
        entry  = pair["entry_point"]
        tc     = pair.get("test_cases", []) if baseline_name == "TestCases" else []

        try:
            tests = baseline_obj.generate_tests(golden, entry, tc, num_tests=TESTS_PER_MUTANT)
        except Exception:
            tests = []

        mutant_killed = False
        for test in tests:
            total_tests += 1
            try:
                compile(test, "<t>", "exec")
            except SyntaxError:
                continue
            if kills_mutant(test, golden, mutant):
                mutant_killed = True
                break

        if mutant_killed:
            killed += 1
        else:
            survived += 1

        if (i + 1) % 500 == 0:
            gc.collect()

    elapsed = time.time() - t0
    kill_rate = killed / max(len(pairs), 1)

    log(f"      {kill_rate:.1%} kill rate ({killed}/{len(pairs)})  {elapsed:.1f}s")

    return {
        "kill_rate": round(kill_rate, 4),
        "killed": killed,
        "survived": survived,
        "total_tests": total_tests,
        "wall_time": round(elapsed, 1),
    }

def main():
    log("=" * 60)
    log("ALL BASELINES — PER-SPLIT EVALUATION")
    log("=" * 60)

    baselines = {
        "TestCases": TestCasesBaseline(),
        "Random": RandomBaseline(),
        "Static": StaticBaseline(),
        "Grammar": GrammarBaseline(),
        "Coverage": CoverageBaseline(),
    }

    all_results = {"train": {}, "val": {}, "test": {}}

    for split_name in ["train", "val", "test"]:
        split_file = SPLITS_DIR / f"{split_name}_pairs.json"
        if not split_file.exists():
            log(f"[SKIP] {split_file} not found")
            continue

        with open(split_file, "r", encoding="utf-8") as f:
            pairs = json.load(f)

        log(f"\n{'='*60}")
        log(f"EVALUATING SPLIT: {split_name.upper()} ({len(pairs)} pairs)")
        log(f"{'='*60}")

        for bl_name, bl_obj in baselines.items():
            result = evaluate_baseline_on_split(split_name, pairs, bl_name, bl_obj)
            all_results[split_name][bl_name] = result
            gc.collect()

    # Save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "all_baselines_splits.json"
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    log(f"\nSaved master results to {out}")

if __name__ == "__main__":
    main()
