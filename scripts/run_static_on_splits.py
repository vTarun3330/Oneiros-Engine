"""
Run the Static baseline on each data split (Train / Val / Test).

Reuses the StaticBaseline, safe_exec, kills_mutant, and make_assert
from baseline/benchmark_runner.py so results are directly comparable
with the full-dataset 10K benchmark.

Outputs:
    results/static_baseline_splits.json

Usage:
    py scripts/run_static_on_splits.py
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
    StaticBaseline, safe_exec, kills_mutant, make_assert, log
)

DATA_DIR    = Path(__file__).parent.parent / "data"
SPLITS_DIR  = DATA_DIR / "splits"
RESULTS_DIR = Path(__file__).parent.parent / "results"

TESTS_PER_MUTANT = 8
SEED = 42


def evaluate_split(split_name: str, pairs: list, baseline) -> dict:
    """Run a baseline against a list of mutation pairs and return metrics."""
    log(f"\n{'='*60}")
    log(f"  Static Baseline on {split_name.upper()} split  ({len(pairs):,} pairs)")
    log(f"{'='*60}")

    t0 = time.time()
    random.seed(SEED)

    killed = 0
    survived = 0
    total_tests = 0
    skipped_gen = 0
    skipped_comp = 0
    kill_by_type = Counter()
    total_by_type = Counter()

    for i, pair in enumerate(pairs):
        golden = pair["golden_code"]
        mutant = pair["mutant_code"]
        entry  = pair["entry_point"]
        mtype  = pair["mutation_type"]
        total_by_type[mtype] += 1

        try:
            tests = baseline.generate_tests(golden, entry, [],
                                            num_tests=TESTS_PER_MUTANT)
        except Exception:
            skipped_gen += 1
            tests = []

        mutant_killed = False
        for test in tests:
            total_tests += 1
            try:
                compile(test, "<t>", "exec")
            except SyntaxError:
                skipped_comp += 1
                continue
            if kills_mutant(test, golden, mutant):
                mutant_killed = True
                break

        if mutant_killed:
            killed += 1
            kill_by_type[mtype] += 1
        else:
            survived += 1

        if (i + 1) % 500 == 0 or (i + 1) == len(pairs):
            elapsed = time.time() - t0
            rate = killed / (i + 1) * 100
            log(f"    [{i+1:>5}/{len(pairs)}]  killed={killed:<5} "
                f"rate={rate:>5.1f}%  elapsed={elapsed:>6.1f}s")

        if (i + 1) % 500 == 0:
            gc.collect()

    elapsed = time.time() - t0
    kill_rate = killed / max(len(pairs), 1)

    result = {
        "split": split_name,
        "total_pairs": len(pairs),
        "kill_rate": round(kill_rate, 4),
        "killed": killed,
        "survived": survived,
        "total_tests": total_tests,
        "skipped_generation": skipped_gen,
        "skipped_compile_tests": skipped_comp,
        "wall_time": round(elapsed, 1),
        "kill_by_type": dict(kill_by_type),
        "total_by_type": dict(total_by_type),
    }

    log(f"  >> {split_name}: {kill_rate:.1%} kill rate "
        f"({killed}/{len(pairs)})  {elapsed:.1f}s")
    return result


def main():
    log("=" * 60)
    log("STATIC BASELINE — PER-SPLIT EVALUATION")
    log("=" * 60)

    baseline = StaticBaseline()
    all_results = {}

    for split_name in ["train", "val", "test"]:
        split_file = SPLITS_DIR / f"{split_name}_pairs.json"
        if not split_file.exists():
            log(f"  [SKIP] {split_file} not found — run split_dataset.py first")
            continue

        with open(split_file, "r", encoding="utf-8") as f:
            pairs = json.load(f)

        result = evaluate_split(split_name, pairs, baseline)
        all_results[split_name] = result
        gc.collect()

    # ── Summary table ─────────────────────────────────────────
    log(f"\n{'='*60}")
    log("SUMMARY")
    log(f"{'='*60}")
    log(f"{'Split':<8} {'Pairs':>7} {'Kill Rate':>10} {'Killed':>8} "
        f"{'Survived':>10} {'Time':>8}")
    log("-" * 55)
    for name, r in all_results.items():
        log(f"{name:<8} {r['total_pairs']:>7,} {r['kill_rate']:>9.1%} "
            f"{r['killed']:>8,} {r['survived']:>10,} {r['wall_time']:>7.1f}s")

    # ── Save ──────────────────────────────────────────────────
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "static_baseline_splits.json"
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    log(f"\nSaved results to {out}")
    log("=" * 60)

    return all_results


if __name__ == "__main__":
    main()
