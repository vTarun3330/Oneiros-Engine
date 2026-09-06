"""Test whether an arm's improvement holds across seeds, not within one.

The relearning arm is positive in all three seeds it has (+0.0383, +0.0449,
+0.0159) and significant in none of them after Holm correction across the
family of per-seed tests. Those two facts are not in tension: the per-seed
McNemar test asks whether one seed's discordant pairs are unbalanced, and at
~230 discordant pairs an effect of +0.03 simply is not resolvable there.

A different question - does the arm improve on this panel at all - is answered
by the seeds themselves. Under the null that the arm is neutral, each seed's
sign is a fair coin, so an exact two-sided binomial test over seed signs is a
legitimate test with the seeds as the unit of analysis. It is weak at n=3
(p=0.25 even if all three are positive, which is why three seeds could never
have settled this) and reaches p=0.0078 at n=8.

Two properties this deliberately keeps:

* seeds with a zero delta are excluded rather than counted as successes, which
  is the standard treatment and the conservative one;
* the sign test is added to the SAME Holm family as the per-seed tests. Running
  it alongside them and correcting only the others would be choosing the
  family after seeing which member won.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness.corpus import write_json
from metrics.research_evaluation import wilson_interval
from scripts.compare_base_vs_sft import (
    _arm, _killed_by_record, _mcnemar_exact, holm_bonferroni, paired_by_function,
)
from utils.reproducibility import source_tree_sha256


def sign_test(deltas: list[float]) -> dict[str, Any]:
    """Exact two-sided binomial test over the signs of per-seed deltas."""
    positive = sum(1 for value in deltas if value > 0)
    negative = sum(1 for value in deltas if value < 0)
    ties = sum(1 for value in deltas if value == 0)
    trials = positive + negative

    if trials == 0:
        return {
            "seeds_positive": positive, "seeds_negative": negative,
            "seeds_tied": ties, "effective_trials": 0, "p_value": 1.0,
            "note": "every seed tied; the test has nothing to weigh",
        }

    observed = max(positive, negative)
    tail = sum(
        math.comb(trials, k) for k in range(observed, trials + 1)
    ) / (2 ** trials)
    p_value = min(1.0, 2 * tail)

    return {
        "seeds_positive": positive,
        "seeds_negative": negative,
        "seeds_tied": ties,
        "effective_trials": trials,
        "p_value": round(p_value, 6),
        "smallest_reachable_p_at_this_n": round(2 / (2 ** trials), 6),
        "note": (
            "ties are excluded, not counted as successes"
            if ties else "no tied seeds"
        ),
    }


def analyse(base_run: str, arm_run: str, arm_name: str) -> dict[str, Any]:
    base_seeds = _arm(base_run, "*validation*.json")
    arm_seeds = _arm(arm_run, "sft_validation_*.json")
    shared_seeds = sorted(set(base_seeds) & set(arm_seeds))

    # paired_by_function pairs within each seed and is keyed by seed, so it is
    # called once over the whole arm rather than per seed.
    paired = paired_by_function(base_seeds, arm_seeds)

    per_seed: list[dict[str, Any]] = []
    deltas: list[float] = []
    tests: list[tuple[str, float]] = []

    for seed in shared_seeds:
        row = paired.get(str(seed))
        if row is None:
            continue
        base_killed = _killed_by_record(base_seeds[seed])
        arm_killed = _killed_by_record(arm_seeds[seed])
        shared_records = sorted(set(base_killed) & set(arm_killed))
        functions = len(shared_records)
        if not functions:
            continue

        # Rates are computed over the SHARED records only, so the delta and the
        # paired counts describe the same set of functions. Using each arm's
        # own full panel would let a delta and a net function gain disagree.
        base_rate = sum(1 for r in shared_records if base_killed[r]) / functions
        arm_rate = sum(1 for r in shared_records if arm_killed[r]) / functions
        delta = round(arm_rate - base_rate, 6)
        deltas.append(delta)

        p_value = float(row["mcnemar_exact_p_value"])
        tests.append((f"{arm_name}_seed_{seed}", p_value))
        per_seed.append({
            "seed": seed,
            "paired_functions": functions,
            "base_kill_rate": round(base_rate, 6),
            "base_kill_rate_wilson_95": wilson_interval(
                sum(1 for r in shared_records if base_killed[r]), functions
            ),
            "arm_kill_rate": round(arm_rate, 6),
            "arm_kill_rate_wilson_95": wilson_interval(
                sum(1 for r in shared_records if arm_killed[r]), functions
            ),
            "delta": delta,
            "improved": row["improved"],
            "regressed": row["regressed"],
            "functions_moved": row["improved"] + row["regressed"],
            "net_function_gain": row["net_function_gain"],
            "mcnemar_exact_p": p_value,
        })

    signs = sign_test(deltas)
    tests.append((f"{arm_name}_sign_test_over_seeds", signs["p_value"]))

    mean_delta = round(sum(deltas) / len(deltas), 6) if deltas else None
    return {
        "schema_version": "oneiros_seed_power_analysis_v1",
        "source_tree_sha256": source_tree_sha256(ROOT),
        "sealed_final_test_accessed": False,
        "arm": arm_name,
        "base_run": base_run,
        "arm_run": arm_run,
        "seeds_compared": shared_seeds,
        "per_seed": per_seed,
        "mean_delta": mean_delta,
        "positive_in_every_seed": bool(deltas) and all(d > 0 for d in deltas),
        "sign_test_over_seeds": signs,
        "holm_bonferroni": holm_bonferroni(tests),
        "family_note": (
            "the sign test is corrected in the same family as the per-seed "
            "tests. Correcting only the per-seed tests and reporting the sign "
            "test raw would be choosing the family after seeing the result."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-run", default="local_base_qwen_val_seed*")
    parser.add_argument("--arm-run", default="local_sft_relearn_v2_seed42")
    parser.add_argument("--arm-name", default="relearning")
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "results" / "v4_2_relearning_seed_power.json",
    )
    arguments = parser.parse_args()

    report = analyse(arguments.base_run, arguments.arm_run, arguments.arm_name)
    write_json(arguments.output, report)
    print(json.dumps({
        "seeds": report["seeds_compared"],
        "per_seed_delta": [row["delta"] for row in report["per_seed"]],
        "mean_delta": report["mean_delta"],
        "positive_in_every_seed": report["positive_in_every_seed"],
        "sign_test": report["sign_test_over_seeds"],
        "significant_after_adjustment":
            report["holm_bonferroni"].get("significant_after_adjustment"),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
