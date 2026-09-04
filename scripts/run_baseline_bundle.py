"""Run every non-LLM baseline over one frozen panel, under one frozen protocol.

Deliverable 15.  The baselines already existed as separate modules with their
own entry points and their own budgets.  Scored that way they are not
comparable to each other or to Oneiros, so this assembles them into a single
bundle: same targets, same candidate budget, same per-execution timeout, same
seeds, same validity rule, one artifact.

Two honesty constraints are structural here rather than editorial:

* The coverage-guided fuzzer in ``baseline/coverage_fuzzer.py`` is a SIMULATED
  coverage fuzzer written for this project.  It is never labelled Atheris.
  Real Atheris results come from ``baseline/atheris_harness.py`` run under
  Linux, and live in their own artifacts.
* A baseline is given the dataset's own tests only where the arm is explicitly
  named for that.  The generative arms get no dataset tests at all, because
  handing them the answer would flatter them exactly where the comparison
  matters.

Reads the development view.  The sealed test split is never opened.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness.candidate_policy import validate_generated_test
from harness.corpus import write_json
from harness.safe_execution import execute_code
from metrics.research_evaluation import wilson_interval
from utils.reproducibility import source_tree_sha256

DEFAULT_VIEW = (
    ROOT / "data" / "corpus" / "v4_1_research_hardened_candidate" / "development_view"
)

#: Matched to the Oneiros protocol so the arms differ only in how candidates
#: are produced, never in how many they get or how long each may run.
CANDIDATES_PER_TARGET = 8
EXECUTION_TIMEOUT_SECONDS = 5.0
SEEDS = (42, 43, 44)


def kills(test_code: str, golden: str, mutant: str) -> tuple[bool, bool]:
    """Return (reference_valid, kills_mutant).

    A candidate that fails on the reference implementation is invalid, not a
    kill: it would 'detect' a defect in correct code too.  This is the same
    two-sided rule the model arms are scored under.

    ``execute_code`` returns a plain (ok, result, error) tuple. Treating it as
    an object with a ``.success`` attribute silently scored every candidate as
    reference-invalid and reported 0.0 for every arm, including the corpus's
    own tests - which is what exposed it.
    """
    reference_ok, _, _ = execute_code(golden, test_code, EXECUTION_TIMEOUT_SECONDS)
    if not reference_ok:
        return False, False
    mutant_ok, _, _ = execute_code(mutant, test_code, EXECUTION_TIMEOUT_SECONDS)
    return True, not mutant_ok


def score_arm(
    records: list[dict[str, Any]],
    generate: Callable[[dict[str, Any], int], list[str]],
    seed: int,
) -> dict[str, Any]:
    killed = 0
    reference_valid = 0
    policy_valid = 0
    candidates_generated = 0
    started = time.time()

    for record in records:
        # Corpus field names, not the trainer's internal pair names: the
        # reference implementation is reference_code and the defective one
        # shown to a generator is code_under_test.
        golden = record.get("reference_code") or ""
        mutant = record.get("code_under_test") or ""
        entry = record.get("entry_point") or ""
        if not (golden and mutant and entry):
            continue
        target_killed = False
        for candidate in generate(record, seed)[:CANDIDATES_PER_TARGET]:
            candidates_generated += 1
            if not validate_generated_test(candidate, entry).valid:
                continue
            policy_valid += 1
            valid, is_kill = kills(candidate, golden, mutant)
            reference_valid += int(valid)
            if is_kill:
                target_killed = True
                break
        killed += int(target_killed)

    total = len(records)
    return {
        "seed": seed,
        "targets": total,
        "killed": killed,
        "kill_rate": round(killed / max(1, total), 6),
        "kill_rate_wilson_95": wilson_interval(killed, max(1, total)),
        "candidates_generated": candidates_generated,
        "policy_valid_candidates": policy_valid,
        "reference_valid_candidates": reference_valid,
        "wall_seconds": round(time.time() - started, 1),
    }


def _call(cls_name: str, record: dict[str, Any], seed: int) -> list[str]:
    """Drive one baseline generator under a pinned global RNG.

    These generators draw from the ``random`` module directly, so the seed has
    to be set here rather than passed in; without it the arms would not be
    reproducible and the per-seed spread would be meaningless.
    """
    import random

    import baseline.benchmark_runner as runner

    random.seed(seed)
    generator = getattr(runner, cls_name)()
    return list(generator.generate_tests(
        record.get("reference_code") or "",
        record.get("entry_point") or "",
        [test["code"] for test in record.get("tests") or [] if test.get("code")],
        CANDIDATES_PER_TARGET,
    ))


def _dataset_tests(record: dict[str, Any], seed: int) -> list[str]:
    return _call("TestCasesBaseline", record, seed)


def _random(record: dict[str, Any], seed: int) -> list[str]:
    return _call("RandomBaseline", record, seed)


def _static(record: dict[str, Any], seed: int) -> list[str]:
    return _call("StaticBaseline", record, seed)


def _grammar(record: dict[str, Any], seed: int) -> list[str]:
    return _call("GrammarBaseline", record, seed)


def _simulated_coverage(record: dict[str, Any], seed: int) -> list[str]:
    return _call("CoverageBaseline", record, seed)


ARMS: dict[str, dict[str, Any]] = {
    "dataset_tests": {
        "generate": _dataset_tests,
        "description": "the corpus's own retained tests; an upper reference, not a generator",
        "sees_dataset_tests": True,
    },
    "random": {
        "generate": _random,
        "description": "random inputs, asserted against the reference",
        "sees_dataset_tests": False,
    },
    "static": {
        "generate": _static,
        "description": "boundary-value templates",
        "sees_dataset_tests": False,
    },
    "grammar": {
        "generate": _grammar,
        "description": "type-aware boundary analysis",
        "sees_dataset_tests": False,
    },
    "simulated_coverage_fuzzer": {
        "generate": _simulated_coverage,
        "description": (
            "output-diversity fuzzing written for this project. NOT Atheris, "
            "NOT libFuzzer, and never to be reported as either"
        ),
        "sees_dataset_tests": False,
    },
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="val")
    parser.add_argument("--view-dir", type=Path, default=DEFAULT_VIEW)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--arm", action="append", default=None)
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "results" / "v4_2_baseline_bundle_val.json",
    )
    arguments = parser.parse_args()

    if arguments.split in {"test", "final", "sealed"}:
        raise SystemExit(
            f"Refusing split {arguments.split!r}: the sealed final test is not "
            "opened by a baseline sweep"
        )

    path = arguments.view_dir / f"{arguments.split}.records.json"
    records = json.loads(path.read_text(encoding="utf-8"))
    records = [
        record for record in records if record.get("task_mode") != "repository"
    ]
    if arguments.limit:
        records = records[: arguments.limit]

    selected = arguments.arm or list(ARMS)
    report: dict[str, Any] = {
        "schema_version": "oneiros_baseline_bundle_v1",
        "source_tree_sha256": source_tree_sha256(ROOT),
        "sealed_final_test_accessed": False,
        "split": arguments.split,
        "targets": len(records),
        "protocol": {
            "candidates_per_target": CANDIDATES_PER_TARGET,
            "execution_timeout_seconds": EXECUTION_TIMEOUT_SECONDS,
            "seeds": arguments.seeds,
            "validity_rule": "harness.candidate_policy.validate_generated_test",
            "kill_rule": (
                "candidate must pass on the reference implementation AND fail "
                "on the mutant; a candidate that fails on the reference is "
                "invalid, not a kill"
            ),
            "matched_to_oneiros": (
                "same targets, same candidate budget, same per-execution "
                "timeout, same validity and kill rules"
            ),
            "advantage_given_to_the_baselines": (
                "every generative arm derives its expected value by EXECUTING "
                "the reference implementation, so each is handed a perfect "
                "oracle and only has to choose inputs. Oneiros is never shown "
                "the reference. A baseline that still loses is not losing for "
                "want of an oracle."
            ),
        },
        "naming_honesty": {
            "simulated_coverage_fuzzer_is_not_atheris": True,
            "actual_atheris_artifacts": [
                "results/v4_2_atheris_vs_historical_oneiros_val_matched8_seed42.json",
                "results/v4_2_atheris_vs_oneiros_val_20000_seed42.json",
            ],
        },
        "arms": {},
    }

    for name in selected:
        arm = ARMS[name]
        print(f"[{name}] {len(records)} targets", flush=True)
        seeds = {}
        for seed in arguments.seeds:
            try:
                seeds[str(seed)] = score_arm(records, arm["generate"], seed)
            except Exception as exc:
                seeds[str(seed)] = {
                    "error": f"{type(exc).__name__}: {exc}", "seed": seed,
                }
            print(f"  seed {seed}: {seeds[str(seed)]}", flush=True)
        rates = [
            row["kill_rate"] for row in seeds.values() if "kill_rate" in row
        ]
        report["arms"][name] = {
            "description": arm["description"],
            "sees_dataset_tests": arm["sees_dataset_tests"],
            "by_seed": seeds,
            "mean_kill_rate": (
                round(sum(rates) / len(rates), 6) if rates else None
            ),
            "range_across_seeds": (
                round(max(rates) - min(rates), 6) if len(rates) > 1 else None
            ),
        }
        write_json(arguments.output, report)

    write_json(arguments.output, report)
    print(json.dumps({
        name: {
            "mean_kill_rate": arm["mean_kill_rate"],
            "sees_dataset_tests": arm["sees_dataset_tests"],
        }
        for name, arm in report["arms"].items()
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
