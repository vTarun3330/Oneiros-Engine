"""Declare the successor evaluation protocol before anything runs under it.

The frozen protocol scores each generation by its FIRST assertion. The
successor judges the whole output, so a multi-assertion test can be executed
and a mutant that only the fifth assertion catches can die. That changes what
Kill@8 means, which is exactly why it needs a name, a version and a receipt
rather than a flag someone remembers to pass.

Two properties this exists to guarantee:

* legacy results are HISTORICAL, not wrong, and are never overwritten. The
  successor writes to its own run directory, and `results_directory_conflict`
  refuses a destination that already holds legacy artifacts.
* a successor artifact can never be mistaken for a legacy one. Every successor
  result carries the protocol name and version, and the receipt records the
  evaluator, parser and candidate-policy source hashes that produced it.

Nothing here evaluates anything. It writes the declaration that a later
evaluation is checked against.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness.candidate_policy import (
    CANDIDATE_SHAPES, MAX_TEST_FUNCTION_ASSERTS,
)
from harness.safe_execution import DEFAULT_TIMEOUT_SECONDS
from harness.corpus import write_json

PROTOCOL_NAME = "oneiros_whole_output_successor"
PROTOCOL_VERSION = "1.0.1"

LEGACY_PROTOCOL_NAME = "oneiros_first_assertion_frozen"

#: Files whose content defines the protocol's behaviour. A change to any of
#: them changes what a successor number means, so each is hashed into the
#: receipt rather than named and trusted.
DEFINING_SOURCES = {
    "evaluator": "metrics/research_evaluation.py",
    "parser": "engine/generator.py",
    "candidate_policy": "harness/candidate_policy.py",
    "execution_harness": "harness/safe_execution.py",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def results_directory_conflict(destination: Path) -> str | None:
    """Why this destination must not be used, or None if it is safe.

    A successor run writing beside legacy artifacts would leave a directory
    whose files mean two different things under the same field names.
    """
    if not destination.exists():
        return None
    legacy = sorted(
        path.name for path in destination.glob("*_validation_*.json")
        if "successor" not in path.name
    )
    if legacy:
        return (
            f"{destination.name} already holds legacy first_assertion "
            f"artifacts ({', '.join(legacy[:3])}"
            f"{'...' if len(legacy) > 3 else ''}); the successor protocol must "
            "write to its own run directory so neither set is overwritten and "
            "neither can be read as the other"
        )
    return None


def build(candidates: int, seeds: list[int], prompt_budget: int,
          completion_budget: int, sequence_budget: int,
          raw_output_dir: str) -> dict[str, Any]:
    sources = {
        name: {"path": relative, "sha256": _sha256(ROOT / relative)}
        for name, relative in DEFINING_SOURCES.items()
    }
    return {
        "schema_version": "oneiros_successor_protocol_receipt_v1",
        "protocol_name": PROTOCOL_NAME,
        "protocol_version": PROTOCOL_VERSION,
        "supersedes": {
            "protocol_name": LEGACY_PROTOCOL_NAME,
            "status": (
                "HISTORICAL, NOT INVALID. Every reported result to date was "
                "produced under it and remains a correct measurement of what "
                "it measured: eight single assertions. Legacy artifacts are "
                "never overwritten, never regenerated in place, and never "
                "compared to successor numbers as though they were the same "
                "metric."
            ),
        },
        "generation": {
            "candidate_parse_mode": "whole_output",
            "retain_raw_output": True,
            "candidates_per_target": candidates,
            "seeds": list(seeds),
            "no_reranking": True,
            "no_reranking_rule": (
                "candidates are scored in generation order and none is "
                "reordered, filtered or re-sampled on any signal derived from "
                "the reference, the mutant or the evaluator"
            ),
            "generated_once": (
                "each candidate is generated exactly once; the legacy and "
                "successor interpretations are both computed from the SAME "
                "retained raw output, so the comparison carries no generation "
                "randomness"
            ),
        },
        "candidate_policy": {
            "permitted_shapes": list(CANDIDATE_SHAPES),
            "max_assertions_per_test_function": MAX_TEST_FUNCTION_ASSERTS,
            "assertion_count_implementation":
                "harness.candidate_policy.count_assertions, ast.walk over all "
                "ast.Assert nodes, unparseable code counts zero",
            "refuses": [
                "helper functions beside the test", "imports",
                "network names", "filesystem names", "process names",
                "dunder access", "decorated tests", "parameterised tests",
                "unbounded string, integer and container literals",
                "candidates that never call the target entry point",
            ],
        },
        "budgets": {
            "prompt_tokens": prompt_budget,
            "completion_tokens": completion_budget,
            "total_sequence_tokens": sequence_budget,
            "fits": prompt_budget + completion_budget <= sequence_budget,
            "note": (
                "prompt + completion must fit the sequence budget before chat "
                "template overhead, which is additional. A configuration that "
                "does not fit is refused rather than silently truncated."
            ),
        },
        "timeout_policy": {
            # READ from the harness, never asserted here. The first version of
            # this receipt claimed 5.0 while the evaluator actually used 0.5,
            # a tenfold error in a field whose whole purpose is to say what
            # the protocol did. A receipt that states a constant by hand is a
            # second source of truth, which is how it went wrong.
            "per_candidate_seconds": DEFAULT_TIMEOUT_SECONDS,
            "source": "harness.safe_execution.DEFAULT_TIMEOUT_SECONDS",
            "read_from_source": True,
            "on_timeout": "recorded as a timeout outcome, never as a kill",
        },
        "raw_output": {
            "storage": raw_output_dir,
            "retained": True,
            "hash_field": "raw_output_sha256",
            "why": (
                "the legacy protocol kept only the hash, so what the model "
                "actually emitted could not be recovered and the collapse "
                "could not be quantified after the fact"
            ),
        },
        "defining_sources": sources,
        "reference_validity_denominators": {
            "reference_valid_rate": "valid / requested (the frozen field)",
            "reference_valid_rate_per_parsed": "valid / parsed",
            "reference_valid_rate_per_executed": "valid / executed",
            "function_reference_valid_rate":
                "functions with >=1 valid candidate / evaluated functions",
            "why": (
                "these are four different questions with four denominators; "
                "quoting one while meaning another overstated candidate "
                "health by five points on locked validation seed 42"
            ),
        },
        "locked_validation_policy": (
            "ablation_dev only for P0 diagnosis. Locked validation is not used "
            "to debug or redesign the protocol, and the three-seed successor "
            "evaluation runs only once token limits, prompt budgets, the "
            "whole-output policy, candidate-validity rules, the curriculum, "
            "the dataset and the selected adapters are all frozen."
        ),
        "sealed_final_test_accessed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=int, default=8)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--prompt-budget", type=int, default=1024)
    parser.add_argument("--completion-budget", type=int, default=128)
    parser.add_argument("--sequence-budget", type=int, default=2048)
    parser.add_argument("--raw-output-dir",
                        default="results/{run}/raw_generations")
    parser.add_argument("--check-destination", type=Path, default=None,
                        help="refuse a directory that already holds legacy artifacts")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results" / "v4_2_successor_protocol_receipt.json")
    arguments = parser.parse_args()

    if arguments.check_destination is not None:
        conflict = results_directory_conflict(arguments.check_destination)
        if conflict:
            raise SystemExit(conflict)

    receipt = build(arguments.candidates, arguments.seeds,
                    arguments.prompt_budget, arguments.completion_budget,
                    arguments.sequence_budget, arguments.raw_output_dir)
    if not receipt["budgets"]["fits"]:
        raise SystemExit(
            f"prompt {arguments.prompt_budget} + completion "
            f"{arguments.completion_budget} exceeds sequence budget "
            f"{arguments.sequence_budget}; choose an allocation that fits "
            "rather than truncating silently"
        )
    write_json(arguments.output, receipt)
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
