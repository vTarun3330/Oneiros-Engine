"""Four questions were being read off one field name.

`reference_valid_rate` is valid / REQUESTED. Reading it as valid / executed
gives a materially different number on the same artifact - 0.3590 against
0.4114 for the relearning arm on locked validation seed 42 - and quoting one
while meaning the other overstates candidate health by five points.

Each denominator now carries its name. `reference_valid_rate` keeps its
original meaning so every historical artifact and every comparison already
made against it stays correct.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from metrics.research_evaluation import summarise_function_results

RESULTS = ROOT / "results"


def _function(requested: int, parsed: int, executed: int, valid: int,
              killed: bool) -> dict:
    return {
        "requested_candidates": requested,
        "parsed_candidates": parsed,
        "execution_valid_candidates": executed,
        "valid_candidates": valid,
        "killing_candidates": 1 if killed else 0,
        "generation_invalid_candidates": requested - parsed,
        "execution_invalid_candidates": parsed - executed,
        "killed": killed,
        "candidate_outcomes": [],
    }


def test_the_four_denominators_are_distinct_fields():
    summary = summarise_function_results([
        _function(requested=8, parsed=6, executed=4, valid=2, killed=True),
        _function(requested=8, parsed=8, executed=8, valid=0, killed=False),
    ])

    assert summary["reference_valid_rate_per_requested"] == round(2 / 16, 6)
    assert summary["reference_valid_rate_per_parsed"] == round(2 / 14, 6)
    assert summary["reference_valid_rate_per_executed"] == round(2 / 12, 6)
    assert summary["function_reference_valid_rate"] == 0.5
    assert summary["functions_with_a_reference_valid_candidate"] == 1


def test_the_frozen_field_still_means_valid_over_requested():
    """Changing it would silently rewrite every historical comparison."""
    summary = summarise_function_results([
        _function(requested=8, parsed=4, executed=4, valid=2, killed=True),
    ])
    assert summary["reference_valid_rate"] == round(2 / 8, 6)
    assert summary["reference_valid_rate"] == \
        summary["reference_valid_rate_per_requested"]
    assert summary["reference_valid_rate"] != \
        summary["reference_valid_rate_per_parsed"]


def test_every_denominator_is_documented_in_the_artifact():
    summary = summarise_function_results([
        _function(requested=8, parsed=8, executed=8, valid=8, killed=True),
    ])
    documented = summary["reference_valid_denominators"]
    for field in ("reference_valid_rate", "reference_valid_rate_per_requested",
                  "reference_valid_rate_per_parsed",
                  "reference_valid_rate_per_executed",
                  "function_reference_valid_rate"):
        assert field in documented, f"{field} has no stated denominator"
        assert field in summary


def test_the_committed_locked_validation_numbers_are_what_is_reported():
    """Pins the audited figures so a quoted number is traceable to a file."""
    cases = {
        "local_base_qwen_val_seed42/base_validation_standard_seed_42.json": 0.453600,
        "local_sft_relearn_v2_seed42/sft_validation_standard_seed_42.json": 0.358983,
        "local_sft_long_s42/sft_validation_standard_seed_42.json": 0.326123,
    }
    for relative, expected in cases.items():
        path = RESULTS / relative
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["reference_valid_rate"] == expected, (
            f"{relative} reference_valid_rate changed; the writeup quotes "
            "these three figures and a paired regression of -0.094617"
        )
        assert payload["reference_valid_candidates"] / \
            payload["requested_candidates"] == \
            __import__("pytest").approx(expected, abs=1e-6), (
            "reference_valid_rate is no longer valid/requested"
        )


def test_the_paired_regression_is_computed_from_the_same_denominator():
    base = 0.453600
    relearning = 0.358983
    assert round(relearning - base, 6) == -0.094617
