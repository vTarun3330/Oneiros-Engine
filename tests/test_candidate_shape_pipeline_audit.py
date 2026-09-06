"""Pin the finding that the evaluation path collapses output to one assertion.

This audit contradicts a capability the project claims, so it must keep working
and must be able to report the opposite once the pipeline is fixed.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_candidate_shape_pipeline import (
    observed_shapes, parser_collapses_to_one_assertion,
)


def test_the_generator_still_collapses_to_the_first_assertion():
    """If this fails, the pipeline changed and the finding must be re-derived."""
    evidence = parser_collapses_to_one_assertion()
    assert evidence["scans_lines_for_a_leading_assert"] is True
    assert evidence["stops_at_the_first_match"] is True
    assert evidence["collapses_output_to_one_assertion"] is True


def test_a_test_function_candidate_would_be_counted_if_one_existed(tmp_path):
    """The audit must be able to report the opposite of its current finding.

    An audit that can only ever say "zero" proves nothing about the artifacts.
    """
    artifact = tmp_path / "sft_validation_standard_seed_42.json"
    artifact.write_text(json.dumps({
        "evaluation_split": "val", "seed": 42,
        "function_results": [{"candidate_outcomes": [
            {"code": "def test_x():\n    assert f(1) == 2\n    assert f(2) == 3",
             "candidate_shape": "test_function", "raw_output_sha256": "abc"},
            {"code": "assert f(1) == 2", "candidate_shape": "assertion"},
            {"code": "", "candidate_shape": None},
        ]}],
    }), encoding="utf-8")

    row = observed_shapes(artifact)
    assert row["test_function_candidates"] == 1
    assert row["recorded_code_shape"] == {
        "test_function": 1, "bare_assertion": 1, "empty": 1,
    }
    assert row["recorded_candidate_shape"]["test_function"] == 1


def test_raw_output_text_is_never_reported_as_retained(tmp_path):
    """Only the hash survives, so the raw text column must stay zero.

    Reporting retained raw output would imply the model's actual emission
    could be recovered and checked. It cannot.
    """
    artifact = tmp_path / "sft_validation_standard_seed_42.json"
    artifact.write_text(json.dumps({
        "evaluation_split": "val", "seed": 42,
        "function_results": [{"candidate_outcomes": [
            {"code": "assert f(1) == 2", "raw_output_sha256": "deadbeef"},
        ]}],
    }), encoding="utf-8")

    row = observed_shapes(artifact)
    assert row["raw_output_text_retained"] == 0
    assert row["raw_output_hash_retained"] == 1


def test_an_artifact_without_results_is_skipped(tmp_path):
    artifact = tmp_path / "sft_validation_standard_seed_42.json"
    artifact.write_text(json.dumps({"evaluation_split": "val"}), encoding="utf-8")
    assert observed_shapes(artifact) is None
