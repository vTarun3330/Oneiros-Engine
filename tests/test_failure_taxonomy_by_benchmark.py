"""Pin the per-benchmark taxonomy cut and its refusal to read sealed data.

The pooled taxonomy averages a 701-target mbpp panel with a 56-target
humaneval one, and the two behave in opposite directions: SFT halves
wrong_expected_value on humaneval and raises it on mbpp. A pooled number
describes neither, which is how "SFT reduces wrong expected values" survived
as a belief while being false for 92.6% of the panel.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.failure_taxonomy_by_benchmark import taxonomy

REPORT = ROOT / "results" / "v4_2_failure_taxonomy_by_benchmark.json"


def _artifact(tmp_path: Path, split: str = "val", sealed: bool = False) -> Path:
    payload = {
        "evaluation_split": split,
        "final_test_measurement": sealed,
        "function_results": [
            {
                "record_id": "mutation::mbpp_1_mut_1", "dataset_name": "mbpp",
                "bug_family": "boundary", "killed": False,
                "candidate_outcomes": [
                    {"parse_valid": True, "policy_valid": True,
                     "reference_status": "assertion_error", "killed": False},
                ],
            },
            {
                "record_id": "mutation::humaneval_1_mut_1",
                "dataset_name": "humaneval", "bug_family": "boundary",
                "killed": True,
                "candidate_outcomes": [
                    {"parse_valid": True, "policy_valid": True,
                     "reference_status": "passed", "reference_valid": True,
                     "killed": True},
                ],
            },
        ],
    }
    path = tmp_path / "artifact.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_the_two_benchmarks_are_reported_separately(tmp_path):
    report = taxonomy(_artifact(tmp_path))
    by = report["by_benchmark"]
    assert set(by) == {"mbpp", "humaneval"}
    assert by["mbpp"]["function_kill_rate"] == 0.0
    assert by["humaneval"]["function_kill_rate"] == 1.0


def test_a_failing_assertion_on_correct_code_is_a_wrong_expected_value(tmp_path):
    """The category that dominates mbpp; if it is miscounted the diagnosis is wrong."""
    report = taxonomy(_artifact(tmp_path))
    assert report["by_benchmark"]["mbpp"]["counts"]["wrong_expected_value"] == 1


def test_a_sealed_final_test_artifact_is_refused(tmp_path):
    with pytest.raises(SystemExit):
        taxonomy(_artifact(tmp_path, sealed=True))


def test_a_test_split_artifact_is_refused(tmp_path):
    with pytest.raises(SystemExit):
        taxonomy(_artifact(tmp_path, split="test"))


def test_the_committed_report_still_shows_the_mbpp_ceiling():
    """Guards the finding the next phase of work is planned against."""
    if not REPORT.exists():
        return
    artifacts = json.loads(REPORT.read_text(encoding="utf-8"))["artifacts"]
    base = next(v for k, v in artifacts.items() if "base" in k)["by_benchmark"]
    assert base["mbpp"]["functions"] == 701 and base["humaneval"]["functions"] == 56
    assert base["mbpp"]["rates"]["wrong_expected_value"] > \
        base["humaneval"]["rates"]["wrong_expected_value"], (
        "mbpp no longer shows the higher wrong-expected-value rate that the "
        "specification-quality diagnosis rests on"
    )
