"""Corrections must be verified, and must be one assertion.

The decision this encodes: fewer, surer assertions rather than per-assertion
scoring. A test function is reference-valid only if every assertion holds, and
at 3.65 assertions the trained arm's per-requested validity is 0.2749 against
the base model's 0.5655 at 1.01. Scoring per assertion instead would raise the
reported number without the model improving.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_single_assertion_corrections import _pick, _single_assertion_jobs

REFERENCE = "def f(x):\n    return x + 1\n"
MUTANT = "def f(x):\n    return x + 2\n"
CORRECTIONS = ROOT / "data" / "training_views" / "single_assertion_v1" / "corrections.json"


def _records():
    return {"rec": {"id": "rec", "reference_code": REFERENCE,
                    "code_under_test": MUTANT, "support_context": ""}}


def _job(assertions):
    return {"lineage": "g", "entry_point": "f", "assertions": assertions,
            "displayed_record_id": "rec", "source_dataset": "mbpp",
            "mutants_killed": 3, "assertion_count": len(assertions)}


def test_a_killing_assertion_is_preferred_over_a_merely_valid_one():
    picked = _pick(_job(["assert f(1) == 2", "assert isinstance(f(1), int)"]),
                   _records(), 5.0)
    assert picked["chosen"] == "assert f(1) == 2"
    assert picked["kills_displayed_target"] is True
    assert picked["reason"] == "killing"


def test_an_assertion_false_on_the_reference_is_never_chosen():
    """It would teach the model to assert something untrue."""
    picked = _pick(_job(["assert f(1) == 99"]), _records(), 5.0)
    assert picked["chosen"] is None
    assert picked["reason"] == "no_assertion_valid_on_reference"


def test_the_shortest_killing_assertion_wins():
    picked = _pick(_job([
        "assert f(1) == 2 and f(2) == 3 and f(3) == 4",
        "assert f(1) == 2",
    ]), _records(), 5.0)
    assert picked["chosen"] == "assert f(1) == 2"


def test_a_lineage_without_assertions_yields_no_job():
    assert _single_assertion_jobs({"lineage": "g", "assertions": []}) is None


def test_every_committed_correction_is_verified_and_single():
    if not CORRECTIONS.exists():
        return
    report = json.loads(CORRECTIONS.read_text(encoding="utf-8"))
    assert report["mean_assertions_after"] == 1.0
    for item in report["items"]:
        assert item["assertion_count"] == 1
        assert item["completion_shape"] == "assertion"
        assert item["verified"] is True
        assert item["verification_evidence"]["valid_on_reference"] is True
        assert item["completion"].lstrip().startswith("assert ")


def test_the_committed_corrections_all_kill_their_target():
    """A correction that does not distinguish the defect teaches nothing."""
    if not CORRECTIONS.exists():
        return
    report = json.loads(CORRECTIONS.read_text(encoding="utf-8"))
    assert report["kills_displayed_target"] == report["corrections"]
