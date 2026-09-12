"""Cross-split near-duplication must be reported as a sweep, not one threshold.

The first run of this audit reported "0 near-duplicate pairs in val" while the
closest pair sat at 0.7971 - three thousandths under the cut, with 45 records
behind it. That headline was true and useless. These tests pin the shape of the
report that replaced it as much as the numbers in it.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.audit_cross_split_near_duplicates import (
    NEAR_DUPLICATE_JACCARD, THRESHOLD_SWEEP, compare, jaccard, normalise,
    shingles, summarise,
)

ROOT = Path(__file__).resolve().parent.parent
AUDIT = ROOT / "results" / "v4_2_cross_split_near_duplicate_audit.json"


def test_normalisation_removes_docstrings_and_formatting():
    documented = '''def f(n):
    """Return twice n, described at length."""
    return n * 2
'''
    bare = "def f( n ):\n\n    return n*2\n"
    assert normalise(documented) == normalise(bare)


def test_normalisation_does_not_erase_a_real_difference():
    assert normalise("def f(n):\n    return n * 2\n") \
        != normalise("def f(n):\n    return n * 3\n")


def test_identical_functions_score_one():
    left = shingles(normalise("def f(n):\n    return n * 2 + 1\n"))
    right = shingles(normalise("def f(n):\n    return n * 2 + 1\n"))
    assert jaccard(left, right) == 1.0


def test_short_functions_do_not_all_collide():
    """Every one-liner sharing a Jaccard of 1.0 would make the report noise."""
    left = shingles(normalise("def f():\n    return 1\n"))
    right = shingles(normalise("def g():\n    return 2\n"))
    assert jaccard(left, right) < 1.0


def test_summary_reports_the_whole_sweep_not_just_the_cut():
    scored = [
        {"evaluation_record": "a", "nearest_training_record": "t", "jaccard": 0.79},
        {"evaluation_record": "b", "nearest_training_record": "t", "jaccard": 0.91},
    ]
    summary = summarise(scored, {"a": "L1", "b": "L2"}, NEAR_DUPLICATE_JACCARD)
    assert summary["max_similarity"] == 0.91
    assert summary["near_duplicate_records_at_threshold"] == 1
    # The 0.79 record must remain visible below the cut, which is the whole
    # point of the sweep.
    assert summary["threshold_sweep"]["0.75"]["records"] == 2
    assert summary["threshold_sweep"]["0.80"]["records"] == 1
    assert set(summary["threshold_sweep"]) == {f"{c:.2f}" for c in THRESHOLD_SWEEP}


def test_records_and_functions_are_counted_separately():
    """45 records of one function is one duplicated function, not 45."""
    scored = [
        {"evaluation_record": f"m{i}", "nearest_training_record": "t",
         "jaccard": 0.9} for i in range(45)
    ]
    lineage_of = {f"m{i}": "one_lineage" for i in range(45)}
    summary = summarise(scored, lineage_of, NEAR_DUPLICATE_JACCARD)
    assert summary["near_duplicate_records_at_threshold"] == 45
    assert summary["near_duplicate_functions_at_threshold"] == 1


def test_compare_returns_every_entry_scored():
    """Filtering inside compare is what baked the misleading threshold in."""
    train = {"t": shingles(normalise("def f(n):\n    return n * 2\n"))}
    panel = {"p": shingles(normalise("def g(x):\n    return x + 99\n"))}
    assert len(compare(train, panel)) == 1


@pytest.mark.skipif(not AUDIT.exists(), reason="audit not built in this checkout")
def test_committed_audit_figures():
    report = json.loads(AUDIT.read_text(encoding="utf-8"))
    assert report["sealed_final_test_accessed"] is False
    assert report["sealed_split_excluded"] == "test"
    assert "train_vs_test" not in report["comparisons"]

    val = report["comparisons"]["train_vs_val"]["reference_code"]
    # Locked validation is clean at the cut, and the report must still carry
    # the near-miss band that makes that claim honest.
    assert val["near_duplicate_records_at_threshold"] == 0
    assert val["max_similarity"] == 0.7971
    assert val["threshold_sweep"]["0.75"]["distinct_evaluation_functions"] == 3
    assert report["comparisons"]["train_vs_val"]["shared_group_id_lineages"] == 0

    adev = report["comparisons"]["train_vs_ablation_dev"]["reference_code"]
    assert adev["near_duplicate_functions_at_threshold"] == 2
    assert adev["max_similarity"] == 0.9111
