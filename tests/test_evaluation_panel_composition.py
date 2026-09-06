"""Pin the panel-composition audit.

This audit exists to stop a specific misreading: that "757 held-out functions"
means the balanced synthetic/repository corpus was measured. It was not. If
repository targets ever DO enter the evaluation panel, that is a change in what
every reported number means, and the audit must notice rather than keep
printing the reassuring sentence.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_evaluation_panel_composition import (
    corpus_composition, panel_composition,
)


def test_corpus_composition_separates_repository_from_function(tmp_path):
    (tmp_path / "val.records.json").write_text(json.dumps([
        {"id": "mutation::a", "task_mode": "function"},
        {"id": "mutation::b", "task_mode": "function"},
        {"id": "official-repository::django::1::x", "task_mode": "repository"},
    ]), encoding="utf-8")

    row = corpus_composition(tmp_path, "val")
    assert row["records"] == 3
    assert row["function_records"] == 2
    assert row["repository_records"] == 1
    assert row["repository_record_ids"] == ["official-repository::django::1::x"]


def test_a_missing_split_yields_nothing_rather_than_zeros(tmp_path):
    """The sealed split has no file here and must not be reported as empty.

    Reporting it as 0 records would be indistinguishable from having opened it
    and found nothing.
    """
    assert corpus_composition(tmp_path, "test") == {}


def test_a_synthetic_only_panel_reports_zero_repository_targets(tmp_path):
    artifact = tmp_path / "sft_validation_standard_seed_42.json"
    artifact.write_text(json.dumps({
        "evaluation_split": "val", "seed": 42,
        "function_results": [
            {"record_id": "mutation::a", "killed": True},
            {"record_id": "mutation::b", "killed": False},
        ],
    }), encoding="utf-8")

    row = panel_composition(artifact, {"official-repository::django::1::x"})
    assert row["evaluated_targets"] == 2
    assert row["repository_targets_evaluated"] == 0


def test_a_repository_target_in_the_panel_is_detected(tmp_path):
    """The audit must be able to report the opposite of its current finding."""
    artifact = tmp_path / "sft_validation_standard_seed_42.json"
    artifact.write_text(json.dumps({
        "evaluation_split": "val", "seed": 42,
        "function_results": [
            {"record_id": "mutation::a", "killed": True},
            {"record_id": "official-repository::django::1::x", "killed": True},
        ],
    }), encoding="utf-8")

    row = panel_composition(artifact, {"official-repository::django::1::x"})
    assert row["evaluated_targets"] == 2
    assert row["repository_targets_evaluated"] == 1
    assert row["repository_target_ids"] == ["official-repository::django::1::x"]


def test_an_artifact_without_results_is_skipped(tmp_path):
    artifact = tmp_path / "sft_validation_standard_seed_42.json"
    artifact.write_text(json.dumps({"evaluation_split": "val"}), encoding="utf-8")
    assert panel_composition(artifact, set()) is None


def test_an_unreadable_artifact_does_not_break_the_audit(tmp_path):
    artifact = tmp_path / "sft_validation_standard_seed_42.json"
    artifact.write_text("{ truncated", encoding="utf-8")
    assert panel_composition(artifact, set()) is None
