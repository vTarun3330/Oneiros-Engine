import json

import pytest

from scripts.compare_sft_monitor_results import compare_monitor_results


def _result(selection: str, killed_ids):
    outcomes = [
        {"record_id": f"record-{index}", "killed": index in killed_ids}
        for index in range(4)
    ]
    return {
        "selection_sha256": selection,
        "seed": 42,
        "tests_per_function": 8,
        "checkpoint_step": 0,
        "function_validation_killed": len(killed_ids),
        "function_kill_rate": len(killed_ids) / 4,
        "candidate_kill_rate": 0.25,
        "end_to_end_candidate_kill_rate": 0.2,
        "parse_success_rate": 0.95,
        "function_results": outcomes,
    }


def test_common_panel_comparison_reports_paired_transitions(tmp_path):
    base = tmp_path / "base.json"
    trained = tmp_path / "trained.json"
    base.write_text(json.dumps(_result("locked", {0, 1})), encoding="utf-8")
    trained.write_text(json.dumps(_result("locked", {1, 2, 3})), encoding="utf-8")

    comparison = compare_monitor_results([base, trained])

    paired = comparison["comparisons"][1]["paired_vs_reference"]
    assert paired["improved_functions"] == 2
    assert paired["regressed_functions"] == 1
    assert paired["net_function_gain"] == 1


def test_common_panel_comparison_rejects_selection_mismatch(tmp_path):
    base = tmp_path / "base.json"
    changed = tmp_path / "changed.json"
    base.write_text(json.dumps(_result("first", {0})), encoding="utf-8")
    changed.write_text(json.dumps(_result("second", {0, 1})), encoding="utf-8")

    with pytest.raises(ValueError, match="locked common panel"):
        compare_monitor_results([base, changed])
