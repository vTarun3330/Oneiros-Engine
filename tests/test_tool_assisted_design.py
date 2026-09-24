"""Tests for the frozen execution-feedback pilot design (gates, panel, audit, receipt)."""
from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import pytest

from scripts.analyse_tool_assisted_pilot import (
    MIN_GAIN_PP, VALIDITY_MARGIN_PP, cluster_bootstrap, verdict,
)

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"


def _interval(difference, low, high):
    return {"difference": difference, "low": low, "high": high}


def test_verdict_is_three_way_and_semantic():
    assert (MIN_GAIN_PP, VALIDITY_MARGIN_PP) == (5.0, 3.0)
    good_validity = _interval(0.0, -1.0, 1.0)
    assert verdict(_interval(6.0, 1.0, 11.0), good_validity, 0.0, 1.2) == "pass"
    assert verdict(_interval(1.0, -2.0, 4.9), good_validity, 0.0, 1.2) == "fail"
    assert verdict(_interval(4.0, -1.0, 9.0), good_validity, 0.0, 1.2) == "inconclusive"
    # A kill gain cannot pass with a validity collapse, a diversity loss or slow runtime.
    assert verdict(_interval(6.0, 1.0, 11.0), _interval(-5.0, -8.0, -3.5), 0.0, 1.2) == "fail"
    assert verdict(_interval(6.0, 1.0, 11.0), good_validity, -0.1, 1.2) == "inconclusive"
    assert verdict(_interval(6.0, 1.0, 11.0), good_validity, 0.0, 2.0) == "inconclusive"


def test_cluster_bootstrap_is_deterministic_and_clustered():
    pairs = [("L1", 0, 1)] * 3 + [("L2", 1, 1)] * 3 + [("L3", 1, 0)]
    first = cluster_bootstrap(pairs, replicates=500, seed=3)
    assert first == cluster_bootstrap(pairs, replicates=500, seed=3)
    assert first["difference"] == pytest.approx(2 / 7)
    assert first["clusters"] == 3 and first["low"] <= first["difference"] <= first["high"]


def test_analysis_refuses_partial_results(tmp_path):
    from scripts.analyse_tool_assisted_pilot import main
    assert main(["--output", str(tmp_path / "analysis.json")]) == 2
    assert not (tmp_path / "analysis.json").exists()


def test_runner_refuses_an_unready_receipt(tmp_path):
    from scripts.run_tool_assisted_pilot import verify_design_receipt
    receipt = tmp_path / "receipt.json"
    receipt.write_text(json.dumps({"ready": False}), encoding="utf-8")
    with pytest.raises(SystemExit):
        verify_design_receipt(receipt)


def test_buggy_side_executor_has_no_channel_for_hidden_material():
    from harness import buggy_side_execution
    parameters = inspect.signature(buggy_side_execution.execute_on_code_under_test).parameters
    assert list(parameters) == ["code_under_test", "candidate", "timeout_seconds"]
    source = Path(buggy_side_execution.__file__).read_text(encoding="utf-8")
    for token in ("golden", "reference_code", "mutant_code", "test_cases"):
        assert token not in source


def test_tracked_panel_is_disjoint_and_honest():
    path = RESULTS / "v4_3_tool_assisted_panel.json"
    if not path.exists():
        pytest.skip("tool-assisted panel not present")
    assert b"\r" not in path.read_bytes()
    panel = json.loads(path.read_text(encoding="utf-8"))
    assert all(panel["checks"].values())
    assert panel["records"] == len(panel["record_ids"]) == 264
    digest = hashlib.sha256(json.dumps(panel["record_ids"], separators=(",", ":"))
                            .encode("utf-8")).hexdigest()
    assert digest == panel["record_ids_sha256"]
    qualification = panel["qualification"]
    assert qualification["previously_inspected"] is True
    assert qualification["repository_available"] is False
    assert qualification["untouched_train_function_lineages_available"] == 0
    assert "not confirmatory" in panel["label"]
    retention = json.loads((RESULTS / "v4_3_execution_dose_retention_panel.json")
                           .read_text(encoding="utf-8"))
    assert not set(panel["record_ids"]) & set(retention["record_ids"])
    assert not set(panel["record_lineages"].values()) & set(
        retention["record_lineages"].values())
    assert not any(panel["leakage"][key] for key in panel["leakage"] if key != "splits_opened")


def test_tracked_audit_classifies_every_component():
    path = RESULTS / "v4_3_tool_assisted_audit.json"
    if not path.exists():
        pytest.skip("component audit not present")
    audit = json.loads(path.read_text(encoding="utf-8"))
    for relative, entry in audit["components"].items():
        assert entry["classification"] in audit["classes"]
        assert entry["static"]["exists"]
    components = audit["components"]
    assert components["harness/verifier_guided_repair.py"]["classification"] == \
        "unsafe_for_this_experiment"
    assert components["baseline/atheris_harness.py"]["static"]["imports_real_atheris"]
    assert components["scripts/measure_repair_loop_headroom.py"]["static"][
        "reads_canonical_records_json"]
    assert audit["readiness"]["native_repository"]["status"] == \
        "not_ready_for_reportable_comparison"


def test_new_components_never_read_canonical_records_json():
    for relative in ("harness/buggy_side_execution.py", "harness/execution_feedback.py",
                     "harness/tool_assisted_generation.py", "scripts/run_tool_assisted_pilot.py",
                     "scripts/freeze_tool_assisted_panel.py",
                     "scripts/preflight_tool_assisted_pilot.py"):
        text = (ROOT / relative).read_text(encoding="utf-8")
        assert "records.json\"" not in text and "'records.json'" not in text, relative


def test_tracked_design_receipt_binds_sources_and_budgets():
    path = RESULTS / "v4_3_tool_assisted_design_receipt.json"
    if not path.exists():
        pytest.skip("design receipt not present")
    from harness.source_identity import canonical_sha256
    receipt = json.loads(path.read_text(encoding="utf-8"))
    assert receipt["ready"] is True and receipt["generation_launched"] is False
    assert receipt["budgets"]["sequences_per_target_b_and_c"] == 16
    assert receipt["budgets"]["final_slots"] == 8
    assert receipt["panel"]["prompt_from_permitted_view_identical"] is True
    assert not any(value for key, value in receipt["leakage"].items() if key != "splits_opened")
    for relative, expected in receipt["source_files_sha256"].items():
        assert canonical_sha256(ROOT / relative) == expected, relative
    for relative, expected in receipt["tracked_artifacts_sha256"].items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected
