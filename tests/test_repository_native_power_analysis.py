"""Tests for the prospective repository-native power analysis and its evidence checks."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts import build_repository_native_power_analysis as power_module
from scripts.build_repository_native_power_analysis import (
    GATE_PP, N_VALUES, EvidenceRefused, clustered_monte_carlo_power, encode,
    minimum_detectable_effect, monte_carlo_power, paired_evidence, power, se_pp,
    verify_power_artifact,
)

ROOT = Path(__file__).resolve().parent.parent
TRACKED = ROOT / "results" / "v4_3_repository_native_power_analysis.json"


def test_gate_is_the_frozen_five_points():
    assert GATE_PP == 5.0
    assert set(N_VALUES) >= {100, 150, 200, 300, 400, 500}


def test_power_rises_with_n_and_effect_and_falls_with_clustering():
    assert power(100, 0.21, 10, 0.05, 4) < power(300, 0.21, 10, 0.05, 12)
    assert power(200, 0.21, 8, 0.05, 8) < power(200, 0.21, 12, 0.05, 8)
    assert se_pp(200, 0.21, 0, 0.2, 8) > se_pp(200, 0.21, 0, 0.0, 8)


@pytest.mark.parametrize("n", N_VALUES)
def test_a_true_effect_exactly_at_the_gate_never_reaches_80_percent(n):
    for rho in (0.0, 0.05, 0.2):
        assert power(n, 0.21, GATE_PP, rho, 12) <= 0.5 + 1e-9
    assert minimum_detectable_effect(n, 0.21, 0.0, 1) > GATE_PP


def test_minimum_detectable_effect_is_consistent_with_power():
    mde = minimum_detectable_effect(300, 0.21, 0.05, 12)
    assert power(300, 0.21, mde, 0.05, 12) >= 0.8
    assert power(300, 0.21, mde - 0.5, 0.05, 12) < 0.8


def test_monte_carlo_agrees_with_the_analytic_approximation():
    assert monte_carlo_power(200, 0.21, 10.0, runs=2000) == pytest.approx(
        power(200, 0.21, 10.0, 0.0, 1.0), abs=0.05)
    assert clustered_monte_carlo_power(400, 0.21, 8.0, 0.05, 34, runs=3000) == pytest.approx(
        power(400, 0.21, 8.0, 0.05, 400 / 34), abs=0.06)


def test_clustered_monte_carlo_power_falls_with_clustering_and_is_deterministic():
    values = [clustered_monte_carlo_power(300, 0.21, 10.0, rho, 25, runs=3000)
              for rho in (0.0, 0.05, 0.10, 0.20)]
    assert values == sorted(values, reverse=True)
    assert clustered_monte_carlo_power(300, 0.21, 10.0, 0.1, 25, runs=500) == \
        clustered_monte_carlo_power(300, 0.21, 10.0, 0.1, 25, runs=500)


# --- evidence verification on a synthetic root ---------------------------------------

def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(root: Path, relative: str, value) -> str:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return _sha(path)


def _result(kills: dict[str, int]) -> dict:
    return {"function_results": [{"record_id": record, "candidate_outcomes": [
        {"rank": 1, "killed": bool(value)}]} for record, value in kills.items()]}


def _fake_evidence_root(root: Path, a: dict, b: dict, lineages: dict) -> str:
    panel_path = power_module.TRAIN_DERIVED_PANELS[1]
    panel_sha = _write(root, panel_path, {
        "label": "train-derived exploratory pilot panel", "record_ids": sorted(lineages),
        "record_ids_sha256": "p" * 64, "record_lineages": lineages,
        "leakage": {"test_accessed": False}})
    evaluations = {}
    for arm, kills in (("A", a), ("B", b)):
        result = f"results/fake/eval_{arm}/rehearsal_result.json"
        result_sha = _write(root, result, _result(kills))
        envelope = f"results/fake/eval_{arm}.json"
        envelope_sha = _write(root, envelope, {
            "status": "complete", "panel_record_ids_sha256": "p" * 64,
            "rehearsal_result": {"path": result, "sha256": result_sha}})
        evaluations[arm] = {"envelope": {"path": envelope, "sha256": envelope_sha},
                            "rehearsal_result": {"path": result, "sha256": result_sha}}
    _write(root, power_module.TOOL_ASSISTED_RECEIPT,
           {"evaluations": evaluations, "panel": {"sha256": panel_sha}})
    _write(root, power_module.EXECUTION_DOSE_RECEIPT,
           {"retention": {}, "tracked_design_artifacts": {}})
    return panel_path


RECORDS = {"r1": 0, "r2": 1, "r3": 0, "r4": 1}
LINEAGES = {"r1": "L1", "r2": "L1", "r3": "L2", "r4": "L3"}


def _evidence(root: Path, panel: str, train_ids=frozenset(RECORDS)):
    return paired_evidence(root, "tool_assisted/A", "tool_assisted/B", panel, set(train_ids))


def test_consistent_evidence_is_accepted_with_relative_paths(tmp_path):
    panel = _fake_evidence_root(tmp_path, RECORDS, {**RECORDS, "r1": 1}, LINEAGES)
    evidence = _evidence(tmp_path, panel)
    assert all(evidence["checks"].values())
    assert all(not Path(path).is_absolute() and "\\" not in path
               for path in evidence["inputs_sha256"])


def test_mismatched_record_ids_between_arms_are_refused(tmp_path):
    panel = _fake_evidence_root(tmp_path, RECORDS, {**RECORDS, "r9": 1}, LINEAGES)
    with pytest.raises(EvidenceRefused, match="paired_arms_have_identical_record_ids"):
        _evidence(tmp_path, panel)


def test_panel_lineages_must_cover_exactly_the_evaluated_records(tmp_path):
    lineages = {key: value for key, value in LINEAGES.items() if key != "r4"}
    panel = _fake_evidence_root(tmp_path, RECORDS, RECORDS, lineages)
    with pytest.raises(EvidenceRefused, match="lineages_cover_exactly"):
        _evidence(tmp_path, panel)


def test_a_non_train_record_is_refused(tmp_path):
    panel = _fake_evidence_root(tmp_path, RECORDS, RECORDS, LINEAGES)
    with pytest.raises(EvidenceRefused, match="every_record_is_a_train_view_record"):
        _evidence(tmp_path, panel, train_ids={"r1", "r2", "r3"})


def test_changed_evidence_files_are_refused(tmp_path):
    panel = _fake_evidence_root(tmp_path, RECORDS, RECORDS, LINEAGES)
    (tmp_path / "results/fake/eval_B/rehearsal_result.json").write_text(
        json.dumps(_result({**RECORDS, "r3": 1})), encoding="utf-8")
    with pytest.raises(EvidenceRefused, match="rehearsal_result hash differs"):
        _evidence(tmp_path, panel)
    panel = _fake_evidence_root(tmp_path / "second", RECORDS, RECORDS, LINEAGES)
    (tmp_path / "second" / panel).write_text("{}", encoding="utf-8")
    with pytest.raises(EvidenceRefused, match="panel hash differs"):
        _evidence(tmp_path / "second", panel)


# --- the tracked artifact -------------------------------------------------------------

@pytest.fixture(scope="module")
def tracked():
    if not TRACKED.exists():
        pytest.skip("power analysis not present")
    return TRACKED.read_bytes()


def test_tracked_power_analysis_verifies_by_recomputation(tracked):
    assert b"\r" not in tracked
    report = verify_power_artifact(ROOT, tracked)
    assert sorted(report["power_by_n"], key=int) == [str(n) for n in N_VALUES]
    for evidence in report["evidence"].values():
        assert 0 < evidence["discordance"] < 1 and all(evidence["checks"].values())
        for path in evidence["inputs_sha256"]:
            assert not Path(path).is_absolute() and "\\" not in path
    recommendation = report["recommendation"]
    assert "minimum_acceptable_n" not in recommendation
    assert recommendation["feasibility_option_n"] == 200
    assert "compromise" in recommendation["feasibility_option_label"]
    assert recommendation["planning_scenario"]["true_effect_pp"] == 8.0
    assert min(recommendation["power_at_recommendation"].values()) >= 0.8
    assert all(value <= 0.5 + 1e-9 for value in
               report["power_at_true_effect_equal_to_gate"].values())
    for n in ("400", "500"):
        assert "resources" in report["power_by_n"][n]


@pytest.mark.parametrize("tamper", [
    lambda r: r["evidence"]["toolassist_b_vs_a"].update(discordance=0.5),
    lambda r: r["gate"].update(min_gain_pp=4.0),
    lambda r: r["recommendation"].update(recommended_final_n=200),
    lambda r: r.update(source_sha256="0" * 64),
    lambda r: r["evidence"]["toolassist_b_vs_a"]["inputs_sha256"].update(
        {"C:\\absolute\\path.json": "0" * 64}),
    lambda r: r["evidence"]["toolassist_b_vs_a"]["inputs_sha256"].update(
        {"results/v4_3_tool_assisted_v1/eval_C/rehearsal_result.json": "0" * 64}),
])
def test_a_tampered_power_artifact_is_refused(tracked, tamper):
    report = json.loads(tracked)
    tamper(report)
    with pytest.raises(EvidenceRefused):
        verify_power_artifact(ROOT, encode(report))
