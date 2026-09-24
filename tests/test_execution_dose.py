"""Tests for the execution-dose-and-retention experiment (design, gates, receipts)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from harness.execution_dose import (
    cluster_bootstrap_difference, dose_summary, largest_remainder, plan_replay,
    removal_ceilings, replay_balance_report, replay_removal_quotas, select_execution_rows,
)
from scripts.analyse_execution_dose_pilot import (
    MIN_GAIN_PP, RETENTION_MARGIN_PP, decide, mechanism_section, paired_comparison,
    retention_verdict,
)

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"


def _control_rows():
    rows = []
    layout = [("mbpp", "simple", "arith", "function_assertion", 12),
              ("mbpp", "complex", "bound", "function_assertion", 8),
              ("humaneval", "moderate", "arith", "function_assertion", 6),
              ("SWE-bench Verified", "complex", "unknown", "repository_pytest_fragment", 6),
              ("BugsInPy", "simple", "unknown", "repository_unittest_fragment", 4)]
    for source, tier, family, mode, count in layout:
        rows += [{"source_dataset": source, "complexity_tier": tier, "bug_family": family,
                  "execution_mode": mode} for _ in range(count)]
    return rows


def test_largest_remainder_is_exact_and_deterministic():
    quotas = largest_remainder(10, {"a": 1, "b": 1, "c": 1})
    assert sum(quotas.values()) == 10
    assert quotas == {"a": 4, "b": 3, "c": 3}
    assert largest_remainder(256, {"simple": 37, "moderate": 46, "complex": 39}) == {
        "simple": 78, "moderate": 96, "complex": 82}


def test_replay_quotas_are_exact_and_never_empty_a_stratum():
    rows = _control_rows() + [{"source_dataset": "manual", "complexity_tier": "simple",
                               "bug_family": "x", "execution_mode": "function_assertion"}]
    quotas = replay_removal_quotas(rows, 18)
    assert sum(quotas.values()) == 18
    assert quotas[("manual", "simple")] == 0


def test_plan_replay_keeps_every_category_and_respects_order():
    rows = _control_rows()
    lengths = list(range(len(rows)))
    removed = plan_replay(rows, 9, lambda position: -lengths[position])
    assert len(removed) == 9 == len(set(removed))
    report = replay_balance_report(rows, removed)
    assert report["no_category_removed"]
    # Longest-first within a stratum: the removed mbpp/simple rows are its last ones.
    simple = [p for p in removed if rows[p]["source_dataset"] == "mbpp"
              and rows[p]["complexity_tier"] == "simple"]
    assert simple == sorted(simple) and simple[-1] == 11


def test_removal_ceilings_bound_loss_and_keep_one_row():
    rows = _control_rows()
    ceilings = removal_ceilings(rows, 18, "execution_mode")
    assert ceilings["repository_unittest_fragment"] <= 3
    assert all(value >= 0 for value in ceilings.values())


def test_plan_replay_refuses_rather_than_relaxing():
    rows = [{"source_dataset": "s", "complexity_tier": "t", "bug_family": "f",
             "execution_mode": "m"} for _ in range(3)]
    rows.append({"source_dataset": "u", "complexity_tier": "t", "bug_family": "g",
                 "execution_mode": "m"})
    with pytest.raises(ValueError, match="emptying a stratum"):
        plan_replay(rows, 3, lambda position: position)


def _pool_row(index, tier, source="mbpp", lineage=None, family="arith", target=None):
    return {"record_id": f"r{index}", "complexity_tier": tier, "source_dataset": source,
            "function_lineage": lineage or f"L{index}", "bug_family": family,
            "trace_completion_sha256": target or f"t{index}"}


def test_select_execution_rows_honours_caps_quotas_and_unique_targets():
    pool = [_pool_row(i, "simple", lineage="L0") for i in range(5)]
    pool += [_pool_row(10 + i, "complex", source="humaneval") for i in range(4)]
    pool += [_pool_row(20, "complex", target="t10")]  # duplicate target
    chosen = select_execution_rows(pool, 4, lineage_cap=2, family_cap=10,
                                   tier_quotas={"simple": 2, "complex": 2},
                                   order_key=lambda row: row["record_id"])
    assert len(chosen) == 4
    assert sum(row["function_lineage"] == "L0" for row in chosen) == 2
    assert len({row["trace_completion_sha256"] for row in chosen}) == 4
    with pytest.raises(ValueError):
        select_execution_rows(pool, 5, lineage_cap=1, family_cap=10,
                              tier_quotas={"simple": 3, "complex": 2},
                              order_key=lambda row: row["record_id"])


def test_dose_summary_reports_both_units():
    summary = dose_summary([10, 10, 10, 10], [0], [30])
    assert summary["execution_example_share"] == 0.25
    assert summary["treatment_supervised_tokens"] == 60
    assert summary["execution_token_share"] == 0.5
    assert summary["treatment_to_control_mass_ratio"] == 1.5


def test_cluster_bootstrap_is_deterministic_and_resamples_clusters():
    pairs = [("A", False, True)] * 5 + [("B", True, True)] * 5 + [("C", True, False)] * 2
    first = cluster_bootstrap_difference(pairs, replicates=2000, seed=7)
    again = cluster_bootstrap_difference(pairs, replicates=2000, seed=7)
    assert first == again
    assert first["difference_pp"] == pytest.approx(100 * 3 / 12)
    assert first["low_pp"] <= first["difference_pp"] <= first["high_pp"]
    assert first["clusters"] == 3 and first["units"] == 12
    flat = cluster_bootstrap_difference([("A", True, True)] * 4, replicates=100, seed=1)
    assert flat["low_pp"] == flat["high_pp"] == 0.0


def test_retention_verdict_is_three_way_at_the_frozen_margin():
    assert RETENTION_MARGIN_PP == 3.0
    assert retention_verdict(-3.0, 1.0) == "pass"
    assert retention_verdict(-3.01, 1.0) == "inconclusive"
    assert retention_verdict(-9.0, -3.01) == "fail"


def _mechanism_artifact(arm, correct_intended, correct_actual, strict_rate=0.0, hits=0):
    def rows(correct):
        return [{"record_id": f"i{n:03d}", "lenient": {
            "verdict": "correct" if n < correct else "wrong_value"}} for n in range(97)]
    return {"status": "complete", "arm": arm, "model": "m", "model_revision": "r",
            "pilot_development_sha256": "p", "items": 97,
            "conditions": ["intended_output", "shown_actual_output"],
            "decoding": {"strategy": "greedy", "max_new_tokens": 128, "batch_size": 8,
                         "system_prompt": None},
            "detail": {"intended_output": rows(correct_intended),
                       "shown_actual_output": rows(correct_actual)},
            "summary": {condition: {"strict_answer_rate": strict_rate,
                                    "strict_accuracy_per_requested": 0.0,
                                    "lenient_accuracy_per_requested": 0.0,
                                    "completion_limit_hits": hits}
                        for condition in ("intended_output", "shown_actual_output")}}


def test_mechanism_gate_requires_both_conditions_and_guards():
    control = _mechanism_artifact("control", 26, 23)
    strong = _mechanism_artifact("dose_treatment", 40, 40)
    result = mechanism_section(control, strong, None)
    assert result["passed"]
    assert result["treatment_minus_control"]["intended_output"]["gained"] == 14
    one_condition = _mechanism_artifact("dose_treatment", 40, 24)
    assert not mechanism_section(control, one_condition, None)["passed"]
    capped = _mechanism_artifact("dose_treatment", 40, 40, hits=1)
    assert not mechanism_section(control, capped, None)["checks"]["no_completion_limit_hits"]
    drifted = dict(strong, decoding={"strategy": "sampled"})
    with pytest.raises(ValueError):
        mechanism_section(control, drifted, None)


def test_paired_comparison_counts_discordant_pairs():
    pairs = [(True, True)] * 23 + [(False, True)] * 2 + [(False, False)] * 72
    result = paired_comparison(pairs)
    assert (result["gained"], result["lost"]) == (2, 0)
    assert result["newcombe90_low_pp"] == pytest.approx(-0.5622, abs=1e-3)
    assert result["exact_mcnemar_p"] == pytest.approx(0.5)


def test_decision_paths_never_open_confirmation():
    mech = {"passed": True, "treatment_minus_control": {}}
    assert decide(mech, {"passed": True, "verdict": "pass"})["outcome"] == \
        "mechanism_supported_at_25pct_dose"
    assert decide(mech, {"passed": False, "verdict": "inconclusive"})["outcome"] == \
        "mechanism_gain_with_retention_inconclusive"
    null = {"passed": False, "treatment_minus_control": {
        c: {"newcombe90_high_pp": MIN_GAIN_PP - 0.1}
        for c in ("intended_output", "shown_actual_output")}}
    decision = decide(null, {"passed": True, "verdict": "pass"})
    assert decision["outcome"] == "null_at_25pct_dose_5pp_gain_excluded"
    for outcome in (decision, decide(mech, {"passed": True, "verdict": "pass"})):
        assert outcome["confirmation_opening_permitted"] is False
        assert outcome["promotion_permitted"] is False


def test_mechanism_evaluator_decoding_matches_the_control_evaluator():
    from scripts.evaluate_execution_dose_mechanism import DECODING
    from scripts.evaluate_execution_supervision_pilot import BATCH_SIZE, MAX_NEW_TOKENS
    assert DECODING == {"strategy": "greedy", "max_new_tokens": MAX_NEW_TOKENS,
                        "batch_size": BATCH_SIZE, "system_prompt": None}
    assert (MAX_NEW_TOKENS, BATCH_SIZE) == (128, 8)


def test_launch_gate_refuses_an_unready_receipt(tmp_path):
    from scripts.preflight_execution_dose_ab import verify_receipt_for_launch
    receipt = tmp_path / "receipt.json"
    receipt.write_text(json.dumps({"ready": False}), encoding="utf-8")
    with pytest.raises(SystemExit):
        verify_receipt_for_launch(receipt)


def test_tracked_dose_design_artifacts_are_consistent():
    """Census -> design -> manifest -> panel verify from tracked bytes alone."""
    census = RESULTS / "v4_3_execution_dose_census.json"
    design = RESULTS / "v4_3_execution_dose_design.json"
    manifest = RESULTS / "v4_3_execution_dose_dataset_manifest.json"
    panel = RESULTS / "v4_3_execution_dose_retention_panel.json"
    if not all(path.exists() for path in (census, design, manifest, panel)):
        pytest.skip("execution-dose design artifacts not present")
    for path in (census, design, manifest, panel):
        assert b"\r" not in path.read_bytes(), f"{path.name} contains CR bytes"
    census_data = json.loads(census.read_text(encoding="utf-8"))
    design_data = json.loads(design.read_text(encoding="utf-8"))
    manifest_data = json.loads(manifest.read_text(encoding="utf-8"))
    panel_data = json.loads(panel.read_text(encoding="utf-8"))
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    assert design_data["dataset_manifest_sha256"] == digest
    assert design_data["census_sha256"] == hashlib.sha256(census.read_bytes()).hexdigest()
    assert design_data["chosen_design"] == "d25" == manifest_data["chosen_design"]
    assert design_data["designs"]["d50"]["feasible"] is False
    assert manifest_data["execution_examples"] == 256
    assert manifest_data["replay_balance"]["no_category_removed"] is True
    assert manifest_data["dose"]["treatment_to_control_mass_ratio"] <= 1.25
    assert all(manifest_data["leakage_checks"].values())
    assert not any(value for key, value in census_data["leakage"].items()
                   if key != "splits_opened")
    assert census_data["leakage"]["splits_opened"] == ["train"]
    assert all(panel_data["checks"].values())
    assert panel_data["records"] == len(panel_data["record_ids"]) == 613
    ids_digest = hashlib.sha256(json.dumps(
        panel_data["record_ids"], separators=(",", ":")).encode("utf-8")).hexdigest()
    assert ids_digest == panel_data["record_ids_sha256"]
    assert not set(manifest_data["execution_record_ids"]) & set(panel_data["record_ids"])
