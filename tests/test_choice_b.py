"""Choice B: frozen split isolation, matched arms, one-look analysis rules."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from harness import choice_b as cb

ROOT = Path(__file__).resolve().parent.parent
SPLIT = ROOT / "results" / "sft_root_cause_phase4_choice_b_split_v2.json"
SPLIT_V1 = ROOT / "results" / "sft_root_cause_phase4_choice_b_split_v1.json"
REFUSED = ROOT / "results" / "sft_root_cause_phase4_choice_b_preflight_split_v1_REFUSED.json"
RECEIPT = ROOT / "results" / "sft_root_cause_phase4_choice_b_preflight_receipt.json"


def _schema(point, low, high, answer_ok=True):
    return {"accuracy": {"point": point, "low": low, "high": high}, "answer_rate_gate": answer_ok}


def test_outcomes_follow_the_frozen_rules():
    ok = {"validity": True, "diversity": True}
    promising = {"a": _schema(6, 1, 11), "b": _schema(4, 0.5, 8)}
    assert cb.decide(promising, ok)["outcome"] == "promising"
    assert cb.decide({"a": _schema(-6, -11, -1), "b": _schema(2, -2, 6)}, ok)["outcome"] == "harm"
    assert cb.decide(promising, {"validity": False, "diversity": True})["outcome"] == "harm"
    no_answers = {"a": _schema(6, 1, 11, answer_ok=False), "b": _schema(4, 0.5, 8)}
    assert cb.decide(no_answers, ok)["outcome"] == "harm"
    flip = {"a": _schema(3, -1, 7), "b": _schema(-1, -5, 3)}
    decided = cb.decide(flip, ok)
    assert decided["outcome"] == "inconclusive_power" and decided["schema_sign_flip"]
    assert cb.decide({"a": _schema(2, -1, 5), "b": _schema(3, -1, 7)}, ok)["outcome"] == \
        "inconclusive_power"


def test_cluster_bootstrap_is_paired_by_group_and_deterministic():
    groups = [f"g{i // 2}" for i in range(200)]
    control = [float(i % 3 == 0) for i in range(200)]
    treatment = [1.0 if i % 10 == 0 else c for i, c in enumerate(control)]
    first = cb.cluster_bootstrap_difference(groups, control, treatment, resamples=2000)
    assert first == cb.cluster_bootstrap_difference(groups, control, treatment, resamples=2000)
    expected = (sum(treatment) - sum(control)) / 200 * 100
    assert first["point"] == pytest.approx(expected, abs=1e-3)
    assert first["groups"] == 100 and first["low"] <= first["point"] <= first["high"]


def test_gate_assignment_is_seeded_and_refuses_small_pools():
    feasible = {f"g{i:03d}": {} for i in range(300)}
    gate = cb.assign_gate(feasible, 200)
    assert gate == cb.assign_gate(dict(reversed(list(feasible.items()))), 200)
    assert len(gate) == 200 == len(set(gate))
    with pytest.raises(ValueError):
        cb.assign_gate({"g": {}}, 2)


def test_cap_and_order_removes_repeats_and_caps_groups():
    rows = [{"record_id": f"r{i}", "group_id": "big", "call": f"f({i})", "value": "1"}
            for i in range(cb.GROUP_CAP + 5)]
    rows += [dict(rows[0])]
    out = cb.cap_and_order(rows)
    assert len(out) == cb.GROUP_CAP
    assert len({(r["record_id"], r["call"], r["value"]) for r in out}) == len(out)


def test_launch_commands_differ_only_in_the_arm():
    control = cb.arm_command("py", "control")
    treatment = cb.arm_command("py", "treatment")
    assert [d[1:] for d in cb.command_difference(control, treatment)] == [
        ("phase4_choice_b_control", "phase4_choice_b_treatment"), ("control", "treatment")]


def test_frozen_split_is_isolated_unique_and_bound_to_the_spec():
    split = json.loads(SPLIT.read_text(encoding="utf-8"))
    gate = set(split["gate"]["group_ids"])
    rows = split["rows"]
    assert len(gate) == cb.GATE_GROUPS == len(split["gate"]["functions"])
    assert split["gate"]["items"] == 2 * cb.GATE_GROUPS
    assert not gate & {r["group_id"] for r in rows}
    assert not {f["record_id"] for f in split["gate"]["functions"]} & {r["record_id"] for r in rows}
    assert len({(r["record_id"], r["call"], r["value"]) for r in rows}) == len(rows)
    assert split["splits_read"] == ["train"]
    assert split["protected_access_audit"]["protected_paths_opened"] == []
    assert split["evaluation_spec_sha256"] == cb.spec_sha256(cb.frozen_evaluation_spec())
    assert split["evaluation_spec"]["old_arm_A_evaluated"] is False
    assert set(split["evaluation_spec"]["outcomes"]) == set(cb.OUTCOMES)
    for rel, digest in split["inputs"].items():
        assert hashlib.sha256((ROOT / rel).read_bytes()).hexdigest() == digest, rel


def test_run_contracts_differ_only_in_arm_objective_and_output():
    from scripts import phase4_choice_b as runner
    contracts = {arm: runner.run_contract(arm, split_sha="s", receipt_sha="r")
                 for arm in runner.ARMS}
    assert sorted(k for k in contracts["control"]
                  if contracts["control"][k] != contracts["treatment"][k]) == [
        "arm", "objective_mode", "output_dir"]
    assert runner.checkpoint_dir("control") != runner.checkpoint_dir("treatment")
    assert runner.ARMS == {"control": "full_completion", "treatment": "value_only"}


def test_preflight_receipt_is_green_and_launches_nothing():
    if not RECEIPT.exists():
        pytest.skip("preflight receipt not yet written")
    receipt = json.loads(RECEIPT.read_text(encoding="utf-8"))
    assert receipt["ready"] is True and receipt["problems"] == []
    assert receipt["training_launched"] is False and receipt["evaluation_launched"] is False
    assert all(receipt["isolation"].values()) and all(receipt["arm_equality"].values())
    assert receipt["run_contract_differences"] == ["arm", "objective_mode", "output_dir"]
    assert receipt["split"]["sha256"] == hashlib.sha256(SPLIT.read_bytes()).hexdigest()
    assert receipt["protected_access_audit"]["protected_paths_opened"] == []
    assert "COMPOSITE" in receipt["estimand"]


def test_v1_split_is_preserved_superseded_and_its_refusal_recorded():
    split = json.loads(SPLIT.read_text(encoding="utf-8"))
    assert split["supersedes"]["sha256"] == hashlib.sha256(SPLIT_V1.read_bytes()).hexdigest()
    refused = json.loads(REFUSED.read_text(encoding="utf-8"))
    assert refused["ready"] is False
    assert any("token fit" in problem for problem in refused["problems"])
    v1 = json.loads(SPLIT_V1.read_text(encoding="utf-8"))
    # Only the token-fit rule changed: same pool, same gate.
    assert v1["gate"]["group_ids"] == split["gate"]["group_ids"]
    assert v1["pool"] == split["pool"]
    assert split["training"]["rows_excluded_by_token_fit"] > 0
