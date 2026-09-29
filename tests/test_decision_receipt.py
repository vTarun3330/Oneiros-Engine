"""Phase 6 decision receipt: honest roles for A and B, nothing launched, no root-cause claim."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RECEIPT = ROOT / "results" / "sft_root_cause_decision_receipt_2026-09-29_v1.json"


def test_decision_receipt_roles_and_limits():
    r = json.loads(RECEIPT.read_text(encoding="utf-8"))
    assert r["launches_nothing"] is True
    assert r["state_of_evidence"]["root_cause_established"] is False
    assert "ONLY path" in r["choice_A"]["role"]
    assert r["choice_B"]["outcomes"] == ["promising", "harm", "inconclusive_power"]
    assert "replace Choice A" in r["choice_B"]["cannot"]
    assert "NOT untouched" in r["choice_B"]["material"]
    assert r["recommendation"]["next_authorized_gpu_experiment"]["requires_user_authorisation"]
    for source in r["sources"].values():
        assert hashlib.sha256((ROOT / source["path"]).read_bytes()).hexdigest() == \
            source["sha256"]


RECEIPT_V2 = ROOT / "results" / "sft_root_cause_decision_receipt_2026-09-29_v2.json"


def test_decision_receipt_v2_freezes_the_composite_estimand_and_corrections():
    r = json.loads(RECEIPT_V2.read_text(encoding="utf-8"))
    assert r["launches_nothing"] is True
    assert r["state_of_evidence"]["root_cause_established"] is False
    smoke = r["objective_smoke_interpretation"]
    assert smoke["frozen_interpretation"] == "A"
    assert smoke["supervised_tokens"] == {"control_full_completion": 952,
                                          "treatment_value_only": 220}
    assert "COMPOSITE" in r["recommendation"]["next_authorized_gpu_experiment"]["estimand"]
    assert r["choice_A"]["readiness"]["choice_A_ready"] is False
    assert "single seed cannot establish efficacy" in r["choice_B"]["labels"]
    assert r["choice_B"]["old_arm_A_compared_on_panel"] is False
    assert "token-count-matched third arm" in r["not_launched"]
    assert r["recommendation"]["next_authorized_gpu_experiment"]["requires_user_authorisation"]
    for source in r["sources"].values():
        assert hashlib.sha256((ROOT / source["path"]).read_bytes()).hexdigest() == \
            source["sha256"]


RECEIPT_V3 = ROOT / "results" / "sft_root_cause_decision_receipt_2026-09-29_v3.json"


def test_decision_receipt_v3_corrects_choice_a_and_supersedes_preflight_v1():
    r = json.loads(RECEIPT_V3.read_text(encoding="utf-8"))
    pilot = r["choice_A"]["fixed_call_yield"]["runtime_argument_capture_pilot"]
    assert pilot["targets_with_usable_fixed_call"] == "3/24"
    assert pilot["repositories_with_usable_fixed_call"] == "1/8"
    assert pilot["all_successes_in_one_repository"] is True
    assert "DESCRIPTIVE" in pilot["interval_status"]
    assert r["choice_A"]["readiness"]["choice_A_ready"] is False
    assert r["choice_A"]["mass_acquisition"] == "REFUSED"
    assert "SUPERSEDED" in r["choice_B"]["preflight_v1"]["status"]
    assert r["state_of_evidence"]["root_cause_established"] is False
    assert r["state_of_evidence"]["generalization_established"] is False
    for source in r["sources"].values():
        assert hashlib.sha256((ROOT / source["path"]).read_bytes()).hexdigest() == source["sha256"]
