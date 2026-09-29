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
