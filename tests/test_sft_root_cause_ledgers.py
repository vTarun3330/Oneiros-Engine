"""The root-cause hypothesis ledger obeys its own declared status schema."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LEDGER = ROOT / "results" / "sft_root_cause_hypotheses.json"


def test_every_status_and_status_after_is_declared():
    ledger = json.loads(LEDGER.read_text(encoding="utf-8"))
    allowed = set(ledger["status_values"])
    for hypothesis in ledger["hypotheses"]:
        assert hypothesis["status"] in allowed, (hypothesis["id"], hypothesis["status"])
        for evidence in hypothesis["evidence"]:
            assert evidence["status_after"] in allowed, (hypothesis["id"], evidence)


def test_withdrawn_evidence_names_the_correction_that_withdrew_it():
    ledger = json.loads(LEDGER.read_text(encoding="utf-8"))
    for hypothesis in ledger["hypotheses"]:
        for evidence in hypothesis["evidence"]:
            if evidence.get("withdrawn"):
                assert evidence.get("withdrawn_by", "").startswith("results/"), evidence
