"""Phase 4 state pointers: idempotent, resolvable, history kept, nothing frozen."""
from __future__ import annotations

import json
from pathlib import Path
import shutil

from scripts import update_phase4_pointers as pointers

ROOT = Path(__file__).resolve().parent.parent


def test_pointers_are_idempotent_and_resolve(tmp_path):
    for rel in [pointers.STATE, pointers.V1_DRAFT, *pointers.CURRENT.values()]:
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / rel, tmp_path / rel)
    first = pointers.run(tmp_path)
    data = (tmp_path / pointers.STATE).read_bytes()
    assert pointers.run(tmp_path) == first
    assert (tmp_path / pointers.STATE).read_bytes() == data
    state = json.loads(data)
    phase4 = state["phases"][4]
    assert phase4["frozen"] is False and phase4["training_started"] is False
    assert phase4["current"]["design"]["path"].endswith("DRAFT_V2.md")
    assert phase4["superseded_designs"][0]["path"] == pointers.V1_DRAFT
    assert [e["event"] for e in state["log"]].count(pointers.EVENT) == 1
    assert "design_draft" not in phase4


def test_power_analysis_rules_are_honest():
    report = json.loads((ROOT / pointers.CURRENT["power_analysis"]).read_text(encoding="utf-8"))
    assert report["cohort_frozen"] is False and report["gpu_used"] is False
    required = report["required_groups_for_80pct_power"]
    for scenario in ("item", "group"):
        assert required[scenario]["confirmatory_5"]["5"] is None
        assert required[scenario]["confirmatory_5"]["3"] is None
    assert "NOT untouched" in report["pools"]["arm_a_exposed_remainder"]["label"]
    assert "UNRESOLVED" in report["repository_disjoint_phase6"]
    assert "NOT estimated" in report["method"]["not_modelled"][0]
