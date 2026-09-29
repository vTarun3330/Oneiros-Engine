"""Reporting addendum 2026-09-29: honest multiplicity, power and acquisition statements."""
from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ADDENDUM = ROOT / "results" / "sft_root_cause_reporting_addendum_2026-09-29_v1.json"


def _load():
    return json.loads(ADDENDUM.read_text(encoding="utf-8"))


def test_six_test_family_is_labelled_post_hoc_without_familywise_claim():
    m = _load()["multiplicity_3a"]
    assert m["observed"]["design_enumerates_contrasts"] is False
    assert "POST-HOC" in m["status"]
    assert "NO familywise confirmatory claim" in m["correction"]
    assert "unchanged" in m["decisions_unchanged"]
    assert len(m["observed"]["six_test_family_used_in_v3"]) == 6


def test_power_is_planning_evidence_with_every_limitation_listed():
    p = _load()["power_evidence_status"]
    assert p["status"] == "PLANNING EVIDENCE, NOT A POWER GUARANTEE"
    text = " ".join(p["simulation_assumptions"])
    for needle in ("sandwich", "cross-schema", "synthetic", "training-seed", "guardrails"):
        assert needle in text


def test_acquisition_totals_are_explicit_and_consistent():
    a = _load()["acquisition_arithmetic"]
    obs, proj = a["observed"], a["projection"]
    per = proj["candidates_per_cohort"]["point"]
    assert per == math.ceil(350 / (obs["admitted"] / obs["inspected"])) == 8400
    assert proj["candidates_gate_plus_phase6_total"]["point"] == 2 * per == 16800
    assert proj["unauthenticated_api_hours_both_cohorts"]["point"] == \
        round(2 * per * obs["api_calls_per_candidate"] / 60, 1)
    repos = proj["repository_minimum"]
    assert repos["cap_only_minimum_per_cohort"] == math.ceil(350 / 12) == 30
    assert repos["required_distinct_repositories_both_disjoint_cohorts"] == 68
    assert "before native-qualification losses" in proj["statement"]
