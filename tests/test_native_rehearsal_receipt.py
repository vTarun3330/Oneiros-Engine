"""Phase 4 native rehearsal receipt: complete, environment failures separated, honest caveats."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RECEIPT = ROOT / "results" / "sft_root_cause_phase4_native_rehearsal_receipt_v1.json"
MANIFEST = ROOT / "results" / "sft_root_cause_phase4_native_rehearsal_manifest_v1.json"


def _load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_manifest_boundaries_and_audit():
    m = _load(MANIFEST)
    assert m["boundary_checks_passed"] is True
    assert m["protected_access_audit"]["protected_accesses"] == []
    checks = m["boundary_checks"]
    assert checks["per_repository_cap_respected"] and checks["max_targets_per_repository"] <= 12
    assert not checks["pool_repositories_in_corpus_lineages"]
    assert all(not v for v in checks["pool_repositories_shared_with_excluded_confirmation_stores"]
               .values())


def test_rehearsal_is_complete_and_separates_environment_failures():
    r = _load(RECEIPT)
    assert r["complete"] and r["targets"] == len(_load(MANIFEST)["targets"])
    env = r["environment_failures"]["count"]
    rates = r["rates"]
    assert rates["qualified_over_evaluable_admitted"]["n"] == r["targets"] - env
    assert sum(r["categories"].values()) == r["targets"]
    for row in r["per_target"]:
        if row["environment_failure"]:
            assert row["category"] != "official_tests_do_not_distinguish"


def test_projection_is_a_range_and_blockers_are_recorded():
    r = _load(RECEIPT)
    per = r["projection"]["mined_candidates_per_cohort"]
    assert per["optimistic"] <= per["point"] <= per["pessimistic"]
    assert r["projection"]["distinct_repositories_gate_plus_phase6"] == 68
    assert any(b["blocker"] == "fixed_call_construction" for b in r["blockers"])
    assert any("optimistic" in c for c in r["caveats"])
