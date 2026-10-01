"""v2.6 decision semantics: observed-only gates, projections never panels or bounds, panels
built underpowered rather than invalid, and the v2.5 history unchanged."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from harness import v26_readiness as rd

ROOT = Path(__file__).resolve().parent.parent


def rows(n, repos=8, lineages=None):
    lineages = lineages or n
    return [{"canonical_test": f"c{i}", "repository": f"r{i % repos}",
             "lineage": f"l{i % lineages}"} for i in range(n)]


@pytest.mark.parametrize("n, status", [(149, "FAIL"), (150, "PASS"), (151, "PASS")])
def test_training_gate_boundaries_on_tests(n, status):
    assert rd.observed_training_gate(rows(n))["status"] == status


@pytest.mark.parametrize("repos, status", [(7, "FAIL"), (8, "PASS"), (9, "PASS")])
def test_training_gate_boundaries_on_repositories(repos, status):
    assert rd.observed_training_gate(rows(150, repos=repos))["status"] == status


@pytest.mark.parametrize("lineages, status", [(59, "FAIL"), (60, "PASS"), (61, "PASS")])
def test_training_gate_boundaries_on_lineages(lineages, status):
    assert rd.observed_training_gate(rows(150, lineages=lineages))["status"] == status


def test_duplicates_cannot_inflate_any_count():
    dup = rows(100) * 2
    out = rd.observed_training_gate(dup)
    assert out["observed"]["repository_tests"] == 100 and out["duplicates_ignored"] == 100
    assert out["status"] == "FAIL"


def test_missing_or_empty_inputs_fail_closed():
    assert rd.observed_training_gate(None)["status"] == "FAIL"
    assert rd.observed_training_gate([])["status"] == "FAIL"
    assert rd.observed_training_gate([{"canonical_test": "c", "repository": "r"}])[
        "status"] == "FAIL"


def test_only_observed_counts_can_pass():
    projected = [{**r, "projected": True} for r in rows(200)]
    with pytest.raises(rd.ProjectionNotEvidence):
        rd.observed_training_gate(projected)


def test_the_historical_v25_result_is_unchanged():
    r = json.loads((ROOT / "results/sft_root_cause_v25_repository_verification_r2.json")
                   .read_text(encoding="utf-8"))["attainable_gate"]
    assert (r["repository_tests"], r["repositories"], r["lineages"]) == (28, 3, 28)
    assert r["passed"] is False


def target(i, repo=None, cx="simple", fam="f0"):
    return {"target_id": f"t{i}", "repository": repo or f"r{i % 10}", "lineage": f"l{i}",
            "function_fingerprint": f"fp{i}", "buggy_commit": "a", "fixed_commit": "b",
            "environment_lock_sha256": "e", "complexity": cx, "defect_family": fam}


def panel(targets):
    return {"frozen": True, "targets_sha256": "h", "targets": targets}


def full_panel():
    cxs, fams = ("simple", "moderate", "complex"), ("f0", "f1", "f2", "f3")
    return panel([target(i, cx=cxs[i % 3], fam=fams[i % 4]) for i in range(80)])


def test_a_projection_is_never_a_panel():
    with pytest.raises(rd.ProjectionNotEvidence):
        rd.panel_status({"kind": "projection", "expected_qualified_targets": 11.7})
    with pytest.raises(rd.ProjectionNotEvidence):
        rd.panel_status({"expected": 90})
    assert rd.panel_status(None)["status"] == "NOT_BUILT"


def test_a_valid_but_small_panel_is_built_underpowered_not_invalid():
    out = rd.panel_status(panel([target(i, repo=f"r{i % 3}") for i in range(12)]))
    assert out["status"] == "BUILT_UNDERPOWERED" and "exploratory" in out["claim_scope"]


def test_a_full_panel_is_confirmation_ready_and_missing_stratum_is_underpowered():
    assert rd.panel_status(full_panel())["status"] == "CONFIRMATION_READY"
    p = full_panel()
    for t in p["targets"]:
        t["complexity"] = "simple"
    assert rd.panel_status(p)["status"] == "BUILT_UNDERPOWERED"


@pytest.mark.parametrize("mutate, fragment", [
    (lambda p: p["targets"][1].update(lineage="l0"), "duplicate lineage"),
    (lambda p: p["targets"][0].pop("fixed_commit"), "missing"),
    (lambda p: p.update(frozen=False), "not frozen"),
    (lambda p: [t.update(repository="r0") for t in p["targets"][:9]], "more than 8"),
])
def test_invalid_panels_are_invalid(mutate, fragment):
    p = full_panel()
    mutate(p)
    out = rd.panel_status(p)
    assert out["status"] == "INVALID" and any(fragment in x for x in out["problems"])


def test_training_overlap_invalidates_the_panel():
    out = rd.panel_status(full_panel(), training_fingerprints=["fp3"])
    assert out["status"] == "INVALID"
    out = rd.panel_status(full_panel(), training_repositories=["r1"])
    assert out["status"] == "INVALID"


def test_readiness_states_are_explicit_and_gpu_defaults_to_false():
    out = rd.readiness({"artifact_integrity": "PASS"})
    assert set(out) == set(rd.STATES)
    assert out["training_corpus_gate"] == "UNKNOWN"
    assert out["gpu_evaluation_authorized"] is False and out["gpu_training_authorized"] is False
    assert rd.readiness({"gpu_training_authorized": "yes"})["gpu_training_authorized"] is False
