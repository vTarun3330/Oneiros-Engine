"""Projection wording and the v2.5 decision scripts (previously untested)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import native_rehearsal_rebuild_v22 as pub
from scripts import v26_acquisition_projection as proj

ROOT = Path(__file__).resolve().parent.parent


def _capture(monkeypatch):
    seen = {}

    def fake(rel, payload):
        seen[rel] = payload
        return "captured"
    monkeypatch.setattr(pub, "publish_once", fake)
    return seen


def _keys(obj, prefix=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield f"{prefix}{k}"
            yield from _keys(v, f"{prefix}{k}.")


def test_v26_projection_never_calls_an_expectation_a_bound(monkeypatch):
    monkeypatch.setattr(proj, "DRAWS", 200)
    seen = _capture(monkeypatch)
    proj.main()
    r = seen[proj.RECEIPT]
    keys = list(_keys({k: v for k, v in r.items() if k != "supersedes_wording_of"}))
    assert not any("upper_bound" in k or "maximum" in k.split(".")[-1] for k in keys
                   if not k.startswith("deterministic_structural_maximum"))
    assert r["is_observed_evidence"] is False
    tp = r["training_pool"]["verified_positives"]
    assert tp["p05"] <= tp["median"] <= tp["p95"]
    assert set(r["scenarios"]) == {"pessimistic", "central", "optimistic"}
    s = r["scenarios"]
    assert s["pessimistic"]["expected_positives_training_pool"] <= \
        s["central"]["expected_positives_training_pool"] <= \
        s["optimistic"]["expected_positives_training_pool"]
    assert 0.0 <= r["training_pool"]["probability"]["positives_>=_shortfall_122"] <= 1.0
    assert any("may not transfer" in x for x in r["limitations"])


def test_simulation_is_seeded_and_deterministic():
    c = proj.clusters()
    a = proj.simulate(5, c, draws=50, seed=1)["draws"]
    b = proj.simulate(5, c, draws=50, seed=1)["draws"]
    assert a == b
    assert c["admissions_per_scanned_repository"].count(1) == 5
    assert len(c["admissions_per_scanned_repository"]) == 20


def test_v25_projection_receipt_is_kept_and_its_mislabel_is_documented(monkeypatch):
    from scripts import v25_acquisition_projection as old
    seen = _capture(monkeypatch)
    old.main()
    payload = seen[old.RECEIPT]
    tracked = json.loads((ROOT / old.RECEIPT).read_text(encoding="utf-8"))
    assert payload == tracked                       # the historical receipt reproduces
    assert "upper_bound_if_every_admitted_commit_became_a_positive" in \
        tracked["projection_training"]              # the documented v2.5 mislabel


def test_v25_preflight_decision_refuses_and_authorizes_nothing(monkeypatch):
    from scripts import native_generated_tests_generate as gen
    from scripts import v25_cpu2_preflight_decision as dec
    monkeypatch.setattr(gen, "model_identity", lambda: {"snapshot_manifest_sha256": "m"})
    monkeypatch.setattr(gen, "adapter_sha256", lambda: "a")
    seen = _capture(monkeypatch)
    dec.main(["--suite", "results/sft_root_cause/native_v25_cpu2_full_suite_fb04648.json"])
    r = seen[dec.RECEIPT]
    assert r["decision"].startswith("REFUSED")
    assert r["gpu_authorization_created"] is False and r["anything_launched"] is False
    assert r["preflight_conditions"]["universe_8_repositories_60_lineages"] is False
    assert len(r["blockers"]) == 5 and not any(r["claims"].values())
