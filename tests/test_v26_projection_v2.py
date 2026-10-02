"""Projection v2: support-aware probabilities, dynamic shortfalls, precise method naming."""
from __future__ import annotations

import pytest

from harness import v26_readiness as rd
from scripts import native_rehearsal_rebuild_v22 as pub
from scripts import v26_projection_v2 as p2


def test_outside_support_is_null_not_zero():
    out = p2.threshold_probability([0, 1, 2, 3], 117, (0, 36))
    assert out["probability"] is None and out["status"] == "NOT_ESTIMABLE_OUTSIDE_SUPPORT"
    assert out["simulated_exceedances"] == 0 and "NOT impossible" in out["meaning"]


def test_inside_support_zero_is_a_real_estimate():
    out = p2.threshold_probability([0, 1, 2, 3], 10, (0, 36))
    assert out["monte_carlo_estimate"] == 0.0 and out["status"] == "ESTIMATED"
    assert "probability" not in out and 0 < out["upper_95_one_sided"] < 1
    assert "NOT an exact probability of zero" in out["interpretation"]
    big = p2.threshold_probability([0] * 10_000, 10, (0, 36))
    assert abs(big["upper_95_one_sided"] - 0.0003) < 0.00002          # rule of three
    assert p2.threshold_probability([5, 5], 4, (0, 36))["monte_carlo_estimate"] == 1.0
    mid = p2.threshold_probability([1] * 50 + [0] * 50, 1, (0, 36))
    assert mid["monte_carlo_estimate"] == 0.5 and 0.5 < mid["upper_95_one_sided"] < 0.7


def test_shortfall_is_derived_from_the_current_counts():
    assert p2.shortfall({"repository_tests": 33, "repositories": 4, "lineages": 33}) == \
        {"repository_tests": 117, "repositories": 4, "lineages": 27}
    assert p2.shortfall({"repository_tests": 28, "repositories": 3, "lineages": 28}) == \
        {"repository_tests": 122, "repositories": 5, "lineages": 32}
    assert p2.shortfall({"repository_tests": 200, "repositories": 9, "lineages": 70}) == \
        {"repository_tests": 0, "repositories": 0, "lineages": 0}


def test_generated_receipt_uses_current_base_and_honest_labels(monkeypatch):
    monkeypatch.setattr(p2, "DRAWS", 300)
    seen = {}
    monkeypatch.setattr(pub, "publish_once", lambda rel, payload: seen.setdefault(rel, payload))
    p2.main()
    r = seen[p2.RECEIPT]
    assert r["observed_base"]["repository_tests"] == 33
    assert r["derived_shortfall"]["repository_tests"] == 117          # not a stale 122
    assert r["method"] == "repository-cluster bootstrap planning simulation"
    assert "posterior" not in r["method"] and r["is_observed_evidence"] is False
    t = r["training_pool"]["thresholds"]["additional_tests"]
    assert t["probability"] is None and t["status"] == "NOT_ESTIMABLE_OUTSIDE_SUPPORT"
    assert r["confirmation_pool"]["thresholds"]["targets_80"]["probability"] is None
    flat = repr(r)
    assert "upper_bound" not in flat and "'maximum'" not in flat


def test_a_projection_cannot_pass_or_fail_an_observed_gate():
    with pytest.raises(rd.ProjectionNotEvidence):
        rd.observed_training_gate([{"canonical_test": "c", "repository": "r", "lineage": "l",
                                    "kind": "projection"}])


def test_confirmation_repositories_count_qualified_targets_not_training_positives():
    c = p2.clusters()
    d = p2.simulate(30, c, draws=200, seed=3)
    assert all(q >= p for q, p in zip(d["repos_with_qualified"], d["repos_with_positive"]))
