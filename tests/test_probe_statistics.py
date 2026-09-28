"""Phase 3 v2 inference: finite p-values, Holm, clustered interactions, stratum-exact H4 rule."""
from __future__ import annotations

import random

import pytest

from harness.probe_statistics import (
    H4_STRATA, bootstrap_contrast, bootstrap_interaction, decide_h4, holm, schema_dependent,
    sign_flip_test,
)


def _rows(p_a, p_b, groups=60, per_group=4, seed=0, levels=("L",)):
    rng = random.Random(seed)
    rows = []
    for g in range(groups):
        for i in range(per_group):
            for level in levels:
                for arm, p in (("a", p_a), ("b", p_b)):
                    rows.append({"arm": arm, "level": level, "group_id": f"g{g}",
                                 "correct": rng.random() < p, "item": i})
    return rows


def test_finite_resample_p_values_are_never_zero():
    rows = _rows(0.05, 0.95)
    result = sign_flip_test(rows, ("a", "L"), ("b", "L"), resamples=2000)
    assert result["extreme"] == 0 and result["p_two_sided"] == 1 / 2001 > 0
    assert "smallest attainable" in result["report"]


def test_identical_arms_are_not_significant():
    rows = _rows(0.4, 0.4)
    for r in rows:                      # make the arms literally identical
        if r["arm"] == "b":
            r["correct"] = next(x["correct"] for x in rows if x["arm"] == "a"
                                and x["group_id"] == r["group_id"] and x["item"] == r["item"])
    assert sign_flip_test(rows, ("a", "L"), ("b", "L"), resamples=2000)["p_two_sided"] == 1.0


def test_a_large_paired_effect_is_detected():
    rows = _rows(0.2, 0.7)
    assert sign_flip_test(rows, ("a", "L"), ("b", "L"), resamples=2000)["p_two_sided"] < 0.01
    assert bootstrap_contrast(rows, ("a", "L"), ("b", "L"), resamples=2000)["ci95_points"][0] > 0


def test_holm_is_monotone_and_never_below_raw():
    raw = {"C1": 0.01, "C2": 0.04, "C3": 0.03}
    adjusted = holm(raw)
    assert all(adjusted[k] >= raw[k] for k in raw)
    order = sorted(raw, key=raw.get)
    assert [adjusted[k] for k in order] == sorted(adjusted[k] for k in order)
    assert adjusted["C1"] == pytest.approx(0.03) and adjusted["C2"] == pytest.approx(0.06)


def test_interaction_is_clustered_by_semantic_group():
    rows = _rows(0.3, 0.5, groups=30, per_group=2, levels=("P", "Q"))
    cells = ((("a", "P"), ("b", "P")), (("a", "Q"), ("b", "Q")))
    once = bootstrap_interaction(rows, *cells, resamples=1000)
    # Duplicating every item inside its group adds items but no clusters: an
    # item-level bootstrap would narrow; a group-level one must not change.
    doubled = rows + [{**r, "item": r["item"] + 100} for r in rows]
    twice = bootstrap_interaction(doubled, *cells, resamples=1000)
    assert once["groups"] == twice["groups"] == 30
    assert once["ci95_points"] == twice["ci95_points"]


def _contrast(points, low, high):
    return {"difference_points": points, "ci95_points": [low, high]}


def test_an_exposed_sign_flip_blocks_h4_even_when_the_pooled_stratum_does_not_flip():
    strata = {s: {"prefill": _contrast(1.5, -2.5, 5.6), "answer": _contrast(2.7, -1.0, 6.5)}
              for s in H4_STRATA}
    strata["exposed"] = {"prefill": _contrast(-1.667, -7.6, 4.2),
                         "answer": _contrast(5.0, 0.42, 9.8)}
    assert not schema_dependent(1.5, 2.7)
    decision = decide_h4(strata)
    assert decision["status"] == "open"
    assert decision["qualifier"] == "schema_dependent_exposure_result"
    assert decision["schema_dependent_strata_used"] == ["exposed"]


def test_without_a_flip_rule_one_still_applies_and_missing_strata_are_refused():
    strata = {s: {"prefill": _contrast(-1.0, -6.0, 4.0), "answer": _contrast(-0.5, -5, 4)}
              for s in H4_STRATA}
    assert decide_h4(strata)["status"] == "weakened"
    del strata["exposed_novel"]
    with pytest.raises(ValueError):
        decide_h4(strata)
