"""Phase 4 power simulation: deterministic, calibrated, and honest about the 5-point rule."""
from __future__ import annotations

import random

import numpy as np

from harness.power_simulation import empirical_groups, required_groups, simulate


def _rows(groups=120, seed=0):
    rng = random.Random(seed)
    rows = []
    for g in range(groups):
        skill = rng.random()
        for i in range(2):
            for arm in ("base", "sft"):
                for level in ("A0", "A0_answer_schema"):
                    rows.append({"arm": arm, "level": level, "group_id": f"g{g}",
                                 "item_key": f"g{g}-{i}", "correct": rng.random() < skill})
    return rows


def test_empirical_groups_keep_items_and_schemas_paired():
    c, t, groups = empirical_groups(_rows(), "base", "sft")
    assert c.shape == t.shape == (120, 2, 2) and len(groups) == 120
    assert not np.isnan(c).any()


def test_simulation_is_deterministic_and_hits_the_target_effect():
    c, t, _ = empirical_groups(_rows(), "base", "sft")
    first = simulate(c, t, groups=150, delta_points=10, sims=300, seed=7)
    assert first == simulate(c, t, groups=150, delta_points=10, sims=300, seed=7)
    for effect in first["mean_simulated_effect_points"]:
        assert abs(effect - 10) < 1.0


def test_group_scenario_hits_the_target_and_is_noisier_than_item_scenario():
    c, t, _ = empirical_groups(_rows(), "base", "sft")
    item = simulate(c, t, groups=150, delta_points=10, sims=300, seed=9, scenario="item")
    group = simulate(c, t, groups=150, delta_points=10, sims=300, seed=9, scenario="group")
    for effect in group["mean_simulated_effect_points"]:
        assert abs(effect - 10) < 1.5
    assert group["mean_se_points"][0] >= item["mean_se_points"][0]


def test_zero_effect_has_about_nominal_false_positive_rate():
    c, t, _ = empirical_groups(_rows(), "base", "sft")
    res = simulate(c, t, groups=200, delta_points=0, sims=600, seed=3)
    assert res["positive_both"]["power"] < 0.05


def test_a_true_five_point_effect_cannot_pass_the_confirmatory_five_point_rule():
    c, t, _ = empirical_groups(_rows(), "base", "sft")
    res = simulate(c, t, groups=2000, delta_points=5, sims=300, seed=5)
    assert res["confirmatory_5"]["power"] < 0.1
    big = simulate(c, t, groups=2000, delta_points=10, sims=300, seed=5)
    assert big["confirmatory_5"]["power"] > 0.8


def test_required_groups_picks_the_smallest_passing_count():
    results = [{"groups": g, "positive_both": {"power": p}} for g, p in
               ((50, 0.3), (100, 0.79), (150, 0.81), (200, 0.95))]
    assert required_groups(results, "positive_both") == 150
    assert required_groups(results[:2], "positive_both") is None
