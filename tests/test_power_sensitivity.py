"""Power sensitivity: coupling works, the exact gate agrees with the fast method, rules honest."""
from __future__ import annotations

import random

import numpy as np

from harness.power_sensitivity import (
    bootstrap_decisions, rules, sandwich_decisions, simulate_datasets, summarise,
)
from harness.power_simulation import empirical_groups


def _arrays(groups=150, seed=0):
    rng = random.Random(seed)
    rows = []
    for g in range(groups):
        skill = rng.random()
        for i in range(2):
            for arm in ("base", "sft"):
                for level in ("A0", "A0_answer_schema"):
                    rows.append({"arm": arm, "level": level, "group_id": f"g{g}",
                                 "item_key": f"g{g}-{i}", "correct": rng.random() < skill})
    c, t, _ = empirical_groups(rows, "base", "sft")
    return c, t


def test_injected_effect_hits_the_target_under_every_coupling():
    c, t = _arrays()
    for level in ("item", "group"):
        for coupling in (1.0, 0.5, 0.0):
            cs, ts, mask = simulate_datasets(c, t, 200, 10, 400, 3, level, coupling)
            effect = sandwich_decisions(cs, ts, mask)["point"].mean(axis=0)
            assert np.all(np.abs(effect - 10) < 1.5), (level, coupling, effect)


def test_shared_coupling_makes_schema_effects_more_correlated_than_independent():
    c, t = _arrays()
    corr = {}
    for coupling in (1.0, 0.0):
        cs, ts, mask = simulate_datasets(c, t, 150, 7, 600, 11, "item", coupling)
        point = sandwich_decisions(cs, ts, mask)["point"]
        corr[coupling] = np.corrcoef(point[:, 0], point[:, 1])[0, 1]
    assert corr[1.0] > corr[0.0]


def test_exact_bootstrap_and_fast_sandwich_decisions_mostly_agree():
    c, t = _arrays()
    cs, ts, mask = simulate_datasets(c, t, 150, 7, 60, 5, "item", 1.0)
    fast = rules(sandwich_decisions(cs, ts, mask))["positive_both"]
    exact = rules(bootstrap_decisions(cs, ts, mask, 400, 9))["positive_both"]
    assert (fast == exact).mean() >= 0.85


def test_summary_reports_monte_carlo_intervals():
    hits = {"x": np.array([True] * 30 + [False] * 70)}
    s = summarise(hits)["x"]
    assert s["probability"] == 0.3
    low, high = s["monte_carlo_wilson_95"]
    assert low < 0.3 < high
