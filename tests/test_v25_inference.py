"""v2.5 confirmation inference on known null, positive and underpowered synthetic cases."""
from __future__ import annotations

import random

from harness import v25_inference as inf
from scripts import v25_power_simulation as ps


def test_exact_sign_flip_on_a_clear_effect_and_on_symmetry():
    assert inf.sign_flip_test([0.2] * 10)["p_value"] == 2 / 1024          # only +/- all
    assert inf.sign_flip_test([0.2, -0.2] * 5)["p_value"] == 1.0
    out = inf.sign_flip_test([0.1] * 20, flips=1000)
    assert out["method"] == "monte_carlo" and out["p_value"] < 0.01


def test_repository_means_and_bootstrap():
    rows = [("a", 1.0), ("a", 0.0), ("b", 0.5)]
    assert inf.repository_means(rows) == {"a": 0.5, "b": 0.5}
    ci = inf.cluster_bootstrap_ci([(f"r{i}", 0.1) for i in range(10)], resamples=500)
    assert ci["lower"] == ci["upper"] == ci["estimate"] == 0.1
    assert inf.cluster_bootstrap_ci(rows, resamples=200, seed=1) == \
        inf.cluster_bootstrap_ci(rows, resamples=200, seed=1)


def test_null_is_calibrated_positive_is_powered_small_panel_is_underpowered():
    params = {**ps.PARAMS, "simulations": 400}
    null = ps.power(10, 8, 0.10, 0.0, params)
    assert null <= 0.08                                                   # ~alpha
    strong = ps.power(10, 8, 0.05, 0.30, params)
    assert strong >= 0.9
    small = ps.power(4, 3, 0.05, 0.10, params)
    assert small < 0.2                                     # 2^4 flips: min p = 0.125


def test_simulation_is_deterministic():
    params = {**ps.PARAMS, "simulations": 50}
    assert ps.power(10, 8, 0.05, 0.10, params) == ps.power(10, 8, 0.05, 0.10, params)
    rng = random.Random(0)
    rows = ps.simulate_panel(rng, 3, 2, 0.05, 0.1, 1.1, 3)
    assert len(rows) == 6 and {r for r, _ in rows} == {"r0", "r1", "r2"}
