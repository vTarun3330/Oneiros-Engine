"""Tests for the prospective repository-native power analysis."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.build_repository_native_power_analysis import (
    GATE_PP, minimum_detectable_effect, monte_carlo_power, power, se_pp,
)

ROOT = Path(__file__).resolve().parent.parent


def test_gate_is_the_frozen_five_points():
    assert GATE_PP == 5.0


def test_power_rises_with_n_and_effect_and_falls_with_clustering():
    assert power(100, 0.21, 10, 0.05, 4) < power(300, 0.21, 10, 0.05, 12)
    assert power(200, 0.21, 8, 0.05, 8) < power(200, 0.21, 12, 0.05, 8)
    assert se_pp(200, 0.21, 0, 0.2, 8) > se_pp(200, 0.21, 0, 0.0, 8)
    # A true effect exactly at the gate can never reach 80% power.
    assert power(300, 0.21, GATE_PP, 0.0, 1) <= 0.5 + 1e-9


def test_minimum_detectable_effect_is_consistent_with_power():
    mde = minimum_detectable_effect(300, 0.21, 0.05, 12)
    assert power(300, 0.21, mde, 0.05, 12) >= 0.8
    assert power(300, 0.21, mde - 0.5, 0.05, 12) < 0.8
    assert mde > GATE_PP


def test_monte_carlo_agrees_with_the_analytic_approximation():
    simulated = monte_carlo_power(200, 0.21, 10.0, runs=1500)
    assert simulated == pytest.approx(power(200, 0.21, 10.0, 0.0, 1.0), abs=0.05)


def test_tracked_power_analysis_is_complete():
    path = ROOT / "results" / "v4_3_repository_native_power_analysis.json"
    if not path.exists():
        pytest.skip("power analysis not present")
    assert b"\r" not in path.read_bytes()
    report = json.loads(path.read_text(encoding="utf-8"))
    assert report["gate"]["min_gain_pp"] == 5.0
    assert sorted(report["power_by_n"], key=int) == ["100", "150", "200", "300"]
    assert report["recommendation"]["recommended_final_n"] == 300
    assert report["recommendation"]["minimum_acceptable_n"] == 200
    for evidence in report["evidence"].values():
        assert 0 < evidence["discordance"] < 1
