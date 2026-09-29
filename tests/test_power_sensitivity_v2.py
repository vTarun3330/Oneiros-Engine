"""Power sensitivity v2: one common selector gives the declared coupling mixture."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from harness import power_sensitivity as v1
from harness import power_sensitivity_v2 as v2
from tests.test_power_sensitivity import _arrays

ROOT = Path(__file__).resolve().parent.parent
V1_RECEIPT = ROOT / "results" / "sft_root_cause_phase4_power_sensitivity_v1.json"
SHAPE = (40, 60, 2, 2)


def _draws(level, coupling, seed=7):
    return v2.coupled_draws(np.random.default_rng(seed), SHAPE, level, coupling,
                            return_selector=True)


def test_coupling_zero_gives_independent_draws():
    for level in ("item", "group"):
        u, selector = _draws(level, 0.0)
        assert not selector.any()
        assert not np.any(u[..., 0] == u[..., 1])
        assert abs(np.corrcoef(u[..., 0].ravel(), u[..., 1].ravel())[0, 1]) < 0.05


def test_coupling_one_gives_fully_shared_draws():
    for level in ("item", "group"):
        u, selector = _draws(level, 1.0)
        assert selector.all()
        assert np.array_equal(u[..., 0], u[..., 1])


def test_partial_coupling_matches_the_requested_mixture_not_its_square():
    for coupling in (0.25, 0.5, 0.8):
        u, selector = _draws("item", coupling, seed=11)
        both_shared = float((u[..., 0] == u[..., 1]).mean())
        assert abs(both_shared - coupling) < 0.03, (coupling, both_shared)
        assert abs(float(selector.mean()) - coupling) < 0.03
        # The v1 defect: independent per-schema selectors share with probability c**2.
        rng = np.random.default_rng(11)
        legacy = v1._draws(rng, SHAPE, "item", coupling)
        assert abs(float((legacy[..., 0] == legacy[..., 1]).mean()) - coupling ** 2) < 0.03


def test_item_and_group_levels_behave_as_declared():
    u, selector = _draws("group", 0.5)
    # Group level: every item of a group carries the same draw and the same selector.
    assert np.array_equal(u[:, :, 0, :], u[:, :, 1, :])
    assert np.array_equal(selector[:, :, 0], selector[:, :, 1])
    u, selector = _draws("item", 0.5)
    assert not np.array_equal(u[:, :, 0, :], u[:, :, 1, :])
    assert 0.3 < float((selector[:, :, 0] != selector[:, :, 1]).mean()) < 0.7
    with pytest.raises(ValueError):
        v2.coupled_draws(np.random.default_rng(0), SHAPE, "schema", 0.5)
    with pytest.raises(ValueError):
        v2.coupled_draws(np.random.default_rng(0), SHAPE, "item", 1.5)


def test_v2_reproduces_v1_exactly_at_coupling_zero_and_one_only():
    c, t = _arrays()
    for level in ("item", "group"):
        for coupling in (0.0, 1.0):
            old = v1.simulate_datasets(c, t, 120, 7, 50, 3, level, coupling)
            new = v2.simulate_datasets(c, t, 120, 7, 50, 3, level, coupling)
            for a, b in zip(old, new):
                assert np.array_equal(a, b, equal_nan=True)
    old = v1.simulate_datasets(c, t, 120, 7, 50, 3, "item", 0.5)
    new = v2.simulate_datasets(c, t, 120, 7, 50, 3, "item", 0.5)
    assert not np.array_equal(old[1], new[1], equal_nan=True)


def test_injected_effect_still_hits_the_target_and_summaries_are_deterministic():
    c, t = _arrays()
    for level in ("item", "group"):
        cs, ts, mask = v2.simulate_datasets(c, t, 200, 10, 400, 3, level, 0.5)
        effect = v2.sandwich_decisions(cs, ts, mask)["point"].mean(axis=0)
        assert np.all(np.abs(effect - 10) < 1.5), (level, effect)
    runs = [v2.summarise(v2.rules(v2.sandwich_decisions(
        *v2.simulate_datasets(c, t, 150, 7, 200, 5, "item", 0.5)))) for _ in range(2)]
    assert runs[0] == runs[1]


def test_partial_coupling_correlation_sits_between_independent_and_shared():
    c, t = _arrays()
    corr = {}
    for coupling in (0.0, 0.5, 1.0):
        point = v2.sandwich_decisions(*v2.simulate_datasets(
            c, t, 150, 7, 800, 11, "item", coupling))["point"]
        corr[coupling] = np.corrcoef(point[:, 0], point[:, 1])[0, 1]
    assert corr[0.0] < corr[0.5] < corr[1.0]


def test_v1_receipt_and_module_are_unchanged():
    import subprocess
    for rel in ("results/sft_root_cause_phase4_power_sensitivity_v1.json",
                "harness/power_sensitivity.py", "scripts/phase4_power_sensitivity.py"):
        committed = subprocess.run(["git", "show", f"3c4f8f8:{rel}"], cwd=ROOT,
                                   capture_output=True, check=True).stdout
        current = (ROOT / rel).read_bytes().replace(b"\r\n", b"\n")
        assert hashlib.sha256(current).hexdigest() == hashlib.sha256(committed).hexdigest(), rel
    v2_receipt = ROOT / "results" / "sft_root_cause_phase4_power_sensitivity_v2.json"
    if v2_receipt.exists():
        data = json.loads(v2_receipt.read_text(encoding="utf-8"))
        v1_sha = hashlib.sha256(V1_RECEIPT.read_bytes()).hexdigest()
        assert data["supersedes"]["path"] == V1_RECEIPT.relative_to(ROOT).as_posix()
        assert data["supersedes"]["sha256"] == v1_sha
        assert data["checks"]["shared_and_independent_cells_identical_to_v1"] is True


def test_v2_receipt_states_gates_limits_and_supersession():
    v2_path = ROOT / "results" / "sft_root_cause_phase4_power_sensitivity_v2.json"
    addendum = ROOT / "results" / "sft_root_cause_phase4_power_sensitivity_v1_supersession.json"
    data = json.loads(v2_path.read_text(encoding="utf-8"))
    assert data["status"] == "PLANNING EVIDENCE, NOT A POWER GUARANTEE"
    assert "EXPLORATORY POSITIVE-EFFECT SCREEN" in data["scientific_gates"]["reconciliation"]
    assert any("training-seed" in item for item in data["assumptions"]["not_modelled"])
    assert any("repository-native" in item for item in data["assumptions"]["not_modelled"])
    # No scenario confirms a 5-point effect at 352 groups when the true effect is 5 points.
    assert data["power_at_planned_gate_sizes"]["352"]["confirmatory_5"]["5"] == [0.0, 0.0]
    sup = json.loads(addendum.read_text(encoding="utf-8"))
    assert sup["superseded"]["sha256"] == hashlib.sha256(V1_RECEIPT.read_bytes()).hexdigest()
    assert sup["successor"]["sha256"] == hashlib.sha256(v2_path.read_bytes()).hexdigest()
