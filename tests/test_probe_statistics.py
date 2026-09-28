"""Phase 3 v3 inference: exact pairing, Monte Carlo wording, Holm, clustered interactions,
route-exact H4, recomputed and enforced capacity gate."""
from __future__ import annotations

import json
from pathlib import Path
import random

import pytest

from harness.probe_statistics import (
    H4_STRATA, PairingError, answer_rate_gate, bootstrap_contrast, bootstrap_interaction,
    capacity_interpretation, decide_h4, holm, monte_carlo_report, paired_cells,
    schema_dependent, sign_flip_test,
)

ROOT = Path(__file__).resolve().parent.parent


def _rows(p_a, p_b, groups=60, per_group=4, seed=0, levels=("L",)):
    rng = random.Random(seed)
    rows = []
    for g in range(groups):
        for i in range(per_group):
            for level in levels:
                for arm, p in (("a", p_a), ("b", p_b)):
                    rows.append({"arm": arm, "level": level, "group_id": f"g{g}",
                                 "item_key": f"g{g}-i{i}", "correct": rng.random() < p,
                                 "answer": "1", "method": "strict"})
    return rows


# --- exact pairing -----------------------------------------------------------------------------

def test_items_that_differ_but_keep_equal_group_counts_are_refused():
    rows = _rows(0.4, 0.4)
    for r in rows:        # substitute one item in arm b with a different item in the SAME group
        if r["arm"] == "b" and r["item_key"] == "g0-i0":
            r["item_key"] = "g0-i9"
    with pytest.raises(PairingError, match="not paired"):
        bootstrap_contrast(rows, ("a", "L"), ("b", "L"), resamples=100)
    with pytest.raises(PairingError):
        sign_flip_test(rows, ("a", "L"), ("b", "L"), resamples=100)


def test_duplicates_regrouping_and_missing_identity_are_refused():
    rows = _rows(0.4, 0.4)
    with pytest.raises(PairingError, match="duplicate"):
        paired_cells(rows + [dict(rows[0])], (("a", "L"), ("b", "L")))
    moved = [dict(r, group_id="g1") if r["arm"] == "b" and r["item_key"] == "g0-i0" else r
             for r in rows]
    with pytest.raises(PairingError, match="change semantic group"):
        paired_cells(moved, (("a", "L"), ("b", "L")))
    keyless = [{k: v for k, v in r.items() if k != "item_key"} for r in rows]
    with pytest.raises(PairingError, match="no item_key"):
        paired_cells(keyless, (("a", "L"), ("b", "L")))


def test_interaction_needs_the_same_items_in_all_four_cells():
    rows = _rows(0.3, 0.5, groups=10, levels=("P", "Q"))
    rows = [r for r in rows if not (r["level"] == "Q" and r["item_key"] == "g0-i0")]
    with pytest.raises(PairingError):
        bootstrap_interaction(rows, (("a", "P"), ("b", "P")), (("a", "Q"), ("b", "Q")),
                              resamples=100)


# --- Monte Carlo p, Holm -----------------------------------------------------------------------

def test_finite_resample_p_is_never_zero_and_is_worded_exactly():
    result = sign_flip_test(_rows(0.05, 0.95), ("a", "L"), ("b", "L"), resamples=10000)
    assert result["extreme"] == 0 and result["monte_carlo_p"] == 1 / 10001 > 0
    assert result["report"] == ("Monte Carlo p = 0.00010; 0/10000 sampled permutations were as "
                                "extreme; 0.00010 is the minimum reportable resolution at this "
                                "resample count.")
    for banned in ("p = 0 ", "<=", "upper bound"):
        assert banned not in result["report"]
    assert monte_carlo_report(5, 100).startswith("Monte Carlo p = 0.05941; 5/100")


def test_identical_arms_are_not_significant():
    rows = _rows(0.4, 0.4)
    first = {r["item_key"]: r["correct"] for r in rows if r["arm"] == "a"}
    for r in rows:
        if r["arm"] == "b":
            r["correct"] = first[r["item_key"]]
    assert sign_flip_test(rows, ("a", "L"), ("b", "L"), resamples=2000)["monte_carlo_p"] == 1.0


def test_a_large_paired_effect_is_detected():
    rows = _rows(0.2, 0.7)
    assert sign_flip_test(rows, ("a", "L"), ("b", "L"), resamples=2000)["monte_carlo_p"] < 0.01
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
    doubled = rows + [{**r, "item_key": r["item_key"] + "-copy"} for r in rows]
    twice = bootstrap_interaction(doubled, *cells, resamples=1000)
    assert once["groups"] == twice["groups"] == 30 and twice["paired_items"] == 120
    assert once["ci95_points"] == twice["ci95_points"]


# --- route-exact H4 --------------------------------------------------------------------------

def _c(points, low, high):
    return {"difference_points": points, "ci95_points": [low, high]}


def _strata(exposed, unexposed):
    strata = {s: {"prefill": _c(1.0, -2, 4), "answer": _c(1.0, -2, 4)} for s in H4_STRATA}
    strata["exposed"], strata["unexposed"] = exposed, unexposed
    return strata


UP = {"prefill": _c(6.0, 1.0, 11.0), "answer": _c(5.0, 0.5, 9.0)}
FLAT = {"prefill": _c(-1.0, -6.0, 4.0), "answer": _c(-0.5, -5.0, 4.0)}
FLIP = {"prefill": _c(-1.667, -7.6, 4.2), "answer": _c(5.0, 0.42, 9.8)}
UP_FLIP = {"prefill": _c(6.0, 1.0, 11.0), "answer": _c(-2.0, -7.0, 3.0)}


def test_exposed_flip_returns_open_before_any_rule():
    d = decide_h4(_strata(FLIP, UP))
    assert (d["status"], d["qualifier"], d["route"]) == \
        ("open", "schema_dependent_exposure_result", ["exposed_schema_check"])


def test_exposed_no_improvement_is_rule_one_and_never_consults_unexposed():
    d = decide_h4(_strata(FLAT, FLIP))          # unexposed flips but must not be consulted
    assert d["status"] == "weakened" and d["route"] == ["exposed_schema_check", "rule_1"]


def test_exposed_improvement_then_unexposed_flip_is_open():
    d = decide_h4(_strata(UP, UP_FLIP))
    assert (d["status"], d["qualifier"]) == ("open", "schema_dependent_unexposed_result")
    assert d["route"][-1] == "unexposed_schema_check"


def test_exposed_improvement_without_unexposed_improvement_is_rule_two():
    d = decide_h4(_strata(UP, FLAT))
    assert (d["status"], d["qualifier"]) == ("strengthened", "memorisation_exposure_failure")


def test_both_improve_is_rule_three():
    d = decide_h4(_strata(UP, UP))
    assert (d["status"], d["qualifier"]) == ("open", "distribution_shift_untestable")


def test_a_missing_required_stratum_is_refused():
    strata = _strata(UP, UP)
    del strata["exposed_novel"]
    with pytest.raises(ValueError):
        decide_h4(strata)


def test_pooled_non_flip_does_not_hide_an_exposed_flip():
    strata = _strata(FLIP, UP)
    strata["all"] = {"prefill": _c(1.5, -2.5, 5.6), "answer": _c(2.7, -1.0, 6.5)}
    assert not schema_dependent(1.5, 2.7)
    assert decide_h4(strata)["status"] == "open"


# --- capacity gate ------------------------------------------------------------------------------

def _gate_rows(large_answer_rate):
    rows = []
    for i in range(100):
        for arm, rate in (("small", 1.0), ("large", large_answer_rate)):
            for level in ("A0", "A3", "A4", "CTRL"):
                answered = i < rate * 100
                rows.append({"arm": arm, "level": level, "item_key": f"i{i}",
                             "group_id": f"g{i}", "correct": answered,
                             "answer": "1" if answered else None,
                             "method": "strict" if answered else "nonanswer"})
    return rows


def test_a_failed_recomputed_gate_suppresses_the_interpretation():
    gate = answer_rate_gate(_gate_rows(0.90), "large", "small", ("A0", "A3", "A4"), "CTRL")
    assert gate["passed"] is False
    primary = {"C1": {"ci95_points": [19.0, 30.0]}, "C2": {"ci95_points": [6.0, 16.0]}}
    interpretation = capacity_interpretation(gate, primary)
    assert interpretation["available"] is False and interpretation["H3"] == "unchanged"
    assert "strengthened" not in json.dumps(interpretation)


def test_a_passing_gate_yields_only_a_model_scale_association():
    gate = answer_rate_gate(_gate_rows(1.0), "large", "small", ("A0", "A3", "A4"), "CTRL")
    assert gate["passed"]
    out = capacity_interpretation(gate, {"C1": {"ci95_points": [19.0, 30.0]}})
    assert out["H3"]["qualifier"] == "model_scale_association"
    assert "parameter count" in out["H3"]["not_established"]


# --- v3 receipts --------------------------------------------------------------------------------

def test_v3_receipts_report_every_stratum_and_honest_reproduction():
    a = json.loads((ROOT / "results/sft_root_cause_phase3a_result_receipt_v3.json")
                   .read_text(encoding="utf-8"))
    for name in H4_STRATA:
        block = a["strata"][name]
        assert {"prefill", "answer", "schema_interaction_answer_minus_prefill",
                "schema_dependent"} <= set(block)
        assert block["schema_interaction_answer_minus_prefill"]["clustered_by"] == \
            "semantic group_id"
    assert a["decisions"]["H4"]["status"] == "open"
    assert a["decisions"]["H1"]["status"] == "open"
    assert a["rescoring"]["duplicates"] == 0
    assert "exact row-level reproduction is not claimed" in a["rescoring"]["reproduction"]
    c = json.loads((ROOT / "results/sft_root_cause_phase3c_result_receipt_v3.json")
                   .read_text(encoding="utf-8"))
    assert c["gate"]["recomputed_from_rows"] is True
    for block in c["primary"].values():
        assert block["meaningful_effect_frozen_rule"]["multiplicity_adjusted"] is False
        assert "0.00010 is the minimum reportable resolution" in block["sign_flip"]["report"]
    assert set(c["secondary_exploratory"]["contrasts"]) == {
        "7b_minus_arm_a_A0", "7b_minus_arm_a_A4", "7b_A4_minus_A0", "7b_A3_minus_A0",
        "7b_minus_1_5b_answer_schema", "C1_by_cohort", "C2_by_tier"}
