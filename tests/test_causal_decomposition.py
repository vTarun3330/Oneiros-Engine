"""Phase 2 decomposition: ratios, paired cluster bootstrap and transitions."""
from __future__ import annotations

from harness.causal_decomposition import (
    category_distribution, function_transitions, paired_comparison,
)
from harness.causal_ledger import TERMINAL_CATEGORIES


def _row(record, group, rank, *, disc=True, valid=False, killed=False, dup=False,
         category="discriminating_wrong_oracle"):
    return {"record_id": record, "group_id": group, "rank": rank, "parse_valid": True,
            "policy_valid": True, "execution_valid": True, "discriminating_input": disc,
            "reference_valid": valid, "killed": killed, "duplicate": dup,
            "candidate_shape": "assertion", "assertion_forms": ["compare_Eq"],
            "normalised_code_sha256": f"{record}-{rank}-{dup}", "call_signatures": [str(rank)],
            "terminal_category": category}


def _arm(oracle_right: bool):
    rows = []
    for g in range(6):
        for f in range(3):
            record = f"r{g}_{f}"
            rows.append(_row(record, f"g{g}", 1, disc=True, valid=oracle_right,
                             killed=oracle_right,
                             category="valid_kill" if oracle_right
                             else "discriminating_wrong_oracle"))
            rows.append(_row(record, f"g{g}", 2, disc=False, valid=True,
                             category="non_discriminating_input"))
    return rows


def test_identical_arms_have_zero_difference_and_a_degenerate_interval():
    result = paired_comparison(_arm(False), _arm(False), resamples=200)
    metric = result["metrics"]["p_correct_oracle_given_disc"]
    assert metric["difference_points"] == 0 and metric["ci95_points"] == [0, 0]
    assert result["clusters"] == 6


def test_the_conditional_oracle_rate_uses_discriminating_executed_candidates_only():
    result = paired_comparison(_arm(False), _arm(True), resamples=200)["metrics"]
    assert result["p_correct_oracle_given_disc"]["a"] == 0.0
    assert result["p_correct_oracle_given_disc"]["b"] == 1.0
    assert result["p_correct_oracle_given_disc"]["b_counts"] == [18, 18]
    assert result["p_correct_oracle_given_non_disc"]["difference_points"] == 0
    assert result["kill_at_8"]["difference_points"] == 100.0
    assert result["kill_at_8"]["region"] == "supported"
    assert result["p_disc_given_execute"]["a"] == 0.5


def test_function_transitions_are_paired_by_record():
    transitions = function_transitions(_arm(False), _arm(True))
    assert transitions["paired_functions"] == 18
    assert transitions["killed"] == {"both": 0, "neither": 0, "gained": 18, "lost": 0}
    assert transitions["has_executed_disc_candidate"]["both"] == 18


def test_category_distribution_sums_to_the_candidates():
    rows = _arm(False)
    dist = category_distribution(rows, TERMINAL_CATEGORIES)
    assert sum(dist["counts"].values()) == dist["candidates"] == len(rows)
