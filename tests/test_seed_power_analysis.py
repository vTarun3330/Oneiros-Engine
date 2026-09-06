"""Pin the seed-level sign test that decides whether relearning is real."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_seed_power import sign_test


def test_three_positive_seeds_cannot_reach_significance():
    """The floor at n=3 is 0.25, so three seeds could never have settled this.

    This is the whole justification for spending GPU time on seeds 45-49: the
    existing three-seed result was not weak evidence, it was evidence from a
    test that had no way to return a significant answer.
    """
    result = sign_test([0.038, 0.045, 0.016])
    assert result["seeds_positive"] == 3
    assert result["p_value"] == 0.25
    assert result["p_value"] == result["smallest_reachable_p_at_this_n"]


def test_eight_positive_seeds_reach_significance():
    result = sign_test([0.04] * 8)
    assert result["effective_trials"] == 8
    assert result["p_value"] == 0.007812
    assert result["p_value"] < 0.05


def test_the_test_is_two_sided():
    """Eight consistently negative seeds are as significant as eight positive.

    A one-sided test here would only ever be able to confirm the hypothesis it
    was written to support.
    """
    assert sign_test([-0.04] * 8)["p_value"] == sign_test([0.04] * 8)["p_value"]


def test_one_dissenting_seed_costs_significance_at_eight():
    result = sign_test([0.04] * 7 + [-0.01])
    assert result["seeds_positive"] == 7
    assert result["seeds_negative"] == 1
    assert result["p_value"] > 0.05


def test_a_tied_seed_is_excluded_not_counted_as_a_success():
    """Counting a zero delta as a win would manufacture evidence."""
    result = sign_test([0.04, 0.04, 0.0])
    assert result["seeds_tied"] == 1
    assert result["effective_trials"] == 2
    assert result["p_value"] == 0.5
    assert "excluded" in result["note"]


def test_all_ties_weigh_nothing():
    result = sign_test([0.0, 0.0])
    assert result["effective_trials"] == 0
    assert result["p_value"] == 1.0


def test_a_balanced_split_is_not_evidence():
    assert sign_test([0.04, -0.04])["p_value"] == 1.0


def test_no_seeds_is_not_a_significant_result():
    assert sign_test([])["p_value"] == 1.0
