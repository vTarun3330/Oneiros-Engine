"""Native generated-test analysis: target-level estimand, clustering, refusals."""
from __future__ import annotations

import pytest

from scripts import native_generated_tests_analyse as an

TARGETS = [f"t{i}" for i in range(6)]
REPO = {t: f"r{i // 2}" for i, t in enumerate(TARGETS)}


def records(kill_base=(), kill_sft=(), fail="pass_both"):
    out = []
    for arm, kills in (("base", kill_base), ("sft", kill_sft)):
        for seed in an.SEEDS:
            for t in TARGETS:
                for slot in range(an.SLOTS):
                    cls = "semantic_kill" if (t, seed, slot) in kills else fail
                    out.append({"arm": arm, "seed": seed, "target_key": t, "slot": slot,
                                "class": cls})
    return out


def test_incomplete_or_duplicate_grids_refuse():
    rows = records()
    with pytest.raises(an.AnalysisRefused, match="incomplete"):
        an.analyse(rows[:-1], TARGETS, REPO)
    with pytest.raises(an.AnalysisRefused, match="duplicate"):
        an.analyse(rows + rows[:1], TARGETS, REPO)


def test_target_level_estimand_averages_seeds_and_never_pools():
    # SFT kills t0 at one of three seeds (slot 7) and t1 at all seeds (slot 0).
    kills = {("t0", 42, 7)} | {("t1", s, 0) for s in an.SEEDS}
    result = an.analyse(records(kill_sft=kills), TARGETS, REPO)
    expected = (1 / 3 + 1) / 6 * 100
    assert result["kill_at_8"]["sft"] == pytest.approx(expected, abs=1e-3)
    assert result["primary"]["point"] == pytest.approx(expected, abs=1e-3)
    assert result["kill_at_1"]["sft"] == pytest.approx(1 / 6 * 100, abs=1e-3)   # slot 7 not in @1
    assert result["unique_bugs_killed"] == {"base": 0, "sft": 2}
    assert result["primary"]["targets"] == 6 and result["primary"]["repositories"] == 3
    assert set(result["primary"]["leave_one_repository_out"]) == {"r0", "r1", "r2"}


def test_denominators_follow_the_admission_stages():
    rows = records()
    rows[0]["class"] = "syntax_failure"
    rows[1]["class"] = "fabricated_import"
    rows[2]["class"] = "target_not_reached"
    rows[3]["class"] = "harness_failure"
    d = an.analyse(rows, TARGETS, REPO)["denominators"]["base"]
    assert d["requested"] == 3 * 6 * 8
    assert d["requested"] - d["parsed"] == 1
    assert d["requested"] - d["collected"] == 2
    assert d["requested"] - d["reached"] == 3
    assert d["infrastructure_excluded"] == 1


def test_validity_non_inferiority_uses_the_lower_bound():
    rows = records()
    for r in rows:
        if r["arm"] == "sft" and r["slot"] < 2:
            r["class"] = "fixed_side_failure"          # SFT loses 25 points of validity
    v = an.analyse(rows, TARGETS, REPO)["validity_non_inferiority"]
    assert v["point"] == pytest.approx(-25.0) and v["non_inferiority_shown"] is False
    assert v["operational_stop"] is True


def test_atheris_jointly_eligible_summary():
    atheris = [{"target_key": "t0", "eligible": True, "mode": "ordinary", "kill": True},
               {"target_key": "t1", "eligible": False, "reason": "instance_method_receiver",
                "mode": "ordinary", "kill": False}]
    kills = {("t0", 42, 0)}
    a = an.analyse(records(kill_sft=kills), TARGETS, REPO, atheris)["atheris_jointly_eligible"]
    assert a["targets"] == 1 and a["ineligible_reasons"] == {"instance_method_receiver": 1}
    assert a["oneiros_unique_kills"] == {"base": 0, "sft": 1}
    assert a["atheris_unique_kills"] == {"ordinary": 1}
