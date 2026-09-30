"""Native generated-test analysis v2: infrastructure handling, validity evidence, study mode."""
from __future__ import annotations

import json

import pytest

from scripts import native_generated_tests_analyse as an

TARGETS = [f"t{i}" for i in range(6)]
REPO = {t: f"r{i // 2}" for i, t in enumerate(TARGETS)}


def records(kill_sft=(), fail="pass_both"):
    out = []
    for arm in an.ARMS:
        for seed in an.SEEDS:
            for t in TARGETS:
                for slot in range(an.SLOTS):
                    cls = "semantic_kill" if arm == "sft" and (t, seed, slot) in kill_sft else fail
                    out.append({"arm": arm, "seed": seed, "target_key": t, "slot": slot,
                                "class": cls, "fixed_valid": cls in ("pass_both", "semantic_kill"),
                                "canary_failed": False, "contract_sha256": "c"})
    return out


def test_incomplete_or_duplicate_grids_refuse():
    rows = records()
    with pytest.raises(an.AnalysisRefused, match="incomplete"):
        an.analyse(rows[:-1], TARGETS, REPO, "confirmation")
    with pytest.raises(an.AnalysisRefused, match="duplicate"):
        an.analyse(rows + rows[:1], TARGETS, REPO, "confirmation")


def test_target_level_estimand_in_confirmation_mode():
    kills = {("t0", 42, 7)} | {("t1", s, 0) for s in an.SEEDS}
    result = an.analyse(records(kill_sft=kills), TARGETS, REPO, "confirmation")
    expected = (1 / 3 + 1) / 6 * 100
    assert result["kill_at_8"]["sft"] == pytest.approx(expected, abs=1e-3)
    assert result["primary"]["point"] == pytest.approx(expected, abs=1e-3)
    assert result["kill_at_1"]["sft"] == pytest.approx(1 / 6 * 100, abs=1e-3)
    assert result["unique_bugs_killed"] == {"base": 0, "sft": 2}
    assert set(result["primary"]["leave_one_repository_out"]) == {"r0", "r1", "r2"}


def test_engineering_mode_suppresses_decisions():
    result = an.analyse(records(kill_sft={("t0", 42, 0)}), TARGETS, REPO,
                        "engineering_dress_rehearsal")
    assert "SUPPRESSED" in result["decisions"]
    assert "primary" not in result and "validity_non_inferiority" not in result
    assert "sft_minus_base_points" not in result["kill_at_8"]
    assert result["root_cause_established"] is False
    assert result["generalization_established"] is False


def test_asymmetric_infrastructure_excludes_the_target_from_both_arms():
    rows = records()
    for r in rows:
        if r["target_key"] == "t2" and r["arm"] == "sft":
            r.update(**{"class": "environment_failure", "canary_failed": True})
    result = an.analyse(rows, TARGETS, REPO, "confirmation")
    assert result["infrastructure"]["excluded_targets"] == ["t2"]
    d = result["denominators"]
    assert d["base"]["requested"] == 3 * 6 * 8 and d["base"]["eligible"] == 3 * 5 * 8
    assert d["sft"]["eligible"] == d["base"]["eligible"]
    assert "environment_failure" not in d["sft"]["classes"]


def test_infrastructure_without_a_failed_canary_refuses():
    rows = records()
    rows[0]["class"] = "harness_failure"            # canary_failed stays False
    with pytest.raises(an.AnalysisRefused, match="not proven arm-independent"):
        an.analyse(rows, TARGETS, REPO, "confirmation")


def test_infrastructure_is_never_counted_as_parsed_or_reached():
    rows = records()
    for r in rows:
        if r["target_key"] == "t0":
            r.update(**{"class": "harness_failure", "canary_failed": True})
    d = an.analyse(rows, TARGETS, REPO, "engineering_dress_rehearsal")["denominators"]["base"]
    assert d["parsed"] == d["eligible"] == 3 * 5 * 8 and d["infrastructure_excluded"] == 24


def test_fixed_validity_needs_evidence_and_nondeterminism_is_never_valid():
    rows = records()
    for r in rows:
        if r["arm"] == "sft" and r["slot"] < 2:
            r.update(**{"class": "nondeterminism", "fixed_valid": True})
        if r["arm"] == "sft" and r["slot"] == 2:
            r["fixed_valid"] = False                  # class says pass_both, evidence says no
    result = an.analyse(rows, TARGETS, REPO, "confirmation")
    assert result["fixed_valid_rate"]["base"] == pytest.approx(100.0)
    assert result["fixed_valid_rate"]["sft"] == pytest.approx(5 / 8 * 100)
    assert result["validity_non_inferiority"]["non_inferiority_shown"] is False


def test_cli_writes_a_hash_bound_artifact_and_refuses_mixed_contracts(tmp_path):
    manifest = {"kept_targets": TARGETS, "targets": [{"key": t, "repository": REPO[t]}
                                                     for t in TARGETS]}
    (tmp_path / "m.json").write_text(json.dumps(manifest))
    rows = records()
    (tmp_path / "r.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    out = tmp_path / "a.json"
    assert an.main(["analyse", "--manifest", str(tmp_path / "m.json"), "--results",
                    str(tmp_path / "r.jsonl"), "--condition", "primary_whole_module",
                    "--study-mode", "engineering_dress_rehearsal", "--out", str(out)]) == 0
    artifact = json.loads(out.read_text())
    assert set(artifact["inputs"]) == {"manifest", "results"}
    rows[0]["contract_sha256"] = "other"
    (tmp_path / "r.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    with pytest.raises(an.AnalysisRefused, match="mix execution contracts"):
        an.main(["analyse", "--manifest", str(tmp_path / "m.json"), "--results",
                 str(tmp_path / "r.jsonl"), "--condition", "primary_whole_module",
                 "--study-mode", "engineering_dress_rehearsal", "--out", str(out)])
