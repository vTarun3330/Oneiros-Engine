"""Native generated-test analysis v2: infrastructure handling, validity evidence, study mode."""
from __future__ import annotations

import json

import pytest

from scripts import native_generated_tests_analyse as an

TARGETS = [f"t{i}" for i in range(6)]
REPO = {t: f"r{i}" for i, t in enumerate(TARGETS)}      # six repositories: gate passes


def records(kill_sft=(), fail="pass_both", targets=None):
    out = []
    for arm in an.ARMS:
        for seed in an.SEEDS:
            for t in targets or TARGETS:
                for slot in range(an.SLOTS):
                    cls = "semantic_kill" if arm == "sft" and (t, seed, slot) in kill_sft else fail
                    out.append({"arm": arm, "seed": seed, "target_key": t, "slot": slot,
                                "key": f"{arm}::{t}::{seed}::{slot}",
                                "module_sha256": f"{arm}{t}{seed}{slot}",
                                "class": cls, "fixed_valid": cls in ("pass_both", "semantic_kill"),
                                "canary_failed": False, "contract_sha256": "c",
                                "generation": {"generated_tokens": 40 + slot,
                                               "eos_reached": slot != 7,
                                               "finish_reason": "eos" if slot != 7 else "length",
                                               "hit_completion_limit": slot == 7,
                                               "prompt_tokens": 900, "row_wall_seconds": 12.0}})
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
    assert set(result["primary"]["leave_one_repository_out"]) == {f"r{i}" for i in range(6)}


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


def _cohort_files(tmp_path):
    import hashlib
    from scripts import native_generated_tests_generate as gen
    from scripts import native_generation_io as gio
    sealed = [{"target_key": t, "prompt": f"p {t}", "condition": "whole_module",
               "prompt_sha256": hashlib.sha256(f"p {t}".encode()).hexdigest()} for t in TARGETS]
    job = gen.build_job(sealed, {t: {"ok": True} for t in TARGETS}, gen.sequence_fit(sealed, len))
    data = json.dumps({"primary_whole_module": job}).encode()
    (tmp_path / "job.json").write_bytes(data)
    targets = [{"key": t, "repository": REPO[t]} for t in TARGETS]
    manifest = {"kept_targets": TARGETS, "targets": targets,
                "nature": "ENGINEERING DRESS REHEARSAL ONLY",
                "study_mode": "engineering_dress_rehearsal",
                **gio.cohort_fields(targets, "job.json", data, "primary_whole_module", {})}
    (tmp_path / "m.json").write_text(json.dumps(manifest))


def test_cli_writes_a_hash_bound_artifact_and_refuses_mixed_contracts(tmp_path):
    from tests.native_fake_sandbox import evidence_fields
    _cohort_files(tmp_path)
    rows = records(kill_sft={("t0", 42, 0)})
    fields = {kill: evidence_fields(kill) for kill in (False, True)}
    for r in rows:                              # real, verifiable evidence for every row
        r.update(fields[r["class"] == "semantic_kill"])
    (tmp_path / "r.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    out = tmp_path / "a.json"
    args = ["analyse", "--manifest", str(tmp_path / "m.json"), "--job", str(tmp_path / "job.json"),
            "--results", str(tmp_path / "r.jsonl"), "--condition", "primary_whole_module",
            "--study-mode", "engineering_dress_rehearsal", "--out", str(out)]
    assert an.main(args) == 0
    artifact = json.loads(out.read_text())
    assert set(artifact["inputs"]) == {"manifest", "job", "results"}
    assert artifact["cohort"]["qualified_targets"] == 6 and artifact["job_sha256"]
    rows[0]["contract_sha256"] = "other"
    (tmp_path / "r.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(an.AnalysisRefused, match="mix execution contracts"):
        an.main(args)


def test_rows_without_generation_telemetry_refuse_and_validity_carries_limit_rates():
    rows = records()
    result = an.analyse(rows, TARGETS, REPO, "engineering_dress_rehearsal")
    assert result["generation_telemetry"]["base"]["completion_limit_hit_rate"] == 12.5
    assert result["fixed_valid_rate"]["completion_limit_hit_rate"] == {"base": 12.5, "sft": 12.5}
    del rows[5]["generation"]
    with pytest.raises(an.AnalysisRefused, match="without generation telemetry"):
        an.analyse(rows, TARGETS, REPO, "engineering_dress_rehearsal")


# --- amendment v2.4: inherited engineering gate, duplication --------------------------------

QUALIFIED24 = [f"q{i:02d}" for i in range(24)]
GENERATED23 = QUALIFIED24[:23]


def _cohort(generated=GENERATED23):
    return {"qualified": QUALIFIED24, "generation": list(generated),
            "pre_generation_exclusions": [{"target_key": "q23", "stage": "pre_generation",
                                           "reasons": ["sequence_overflow"]}]}


def _infra(rows, target, arm=None):
    for r in rows:
        if r["target_key"] == target and (arm is None or r["arm"] == arm):
            r.update(**{"class": "environment_failure", "canary_failed": True})


def _gate(rows, repos):
    return an.analyse(rows, GENERATED23, repos, "engineering_dress_rehearsal", cohort=_cohort())


def test_gate_passes_with_22_of_24_across_five_repositories():
    rows = records(targets=GENERATED23)
    _infra(rows, "q05", arm="sft")                  # one arm only -> excluded from BOTH arms
    result = _gate(rows, {t: f"r{i % 5}" for i, t in enumerate(QUALIFIED24)})
    gate = result["engineering_gate"]
    assert result["engineering_gate_passed"] is True and gate["required_eligible"] == 22
    assert (gate["eligible"], gate["repositories"]) == (22, 5)
    assert result["infrastructure"]["excluded_targets"] == ["q05"]
    assert result["denominators"]["base"]["eligible"] == result["denominators"]["sft"]["eligible"]
    c = result["cohort"]
    assert (c["qualified_targets"], c["generation_targets"], c["pre_generation_excluded"],
            c["post_generation_infrastructure_excluded"], c["final_infrastructure_eligible"],
            c["represented_repositories"]) == (24, 23, 1, 1, 22, 5)
    assert "kill_at_8" in result and "arm_comparison" not in result


def test_gate_fails_with_21_of_24_and_suppresses_arm_comparison():
    rows = records(targets=GENERATED23)
    _infra(rows, "q05")
    _infra(rows, "q06", arm="base")
    result = _gate(rows, {t: f"r{i % 5}" for i, t in enumerate(QUALIFIED24)})
    assert result["engineering_gate_passed"] is False
    assert result["engineering_gate"]["failures"] == ["21 of 24 qualified targets eligible (< 22)"]
    assert result["arm_comparison"].startswith("SUPPRESSED")
    for suppressed in ("kill_at_1", "kill_at_8", "unique_bugs_killed", "fixed_valid_rate"):
        assert suppressed not in result
    assert result["root_cause_established"] is False


def test_gate_fails_with_22_targets_but_only_four_repositories():
    rows = records(targets=GENERATED23)
    _infra(rows, "q05")
    result = _gate(rows, {t: f"r{i % 4}" for i, t in enumerate(QUALIFIED24)})
    assert result["engineering_gate_passed"] is False
    assert result["engineering_gate"]["failures"] == ["4 repositories (< 5)"]
    assert "kill_at_8" not in result


def test_duplicates_across_seeds_are_reported_separately_from_within_row_duplicates():
    rows = records()
    for r in rows:                                  # unique within each seed, same across seeds
        r["module_sha256"] = f"{r['arm']}-{r['target_key']}-{r['slot']}"
    result = an.analyse(rows, TARGETS, REPO, "engineering_dress_rehearsal")
    dup = result["generation_telemetry"]["base"]["duplication"]
    assert dup["within_target_seed_duplicates"] == 0
    assert dup["within_target_seed_denominator"] == 3 * 6 * 8
    assert dup["unique_candidates"] == 6 * 8 and dup["unique_rate"] == pytest.approx(100 / 3, abs=1e-3)
    assert dup["cross_seed_repeated_hashes"] == 6 * 8
    assert dup["cross_seed_repeated_candidates"] == 3 * 6 * 8
    assert dup["per_target"]["t0"] == {"candidates": 24, "unique": 8,
                                       "cross_seed_repeated_hashes": 8}
    for r in rows:                                  # a within-row repeat, no cross-seed repeat
        r["module_sha256"] = f"{r['arm']}-{r['target_key']}-{r['seed']}-{min(r['slot'], 6)}"
    dup = an.analyse(rows, TARGETS, REPO, "engineering_dress_rehearsal")[
        "generation_telemetry"]["base"]["duplication"]
    assert dup["within_target_seed_duplicates"] == 3 * 6
    assert dup["cross_seed_repeated_hashes"] == 0
