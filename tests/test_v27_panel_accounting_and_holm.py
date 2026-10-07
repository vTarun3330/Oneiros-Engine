"""v2.7 r5 corrective pass: the authoritative cohort preserves the frozen 34 -> 29 + 5 panel
accounting, and Holm's multiplicity family stays fixed at the six planned hypotheses."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts import native_generated_tests_analyse as analysis  # noqa: E402
from scripts import native_generation_io as gio  # noqa: E402

JOB = ROOT / "results/sft_root_cause_v27_generation_job_r5.json"
MANIFEST = ROOT / "results/sft_root_cause_v27_generation_manifest_r5.json"


def _manifest():
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def _resolve_with(tmp_path, manifest):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return gio.resolve_cohort(JOB, path)


# --- Fix 1: panel accounting --------------------------------------------------------------------

def test_r5_cohort_carries_the_exact_34_29_5_accounting():
    c = gio.resolve_cohort(JOB, MANIFEST)
    assert c["panel"]["targets"] == 34
    assert len(c["qualified"]) == 29 and len(c["generation"]) == 29
    assert len(c["panel_policy_exclusions"]) == 5
    ids = {e["target_key"] for e in c["panel_policy_exclusions"]}
    assert not ids & set(c["qualified"])
    assert all(e["reason"].startswith("sandbox_control_failed") for e in c["panel_policy_exclusions"])


def test_missing_panel_metadata_is_refused_for_a_registry_cohort(tmp_path):
    m = _manifest()
    del m["panel"], m["panel_policy_exclusions"]
    with pytest.raises(SystemExit, match="panel accounting missing"):
        _resolve_with(tmp_path, m)


def test_modified_panel_hash_is_refused(tmp_path):
    m = _manifest()
    m["panel"]["sha256"] = "0" * 64
    with pytest.raises(SystemExit, match="hash differs"):
        _resolve_with(tmp_path, m)


def test_incorrect_panel_count_is_refused(tmp_path):
    m = _manifest()
    m["panel"]["targets"] = 33
    with pytest.raises(SystemExit, match="count"):
        _resolve_with(tmp_path, m)


def test_duplicate_policy_exclusions_are_refused(tmp_path):
    m = _manifest()
    m["panel_policy_exclusions"] = m["panel_policy_exclusions"] + m["panel_policy_exclusions"][:1]
    with pytest.raises(SystemExit, match="duplicate"):
        _resolve_with(tmp_path, m)


def test_overlapping_policy_exclusion_is_refused(tmp_path):
    m = _manifest()
    m["panel_policy_exclusions"][0] = {"target_key": m["qualified_targets"][0], "reason": "x"}
    with pytest.raises(SystemExit, match="overlap"):
        _resolve_with(tmp_path, m)


def test_exclusion_outside_the_panel_is_refused(tmp_path):
    m = _manifest()
    m["panel_policy_exclusions"][0] = {"target_key": "cand:not/in-panel@0", "reason": "x"}
    with pytest.raises(SystemExit, match="outside the frozen panel"):
        _resolve_with(tmp_path, m)


def test_union_not_equal_to_the_panel_is_refused(tmp_path):
    m = _manifest()
    m["panel_policy_exclusions"] = m["panel_policy_exclusions"][:4]
    with pytest.raises(SystemExit, match="do not equal the frozen panel"):
        _resolve_with(tmp_path, m)


def test_exclusion_without_a_reason_is_refused(tmp_path):
    m = _manifest()
    m["panel_policy_exclusions"][0] = {"target_key": m["panel_policy_exclusions"][0]["target_key"]}
    with pytest.raises(SystemExit, match="reason"):
        _resolve_with(tmp_path, m)


def test_legacy_cohorts_without_a_panel_still_resolve():
    assert gio.panel_accounting({}, ["a"], required=False) == {
        "panel": None, "panel_policy_exclusions": [], "subset_receipt": None}


# --- policy-exclusion reasons bound to the authoritative v3 subset receipt --------------------

SUBSET = ROOT / "results/sft_root_cause_v27_common_subset_r4_v3.json"


def test_exclusion_reasons_are_bound_to_the_v3_subset_receipt():
    c = gio.resolve_cohort(JOB, MANIFEST)
    assert c["subset_receipt"]["schema_version"] == "oneiros_v27_common_subset_v3"
    subset = json.loads(SUBSET.read_text(encoding="utf-8"))
    derived = sorted((r["target_id"], r["exclusion_reason"]) for r in subset["targets"]
                     if r["failure_class"] == "policy")
    assert derived == sorted((e["target_key"], e["reason"]) for e in c["panel_policy_exclusions"])


def test_modified_exclusion_reason_is_refused(tmp_path):
    m = _manifest()
    m["panel_policy_exclusions"][0]["reason"] = "a different but plausible reason"
    with pytest.raises(SystemExit, match="differ from the subset receipt"):
        _resolve_with(tmp_path, m)


def test_missing_subset_receipt_is_refused(tmp_path):
    m = _manifest()
    del m["subset_receipt"]
    with pytest.raises(SystemExit, match="subset receipt missing"):
        _resolve_with(tmp_path, m)


def test_modified_subset_hash_is_refused(tmp_path):
    m = _manifest()
    m["subset_receipt"]["sha256"] = "1" * 64
    with pytest.raises(SystemExit, match="hash differs"):
        _resolve_with(tmp_path, m)


def _with_subset(tmp_path, mutate):
    """A copy of the real v3 subset receipt, mutated, bound by the manifest at its new hash."""
    import hashlib
    subset = json.loads(SUBSET.read_text(encoding="utf-8"))
    mutate(subset)
    rel_dir = ROOT / "results" / "sft_root_cause" / "_pytest_tmp_subset"
    rel_dir.mkdir(parents=True, exist_ok=True)
    path = rel_dir / f"{tmp_path.name}.json"
    path.write_bytes(json.dumps(subset).encode("utf-8"))
    m = _manifest()
    m["subset_receipt"] = {"path": path.relative_to(ROOT).as_posix(),
                           "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    try:
        return _resolve_with(tmp_path, m)
    finally:
        path.unlink()
        if not any(rel_dir.iterdir()):
            rel_dir.rmdir()


def test_subset_native_id_mismatch_is_refused(tmp_path):
    def mutate(s):
        s["sets"]["native_executable"] = s["sets"]["native_executable"][1:]
    with pytest.raises(SystemExit, match="native-executable set differs"):
        _with_subset(tmp_path, mutate)


def test_subset_exclusion_reason_mismatch_is_refused(tmp_path):
    def mutate(s):
        next(r for r in s["targets"] if r["failure_class"] == "policy")["exclusion_reason"] = "x"
    with pytest.raises(SystemExit, match="differ from the subset receipt"):
        _with_subset(tmp_path, mutate)


def test_subset_bound_to_a_different_panel_is_refused(tmp_path):
    def mutate(s):
        s["inputs_sha256"]["results/sft_root_cause_v27_confirmation_panel_r4.json"] = "2" * 64
    with pytest.raises(SystemExit, match="different frozen panel"):
        _with_subset(tmp_path, mutate)


def test_a_synthetic_subset_schema_is_refused_for_a_real_cohort(tmp_path):
    def mutate(s):
        s["schema_version"], s["status"] = "oneiros_synthetic_subset_v1", "SYNTHETIC_SOURCE_OF_TRUTH"
    with pytest.raises(SystemExit, match="schema/status"):
        _with_subset(tmp_path, mutate)


# --- full-precision p-values ---------------------------------------------------------------------

def test_tiny_exact_p_values_are_preserved():
    p = analysis.mcnemar_exact(29, 0)
    assert p == 3.725290298461914e-09
    fam = {h["label"]: h for h in analysis.fixed_family_holm({"sft_minus_base:42": p})["hypotheses"]}
    assert fam["sft_minus_base:42"]["raw_p"] == p
    assert fam["sft_minus_base:42"]["holm_adjusted_p"] == pytest.approx(2.2351741790771484e-08)
    assert fam["sft_minus_base:42"]["holm_adjusted_p"] > 0
    assert fam["relearn_minus_base:42"]["raw_p"] is None
    assert fam["relearn_minus_base:42"]["holm_adjusted_p"] is None
    assert len(fam) == 6


def test_per_seed_and_family_values_agree_exactly_and_are_nonzero():
    out = _analyse()
    fam = _family(out)
    for name in ("sft_minus_base", "relearn_minus_base"):
        for s, row in out["contrasts"][name]["per_seed_secondary"].items():
            label = f"{name}:{s}"
            assert row["mcnemar_exact_two_sided_p"] == fam[label]["raw_p"] > 0
            assert row["holm_adjusted_p"] == fam[label]["holm_adjusted_p"] > 0
    assert out["analysis_version"] == "oneiros_native_generated_tests_analyse_v5"


# --- analysis output ------------------------------------------------------------------------

def _records(targets, arms, kill):
    rows = []
    for a in arms:
        for s in analysis.SEEDS:
            for t in targets:
                for k in range(analysis.SLOTS):
                    cls = "semantic_kill" if k == 0 and kill(a, t) else "pass_both"
                    rows.append({"arm": a, "seed": s, "target_key": t, "slot": k, "class": cls,
                                 "fixed_valid": True, "canary_failed": False,
                                 "module_sha256": f"{a}{s}{t}{k}",
                                 "generation": {**{f: 1 for f in analysis.TELEMETRY_FIELDS},
                                                "hit_completion_limit": False,
                                                "eos_reached": True, "row_wall_seconds": 0.1}})
    return rows


def _analyse(gates_ok=True, kill=lambda a, t: a != "base", real_cohort=True,
             synthetic_gate=None, suppress=None):
    cohort = gio.resolve_cohort(JOB, MANIFEST)
    targets, arms = cohort["generation"], cohort["arms"]
    grid = analysis.index(_records(targets, arms, kill), targets, arms)
    evidence = {k: gates_ok for k in analysis.EVIDENCE_SUBGATES}
    if suppress:                            # break the pair-specific gate of one contrast
        for t in targets[:10]:
            for s in analysis.SEEDS:
                for k in range(analysis.SLOTS):
                    grid[(suppress, s, t, k)]["class"] = "environment_failure"
                    grid[(suppress, s, t, k)]["canary_failed"] = True
    return analysis.exploratory(grid, sorted(targets), cohort["repo_of"], arms, cohort,
                                evidence, synthetic_gate=synthetic_gate)


def test_analysis_reports_the_exact_34_29_5_accounting():
    out = _analyse()
    c = out["cohort"]
    assert (c["frozen_panel_targets"], c["native_executable_targets"], c["generation_targets"],
            c["panel_policy_excluded"]) == (34, 29, 29, 5)
    assert len(c["panel_policy_exclusions"]) == 5 and all(e["reason"] for e in
                                                         c["panel_policy_exclusions"])
    assert "never model failures" in c["note"]
    assert set(c["post_generation_infrastructure_exclusions"]) == {"sft_minus_base",
                                                                  "relearn_minus_base"}
    for contrast in out["contrasts"].values():           # policy exclusions in no denominator
        assert contrast["requested_targets"] == 29


# --- Fix 2: fixed six-hypothesis Holm family -------------------------------------------------------

def test_the_family_is_derived_from_the_frozen_plan():
    assert analysis.HOLM_FAMILY == ("sft_minus_base:42", "sft_minus_base:43", "sft_minus_base:44",
                                    "relearn_minus_base:42", "relearn_minus_base:43",
                                    "relearn_minus_base:44")


def _family(out):
    return {h["label"]: h for h in out["multiplicity"]["hypotheses"]}


def test_both_contrasts_computed():
    out = _analyse()
    fam = _family(out)
    assert out["multiplicity"]["holm_family_size"] == 6 and len(fam) == 6
    assert all(h["status"] == "computed" and h["raw_p"] is not None for h in fam.values())


@pytest.mark.parametrize("suppressed, computed", [("relearn", "sft"), ("sft", "relearn")])
def test_one_contrast_suppressed_keeps_the_family_at_six(suppressed, computed):
    out = _analyse(suppress=suppressed)
    fam = _family(out)
    assert out["contrasts"][f"{suppressed}_minus_base"]["status"].startswith("SUPPRESSED")
    assert out["multiplicity"]["holm_family_size"] == 6
    for s in analysis.SEEDS:
        sup, com = fam[f"{suppressed}_minus_base:{s}"], fam[f"{computed}_minus_base:{s}"]
        assert sup["status"] == "suppressed" and sup["raw_p"] is None \
            and sup["holm_adjusted_p"] is None
        assert com["status"] == "computed"
    # the computed contrast is adjusted within the full six-test family (suppressed as p = 1)
    raw = {h: fam[h]["raw_p"] for h in fam if fam[h]["status"] == "computed"}
    expected = analysis.holm({h: raw.get(h, 1.0) for h in analysis.HOLM_FAMILY})
    for h in raw:
        assert fam[h]["holm_adjusted_p"] == expected[h]
        assert out["contrasts"][f"{computed}_minus_base"]["per_seed_secondary"][h[-2:]][
            "holm_family_size"] == 6


def test_both_contrasts_suppressed():
    out = _analyse(gates_ok=False)
    fam = _family(out)
    assert out["multiplicity"]["holm_family_size"] == 6
    assert all(h["status"] == "suppressed" and h["raw_p"] is None for h in fam.values())
    assert all(c["status"].startswith("SUPPRESSED") for c in out["contrasts"].values())


def test_adjustment_uses_six_even_with_three_computed_values():
    computed = {"sft_minus_base:42": 0.01, "sft_minus_base:43": 0.02, "sft_minus_base:44": 0.03}
    fam = {h["label"]: h for h in analysis.fixed_family_holm(computed)["hypotheses"]}
    assert fam["sft_minus_base:42"]["holm_adjusted_p"] == 0.06      # 6 x 0.01, not 3 x 0.01
    with pytest.raises(analysis.AnalysisRefused):
        analysis.fixed_family_holm({"other:42": 0.5})


def test_no_inferential_claims():
    out = _analyse()
    assert out["multiplicity"]["descriptive_only"] is True
    assert not any(out[k] for k in ("generalization_established", "sft_benefit_established",
                                    "relearning_benefit_established"))
