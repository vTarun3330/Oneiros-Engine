"""v2.7 three-arm pipeline: frozen arm registry, named-arm generation, gate/authorisation
versioning, three-arm loading and pairwise exploratory analysis."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from harness import native_arm_registry as reg  # noqa: E402
from harness import native_launch_gate as gate  # noqa: E402
from scripts import native_generated_tests_analyse as analysis  # noqa: E402
from scripts import native_generated_tests_generate as gen  # noqa: E402
from scripts import native_generation_io as gio  # noqa: E402

JOB_R5 = ROOT / "results/sft_root_cause_v27_generation_job_r5.json"
MANIFEST_R5 = ROOT / "results/sft_root_cause_v27_generation_manifest_r5.json"
LOCAL_ADAPTERS = all((ROOT / reg.REGISTRY["arms"][a]["adapter_dir"]).is_dir()
                     for a in ("sft", "relearn"))
needs_adapters = pytest.mark.skipif(not LOCAL_ADAPTERS, reason="local checkpoints not in git")


# --- registry -------------------------------------------------------------------------------

def _fake_adapter(tmp: Path, name: str, base=reg.BASE_MODEL) -> dict:
    d = tmp / name
    d.mkdir(parents=True)
    (d / "adapter_model.safetensors").write_bytes(name.encode() * 64)
    (d / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": base}))
    files = reg.adapter_manifest(d)
    return {"label": name, "adapter_dir": name,
            "adapter_manifest_sha256": reg.manifest_sha256(files),
            "adapter_model_sha256": files["adapter_model.safetensors"],
            "adapter_config_sha256": files["adapter_config.json"], "receipts": {}}


def _fake_registry(tmp: Path) -> dict:
    r = copy.deepcopy(reg.REGISTRY)
    r["arms"]["sft"] = _fake_adapter(tmp, "a431")
    r["arms"]["relearn"] = _fake_adapter(tmp, "relearn141")
    return r


def test_registry_hash_is_canonical_and_bound_everywhere():
    s = reg.summary()
    assert s["sha256"] == reg.registry_sha256() and s["arms"] == ["base", "sft", "relearn"]
    manifest = json.loads(MANIFEST_R5.read_text(encoding="utf-8"))
    job = json.loads(JOB_R5.read_text(encoding="utf-8"))
    assert manifest["arm_registry"] == s == job["arm_registry"]


@needs_adapters
def test_frozen_registry_verifies_against_disk():
    assert reg.verify(ROOT) == []


def test_swapped_adapters_are_refused(tmp_path):
    r = _fake_registry(tmp_path)
    assert reg.verify(tmp_path, registry=r) == []
    swapped = copy.deepcopy(r)
    swapped["arms"]["sft"]["adapter_dir"], swapped["arms"]["relearn"]["adapter_dir"] = \
        r["arms"]["relearn"]["adapter_dir"], r["arms"]["sft"]["adapter_dir"]
    problems = reg.verify(tmp_path, registry=swapped)
    assert any("swapped" in p for p in problems)


@pytest.mark.parametrize("victim", ["adapter_model.safetensors", "adapter_config.json"])
def test_modified_adapter_or_config_is_refused(tmp_path, victim):
    r = _fake_registry(tmp_path)
    path = tmp_path / "a431" / victim
    path.write_bytes(path.read_bytes() + b" ")
    assert reg.verify_arm(tmp_path, "sft", registry=r)


def test_adapter_trained_on_another_base_is_refused(tmp_path):
    r = copy.deepcopy(reg.REGISTRY)
    r["arms"]["sft"] = _fake_adapter(tmp_path, "other", base="some/other-model")
    assert any("base model" in p for p in reg.verify_arm(tmp_path, "sft", registry=r))


# --- cohort / generator ------------------------------------------------------------------------

def test_relearn_requires_the_registry():
    with pytest.raises(SystemExit, match="registry"):
        gio.cohort_arms({"arms": ["base", "sft", "relearn"]})
    with pytest.raises(SystemExit, match="differs"):
        gio.cohort_arms({"arms": ["base", "sft", "relearn"],
                         "arm_registry": {**reg.summary(), "sha256": "0" * 64}})
    assert gio.cohort_arms({}) == (("base", "sft"), None)        # legacy unchanged


def test_r5_cohort_dimensions():
    c = gio.resolve_cohort(JOB_R5, MANIFEST_R5)
    e = c["expected"]
    assert c["arms"] == ("base", "sft", "relearn")
    assert (e["rows_per_arm"], e["candidates_per_arm"]) == (87, 696)
    assert (e["rows_total"], e["candidates_total"]) == (261, 2088)
    assert c["study_mode"] == analysis.EXPLORATORY


def test_generator_refuses_relearn_without_a_registry_bound_job():
    with pytest.raises(SystemExit, match="registry"):
        gen.arm_adapter_dir("relearn", None)
    assert gen.arm_adapter_dir("base", None) is None
    assert gen.arm_adapter_dir("sft", None) == ROOT / gen.CONTRACT["sft_adapter"]


def test_generator_resolves_adapters_only_through_the_registry():
    s = reg.summary()
    assert gen.arm_adapter_dir("relearn", s) == ROOT / reg.REGISTRY["arms"]["relearn"]["adapter_dir"]
    with pytest.raises(SystemExit, match="differs"):
        gen.job_registry({"arm_registry": {**s, "sha256": "f" * 64}})


def _identity(arm, registry_sha="r1", adapter="x"):
    return {"backend": "synthetic", "arm": arm, "condition": "primary_whole_module",
            "job_sha256": "j", "job_file_sha256": "jf",
            "adapter_manifest_sha256": None if arm == "base" else adapter,
            "arm_registry_sha256": registry_sha}


def test_a_registry_change_invalidates_resume(tmp_path):
    job = {"items": [{"target_key": "t", "prompt": "p", "prompt_sha256": gen.sha256_text("p")}]}
    job["job_sha256"] = gen.contract_sha({"items": job["items"]})
    gen.run(job, "relearn", "primary_whole_module", tmp_path, _identity("relearn"),
            gen.mock_backend)
    with pytest.raises(SystemExit, match="identity differs"):
        gen.run(job, "relearn", "primary_whole_module", tmp_path,
                _identity("relearn", registry_sha="r2"), gen.mock_backend)


# --- launch gate / authorisation ---------------------------------------------------------------

def test_versioned_preflight_and_authorisation_never_cross():
    assert gate.AUTH_FOR[gate.PREFLIGHT_SCHEMA_V27] == "oneiros_native_gpu_authorization_v4"
    assert gate.AUTH_FOR[gate.PREFLIGHT_SCHEMA] == "oneiros_native_gpu_authorization_v3"


def test_v27_authorisation_binds_registry_adapters_arms_and_outputs(tmp_path):
    pre = {"schema_version": gate.PREFLIGHT_SCHEMA_V27, "source": {"commit": "c"},
           "arm_registry": reg.summary(), "adapters": reg.summary()["adapter_manifest_sha256"],
           "job": {"file_sha256": "j"}, "model": {"m": 1}}
    pre_path = tmp_path / "pre.json"
    pre_path.write_text(json.dumps(pre))
    job = tmp_path / "job.json"
    job.write_text("{}")
    identity = {"executable_tree_sha256": "e", "protocol_sha256": {}}
    good = {"schema_version": "oneiros_native_gpu_authorization_v4", "approved_by_user": True,
            "created_utc": "2026-01-01T00:00:00Z", "preflight_path": "pre.json",
            "preflight_sha256": gate._sha_bytes(pre_path.read_bytes()),
            "source": identity, "source_commit": "c", "protocol_sha256": {},
            "job_file_sha256": gate._sha_bytes(job.read_bytes()), "model": {"m": 1},
            "arm_registry_sha256": reg.registry_sha256(),
            "adapters": reg.summary()["adapter_manifest_sha256"],
            "adapter_manifest_sha256": None, "allowed_conditions": ["primary_whole_module"],
            "allowed_arms": ["base", "sft", "relearn"],
            "output_dirs": {a: f"out/{a}" for a in ("base", "sft", "relearn")}}
    pre["job"]["file_sha256"] = gate._sha_bytes(job.read_bytes())
    pre_path.write_text(json.dumps(pre))
    good["preflight_sha256"] = gate._sha_bytes(pre_path.read_bytes())

    def problems(auth, arm="relearn", out="out/relearn"):
        p = tmp_path / "auth.json"
        p.write_text(json.dumps(auth))
        return gate.authorization_problems(tmp_path, p, pre_path, identity=identity,
                                           job_path=job, condition="primary_whole_module",
                                           arm=arm, out_dir=tmp_path / out)
    assert problems(good) == []
    assert problems({**good, "schema_version": "oneiros_native_gpu_authorization_v3"})
    assert problems({**good, "arm_registry_sha256": "0" * 64})
    assert problems({**good, "allowed_arms": ["base", "sft"]})
    assert problems(good, out="out/sft")


def test_trained_arms_are_blocked_until_the_base_arm_verifies(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(gate, "base_arm_problems",
                        lambda *a, **k: (calls.append(1) or ["base arm not complete"], None))
    monkeypatch.setattr(gate, "pipeline_problems", lambda *a, **k: [])
    monkeypatch.setattr(gate, "authorization_problems", lambda *a, **k: [])
    monkeypatch.setattr(gate, "checkout_state", lambda root, fetch=None: {"head": "h", "fetch": {}})
    monkeypatch.setattr(gate, "source_identity", lambda root: {"executable_tree_sha256": "e",
                                                                "protocol_sha256": {}})
    for arm in ("sft", "relearn"):
        r = gate.evaluate(tmp_path, tmp_path / "p", tmp_path / "a", job_path=tmp_path / "j",
                          condition="primary_whole_module", arm=arm, out_dir=tmp_path / arm,
                          model_identity=dict, adapter_sha256=lambda: None)
        assert r["launch_ready"] is False and "base arm not complete" in r["pipeline_problems"]
    r = gate.evaluate(tmp_path, tmp_path / "p", tmp_path / "a", job_path=tmp_path / "j",
                      condition="primary_whole_module", arm="base", out_dir=tmp_path / "base",
                      model_identity=dict, adapter_sha256=lambda: None)
    assert r["launch_ready"] is True and len(calls) == 2


# --- loading ------------------------------------------------------------------------------------

def _write_arm(tmp: Path, arm: str, cohort: dict, adapter, registry_sha):
    job = {"items": [cohort["items"][t] for t in cohort["generation"]]}
    job["job_sha256"] = cohort["job_sha256"]
    identity = {**_identity(arm, registry_sha, adapter), "job_sha256": cohort["job_sha256"],
                "job_file_sha256": cohort["job_file_sha256"]}
    gen.run(job, arm, "primary_whole_module", tmp / arm, identity, gen.mock_backend)


def _r5_cohort():
    return gio.resolve_cohort(JOB_R5, MANIFEST_R5)


def test_three_arm_loader_accepts_exact_and_refuses_swapped_adapters(tmp_path):
    cohort = _r5_cohort()
    adapters = cohort["arm_registry"]["adapter_manifest_sha256"]
    rsha = cohort["arm_registry"]["sha256"]
    for arm in cohort["arms"]:
        _write_arm(tmp_path, arm, cohort, adapters.get(arm), rsha)
    paths = gio.arm_dirs(tmp_path, {}, cohort["arms"], "primary_whole_module")
    loaded = gio.load_arm_generations(paths, cohort)
    assert len(loaded["rows"]) == 261
    assert sum(len(r["candidates"]) for r in loaded["rows"].values()) == 2088
    bad = tmp_path / "bad"
    for arm in cohort["arms"]:
        swapped = {"sft": adapters["relearn"], "relearn": adapters["sft"]}.get(arm)
        _write_arm(bad, arm, cohort, swapped, rsha)
    with pytest.raises(SystemExit, match="registered adapter"):
        gio.load_arm_generations(gio.arm_dirs(bad, {}, cohort["arms"],
                                              "primary_whole_module"), cohort)


def test_arm_directories_must_match_the_declared_arms(tmp_path):
    with pytest.raises(SystemExit, match="differ"):
        gio.arm_dirs(None, {"base": tmp_path / "b", "sft": tmp_path / "s"},
                     ("base", "sft", "relearn"), "primary_whole_module")
    with pytest.raises(SystemExit, match="share"):
        gio.arm_dirs(None, {"base": tmp_path, "sft": tmp_path}, ("base", "sft"),
                     "primary_whole_module")


# --- analysis ------------------------------------------------------------------------------------

def _records(targets, arms, kill, infra=()):
    rows = []
    for a in arms:
        for s in analysis.SEEDS:
            for t in targets:
                for k in range(analysis.SLOTS):
                    cls = "pass_both"
                    if (a, t) in infra:
                        cls = "environment_failure"
                    elif k == 0 and kill(a, t):
                        cls = "semantic_kill"
                    rows.append({"arm": a, "seed": s, "target_key": t, "slot": k, "class": cls,
                                 "fixed_valid": cls != "environment_failure",
                                 "canary_failed": cls == "environment_failure",
                                 "module_sha256": f"{a}{s}{t}{k}",
                                 "generation": {f: 1 for f in analysis.TELEMETRY_FIELDS}
                                 | {"hit_completion_limit": False, "eos_reached": True,
                                    "row_wall_seconds": 0.1}})
    return rows


def _cohort(targets, arms):
    return {"arms": arms, "qualified": targets, "generation": targets,
            "pre_generation_exclusions": []}


def test_exploratory_pairwise_contrasts_and_isolated_denominators():
    targets = [f"t{i}" for i in range(10)]
    repo_of = {t: f"r{i % 5}" for i, t in enumerate(targets)}
    arms = ("base", "sft", "relearn")
    # relearn has an arm-specific infrastructure failure on t0: it must not touch sft - base
    rows = _records(targets, arms, lambda a, t: a == "sft" or (a == "relearn" and t < "t5"),
                    infra={("relearn", "t0")})
    out = analysis.analyse(rows, targets, repo_of, analysis.EXPLORATORY, cohort=_cohort(targets, arms),
                           evidence_gates={k: True for k in analysis.EVIDENCE_SUBGATES})
    sft, rel = out["contrasts"]["sft_minus_base"], out["contrasts"]["relearn_minus_base"]
    assert sft["pair_eligible_targets"] == 10 and sft["pair_infrastructure_excluded"] == []
    assert rel["pair_eligible_targets"] == 9 and rel["pair_infrastructure_excluded"] == ["t0"]
    assert sft["primary"]["point"] == 100.0 and sft["primary"]["targets"] == 10
    assert sft["per_seed_secondary"]["42"] == {**sft["per_seed_secondary"]["42"],
                                               "gained": 10, "lost": 0, "tied": 0}
    assert out["unit"].startswith("target") and out["grid_cells"] == 3 * 3 * 10 * 8
    assert all(not out[k] for k in ("generalization_established", "sft_benefit_established",
                                    "relearning_benefit_established"))


def test_exploratory_suppresses_without_global_evidence():
    targets = [f"t{i}" for i in range(10)]
    repo_of = {t: f"r{i % 5}" for i, t in enumerate(targets)}
    arms = ("base", "sft", "relearn")
    out = analysis.analyse(_records(targets, arms, lambda a, t: False), targets, repo_of,
                           analysis.EXPLORATORY, cohort=_cohort(targets, arms))
    assert all(c["status"].startswith("SUPPRESSED") for c in out["contrasts"].values())


def test_the_manifest_binds_the_frozen_analysis_plan():
    m = json.loads(MANIFEST_R5.read_text(encoding="utf-8"))
    assert m["study_mode"] == analysis.EXPLORATORY
    assert m["analysis_plan"]["sha256"] == analysis.analysis_plan_sha256()
    tampered = {**analysis.ANALYSIS_PLAN_V27, "claims": "anything"}
    assert analysis.analysis_plan_sha256(tampered) != m["analysis_plan"]["sha256"]


def test_one_target_per_repository_rule_is_deterministic():
    targets = [f"t{i}" for i in range(10)]
    repo_of = {t: f"r{i % 5}" for i, t in enumerate(targets)}
    chosen = analysis.one_per_repository(targets, repo_of)
    assert chosen == analysis.one_per_repository(list(reversed(targets)), repo_of)
    assert len(chosen) == 5 and len({repo_of[t] for t in chosen}) == 5


def test_legacy_modes_refuse_three_arms():
    targets = ["t0"]
    with pytest.raises(analysis.AnalysisRefused, match="base/sft"):
        analysis.analyse(_records(targets, ("base", "sft", "relearn"), lambda a, t: False),
                         targets, {"t0": "r"}, "engineering_dress_rehearsal",
                         cohort=_cohort(targets, ("base", "sft", "relearn")))
