"""v2.5 job wiring: the generation backend receives exactly the v2.5 builder's prompt; v2.4
jobs, stale builders, leaky or overflowing prompts are refused. No model is loaded."""
from __future__ import annotations

import json
from pathlib import Path
import shutil

import pytest

from harness import native_generated_test_job_v25 as jobs
from harness import native_generated_test_prompt_v25 as v25
from scripts import native_generated_tests_generate as gen
from tests.native_v25_job_helpers import (BUGGY, VERIFIER, count_words, dto, targets,
                                          v25_job_file)

ROOT = Path(__file__).resolve().parent.parent
COND = "primary_whole_module"


def _no_model(monkeypatch):
    monkeypatch.setattr(gen, "model_identity", lambda: {"snapshot_manifest_sha256": "m"})


def test_items_are_exactly_the_v25_builder_prompts(tmp_path):
    data = v25_job_file(tmp_path / "job.json", 2)
    job = data[COND]
    assert data["schema_version"] == jobs.JOB_SCHEMA and len(job["items"]) == 2
    for item in job["items"]:
        built = v25.build_prompt(dto(int(item["target_key"][1:])), BUGGY)
        assert item["prompt"] == built["prompt"] and item["prompt_sha256"] == built["prompt_sha256"]
        assert item["output_type"] == "pytest_module_v1"
        assert item["builder_version"] == v25.BUILDER_VERSION
        assert item["view_sha256"] == built["view_sha256"]
        assert item["target"]["import"] == "from pkg.geo import area"
        assert item["prompt_tokens"] == count_words(item["prompt"])
    assert data["builder"]["builder_sha256"]["v25_pytest_module_v1"]


def test_generation_backend_receives_the_v25_prompt_end_to_end(tmp_path, monkeypatch):
    _no_model(monkeypatch)
    v25_job_file(tmp_path / "job.json", 2)
    seen, mock = [], gen.mock_backend

    def recording_backend(prompt, n, seed):
        seen.append(prompt)
        return mock(prompt, n, seed)
    monkeypatch.setattr(gen, "mock_backend", recording_backend)
    out = tmp_path / "out"
    assert gen.main(["run", "--job", str(tmp_path / "job.json"), "--condition", COND,
                     "--arm", "base", "--out", str(out), "--backend", "mock"]) == 0
    expected = {v25.build_prompt(dto(i), BUGGY)["prompt"] for i in range(2)}
    assert set(seen) == expected and len(seen) == 2 * len(gen.CONTRACT["seeds"])
    assert all("Expected test format: pytest_module_v1" in p and "pytest_fragment" not in p
               for p in seen)
    contract = json.loads((out / f"contract_{COND}_base.json").read_text())
    assert contract["job_schema"] == jobs.JOB_SCHEMA
    assert contract["generator_version"] == "oneiros_native_generated_tests_generate_v5"
    assert contract["identity"]["prompt_builder_sha256"] == \
        jobs.builder_identity(ROOT)["builder_sha256"]


def _v24_job():
    prompts = [{"target_key": "t0", "prompt": "Expected test format: pytest_fragment",
                "condition": "whole_module"}]
    prompts[0]["prompt_sha256"] = gen.sha256_text(prompts[0]["prompt"])
    fit = gen.sequence_fit(prompts, len)
    return {COND: gen.build_job(prompts, {"t0": {"ok": True}}, fit)}


def test_a_v24_job_is_refused_before_anything_is_written(tmp_path, monkeypatch):
    _no_model(monkeypatch)
    (tmp_path / "job.json").write_text(json.dumps(_v24_job()))
    with pytest.raises(jobs.JobRefused, match="not a v2.5 job"):
        gen.main(["run", "--job", str(tmp_path / "job.json"), "--condition", COND,
                  "--arm", "base", "--out", str(tmp_path / "o"), "--backend", "mock"])
    assert not (tmp_path / "o").exists()


def test_the_real_historical_v24_job_is_refused():
    path = ROOT / "results/sft_root_cause_native_v24_rehearsal_job_v5.json"
    if not path.is_file():
        pytest.skip("historical v2.4 job not present")
    with pytest.raises(jobs.JobRefused, match="not a v2.5 job"):
        jobs.validate_job_file(json.loads(path.read_text(encoding="utf-8")), COND, ROOT)


def test_changing_the_prompt_builder_invalidates_the_job(tmp_path):
    root = tmp_path / "repo"
    for rel in jobs.BUILDER_FILES.values():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ROOT / rel, root / rel)
    data = v25_job_file(tmp_path / "job.json", 1, root=root)
    assert jobs.validate_job_file(data, COND, root)["items"]
    with (root / jobs.BUILDER_FILES["v25_pytest_module_v1"]).open("a") as fh:
        fh.write("\n# a change\n")
    with pytest.raises(jobs.JobRefused, match="different prompt builder"):
        jobs.validate_job_file(data, COND, root)


def test_a_builder_change_refuses_resume_of_an_existing_output(tmp_path, monkeypatch):
    _no_model(monkeypatch)
    v25_job_file(tmp_path / "job.json", 1)
    argv = ["run", "--job", str(tmp_path / "job.json"), "--condition", COND, "--arm", "base",
            "--out", str(tmp_path / "out"), "--backend", "mock"]
    assert gen.main(argv) == 0
    real = gen.collect_identity

    def changed(*a, **k):
        identity = real(*a, **k)
        identity["prompt_builder_sha256"] = {**identity["prompt_builder_sha256"],
                                             "v25_pytest_module_v1": "0" * 64}
        return identity
    monkeypatch.setattr(gen, "collect_identity", changed)
    with pytest.raises(gen.Refused, match="prompt_builder_sha256"):
        gen.main(argv)


def test_a_v24_output_directory_cannot_be_reused(tmp_path, monkeypatch):
    _no_model(monkeypatch)
    v25_job_file(tmp_path / "job.json", 1)
    out = tmp_path / "out"
    out.mkdir()
    (out / f"contract_{COND}_base.json").write_text(json.dumps(
        {"generator_version": "oneiros_native_generated_tests_generate_v4"}))
    with pytest.raises(gen.Refused, match="identity differs"):
        gen.main(["run", "--job", str(tmp_path / "job.json"), "--condition", COND,
                  "--arm", "base", "--out", str(out), "--backend", "mock"])


@pytest.mark.parametrize("mutate, fragment", [
    (lambda i: i.update(prompt=i["prompt"].replace("pytest_module_v1", "pytest_fragment")),
     "retired label"),
    (lambda i: i.update(output_type="pytest_fragment"), "not built by the v2.5 builder"),
    (lambda i: i.update(prompt_tokens=4096), "token fit"),
    (lambda i: i.update(prompt=i["prompt"] + " "), "seal mismatch"),
])
def test_tampered_items_are_refused(tmp_path, mutate, fragment):
    data = v25_job_file(tmp_path / "job.json", 1)
    item = data[COND]["items"][0]
    mutate(item)
    data[COND]["job_sha256"] = jobs.items_sha(data[COND]["items"])
    with pytest.raises(jobs.JobRefused, match=fragment):
        jobs.validate_job_file(data, COND, ROOT)


@pytest.mark.parametrize("leak, kind", [
    ("    return math.pi * r * r", "fixed_line"),
    ("    return math.pi * r * r", "patch_line"),
    ("    assert area(2.0) == 12.566370614359172", "official_test_line"),
    ("x = 12.566370614359172", "expected_literal"),
])
def test_fixed_code_patch_official_tests_and_expected_values_refuse_the_prompt(
        monkeypatch, leak, kind):
    """A builder defect that let protected material into the prompt is caught by the
    independent scanner; the clean builder output passes."""
    assert "item" in jobs.build_item(dto(0), BUGGY, VERIFIER, count_words)
    real = v25.build_prompt

    def leaky(d, src):
        built = dict(real(d, src))
        built["prompt"] = built["prompt"].replace("### Task", f"{leak}\n\n### Task")
        built["prompt_sha256"] = gen.sha256_text(built["prompt"])
        return built
    monkeypatch.setattr(v25, "build_prompt", leaky)
    out = jobs.build_item(dto(0), BUGGY, VERIFIER, count_words)
    assert out["refused"]["reasons"] == ["leakage"]
    assert any(r.startswith(kind) for r in out["refused"]["details"]["leakage"])


def test_missing_verifier_material_fails_closed():
    out = jobs.build_item(dto(0), BUGGY, None, count_words)
    assert out["refused"]["reasons"] == ["leakage"]


def test_overflow_is_refused_and_never_truncated():
    out = jobs.build_item(dto(0), BUGGY, VERIFIER, lambda text: 2049)
    assert out["refused"]["reasons"] == ["sequence_overflow"]
    out = jobs.build_item(dto(0), BUGGY, VERIFIER, lambda text: 2048)
    assert out["item"]["prompt"] == v25.build_prompt(dto(0), BUGGY)["prompt"]


def test_job_build_records_every_target_once():
    data = jobs.build_job_file(targets(3) + [{"dto": dto(9), "buggy_source": BUGGY}],
                               count_words, ROOT, purpose="test")
    job = data[COND]
    assert [i["target_key"] for i in job["items"]] == ["t0", "t1", "t2"]
    assert [r["target_key"] for r in job["refused"]] == ["t9"]
    with pytest.raises(jobs.JobRefused, match="duplicate"):
        jobs.build_job_file(targets(1) + targets(1), count_words, ROOT, purpose="test")
