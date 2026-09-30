"""Durable generation runner v2: identity, condition selection, source-stable authorisation.

No model is loaded; the mock backend is used throughout."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts import native_generated_tests_generate as gen

ROOT = Path(__file__).resolve().parent.parent


def _job(n=3):
    prompts = [{"target_key": f"t{i}", "prompt": f"prompt {i}", "condition": "whole_module",
                "prompt_sha256": hashlib.sha256(f"prompt {i}".encode()).hexdigest()}
               for i in range(n)]
    fit = gen.sequence_fit(prompts, lambda text: len(text))
    return gen.build_job(prompts, {p["target_key"]: {"ok": True} for p in prompts}, fit)


IDENTITY = {"backend": "mock", "condition": "primary_whole_module", "arm": "base",
            "source_commit": "abc", "model": {"snapshot_manifest_sha256": "m"}}


def test_contract_freezes_every_protocol_value():
    c = gen.CONTRACT
    assert c == {"base_model": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
                 "base_revision": "2e1fd397ee46e1388853d2af2c993145b0f1098a",
                 "sft_adapter": "checkpoints/local_sft_armA_baseline_successor_s42/sft_adapter",
                 "seeds": [42, 43, 44], "candidates": 8, "temperature": 0.7, "top_p": 0.9,
                 "do_sample": True, "max_new_tokens": 1024, "prompt_token_limit": 2048,
                 "sequence_limit": 3072, "batch_size": 2,
                 "extraction": "whole_output_single_fence_strip", "reranking": "none",
                 "duplicates": "kept", "raw_output_retained": True,
                 "attention_implementation": "sdpa"}
    source = (ROOT / "scripts" / "native_generated_tests_generate.py").read_text(encoding="utf-8")
    assert "do_sample=CONTRACT[\"do_sample\"]" in source and "do_sample=True" not in source


def test_job_refuses_leaky_unsealed_and_overflowing_prompts():
    prompts = [{"target_key": k, "prompt": p, "condition": "whole_module",
                "prompt_sha256": hashlib.sha256(p.encode()).hexdigest()}
               for k, p in (("ok", "a"), ("leak", "b"), ("long", "c" * 3000), ("tamper", "d"))]
    prompts[3]["prompt"] = "changed"
    fit = gen.sequence_fit(prompts, lambda text: len(text))
    job = gen.build_job(prompts, {"ok": {"ok": True}, "leak": {"ok": False, "reasons": ["x"]},
                                  "long": {"ok": True}, "tamper": {"ok": True}}, fit)
    assert [i["target_key"] for i in job["items"]] == ["ok"]
    assert {r["target_key"]: r["reason"] for r in job["refused"]} == {
        "leak": "leakage", "long": "sequence_overflow", "tamper": "seal_mismatch"}


def test_condition_selection_and_removed_scaffold():
    job = _job()
    file = {"primary_whole_module": job}
    assert gen.select_condition(file, "primary_whole_module") is job
    with pytest.raises(gen.Refused, match="removed by amendment v2.1"):
        gen.select_condition(file, "secondary_scaffolded_diagnostic")
    tampered = {"primary_whole_module": {**job, "job_sha256": "0" * 64}}
    with pytest.raises(gen.Refused, match="do not match their hash"):
        gen.select_condition(tampered, "primary_whole_module")


def test_crash_and_resume_equals_an_uninterrupted_run(tmp_path):
    job = _job()
    gen.run(job, "base", "primary_whole_module", tmp_path / "clean", IDENTITY, gen.mock_backend)
    with pytest.raises(RuntimeError, match="deliberate crash"):
        gen.run(job, "base", "primary_whole_module", tmp_path / "crashy", IDENTITY,
                gen.mock_backend, crash_after=4)
    resumed = gen.run(job, "base", "primary_whole_module", tmp_path / "crashy", IDENTITY,
                      gen.mock_backend)
    assert resumed["mode"] == "resume" and resumed["lines"] == 9
    name = "generations_primary_whole_module_base.jsonl"
    assert sorted((tmp_path / "clean" / name).read_text().splitlines()) == \
        sorted((tmp_path / "crashy" / name).read_text().splitlines())


@pytest.mark.parametrize("field, value", [
    ("source_commit", "def"), ("backend", "hf"), ("model", {"snapshot_manifest_sha256": "x"}),
    ("condition", "other"),
])
def test_any_identity_change_refuses_and_preserves_prior_output(tmp_path, field, value):
    job = _job()
    gen.run(job, "base", "primary_whole_module", tmp_path, IDENTITY, gen.mock_backend)
    output = tmp_path / "generations_primary_whole_module_base.jsonl"
    before = output.read_bytes()
    with pytest.raises(gen.Refused, match="identity differs"):
        gen.run(job, "base", "primary_whole_module", tmp_path, {**IDENTITY, field: value},
                gen.mock_backend)
    assert output.read_bytes() == before


def test_partial_line_is_quarantined(tmp_path):
    job = _job()
    gen.run(job, "base", "primary_whole_module", tmp_path, IDENTITY, gen.mock_backend)
    path = tmp_path / "generations_primary_whole_module_base.jsonl"
    path.write_bytes(path.read_bytes() + b'{"key": "t0::42"')
    again = gen.run(job, "base", "primary_whole_module", tmp_path, IDENTITY, gen.mock_backend)
    assert again["lines"] == 9 and list((tmp_path / "quarantine").iterdir())


def test_seeds_are_per_target_and_arms_share_them(tmp_path):
    job = _job(2)
    gen.run(job, "base", "primary_whole_module", tmp_path, IDENTITY, gen.mock_backend)
    gen.run(job, "sft", "primary_whole_module", tmp_path, {**IDENTITY, "arm": "sft"},
            gen.mock_backend)
    rows = {arm: {json.loads(l)["key"]: json.loads(l)["target_seed"] for l in
                  (tmp_path / f"generations_primary_whole_module_{arm}.jsonl")
                  .read_text().splitlines()} for arm in ("base", "sft")}
    assert rows["base"] == rows["sft"] and len(set(rows["base"].values())) == len(rows["base"])


def _snapshot_available():
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(gen.CONTRACT["base_model"], revision=gen.CONTRACT["base_revision"],
                          local_files_only=True)
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _snapshot_available(), reason="base snapshot not local")
def test_the_exact_cli_command_works_with_the_mock_backend(tmp_path):
    job_file = tmp_path / "job.json"
    job_file.write_text(json.dumps({"primary_whole_module": _job(2)}))
    out = tmp_path / "out"
    assert gen.main(["run", "--job", str(job_file), "--condition", "primary_whole_module",
                     "--arm", "sft", "--out", str(out), "--backend", "mock"]) == 0
    contract = json.loads((out / "contract_primary_whole_module_sft.json").read_text())
    identity = contract["identity"]
    for field in ("source_commit", "source_tree", "generator_sha256", "prompt_builder_sha256",
                  "protocol_sha256", "job_file_sha256", "job_sha256", "model",
                  "adapter_manifest_sha256", "libraries", "condition", "arm"):
        assert identity.get(field) is not None, field
    assert contract["contract"] == gen.CONTRACT


# --- source-stable GPU authorisation ---------------------------------------------------------

@pytest.fixture
def authorised(tmp_path, monkeypatch):
    monkeypatch.setattr(gen, "_git", lambda *a: "" if a[0] == "status" else "HEADSHA")
    preflight = tmp_path / "preflight.json"
    preflight.write_text(json.dumps({"pipeline_ready": True, "source_commit": "HEADSHA"}))
    job = _job()
    out = tmp_path / "gpu_out"
    auth = {"schema_version": gen.AUTH_SCHEMA, "preflight_path": str(preflight),
            "preflight_sha256": gen.sha256_file(preflight), "source_commit": "HEADSHA",
            "protocol_sha256": {p: gen.sha256_file(ROOT / p) for p in gen.PROTOCOL_FILES},
            "job_sha256": job["job_sha256"], "allowed_conditions": ["primary_whole_module"],
            "allowed_arms": ["base"], "output_dir": str(out)}
    path = tmp_path / "auth.json"
    path.write_text(json.dumps(auth))
    return {"path": path, "auth": auth, "job": job, "out": out, "tmp": tmp_path}


def test_matching_authorisation_is_accepted(authorised):
    got = gen.verify_authorization(authorised["path"], job=authorised["job"],
                                   condition="primary_whole_module", arm="base",
                                   out_dir=authorised["out"])
    assert got["source_commit"] == "HEADSHA"


def test_hf_without_authorisation_refuses(tmp_path):
    with pytest.raises(gen.Refused, match="needs a GPU authorisation receipt"):
        gen.verify_authorization(None, job=_job(), condition="primary_whole_module",
                                 arm="base", out_dir=tmp_path)


@pytest.mark.parametrize("kwargs, fragment", [
    ({"arm": "sft"}, "arm"), ({"condition": "other"}, "condition"),
    ({"out_dir": "elsewhere"}, "output directory"), ({"job": "other"}, "job"),
])
def test_authorisation_does_not_extend_to_other_runs(authorised, kwargs, fragment):
    args = {"job": authorised["job"], "condition": "primary_whole_module", "arm": "base",
            "out_dir": authorised["out"]}
    if kwargs.get("job") == "other":
        kwargs = {"job": _job(4)}
    if kwargs.get("out_dir") == "elsewhere":
        kwargs = {"out_dir": authorised["tmp"] / "elsewhere"}
    args.update(kwargs)
    with pytest.raises(gen.Refused, match=fragment):
        gen.verify_authorization(authorised["path"], **args)


@pytest.mark.parametrize("mutation, fragment", [
    ({"source_commit": "OTHER"}, "source commit"),
    ({"preflight_sha256": "0" * 64}, "preflight hash"),
    ({"protocol_sha256": {}}, "protocol hash"),
    ({"schema_version": "x"}, "schema"),
])
def test_stale_or_wrong_authorisation_refuses(authorised, mutation, fragment):
    authorised["path"].write_text(json.dumps({**authorised["auth"], **mutation}))
    with pytest.raises(gen.Refused, match=fragment):
        gen.verify_authorization(authorised["path"], job=authorised["job"],
                                 condition="primary_whole_module", arm="base",
                                 out_dir=authorised["out"])


def test_preflight_that_is_not_pipeline_ready_refuses(authorised):
    preflight = Path(authorised["auth"]["preflight_path"])
    preflight.write_text(json.dumps({"pipeline_ready": False, "source_commit": "HEADSHA"}))
    authorised["path"].write_text(json.dumps({**authorised["auth"],
                                              "preflight_sha256": gen.sha256_file(preflight)}))
    with pytest.raises(gen.Refused, match="not pipeline_ready"):
        gen.verify_authorization(authorised["path"], job=authorised["job"],
                                 condition="primary_whole_module", arm="base",
                                 out_dir=authorised["out"])


def test_cli_hf_refuses_without_authorisation(tmp_path):
    job_file = tmp_path / "job.json"
    job_file.write_text(json.dumps({"primary_whole_module": _job(1)}))
    with pytest.raises(gen.Refused, match="authorisation"):
        gen.main(["run", "--job", str(job_file), "--condition", "primary_whole_module",
                  "--arm", "sft", "--out", str(tmp_path / "o"), "--backend", "hf"])
