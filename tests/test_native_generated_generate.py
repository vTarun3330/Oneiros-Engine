"""Durable generation runner (mock backend only; no model is loaded)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts import native_generated_tests_generate as gen


def _job(n=3):
    prompts = [{"target_key": f"t{i}", "prompt": f"prompt {i}", "condition": "whole_module",
                "prompt_sha256": hashlib.sha256(f"prompt {i}".encode()).hexdigest()}
               for i in range(n)]
    fit = gen.sequence_fit(prompts, lambda text: len(text))
    scans = {p["target_key"]: {"ok": True} for p in prompts}
    return gen.build_job(prompts, scans, fit), prompts


def test_contract_freezes_the_protocol_values():
    c = gen.CONTRACT
    assert (c["seeds"], c["candidates"], c["temperature"], c["top_p"], c["max_new_tokens"],
            c["prompt_token_limit"], c["sequence_limit"], c["batch_size"]) == (
        [42, 43, 44], 8, 0.7, 0.9, 1024, 2048, 3072, 2)
    assert c["prompt_token_limit"] + c["max_new_tokens"] <= c["sequence_limit"]


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


def test_extraction_strips_only_a_whole_output_fence():
    assert gen.extract("```python\nx = 1\n```")["module"] == "x = 1\n"
    inner = "x = 1\n```python\ny\n```\n"
    assert gen.extract(inner)["fence_stripped"] is False


def test_crash_and_resume_equals_an_uninterrupted_run(tmp_path):
    job, _ = _job()
    clean = gen.run(job, "base", tmp_path / "clean", {"backend": "mock"}, gen.mock_backend)
    with pytest.raises(RuntimeError, match="deliberate crash"):
        gen.run(job, "base", tmp_path / "crashy", {"backend": "mock"}, gen.mock_backend,
                crash_after=4)
    resumed = gen.run(job, "base", tmp_path / "crashy", {"backend": "mock"}, gen.mock_backend)
    assert resumed["mode"] == "resume" and resumed["lines"] == 9
    clean_rows = sorted((tmp_path / "clean" / "generations_base.jsonl").read_text().splitlines())
    resumed_rows = sorted((tmp_path / "crashy" / "generations_base.jsonl").read_text().splitlines())
    assert clean_rows == resumed_rows
    row = json.loads(clean_rows[0])
    assert len(row["raw_outputs"]) == 8 and row["raw_sha256"][0] == hashlib.sha256(
        row["raw_outputs"][0].encode()).hexdigest()


def test_partial_line_is_quarantined_and_changed_contract_refused(tmp_path):
    job, _ = _job()
    out = tmp_path / "o"
    gen.run(job, "base", out, {"backend": "mock"}, gen.mock_backend)
    path = out / "generations_base.jsonl"
    path.write_bytes(path.read_bytes() + b'{"key": "t0::42"')
    again = gen.run(job, "base", out, {"backend": "mock"}, gen.mock_backend)
    assert again["lines"] == 9 and list((out / "quarantine").iterdir())
    with pytest.raises(SystemExit, match="contract differs"):
        gen.run(job, "base", out, {"backend": "other"}, gen.mock_backend)


def test_seeds_are_per_target_and_arms_share_them(tmp_path):
    job, _ = _job(2)
    gen.run(job, "base", tmp_path, {"backend": "mock"}, gen.mock_backend)
    gen.run(job, "sft", tmp_path, {"backend": "mock"}, gen.mock_backend)
    base = {json.loads(l)["key"]: json.loads(l)["target_seed"]
            for l in (tmp_path / "generations_base.jsonl").read_text().splitlines()}
    sft = {json.loads(l)["key"]: json.loads(l)["target_seed"]
           for l in (tmp_path / "generations_sft.jsonl").read_text().splitlines()}
    assert base == sft and len(set(base.values())) == len(base)


def test_gpu_backend_is_refused_in_this_work_block(tmp_path):
    job, _ = _job(1)
    (tmp_path / "job.json").write_text(json.dumps(job))
    with pytest.raises(SystemExit, match="REFUSED"):
        gen.main(["run", "--job", str(tmp_path / "job.json"), "--arm", "sft", "--out",
                  str(tmp_path / "o"), "--backend", "hf"])
