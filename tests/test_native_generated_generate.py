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
    assert {r["target_key"]: r["reasons"] for r in job["refused"]} == {
        "leak": ["leakage"], "long": ["sequence_overflow"], "tamper": ["seal_mismatch"]}


def test_overlapping_leakage_and_overflow_are_both_recorded():
    """Regression (v2.2 B): marshmallow@252090c was leakage-refused AND over the limit,
    but v2.1 recorded only its first failure."""
    prompts = [{"target_key": k, "prompt": p, "condition": "whole_module",
                "prompt_sha256": hashlib.sha256(p.encode()).hexdigest()}
               for k, p in (("ok", "a"), ("both", "b" * 3000), ("leak", "c"),
                            ("long", "d" * 3000))]
    prompts.append({"target_key": "all3", "prompt": "e" * 3000, "condition": "whole_module",
                    "prompt_sha256": "0" * 64})
    fit = gen.sequence_fit(prompts, lambda text: len(text))
    scans = {"ok": {"ok": True}, "both": {"ok": False, "reasons": ["issue_text: x"]},
             "leak": {"ok": False, "reasons": ["issue_text: y"]}, "long": {"ok": True}}
    job = gen.build_job(prompts, scans, fit,
                        builder_refused=[{"target_key": "nobuild", "reason": "not found"}])
    reasons = {r["target_key"]: r["reasons"] for r in job["refused"]}
    assert reasons == {"both": ["leakage", "sequence_overflow"], "leak": ["leakage"],
                       "long": ["sequence_overflow"],
                       "all3": ["seal_mismatch", "leakage", "sequence_overflow"],
                       "nobuild": ["prompt_refused"]}
    acc = gen.refusal_accounting(["ok", "both", "leak", "long", "all3", "nobuild"], job)
    assert acc["admitted"] == 1 and acc["unique_refused"] == 5 and acc["denominator_kept"] == 6
    assert acc["per_reason"] == {"prompt_refused": 1, "seal_mismatch": 1, "leakage": 3,
                                 "sequence_overflow": 3}
    assert sum(acc["per_reason"].values()) == 8 != acc["unique_refused"]
    assert acc["combinations"] == {"leakage": 1, "leakage+seal_mismatch+sequence_overflow": 1,
                                   "leakage+sequence_overflow": 1, "prompt_refused": 1,
                                   "sequence_overflow": 1}
    assert acc["overlapping_targets"] == ["all3", "both"]
    assert acc["unaccounted"] == [] and acc["unexpected"] == [] and \
        acc["admitted_and_refused"] == []
    missing = gen.refusal_accounting(["ok", "both", "leak", "long", "all3", "nobuild", "lost"], job)
    assert missing["unaccounted"] == ["lost"]


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


# --- GPU authorisation: the v2.1 in-generator check was superseded by the read-only launch
# gate of amendment v2.2 section D; its lifecycle is tested in tests/test_native_launch_gate.py.

def test_cli_hf_refuses_without_preflight_and_authorisation(tmp_path):
    job_file = tmp_path / "job.json"
    job_file.write_text(json.dumps({"primary_whole_module": _job(1)}))
    with pytest.raises(gen.Refused, match="authorisation"):
        gen.main(["run", "--job", str(job_file), "--condition", "primary_whole_module",
                  "--arm", "sft", "--out", str(tmp_path / "o"), "--backend", "hf"])
    assert not (tmp_path / "o").exists()


def test_generator_uses_the_launch_gate_and_no_file_existence_shortcut():
    source = (ROOT / "scripts" / "native_generated_tests_generate.py").read_text(encoding="utf-8")
    assert "from harness.native_launch_gate import evaluate" in source
    assert "verify_authorization" not in source and ".exists()" not in source.split(
        "def launch_gate", 1)[1].split("def ", 1)[0]
    assert gen.PROTOCOL_FILES[-1].endswith("PROTOCOL_V2_2.md")
