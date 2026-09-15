"""Evaluating a checkpoint must not require pretending it was trained here.

Before this option existed, ``--phase sft_eval`` could only evaluate
``<run>/sft_adapter``. Measuring any other checkpoint meant building a
directory that looked like a finished training run - copied weights, a copied
dataset manifest, copied metadata, and a ``sft_complete.marker`` whose literal
meaning was false. Three provenance gates were each satisfied by manufacturing
the evidence that gate asked for.

So the tests that matter here are not only "does it load". They are: does it
refuse the wrong weights before touching the GPU, does it leave the source
directory exactly as it found it, and does the artifact say plainly where the
weights came from.
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import train_on_dataset as trainer

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable
REAL_ADAPTER = ROOT / "checkpoints/local_sft_armA_baseline_successor_s42/sft_adapter"
REAL_SHA = "e67dd599a37cbf2a738791c5cdf889cb4938a2ad53c991dca2fefae503b6f9e7"


def cli(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [PY, "scripts/train_on_dataset.py", *args],
        capture_output=True, text=True, cwd=ROOT,
    )


@pytest.fixture
def fake_adapter(tmp_path):
    """A structurally valid adapter with known bytes."""
    d = tmp_path / "sft_adapter"
    d.mkdir()
    payload = b"not real weights, but real bytes"
    (d / "adapter_model.safetensors").write_bytes(payload)
    (d / "adapter_config.json").write_text(json.dumps({"peft_type": "LORA"}), encoding="utf-8")
    return d, hashlib.sha256(payload).hexdigest()


# ------------------------------------------------------------ the resolver

def test_a_valid_adapter_resolves_to_its_own_path_and_hash(fake_adapter):
    d, sha = fake_adapter
    path, resolved = trainer.resolve_external_adapter(d, sha)
    assert path == d.resolve()
    assert resolved == sha


def test_the_real_arm_a_adapter_resolves():
    if not REAL_ADAPTER.is_dir():
        pytest.skip("Arm A checkpoint not present on this machine")
    path, sha = trainer.resolve_external_adapter(REAL_ADAPTER, REAL_SHA)
    assert sha == REAL_SHA
    assert path == REAL_ADAPTER.resolve()


def test_a_hash_mismatch_is_refused(fake_adapter):
    d, _ = fake_adapter
    with pytest.raises(RuntimeError, match="Adapter hash mismatch"):
        trainer.resolve_external_adapter(d, "b" * 64)


def test_an_absent_hash_is_refused(fake_adapter):
    d, _ = fake_adapter
    for empty in (None, ""):
        with pytest.raises(RuntimeError, match="requires --expected-adapter-sha256"):
            trainer.resolve_external_adapter(d, empty)


@pytest.mark.parametrize("bad", ["e67dd599", "not-a-hash", "f" * 63, "f" * 65])
def test_a_malformed_hash_is_refused(fake_adapter, bad):
    d, _ = fake_adapter
    with pytest.raises(RuntimeError, match="64-character hex"):
        trainer.resolve_external_adapter(d, bad)


def test_an_uppercase_or_padded_hash_is_normalised_not_rejected(fake_adapter):
    """A copied-and-pasted hash should not fail on case or stray whitespace.

    The value that gets recorded is always the canonical lowercase digest the
    file actually hashes to, so accepting these costs no precision.
    """
    d, sha = fake_adapter
    for supplied in (sha.upper(), f"  {sha}  ", f"\t{sha.upper()}\n"):
        _path, resolved = trainer.resolve_external_adapter(d, supplied)
        assert resolved == sha


def test_a_missing_directory_is_refused(tmp_path):
    with pytest.raises(RuntimeError, match="is not a directory"):
        trainer.resolve_external_adapter(tmp_path / "nope", "a" * 64)


def test_a_file_instead_of_a_directory_is_refused(tmp_path):
    f = tmp_path / "adapter"
    f.write_text("x", encoding="utf-8")
    with pytest.raises(RuntimeError, match="is not a directory"):
        trainer.resolve_external_adapter(f, "a" * 64)


@pytest.mark.parametrize("missing", ["adapter_model.safetensors", "adapter_config.json"])
def test_a_malformed_adapter_is_refused(fake_adapter, missing):
    d, sha = fake_adapter
    (d / missing).unlink()
    with pytest.raises(RuntimeError, match="not a loadable LoRA adapter"):
        trainer.resolve_external_adapter(d, sha)
    # and the message names what is actually absent
    try:
        trainer.resolve_external_adapter(d, sha)
    except RuntimeError as exc:
        assert missing in str(exc)


def test_an_empty_directory_names_both_missing_files(tmp_path):
    d = tmp_path / "empty"
    d.mkdir()
    with pytest.raises(RuntimeError) as exc:
        trainer.resolve_external_adapter(d, "a" * 64)
    assert "adapter_model.safetensors" in str(exc.value)
    assert "adapter_config.json" in str(exc.value)


def test_resolution_leaves_the_source_directory_untouched(fake_adapter):
    """The whole point: read in place, write nothing."""
    d, sha = fake_adapter
    before = {p.name: (p.stat().st_mtime_ns, hashlib.sha256(p.read_bytes()).hexdigest())
              for p in sorted(d.iterdir())}
    trainer.resolve_external_adapter(d, sha)
    after = {p.name: (p.stat().st_mtime_ns, hashlib.sha256(p.read_bytes()).hexdigest())
             for p in sorted(d.iterdir())}
    assert before == after
    assert set(d.iterdir()) == {d / "adapter_model.safetensors", d / "adapter_config.json"}


def test_no_marker_or_metadata_is_ever_created(fake_adapter):
    """A marker is exactly what this option exists to avoid fabricating."""
    d, sha = fake_adapter
    trainer.resolve_external_adapter(d, sha)
    for forbidden in ("sft_complete.marker", "sft_metadata.json",
                      "dataset_manifest.sha256", "sft_run_config.json"):
        assert not (d / forbidden).exists()
        assert not (d.parent / forbidden).exists()


def test_the_resolver_does_not_import_torch(fake_adapter):
    """Refusal must be cheap: no CUDA, no weights, no model."""
    d, sha = fake_adapter
    source = __import__("inspect").getsource(trainer.resolve_external_adapter)
    assert "torch" not in source
    assert "Phi3Generator" not in source


# --------------------------------------------------------------- provenance

def test_provenance_records_every_required_field(monkeypatch, fake_adapter, tmp_path):
    d, sha = fake_adapter
    monkeypatch.setattr(trainer, "EXTERNAL_ADAPTER_DIR", d)
    monkeypatch.setattr(trainer, "EXTERNAL_ADAPTER_SHA256", sha)
    monkeypatch.setattr(trainer, "EXTERNAL_ADAPTER_SOURCE_RUN", "local_sft_armA_baseline_successor_s42")
    out = tmp_path / "locked_val_armA_ckpt431_s42"
    prov = trainer.external_adapter_provenance(out, out.name)

    assert prov["adapter_provenance"] == "external_evaluation_adapter"
    assert prov["adapter_source_path"] == str(d)
    assert prov["adapter_sha256"] == sha
    assert re.fullmatch(r"[0-9a-f]{40}", prov["expected_base_model_revision"])
    assert prov["source_training_run"] == "local_sft_armA_baseline_successor_s42"
    assert prov["evaluation_run_name"] == "locked_val_armA_ckpt431_s42"
    assert prov["evaluation_output_directory"] == str(out)


def test_provenance_marks_the_adapter_inference_only(monkeypatch, fake_adapter, tmp_path):
    d, sha = fake_adapter
    monkeypatch.setattr(trainer, "EXTERNAL_ADAPTER_DIR", d)
    monkeypatch.setattr(trainer, "EXTERNAL_ADAPTER_SHA256", sha)
    prov = trainer.external_adapter_provenance(tmp_path, "run")
    assert prov["inference_only"] is True
    assert prov["training_forbidden_with_this_adapter"] is True
    assert prov["source_directory_modified"] is False


def test_an_unsupplied_source_run_is_recorded_as_null(monkeypatch, fake_adapter, tmp_path):
    d, sha = fake_adapter
    monkeypatch.setattr(trainer, "EXTERNAL_ADAPTER_DIR", d)
    monkeypatch.setattr(trainer, "EXTERNAL_ADAPTER_SHA256", sha)
    monkeypatch.setattr(trainer, "EXTERNAL_ADAPTER_SOURCE_RUN", None)
    assert trainer.external_adapter_provenance(tmp_path, "run")["source_training_run"] is None


def test_provenance_is_absent_from_a_normal_run(monkeypatch):
    """Adding a key for every run would invalidate existing progress files."""
    monkeypatch.setattr(trainer, "EXTERNAL_ADAPTER_DIR", None)
    context = trainer._adapter_evaluation_context(
        "fingerprint", "sft_adapter", "a" * 64, "ablation_dev", None, "b" * 64, 542)
    assert "external_adapter" not in context


def test_provenance_is_present_when_an_external_adapter_is_used(monkeypatch, fake_adapter):
    d, sha = fake_adapter
    monkeypatch.setattr(trainer, "EXTERNAL_ADAPTER_DIR", d)
    monkeypatch.setattr(trainer, "EXTERNAL_ADAPTER_SHA256", sha)
    monkeypatch.setattr(trainer, "EXTERNAL_ADAPTER_SOURCE_RUN", "src_run")
    context = trainer._adapter_evaluation_context(
        "fingerprint", "external_evaluation_adapter", sha, "val", None, "b" * 64, 700)
    assert context["external_adapter"]["adapter_provenance"] == "external_evaluation_adapter"
    assert context["external_adapter"]["adapter_sha256"] == sha
    assert context["adapter"] == "external_evaluation_adapter"


# ------------------------------------------------- runner bound to contract

def test_the_run_contract_binds_the_runner_and_the_resolution_code(monkeypatch):
    monkeypatch.setattr(trainer, "EXTERNAL_ADAPTER_DIR", None)
    contract = trainer._adapter_evaluation_context(
        "fingerprint", "base_model", "a" * 64, "val", None, "b" * 64, 700)["run_contract"]
    assert contract["runner_source_sha256"] == hashlib.sha256(
        (ROOT / "scripts" / "train_on_dataset.py").read_bytes()).hexdigest()
    assert re.fullmatch(r"[0-9a-f]{64}", contract["adapter_resolution_source_sha256"])


def test_the_resolution_hash_moves_only_when_resolution_changes(monkeypatch):
    before = trainer._adapter_resolution_source_sha256()
    assert before == trainer._adapter_resolution_source_sha256()
    monkeypatch.setattr(trainer, "EXTERNAL_ADAPTER_REQUIRED_FILES",
                        ("adapter_model.safetensors",))
    assert trainer._adapter_resolution_source_sha256() != before


def test_the_contract_still_binds_generation_settings(monkeypatch):
    monkeypatch.setattr(trainer, "EXTERNAL_ADAPTER_DIR", None)
    contract = trainer._adapter_evaluation_context(
        "fingerprint", "base_model", "a" * 64, "val", None, "b" * 64, 700)["run_contract"]
    for field in ("temperature", "top_p", "do_sample", "max_new_tokens",
                  "candidates_per_function", "prompt_token_limit",
                  "candidate_parse_mode", "retain_raw_output",
                  "base_model_name", "base_model_revision"):
        assert field in contract
    assert re.fullmatch(r"[0-9a-f]{40}", contract["base_model_revision"])


# --------------------------------------------------------------- the CLI

def test_the_option_is_rejected_outside_sft_eval():
    for phase in ("base_eval", "sft", "dpo", "sft_then_dpo"):
        result = cli("--phase", phase, "--run-name", "tmp_external_adapter_check",
                     "--adapter-dir", str(REAL_ADAPTER),
                     "--expected-adapter-sha256", REAL_SHA)
        assert result.returncode != 0, phase
        assert "valid only with --phase sft_eval" in result.stderr, phase


def test_the_hash_is_required_by_the_cli():
    result = cli("--phase", "sft_eval", "--run-name", "tmp_external_adapter_check",
                 "--adapter-dir", str(REAL_ADAPTER))
    assert result.returncode != 0
    assert "requires --expected-adapter-sha256" in result.stderr


def test_a_hash_without_an_adapter_is_rejected():
    result = cli("--phase", "sft_eval", "--run-name", "tmp_external_adapter_check",
                 "--expected-adapter-sha256", REAL_SHA)
    assert result.returncode != 0
    assert "requires --adapter-dir" in result.stderr


def test_a_source_run_without_an_adapter_is_rejected():
    result = cli("--phase", "sft_eval", "--run-name", "tmp_external_adapter_check",
                 "--adapter-source-run", "whatever")
    assert result.returncode != 0
    assert "requires --adapter-dir" in result.stderr


def test_the_cli_refuses_a_wrong_hash_before_loading_anything():
    if not REAL_ADAPTER.is_dir():
        pytest.skip("Arm A checkpoint not present on this machine")
    result = cli("--phase", "sft_eval", "--run-name", "tmp_external_adapter_check",
                 "--adapter-dir", str(REAL_ADAPTER),
                 "--expected-adapter-sha256", "c" * 64)
    assert result.returncode != 0
    assert "Adapter hash mismatch" in result.stderr
    # nothing was loaded, so no GPU banner and no corpus verification ran
    assert "Local GPU" not in result.stdout
    assert "Canonical corpus verified" not in result.stdout


def test_a_refused_run_creates_no_output_directories():
    cli("--phase", "sft_eval", "--run-name", "tmp_external_adapter_check",
        "--adapter-dir", str(REAL_ADAPTER), "--expected-adapter-sha256", "c" * 64)
    assert not (ROOT / "results" / "tmp_external_adapter_check").exists()
    assert not (ROOT / "checkpoints" / "tmp_external_adapter_check").exists()


def test_adapter_path_is_accepted_as_an_alias():
    result = cli("--phase", "sft_eval", "--run-name", "tmp_external_adapter_check",
                 "--adapter-path", str(REAL_ADAPTER),
                 "--expected-adapter-sha256", "c" * 64)
    assert "Adapter hash mismatch" in result.stderr


# ------------------------------------------------ marker/metadata not needed

def test_the_external_branch_returns_before_the_marker_gate():
    """Ordering is the mechanism: the gate must be unreachable, not merely
    satisfied. A directory with no marker, no metadata and no manifest must
    evaluate."""
    source = (ROOT / "scripts" / "train_on_dataset.py").read_text(encoding="utf-8")
    external = source.index("if EXTERNAL_ADAPTER_DIR is not None:\n        if TRAINING_PHASE != \"sft_eval\"")
    marker_gate = source.index("has_training_artifacts = sft_marker.exists()")
    metadata_gate = source.index("Verified SFT marker exists without SFT metadata")
    assert external < marker_gate
    assert external < metadata_gate


def _runner_branch() -> str:
    """The external-adapter branch inside run_training.

    Anchored on the phase guard, because the plain `if EXTERNAL_ADAPTER_DIR is
    not None:` also appears in the context builder and matching the first
    occurrence silently tested the wrong block.
    """
    source = (ROOT / "scripts" / "train_on_dataset.py").read_text(encoding="utf-8")
    start = source.index(
        'if EXTERNAL_ADAPTER_DIR is not None:\n        if TRAINING_PHASE != "sft_eval"')
    return source[start:source.index("if not use_mock:", start)]


def test_the_external_branch_refuses_training_phases_in_the_runner_too():
    """Defence in depth: argparse refuses, and so does run_training."""
    block = _runner_branch()
    assert 'TRAINING_PHASE != "sft_eval"' in block
    assert "must never seed training" in block


def test_the_external_branch_evaluates_the_external_directory():
    block = _runner_branch()
    assert "_evaluate_adapter_kill_rate(" in block
    assert "EXTERNAL_ADAPTER_DIR," in block
    assert '"external_evaluation_adapter"' in block


def test_the_external_branch_refuses_mock_inference():
    assert "requires real model inference" in _runner_branch()


# ------------------------------------------------------ sealed-test isolation

def test_the_sealed_split_is_not_selectable_at_all():
    """--adapter-dir must not become a new door to the sealed test."""
    result = cli("--phase", "sft_eval", "--run-name", "tmp_external_adapter_check",
                 "--evaluation-split", "test",
                 "--adapter-dir", str(REAL_ADAPTER),
                 "--expected-adapter-sha256", REAL_SHA)
    assert result.returncode != 0
    assert "invalid choice" in result.stderr


def test_the_evaluation_split_choices_exclude_test():
    source = (ROOT / "scripts" / "train_on_dataset.py").read_text(encoding="utf-8")
    line = source[source.index('"--evaluation-split"'):][:160]
    assert '"test"' not in line
    for allowed in ('"train"', '"ablation_dev"', '"val"'):
        assert allowed in line


# ------------------------------------------ existing behaviour is preserved

def test_base_eval_still_needs_no_adapter(monkeypatch):
    monkeypatch.setattr(trainer, "EXTERNAL_ADAPTER_DIR", None)
    context = trainer._adapter_evaluation_context(
        "fingerprint", "base_model", "a" * 64, "val", None, "b" * 64, 700)
    assert context["adapter"] == "base_model"
    assert "external_adapter" not in context


def test_normal_sft_eval_context_is_unchanged_in_shape(monkeypatch):
    monkeypatch.setattr(trainer, "EXTERNAL_ADAPTER_DIR", None)
    context = trainer._adapter_evaluation_context(
        "fingerprint", "sft_adapter", "a" * 64, "ablation_dev", None, "b" * 64, 542)
    for field in ("format_version", "run_contract", "run_contract_sha256",
                  "dataset_fingerprint", "adapter", "adapter_sha256",
                  "evaluation_split", "final_test_measurement", "seed",
                  "tests_per_function", "evaluation_scope_sha256",
                  "function_validation_records", "evaluation_profile",
                  "reproducibility"):
        assert field in context
    assert context["final_test_measurement"] is False


def test_running_without_the_option_leaves_the_globals_unset():
    assert trainer.EXTERNAL_ADAPTER_DIR is None
    assert trainer.EXTERNAL_ADAPTER_SHA256 is None
    assert trainer.EXTERNAL_ADAPTER_SOURCE_RUN is None
