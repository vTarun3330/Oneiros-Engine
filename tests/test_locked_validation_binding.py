"""A frozen preflight is worth nothing if the run that follows ignores it.

Until this binding existed the link was procedural: a receipt was written, and
then a command was typed which was believed to match it. Nothing checked. These
tests pin the mechanical version - the receipt's path and hash travel in the
command, and the runner refuses to start unless the receipt on disk is that
receipt and the arm being launched is the arm it froze.

The refusals all have to land before CUDA initialises, before a corpus record
is read and before an output directory exists, because a gate that fires after
the GPU has spun up has already let the run begin.
"""
from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from harness.locked_validation_binding import (
    BOUND_SETTINGS, SCHEMA_VERSION, contract_block, load_receipt,
    verification_problems,
)
from harness.source_identity import HASH_SCHEME_VERSION
from scripts import preflight_locked_validation as pf

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable
RECEIPT_PATH = ROOT / "results" / "v4_2_locked_validation_preflight.json"
SHA = "2e1fd397ee46e1388853d2af2c993145b0f1098a"
ADAPTER_SHA = "e67dd599a37cbf2a738791c5cdf889cb4938a2ad53c991dca2fefae503b6f9e7"


def _receipt() -> dict:
    if not RECEIPT_PATH.exists():
        pytest.skip("locked-validation preflight receipt not generated")
    return json.loads(RECEIPT_PATH.read_text(encoding="utf-8"))


def _receipt_sha() -> str:
    return hashlib.sha256(RECEIPT_PATH.read_bytes()).hexdigest()


def _settings(receipt: dict) -> dict:
    return {name: receipt["resolved_generation_settings"][name] for name in BOUND_SETTINGS}


def _verify(receipt: dict, **over):
    kwargs = dict(
        root=ROOT, receipt=receipt, run_name=pf.ARM_A_RUN_NAME, phase="sft_eval",
        evaluation_split="val", model_revision=SHA, adapter_sha256=ADAPTER_SHA,
        protocol_name=receipt["protocol"]["protocol_name"],
        protocol_sha256=receipt["protocol"]["protocol_sha256"],
        resolved_settings=_settings(receipt),
    )
    kwargs.update(over)
    return verification_problems(**kwargs)


# ------------------------------------------------------- the happy path

def test_the_real_receipt_verifies_for_both_arms():
    receipt = _receipt()
    assert _verify(receipt) == []
    assert _verify(receipt, run_name=pf.BASE_RUN_NAME, phase="base_eval",
                   adapter_sha256=None) == []


def test_loading_the_real_receipt_with_its_real_hash_succeeds():
    receipt, problems = load_receipt(RECEIPT_PATH, _receipt_sha())
    assert problems == []
    assert receipt["ready_to_launch"] is True


# --------------------------------------------------- receipt hash rejection

def test_a_changed_receipt_hash_is_rejected():
    _receipt()
    receipt, problems = load_receipt(RECEIPT_PATH, "a" * 64)
    assert receipt == {}
    assert any("hash mismatch" in p for p in problems)


def test_a_tampered_receipt_is_rejected(tmp_path):
    receipt = _receipt()
    receipt["decision_rule"]["criteria"]["1_practical_kill_gain"][
        "kill_at_8_absolute_points"] = ">= +0.1"
    tampered = tmp_path / "tampered.json"
    tampered.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    _loaded, problems = load_receipt(tampered, _receipt_sha())
    assert any("hash mismatch" in p for p in problems)


def test_a_missing_receipt_is_rejected(tmp_path):
    _loaded, problems = load_receipt(tmp_path / "nope.json", "a" * 64)
    assert any("not found" in p for p in problems)


def test_a_receipt_without_an_expected_hash_is_rejected():
    _loaded, problems = load_receipt(RECEIPT_PATH, "")
    assert any("requires --expected-preflight-receipt-sha256" in p for p in problems)


def test_a_receipt_not_marked_ready_is_rejected():
    receipt = copy.deepcopy(_receipt())
    receipt["ready_to_launch"] = False
    assert any("not marked ready_to_launch" in p for p in _verify(receipt))


def test_a_receipt_with_outstanding_problems_is_rejected():
    receipt = copy.deepcopy(_receipt())
    receipt["preflight_problems"] = ["something was wrong"]
    assert any("unresolved problems" in p for p in _verify(receipt))


# ------------------------------------------------------ identity rejections

def test_a_different_adapter_sha_is_rejected():
    found = _verify(_receipt(), adapter_sha256="b" * 64)
    assert any("adapter SHA-256 differs" in p for p in found)


def test_a_missing_adapter_on_the_adapter_arm_is_rejected():
    found = _verify(_receipt(), adapter_sha256=None)
    assert any("adapter SHA-256 differs" in p for p in found)


def test_an_adapter_smuggled_onto_the_base_arm_is_rejected():
    found = _verify(_receipt(), run_name=pf.BASE_RUN_NAME, phase="base_eval",
                    adapter_sha256=ADAPTER_SHA)
    assert any("adapter SHA-256 differs" in p for p in found)


def test_a_different_model_revision_is_rejected():
    found = _verify(_receipt(), model_revision="c" * 40)
    assert any("base model revision differs" in p for p in found)


def test_a_different_split_is_rejected():
    found = _verify(_receipt(), evaluation_split="ablation_dev")
    assert any("evaluation split differs" in p for p in found)


def test_the_sealed_split_is_rejected():
    found = _verify(_receipt(), evaluation_split="test")
    assert any("must not be a sealed-test measurement" in p for p in found)


def test_a_different_protocol_name_is_rejected():
    found = _verify(_receipt(), protocol_name="some_other_protocol")
    assert any("protocol name differs" in p for p in found)


def test_a_different_protocol_sha_is_rejected():
    found = _verify(_receipt(), protocol_sha256="d" * 64)
    assert any("protocol sha256 differs" in p for p in found)


def test_the_wrong_phase_for_an_arm_is_rejected():
    found = _verify(_receipt(), phase="base_eval")
    assert any("frozen as --phase sft_eval" in p for p in found)


def test_an_unknown_run_name_is_rejected():
    found = _verify(_receipt(), run_name="some_other_run")
    assert any("is not an arm in the frozen preflight receipt" in p for p in found)


@pytest.mark.parametrize("setting", BOUND_SETTINGS)
def test_each_bound_generation_setting_is_checked(setting):
    receipt = _receipt()
    settings = _settings(receipt)
    settings[setting] = "tampered"
    found = _verify(receipt, resolved_settings=settings)
    assert any(f"setting {setting} differs" in p for p in found)


# ------------------------------------------------- source identity rejections

def test_a_changed_source_blob_id_is_rejected():
    receipt = copy.deepcopy(_receipt())
    receipt["source_identity"]["sources"]["evaluator"]["git_blob_sha1"] = "0" * 40
    found = _verify(receipt)
    assert any("Git blob identity differs" in p for p in found)


def test_a_changed_canonical_source_hash_is_rejected():
    receipt = copy.deepcopy(_receipt())
    receipt["source_identity"]["sources"]["generator"]["canonical_sha256"] = "0" * 64
    found = _verify(receipt)
    assert any("canonical source hash differs" in p for p in found)


def test_a_changed_entrypoint_is_rejected():
    """The runner itself is an evaluation-defining source."""
    receipt = copy.deepcopy(_receipt())
    receipt["source_identity"]["sources"]["evaluation_entrypoint"]["git_blob_sha1"] = "1" * 40
    assert any("train_on_dataset.py Git blob identity differs" in p
               for p in _verify(receipt))


def test_a_missing_source_identity_is_rejected():
    receipt = copy.deepcopy(_receipt())
    del receipt["source_identity"]["sources"]["candidate_policy"]
    assert any("no identity for candidate_policy" in p for p in _verify(receipt))


def test_a_different_hash_scheme_is_rejected():
    receipt = copy.deepcopy(_receipt())
    receipt["source_identity"]["hash_scheme_version"] = "some_other_scheme_v9"
    assert any("source-identity scheme differs" in p for p in _verify(receipt))


def test_a_view_including_the_sealed_split_is_rejected():
    receipt = copy.deepcopy(_receipt())
    receipt["corpus"]["included_splits"] = ["train", "val", "test"]
    assert any("includes the sealed split" in p for p in _verify(receipt))


def test_a_receipt_claiming_sealed_access_is_rejected():
    receipt = copy.deepcopy(_receipt())
    receipt["sealed_final_test"]["accessed"] = True
    assert any("does not record the sealed test as unopened" in p for p in _verify(receipt))


# ---------------------------------------------------------- contract block

def test_the_contract_block_names_both_commits_without_requiring_equality():
    receipt = _receipt()
    block = contract_block(
        root=ROOT, receipt_path="results/v4_2_locked_validation_preflight.json",
        receipt_sha256=_receipt_sha(), receipt=receipt,
        git_head_at_launch="f" * 40, adapter_sha256=ADAPTER_SHA, model_revision=SHA)
    assert block["git_head_at_launch"] == "f" * 40
    assert block["git_commit_recorded_in_preflight_receipt"] == \
        receipt["source_identity"]["git_commit"]
    assert block["git_head_at_launch"] != block["git_commit_recorded_in_preflight_receipt"]
    assert "NOT required to be equal" in block["why_the_two_commits_differ"]
    assert "blob identity" in block["why_the_two_commits_differ"]


def test_the_contract_block_carries_every_required_field():
    block = contract_block(
        root=ROOT, receipt_path="p.json", receipt_sha256="a" * 64, receipt=_receipt(),
        git_head_at_launch="b" * 40, adapter_sha256=ADAPTER_SHA, model_revision=SHA)
    for field in ("schema_version", "frozen_preflight_receipt_path",
                  "frozen_preflight_receipt_sha256", "git_head_at_launch",
                  "git_commit_recorded_in_preflight_receipt", "adapter_sha256",
                  "base_model_revision", "source_identity_scheme_version"):
        assert field in block, field
    assert block["schema_version"] == SCHEMA_VERSION
    assert block["source_identity_scheme_version"] == HASH_SCHEME_VERSION
    assert block["verified_before_cuda_corpus_or_output_directories"] is True


def test_the_run_contract_records_head_and_receipt_hash(monkeypatch):
    from scripts import train_on_dataset as trainer
    receipt = _receipt()
    monkeypatch.setattr(trainer, "FROZEN_PREFLIGHT_RECEIPT_PATH",
                        "results/v4_2_locked_validation_preflight.json")
    monkeypatch.setattr(trainer, "FROZEN_PREFLIGHT_RECEIPT_SHA256", _receipt_sha())
    monkeypatch.setattr(trainer, "FROZEN_PREFLIGHT_RECEIPT", receipt)
    contract = trainer._adapter_evaluation_context(
        "fingerprint", "external_evaluation_adapter", ADAPTER_SHA, "val", None,
        "b" * 64, 700)["run_contract"]
    binding = contract["locked_validation_binding"]
    assert len(binding["git_head_at_launch"]) == 40
    assert binding["frozen_preflight_receipt_sha256"] == _receipt_sha()
    assert binding["adapter_sha256"] == ADAPTER_SHA
    assert binding["base_model_revision"] == SHA
    # the per-source hashes live in one place only
    assert contract["source_identity"]["hash_scheme_version"] == HASH_SCHEME_VERSION


def test_an_ordinary_run_contract_has_no_binding_block(monkeypatch):
    from scripts import train_on_dataset as trainer
    monkeypatch.setattr(trainer, "FROZEN_PREFLIGHT_RECEIPT_PATH", None)
    contract = trainer._adapter_evaluation_context(
        "fingerprint", "base_model", "a" * 64, "ablation_dev", None, "b" * 64, 542)["run_contract"]
    assert "locked_validation_binding" not in contract


# ------------------------------------------------------------- the real CLI

def _dry_run(command: list[str]) -> subprocess.CompletedProcess:
    cmd = list(command)
    cmd[0] = PY
    cmd.append("--dry-run")
    return subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)


def test_the_exact_generated_commands_pass_under_dry_run():
    receipt = _receipt()
    if receipt.get("ready_to_launch") is not True:
        pytest.skip("preflight receipt is not ready_to_launch")
    for name, command in pf.resolved_commands(receipt, _receipt_sha()).items():
        result = _dry_run(command)
        assert result.returncode == 0, f"{name}: {result.stderr[-1500:]}"
        assert "LOCKED VALIDATION] bound to frozen preflight receipt" in result.stdout, name
        payload = json.loads(result.stdout[result.stdout.index("{"):result.stdout.rindex("}") + 1])
        assert payload["training_launched"] is False, name


def test_the_dry_run_creates_no_output_directories():
    receipt = _receipt()
    if receipt.get("ready_to_launch") is not True:
        pytest.skip("preflight receipt is not ready_to_launch")
    for command in pf.resolved_commands(receipt, _receipt_sha()).values():
        _dry_run(command)
    for run_name in (pf.BASE_RUN_NAME, pf.ARM_A_RUN_NAME):
        assert not (ROOT / "results" / run_name).exists(), run_name
        assert not (ROOT / "checkpoints" / run_name).exists(), run_name


def test_the_dry_run_reads_no_validation_payload():
    receipt = _receipt()
    if receipt.get("ready_to_launch") is not True:
        pytest.skip("preflight receipt is not ready_to_launch")
    for name, command in pf.resolved_commands(receipt, _receipt_sha()).items():
        out = _dry_run(command).stdout
        assert "Canonical corpus verified" not in out, name
        assert "development corpus view verified" not in out, name
        assert "Local GPU" not in out, name


def test_a_wrong_receipt_hash_is_refused_by_the_cli_before_anything_loads():
    receipt = _receipt()
    if receipt.get("ready_to_launch") is not True:
        pytest.skip("preflight receipt is not ready_to_launch")
    command = pf.resolved_commands(receipt, _receipt_sha())["B_arm_a_checkpoint_431"]
    command = [("e" * 64) if part == _receipt_sha() else part for part in command]
    result = _dry_run(command)
    assert result.returncode != 0
    assert "hash mismatch" in result.stderr
    assert "Local GPU" not in result.stdout
    assert "Canonical corpus verified" not in result.stdout
    assert not (ROOT / "results" / pf.ARM_A_RUN_NAME).exists()


def test_the_receipt_sha_flag_requires_the_receipt_flag():
    result = subprocess.run(
        [PY, "scripts/train_on_dataset.py", "--phase", "base_eval",
         "--run-name", "tmp_binding_check",
         "--expected-preflight-receipt-sha256", "a" * 64],
        capture_output=True, text=True, cwd=ROOT)
    assert result.returncode != 0
    assert "requires --frozen-preflight-receipt" in result.stderr


def test_the_placeholder_is_documented_in_the_receipt():
    receipt = _receipt()
    ref = receipt["receipt_self_reference"]
    assert ref["placeholder"] == pf.RECEIPT_SHA_PLACEHOLDER
    assert "cannot contain its own SHA-256" in ref["note"]
    for arm in receipt["arms"].values():
        assert pf.RECEIPT_SHA_PLACEHOLDER in arm["command"]


def test_resolved_commands_substitutes_only_the_placeholder():
    receipt = _receipt()
    resolved = pf.resolved_commands(receipt, "9" * 64)
    for name, command in resolved.items():
        assert pf.RECEIPT_SHA_PLACEHOLDER not in command, name
        assert command[command.index("--expected-preflight-receipt-sha256") + 1] == "9" * 64
        stored = receipt["arms"][name]["command"]
        assert len(command) == len(stored)
