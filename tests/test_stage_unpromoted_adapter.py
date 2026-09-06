"""Staging an unpromoted adapter must never look like a promotion.

The monitor gate decides promotion AND, as a side effect, measurability: a
rejected arm writes no adapter, so it cannot be evaluated at all. Refusing to
promote on candidate health is right; refusing to measure is not. This script
separates the two, and these tests keep the separation honest.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.stage_unpromoted_terminal_adapter as staging


def _arm(tmp_path: Path, name: str, *, gate_passed: bool,
         promoted: bool = False) -> None:
    adapter = tmp_path / "checkpoints" / name
    (adapter / "sft_terminal_adapter").mkdir(parents=True)
    (adapter / "sft_terminal_adapter" / "adapter_config.json").write_text(
        "{}", encoding="utf-8")
    (adapter / "sft_terminal_adapter" / "adapter_model.safetensors").write_text(
        "weights", encoding="utf-8")
    if promoted:
        (adapter / "sft_adapter").mkdir()
        (adapter / "sft_complete.marker").write_text("complete\n", encoding="utf-8")
    results = tmp_path / "results" / name
    results.mkdir(parents=True)
    (results / "training_results.json").write_text(
        json.dumps({"sft_monitor_gate_passed": gate_passed}), encoding="utf-8")


def test_staging_marks_the_adapter_as_unpromoted(tmp_path, monkeypatch):
    monkeypatch.setattr(staging, "ROOT", tmp_path)
    _arm(tmp_path, "arm", gate_passed=False)

    row = staging.stage("arm", "parse health failed")
    assert row["monitor_promoted"] is False

    metadata = json.loads(
        (tmp_path / "checkpoints" / "arm" / "sft_metadata.json").read_text(
            encoding="utf-8")
    )
    assert metadata["monitor_promoted"] is False
    assert metadata["monitor_rejection_reason"] == "parse health failed"
    assert metadata["adapter_source"] == "sft_terminal_adapter"
    assert (tmp_path / "checkpoints" / "arm"
            / "unpromoted_terminal_adapter.marker").exists()


def test_the_marker_says_it_was_staged_not_completed(tmp_path, monkeypatch):
    """A reader of the marker must not conclude the run passed its gate."""
    monkeypatch.setattr(staging, "ROOT", tmp_path)
    _arm(tmp_path, "arm", gate_passed=False)
    staging.stage("arm", "parse health failed")

    marker = (tmp_path / "checkpoints" / "arm" / "sft_complete.marker")
    assert marker.read_text(encoding="utf-8").strip() == (
        "staged_unpromoted_terminal_adapter"
    )


def test_it_refuses_to_touch_a_genuinely_promoted_run(tmp_path, monkeypatch):
    monkeypatch.setattr(staging, "ROOT", tmp_path)
    _arm(tmp_path, "arm", gate_passed=True, promoted=True)
    with pytest.raises(SystemExit, match="already has a promoted adapter"):
        staging.stage("arm", "reason")


def test_it_refuses_a_run_that_passed_the_gate(tmp_path, monkeypatch):
    """An arm that passed should promote normally; staging would hide a bug."""
    monkeypatch.setattr(staging, "ROOT", tmp_path)
    _arm(tmp_path, "arm", gate_passed=True)
    with pytest.raises(SystemExit, match="passed the monitor gate"):
        staging.stage("arm", "reason")


def test_it_refuses_a_run_with_no_terminal_adapter(tmp_path, monkeypatch):
    monkeypatch.setattr(staging, "ROOT", tmp_path)
    adapter = tmp_path / "checkpoints" / "arm"
    adapter.mkdir(parents=True)
    (tmp_path / "results" / "arm").mkdir(parents=True)
    (tmp_path / "results" / "arm" / "training_results.json").write_text(
        json.dumps({"sft_monitor_gate_passed": False}), encoding="utf-8")
    with pytest.raises(SystemExit, match="no sft_terminal_adapter"):
        staging.stage("arm", "reason")


def test_a_directory_without_weights_is_refused(tmp_path, monkeypatch):
    """A checkpoint path that holds no adapter must fail loudly, not stage."""
    monkeypatch.setattr(staging, "ROOT", tmp_path)
    _arm(tmp_path, "arm", gate_passed=False)
    (tmp_path / "checkpoints" / "arm" / "empty").mkdir()
    with pytest.raises(SystemExit, match="not a LoRA adapter directory"):
        staging.stage("arm", "reason", source="empty")


def test_an_intermediate_checkpoint_stages_into_its_own_run(tmp_path, monkeypatch):
    """Staging a checkpoint must not disturb the run it came from.

    That run has a promoted adapter and reported numbers, and the checkpoint is
    being measured precisely to compare against them. Writing into the source
    directory would destroy the comparison.
    """
    monkeypatch.setattr(staging, "ROOT", tmp_path)
    _arm(tmp_path, "arm", gate_passed=True, promoted=True)
    checkpoint = tmp_path / "checkpoints" / "arm" / "sft_tmp" / "checkpoint-100"
    checkpoint.mkdir(parents=True)
    (checkpoint / "adapter_model.safetensors").write_text("w", encoding="utf-8")

    row = staging.stage("arm", "intermediate checkpoint",
                        source="sft_tmp/checkpoint-100",
                        destination_run="arm_ckpt100")
    assert row["staged_into"] == "arm_ckpt100"
    assert (tmp_path / "checkpoints" / "arm_ckpt100" / "sft_adapter").is_dir()
    # the source run is untouched
    assert (tmp_path / "checkpoints" / "arm" / "sft_adapter").is_dir()
    assert (tmp_path / "checkpoints" / "arm" / "sft_complete.marker").read_text(
        encoding="utf-8").strip() == "complete"


def test_staging_into_an_existing_run_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(staging, "ROOT", tmp_path)
    _arm(tmp_path, "arm", gate_passed=False)
    (tmp_path / "checkpoints" / "taken").mkdir()
    with pytest.raises(SystemExit, match="already exists"):
        staging.stage("arm", "reason", destination_run="taken")


def test_the_dataset_manifest_travels_with_a_staged_checkpoint(tmp_path, monkeypatch):
    """Without it every evaluation of the staged run is refused before it runs.

    The trainer compares dataset_manifest.sha256 against the fingerprint it
    computes, to refuse an adapter trained on a different corpus or scope. A
    staged run missing that file looks like it was trained on nothing, and both
    evaluations died in seconds with "different corpus or SFT training scope".
    """
    monkeypatch.setattr(staging, "ROOT", tmp_path)
    _arm(tmp_path, "arm", gate_passed=True, promoted=True)
    (tmp_path / "checkpoints" / "arm" / "dataset_manifest.sha256").write_text(
        "fingerprint", encoding="utf-8")
    checkpoint = tmp_path / "checkpoints" / "arm" / "sft_tmp" / "checkpoint-100"
    checkpoint.mkdir(parents=True)
    (checkpoint / "adapter_model.safetensors").write_text("w", encoding="utf-8")

    staging.stage("arm", "intermediate", source="sft_tmp/checkpoint-100",
                  destination_run="arm_ckpt100")
    staged = tmp_path / "checkpoints" / "arm_ckpt100" / "dataset_manifest.sha256"
    assert staged.exists(), "the staged run cannot be evaluated without it"
    assert staged.read_text(encoding="utf-8") == "fingerprint"
