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
