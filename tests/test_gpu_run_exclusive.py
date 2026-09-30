"""gpu_run --exclusive-key: base and SFT generation can never overlap (amendment v2.4 I.1)."""
from __future__ import annotations

import json
import os

import pytest

from scripts import gpu_run

KEY = "native_v24_generation"
COMMAND = ["--", "python", "scripts/native_generated_tests_generate.py", "run", "--arm", "base"]


class _Popen:
    pid = 424242

    def __init__(self, *a, **k):
        pass


@pytest.fixture
def runs(tmp_path, monkeypatch):
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    monkeypatch.setattr(gpu_run, "RUNS_DIR", runs_dir)
    monkeypatch.setattr(gpu_run.subprocess, "Popen", _Popen)     # never start a supervisor
    monkeypatch.setattr(gpu_run, "build_manifest", lambda run_id, name, command: {
        "run_id": run_id, "name": name, "command": command,
        "config_snapshot": {"run_name": gpu_run._extract_arg(command, "--run-name")}})
    return runs_dir


def _existing(runs_dir, name, key, status):
    d = runs_dir / name
    d.mkdir()
    (d / "manifest.json").write_text(json.dumps({"exclusive_key": key,
                                                 "config_snapshot": {"run_name": None}}))
    if status is not None:
        (d / "status.json").write_text(json.dumps(status))


def _start(name="sft", key=KEY, allow=False):
    args = type("A", (), {"name": name, "allow_concurrent": allow, "resumed_from": None,
                          "lineage": None, "exclusive_key": key, "command": list(COMMAND)})()
    return gpu_run.cmd_start(args)


def test_a_live_run_with_the_same_key_blocks(runs, monkeypatch):
    _existing(runs, "base", KEY, {"state": "running", "supervisor_pid": 11, "child_pid": 12})
    monkeypatch.setattr(gpu_run, "_pid_alive", lambda pid: pid == 12)
    assert _start() == 4
    assert [p.name for p in runs.iterdir() if not p.name.startswith(".")] == ["base"]


def test_allow_concurrent_does_not_override_the_exclusive_key(runs, monkeypatch):
    _existing(runs, "base", KEY, {"state": "running", "supervisor_pid": 11, "child_pid": 12})
    monkeypatch.setattr(gpu_run, "_pid_alive", lambda pid: True)
    assert _start(allow=True) == 4


def test_unknown_state_blocks_fail_closed(runs):
    _existing(runs, "base", KEY, None)                        # launched, no status yet
    assert _start() == 4


@pytest.mark.parametrize("state", ["completed", "failed", "stopped"])
def test_finished_runs_do_not_block(runs, state):
    _existing(runs, "base", KEY, {"state": state, "supervisor_pid": 11, "child_pid": 12})
    assert _start() == 0
    new = [p for p in runs.iterdir() if p.name != "base" and not p.name.startswith(".")]
    assert json.loads((new[0] / "manifest.json").read_text())["exclusive_key"] == KEY


def test_stale_running_state_with_dead_processes_does_not_block(runs, monkeypatch):
    _existing(runs, "base", KEY, {"state": "running", "supervisor_pid": 11, "child_pid": 12})
    monkeypatch.setattr(gpu_run, "_pid_alive", lambda pid: False)
    assert _start() == 0


def test_an_unrelated_key_or_no_key_does_not_block(runs, monkeypatch):
    _existing(runs, "other", "some_other_key", {"state": "running", "child_pid": 12})
    _existing(runs, "plain", None, {"state": "running", "child_pid": 13})
    monkeypatch.setattr(gpu_run, "_pid_alive", lambda pid: True)
    assert _start() == 0


def test_the_key_never_invokes_the_training_artifact_validator(runs, monkeypatch):
    called = []
    monkeypatch.setattr(gpu_run, "validate_artifacts", lambda *a, **k: called.append(a))
    assert _start() == 0
    manifest = json.loads(next(p for p in runs.iterdir() if not p.name.startswith("."))
                          .joinpath("manifest.json").read_text())
    assert manifest["config_snapshot"]["run_name"] is None and not called
    assert gpu_run.validate_artifacts.__name__ == "<lambda>"


def test_pid_alive_is_non_destructive():
    assert gpu_run._pid_alive(os.getpid()) is True
    assert gpu_run._pid_alive(None) is False and gpu_run._pid_alive(-3) is False
