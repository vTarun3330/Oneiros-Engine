"""Atomic exclusive-key reservation for gpu_run (amendment v2.4 I.1).

The reservation is acquired atomically BEFORE conflict inspection or run-directory creation,
is owned by a unique token, is renewed by the supervisor and released only by its owner after
the final status is written. Genuinely concurrent starters race here; exactly one may win.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

import pytest

from scripts import gpu_run

REPO = Path(__file__).resolve().parent.parent
KEY = "native_v24_generation"

RACER = r'''
import json, sys, time, uuid
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from scripts import gpu_run
runs, go, mode, name = Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4], sys.argv[5]
class _Popen:
    pid = 0
    def __init__(self, *a, **k):
        pass
gpu_run.subprocess.Popen = _Popen
gpu_run.RUNS_DIR = runs
while not go.exists():
    pass                                   # spin at the barrier, then start together
if mode == "acquire":
    held = gpu_run.acquire_exclusive(sys.argv[6], runs, name, uuid.uuid4().hex)
    print(json.dumps({"won": held is not None}))
    sys.stdout.flush()
    time.sleep(3)
else:
    args = type("A", (), {"name": name, "allow_concurrent": True, "resumed_from": None,
                          "lineage": None, "exclusive_key": sys.argv[6],
                          "command": ["--", "python", "-c", "pass"]})()
    print(json.dumps({"won": gpu_run.cmd_start(args) == 0}))
sys.stdout.flush()
time.sleep(3)                              # the winner stays alive while losers decide
'''


def _race(tmp_path, mode, racers, rounds):
    script = tmp_path / "racer.py"
    script.write_text(RACER)
    winners = []
    for r in range(rounds):
        runs = tmp_path / f"runs_{mode}_{r}"
        runs.mkdir()
        go = tmp_path / f"go_{mode}_{r}"
        procs = [subprocess.Popen([sys.executable, str(script), str(REPO), str(runs), str(go),
                                   mode, f"racer{i}", KEY], stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE, text=True) for i in range(racers)]
        time.sleep(1.5)                                  # all racers spinning at the barrier
        go.write_text("go")
        results = [json.loads(p.communicate(timeout=120)[0].strip().splitlines()[-1])["won"]
                   for p in procs]
        winners.append(sum(results))
    return winners


def test_concurrent_acquisition_has_exactly_one_winner(tmp_path):
    assert _race(tmp_path, "acquire", racers=6, rounds=8) == [1] * 8


def test_concurrent_cmd_start_has_exactly_one_winner_even_with_allow_concurrent(tmp_path):
    assert _race(tmp_path, "start", racers=4, rounds=4) == [1] * 4


class _Popen:
    pid = 31337

    def __init__(self, *a, **k):
        pass


@pytest.fixture
def runs(tmp_path, monkeypatch):
    d = tmp_path / "runs"
    d.mkdir()
    monkeypatch.setattr(gpu_run, "RUNS_DIR", d)
    monkeypatch.setattr(gpu_run.subprocess, "Popen", _Popen)
    return d


def _start(key=KEY, allow=False, name="sft"):
    args = type("A", (), {"name": name, "allow_concurrent": allow, "resumed_from": None,
                          "lineage": None, "exclusive_key": key,
                          "command": ["--", "python", "-c", "pass"]})()
    return gpu_run.cmd_start(args)


def _owner(runs, record):
    lock = gpu_run.exclusive_lock_path(KEY, runs)
    lock.mkdir(parents=True)
    (lock / "owner.json").write_text(record if isinstance(record, str) else json.dumps(record))
    return lock


def test_lock_path_is_a_safe_hash_of_the_key(runs):
    path = gpu_run.exclusive_lock_path("../../evil key", runs)
    assert path.parent.parent == runs and len(path.name) == 64 and ".." not in path.name


def test_reservation_without_a_manifest_blocks_while_its_starter_lives(runs):
    _owner(runs, {"token": "t", "key": KEY, "run_id": "not-yet-created",
                  "starter_pid": os.getpid()})
    assert _start() == 4


def test_unreadable_reservation_fails_closed(runs):
    _owner(runs, "{not json")
    assert _start() == 4
    lock = gpu_run.exclusive_lock_path(KEY, runs)
    (lock / "owner.json").unlink()                    # reservation dir without an owner file
    assert _start() == 4


def _run(runs, run_id, status):
    d = runs / run_id
    d.mkdir()
    (d / "manifest.json").write_text(json.dumps({"exclusive_key": KEY}))
    if status is not None:
        (d / "status.json").write_text(json.dumps(status))


def test_live_owner_blocks(runs, monkeypatch):
    _run(runs, "base", {"state": "running", "supervisor_pid": 11, "child_pid": 12})
    _owner(runs, {"token": "t", "key": KEY, "run_id": "base", "starter_pid": 10,
                  "supervisor_pid": 11})
    monkeypatch.setattr(gpu_run, "_pid_alive", lambda pid: pid == 12)
    assert _start() == 4


@pytest.mark.parametrize("state", ["completed", "running"])
def test_dead_owner_is_recovered_explicitly(runs, monkeypatch, state):
    _run(runs, "base", {"state": state, "supervisor_pid": 11, "child_pid": 12})
    _owner(runs, {"token": "t", "key": KEY, "run_id": "base", "starter_pid": 10,
                  "supervisor_pid": 11})
    monkeypatch.setattr(gpu_run, "_pid_alive", lambda pid: False)
    assert _start() == 0
    recovered = list((runs / ".exclusive" / "recovered").iterdir())
    assert len(recovered) == 1
    assert json.loads((recovered[0] / "owner.json").read_text())["run_id"] == "base"
    owner = json.loads((gpu_run.exclusive_lock_path(KEY, runs) / "owner.json").read_text())
    assert owner["run_id"] != "base"


def test_a_different_key_is_independent(runs, monkeypatch):
    _owner(runs, {"token": "t", "key": KEY, "run_id": "x", "starter_pid": os.getpid()})
    assert _start(key="another_key") == 0


def test_allow_concurrent_never_bypasses_the_reservation(runs):
    _owner(runs, {"token": "t", "key": KEY, "run_id": "x", "starter_pid": os.getpid()})
    assert _start(allow=True) == 4


def test_only_the_owner_token_releases(runs):
    held = gpu_run.acquire_exclusive(KEY, runs, "run-a", "token-a")
    assert held is not None
    assert gpu_run.release_exclusive(KEY, runs, "token-b") is False
    assert gpu_run.exclusive_lock_path(KEY, runs).exists()
    assert gpu_run.release_exclusive(KEY, runs, "token-a") is True
    assert not gpu_run.exclusive_lock_path(KEY, runs).exists()
    assert gpu_run.acquire_exclusive(KEY, runs, "run-b", "token-c") is not None


def test_started_run_records_its_reservation_token(runs):
    assert _start() == 0
    run_dir = next(p for p in runs.iterdir() if not p.name.startswith("."))
    manifest = json.loads((run_dir / "manifest.json").read_text())
    owner = json.loads((gpu_run.exclusive_lock_path(KEY, runs) / "owner.json").read_text())
    assert manifest["exclusive_key"] == KEY and manifest["exclusive_token"] == owner["token"]
    assert owner["run_id"] == run_dir.name and owner["supervisor_pid"] == _Popen.pid


def test_supervisor_releases_only_after_the_final_status(tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    runs.mkdir()
    monkeypatch.setattr(gpu_run, "RUNS_DIR", runs)
    monkeypatch.setattr(gpu_run, "_gpu_sample", lambda: {})
    token = uuid.uuid4().hex
    run_dir = runs / "base"
    run_dir.mkdir()
    assert gpu_run.acquire_exclusive(KEY, runs, "base", token) is not None
    (run_dir / "manifest.json").write_text(json.dumps({
        "run_id": "base", "name": "base", "created_utc": "now",
        "command": [sys.executable, "-c", "print('done')"], "config_snapshot": {"run_name": None},
        "exclusive_key": KEY, "exclusive_token": token}))
    order = []
    real_release = gpu_run.release_exclusive

    def release(key, runs_dir, tok):
        order.append(json.loads((run_dir / "status.json").read_text())["state"])
        return real_release(key, runs_dir, tok)
    monkeypatch.setattr(gpu_run, "release_exclusive", release)
    assert gpu_run.supervise(run_dir) == 0
    assert order == ["completed"]                     # final status written before release
    assert not gpu_run.exclusive_lock_path(KEY, runs).exists()
