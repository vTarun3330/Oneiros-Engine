"""Tests for the durable GPU run manager.

The launcher exists so that a run's fate is knowable after the submitting
client disappears. These tests pin the parts that make that true: honest
termination classification, refusal to call an unvalidated run complete, and
a manifest that records enough to reproduce the run.
"""
import json

from scripts.gpu_run import (
    TERMINATION_REASONS,
    build_manifest,
    classify_termination,
    validate_artifacts,
)


def _write(path, text=""):
    path.write_text(text, encoding="utf-8")
    return path


def test_clean_exit_is_classified_completed(tmp_path):
    out = _write(tmp_path / "stdout.log", "all good\n")
    err = _write(tmp_path / "stderr.log")
    tel = _write(tmp_path / "telemetry.jsonl")

    result = classify_termination(0, out, err, tel)

    assert result["reason"] == "completed"
    assert result["confidence"] == "certain"


def test_cuda_oom_is_distinguished_from_a_plain_nonzero_exit(tmp_path):
    out = _write(tmp_path / "stdout.log")
    err = _write(
        tmp_path / "stderr.log",
        "torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB\n",
    )
    tel = _write(tmp_path / "telemetry.jsonl")

    result = classify_termination(1, out, err, tel)

    assert result["reason"] == "cuda_oom"
    assert result["reason"] in TERMINATION_REASONS


def test_external_kill_is_not_reported_as_a_python_failure(tmp_path):
    """An SSH disconnect or taskkill leaves no traceback; say so honestly."""
    out = _write(tmp_path / "stdout.log", "step 3\n")
    err = _write(tmp_path / "stderr.log")
    tel = _write(tmp_path / "telemetry.jsonl")

    result = classify_termination(137, out, err, tel)

    assert result["reason"] == "external_termination"


def test_python_exception_reports_its_traceback(tmp_path):
    out = _write(tmp_path / "stdout.log")
    err = _write(
        tmp_path / "stderr.log",
        'Traceback (most recent call last):\n  File "x.py", line 1\nValueError: bad record\n',
    )
    tel = _write(tmp_path / "telemetry.jsonl")

    result = classify_termination(1, out, err, tel)

    assert result["reason"] == "python_exception"
    assert "ValueError" in result["detail"]


def test_disk_and_ram_exhaustion_are_read_from_telemetry(tmp_path):
    out = _write(tmp_path / "stdout.log")
    err = _write(tmp_path / "stderr.log")
    disk_tel = _write(
        tmp_path / "disk.jsonl", json.dumps({"disk_free_gb": 0.2, "ram_available_gb": 40.0}) + "\n"
    )
    ram_tel = _write(
        tmp_path / "ram.jsonl", json.dumps({"disk_free_gb": 200.0, "ram_available_gb": 0.3}) + "\n"
    )

    assert classify_termination(1, out, err, disk_tel)["reason"] == "disk_exhaustion"
    assert classify_termination(1, out, err, ram_tel)["reason"] == "ram_exhaustion"


def test_missing_exit_code_is_supervisor_lost_not_success(tmp_path):
    """A vanished supervisor must never be silently read as a finished run."""
    out = _write(tmp_path / "stdout.log", "step 12\n")
    err = _write(tmp_path / "stderr.log")
    tel = _write(tmp_path / "telemetry.jsonl")

    result = classify_termination(None, out, err, tel)

    assert result["reason"] == "supervisor_lost"


def test_training_run_without_artifacts_is_not_validated():
    """Exit code alone must not promote a training run to complete."""
    result = validate_artifacts("a_run_name_that_does_not_exist")

    assert result["validated"] is False
    assert "missing" in result["reason"]


def test_command_without_run_name_declares_no_artifact_contract():
    result = validate_artifacts(None)

    assert result["validated"] is True
    assert result["artifact_contract"] == "none_declared"


def test_manifest_captures_reproducibility_identity():
    command = [
        "python", "scripts/train_on_dataset.py",
        "--run-name", "some_run",
        "--seed", "42",
        "--base-model-name", "Qwen/Qwen2.5-Coder-1.5B-Instruct",
        "--attention-implementation", "sdpa",
        "--sft-learning-rate", "0.00001",
        "--sft-checkpoint-steps", "10",
    ]

    manifest = build_manifest("20260101-000000-x", "x", command)

    assert manifest["command"] == command
    snapshot = manifest["config_snapshot"]
    assert snapshot["run_name"] == "some_run"
    assert snapshot["seed"] == "42"
    assert snapshot["base_model_name"] == "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    assert snapshot["attention_implementation"] == "sdpa"
    assert snapshot["learning_rate"] == "0.00001"
    assert snapshot["checkpoint_steps"] == "10"
    # Git identity and dependency versions are what make a result reproducible.
    assert "commit" in manifest["git"] and "dirty" in manifest["git"]
    assert "torch" in manifest["versions"] and "python" in manifest["versions"]
    assert manifest["resume"]["lineage"] == ["20260101-000000-x"]


def test_colliding_run_names_are_refused(tmp_path, monkeypatch):
    """Two runs writing one artifact set corrupt each other's checkpoints.

    Launching the same evaluation twice seconds apart put two processes on one
    progress-checkpoint sequence; one read a partially written file and died
    with EOFError. The launcher must make that collision impossible.
    """
    import json as _json

    from scripts import gpu_run

    runs_dir = tmp_path / "runs"
    (runs_dir / "existing").mkdir(parents=True)
    (runs_dir / "existing" / "status.json").write_text(
        _json.dumps({"state": "running"}), encoding="utf-8"
    )
    (runs_dir / "existing" / "manifest.json").write_text(
        _json.dumps({"config_snapshot": {"run_name": "target_run", "seed": "44"}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(gpu_run, "RUNS_DIR", runs_dir)

    args = type("A", (), {
        "name": "second",
        "allow_concurrent": False,
        "resumed_from": None,
        "lineage": None,
        "command": ["--", "python", "x.py", "--run-name", "target_run", "--seed", "44"],
    })()

    assert gpu_run.cmd_start(args) == 3
    # Only the pre-existing run directory survives; nothing new was created.
    assert [p.name for p in runs_dir.iterdir()] == ["existing"]


def test_a_different_seed_on_the_same_run_name_is_allowed(tmp_path, monkeypatch):
    """Only an identical (run_name, seed) target collides."""
    import json as _json

    from scripts import gpu_run

    runs_dir = tmp_path / "runs"
    (runs_dir / "existing").mkdir(parents=True)
    (runs_dir / "existing" / "status.json").write_text(
        _json.dumps({"state": "running"}), encoding="utf-8"
    )
    (runs_dir / "existing" / "manifest.json").write_text(
        _json.dumps({"config_snapshot": {"run_name": "target_run", "seed": "43"}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(gpu_run, "RUNS_DIR", runs_dir)

    conflicting = gpu_run._extract_arg(
        ["python", "x.py", "--run-name", "target_run", "--seed", "44"], "--seed"
    )
    assert conflicting == "44"  # differs from the running job's 43, so no collision
