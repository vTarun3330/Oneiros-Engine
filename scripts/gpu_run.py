"""Durable GPU job launcher, telemetry recorder, and status classifier.

A run launched here is owned entirely by the GPU host. The client that
submitted it may disconnect, close its terminal, or shut down without
affecting the job.

Design
------
``start`` spawns a detached *supervisor* process, which then spawns the actual
training child. The supervisor - not the client - records telemetry, waits for
the child, captures the exit code, classifies the termination reason, and
writes the completion marker. This is what makes the run observable after a
disconnect: nothing about its bookkeeping depends on the submitting session
still existing.

A run directory ``runs/<run_id>/`` holds:

    manifest.json    command, git identity, config, versions, seeds, dataset
    stdout.log       child stdout (never a reused path)
    stderr.log       child stderr
    telemetry.jsonl  periodic GPU/VRAM/RAM/disk samples
    heartbeat.json   supervisor liveness, refreshed each sample
    status.json      final status, exit code, classified termination reason
    .complete        written ONLY after a zero exit and artifact validation

Never infer success from "the SSH command returned" or "a log file exists".
Completion requires exit code 0, the expected artifacts, successful
validation, and the marker.

Usage
-----
    python scripts/gpu_run.py start --name my_run -- python scripts/train_on_dataset.py ...
    python scripts/gpu_run.py status [--run-id ID]
    python scripts/gpu_run.py list
    python scripts/gpu_run.py logs --run-id ID [--follow]
    python scripts/gpu_run.py stop --run-id ID
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = ROOT / "runs"
TELEMETRY_INTERVAL_SECONDS = 45
CHILD_POLL_SECONDS = 2
SCHEMA_VERSION = "oneiros_gpu_run_v1"

TERMINATION_REASONS = (
    "completed",
    "cuda_oom",
    "ram_exhaustion",
    "disk_exhaustion",
    "python_exception",
    "external_termination",
    "supervisor_lost",
    "nonzero_exit",
    "unknown",
)


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _run(cmd: list[str], timeout: int = 20) -> str:
    try:
        out = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, cwd=ROOT
        )
        return out.stdout.strip()
    except Exception:
        return ""


# --------------------------------------------------------------------------
# manifest
# --------------------------------------------------------------------------

def _git_identity() -> dict:
    status = _run(["git", "status", "--porcelain"])
    return {
        "branch": _run(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "commit": _run(["git", "rev-parse", "HEAD"]),
        "dirty": bool(status),
        "dirty_files": [line[3:] for line in status.splitlines()][:50],
    }


def _dependency_versions() -> dict:
    versions = {"python": sys.version.split()[0]}
    for package in (
        "torch", "transformers", "peft", "trl", "datasets",
        "bitsandbytes", "accelerate",
    ):
        try:
            import importlib.metadata as md
            versions[package] = md.version(package)
        except Exception:
            versions[package] = "not-installed"
    try:
        import torch
        versions["cuda_runtime"] = getattr(torch.version, "cuda", None)
        versions["cudnn"] = str(getattr(torch.backends.cudnn, "version", lambda: None)())
        versions["gpu_name"] = (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        )
    except Exception:
        pass
    versions["driver"] = _run([
        "nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"
    ])
    return versions


def _dataset_identity(corpus_version: str | None) -> dict:
    """Record corpus identity by hash, without copying record bodies."""
    if not corpus_version:
        return {"corpus_version": None}
    corpus_dir = ROOT / "data" / "corpus" / corpus_version
    identity: dict = {"corpus_version": corpus_version, "corpus_dir": str(corpus_dir)}
    try:
        manifest = json.loads((corpus_dir / "manifest.json").read_text(encoding="utf-8"))
        identity["corpus_id"] = manifest.get("corpus_id")
        identity["record_count"] = manifest.get("record_count")
        identity["files"] = {
            name: meta.get("sha256")
            for name, meta in (manifest.get("files") or {}).items()
        }
        identity["splits"] = {
            split: {
                "record_count": meta.get("record_count"),
                "record_ids_sha256": meta.get("record_ids_sha256"),
            }
            for split, meta in (manifest.get("splits") or {}).items()
        }
    except Exception as exc:
        identity["error"] = f"corpus manifest unreadable: {exc}"
    return identity


def _extract_arg(command: list[str], flag: str) -> str | None:
    if flag in command:
        index = command.index(flag)
        if index + 1 < len(command):
            return command[index + 1]
    return None


def build_manifest(run_id: str, name: str, command: list[str]) -> dict:
    corpus_version = _extract_arg(command, "--corpus-version")
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "name": name,
        "created_utc": _utc(),
        "command": command,
        "command_string": subprocess.list2cmdline(command),
        "cwd": str(ROOT),
        "git": _git_identity(),
        "versions": _dependency_versions(),
        "dataset": _dataset_identity(corpus_version),
        "config_snapshot": {
            "seed": _extract_arg(command, "--seed"),
            "base_model_name": _extract_arg(command, "--base-model-name"),
            "base_model_revision": _extract_arg(command, "--base-model-revision"),
            "attention_implementation": _extract_arg(command, "--attention-implementation"),
            "selection_tokenizer": _extract_arg(command, "--sft-selection-tokenizer-name"),
            "prompt_token_limit": _extract_arg(command, "--sft-prompt-token-limit"),
            "selection_prompt_token_limit": _extract_arg(command, "--sft-selection-prompt-token-limit"),
            "learning_rate": _extract_arg(command, "--sft-learning-rate"),
            "max_pairs": _extract_arg(command, "--max-pairs"),
            "checkpoint_steps": _extract_arg(command, "--sft-checkpoint-steps"),
            "monitor_validation_functions": _extract_arg(command, "--sft-monitor-validation-functions"),
            "complex_target_fraction": _extract_arg(command, "--sft-complex-target-fraction"),
            "evaluation_split": _extract_arg(command, "--evaluation-split"),
            "run_name": _extract_arg(command, "--run-name"),
        },
        "resume": {"resumed": False, "resumed_from_checkpoint": None, "lineage": [run_id]},
    }


# --------------------------------------------------------------------------
# telemetry
# --------------------------------------------------------------------------

def _gpu_sample() -> dict:
    raw = _run([
        "nvidia-smi",
        "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
        "--format=csv,noheader,nounits",
    ])
    if not raw:
        return {}
    parts = [p.strip() for p in raw.split(",")]
    keys = ["gpu_util_pct", "vram_used_mib", "vram_total_mib", "temp_c", "power_w"]
    sample: dict = {}
    for key, value in zip(keys, parts):
        try:
            sample[key] = float(value)
        except ValueError:
            sample[key] = None
    return sample


def _host_sample() -> dict:
    sample: dict = {}
    try:
        import psutil  # optional
        sample["ram_used_gb"] = round(psutil.virtual_memory().used / 1e9, 2)
        sample["ram_available_gb"] = round(psutil.virtual_memory().available / 1e9, 2)
    except Exception:
        pass
    try:
        usage = shutil.disk_usage(str(ROOT))
        sample["disk_free_gb"] = round(usage.free / 1e9, 2)
        sample["disk_total_gb"] = round(usage.total / 1e9, 2)
    except Exception:
        pass
    return sample


# --------------------------------------------------------------------------
# progress + classification
# --------------------------------------------------------------------------

_STEP_PATTERNS = (
    re.compile(r"\[SFT MONITOR\]\s+step=(\d+)"),
    re.compile(r"SFT monitor step=(\d+)"),
)


def _scan_progress(stdout_path: Path) -> dict:
    """Best-effort last completed step and checkpoint from the child's log."""
    progress: dict = {"last_monitored_step": None, "last_log_line": None}
    try:
        text = stdout_path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return progress
    steps: list[int] = []
    for pattern in _STEP_PATTERNS:
        steps.extend(int(m) for m in pattern.findall(text))
    if steps:
        progress["last_monitored_step"] = max(steps)
    lines = [line for line in text.splitlines() if line.strip()]
    if lines:
        progress["last_log_line"] = lines[-1][:300]
    return progress


def _latest_checkpoint(run_name: str | None) -> str | None:
    if not run_name:
        return None
    adapter_dir = ROOT / "checkpoints" / run_name
    if not adapter_dir.exists():
        return None
    candidates = sorted(adapter_dir.glob("**/checkpoint-*"), key=lambda p: p.name)
    return str(candidates[-1]) if candidates else None


def _extract_traceback(stderr_path: Path) -> str | None:
    try:
        text = stderr_path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return None
    index = text.rfind("Traceback (most recent call last)")
    return text[index:index + 4000] if index != -1 else None


def classify_termination(
    exit_code: int | None,
    stdout_path: Path,
    stderr_path: Path,
    telemetry_path: Path,
) -> dict:
    """Distinguish OOM from disconnect, exception, exhaustion, or external kill."""
    combined = ""
    for path in (stderr_path, stdout_path):
        try:
            combined += path.read_text(encoding="utf-8", errors="replace")[-200_000:]
        except Exception:
            pass
    lowered = combined.lower()
    traceback_text = _extract_traceback(stderr_path)

    last_disk = None
    last_ram = None
    try:
        lines = telemetry_path.read_text(encoding="utf-8").strip().splitlines()
        if lines:
            final = json.loads(lines[-1])
            last_disk = final.get("disk_free_gb")
            last_ram = final.get("ram_available_gb")
    except Exception:
        pass

    if exit_code == 0:
        return {"reason": "completed", "confidence": "certain", "detail": "child exited 0"}
    if "cuda out of memory" in lowered or "torch.cuda.outofmemoryerror" in lowered:
        return {"reason": "cuda_oom", "confidence": "certain",
                "detail": "CUDA OOM text present in child output"}
    if "memoryerror" in lowered or (last_ram is not None and last_ram < 1.0):
        return {"reason": "ram_exhaustion", "confidence": "high",
                "detail": f"MemoryError or last available RAM {last_ram} GB"}
    if "no space left" in lowered or (last_disk is not None and last_disk < 1.0):
        return {"reason": "disk_exhaustion", "confidence": "high",
                "detail": f"disk full text or last free disk {last_disk} GB"}
    if traceback_text:
        return {"reason": "python_exception", "confidence": "certain",
                "detail": traceback_text.splitlines()[-1][:300]}
    if exit_code is None:
        return {"reason": "supervisor_lost", "confidence": "medium",
                "detail": "supervisor did not record an exit code; host restart or supervisor kill"}
    if exit_code in (1, -1, 137, 143, 3221225786, 0xC000013A):
        return {"reason": "external_termination", "confidence": "high",
                "detail": f"exit code {exit_code} with no traceback; typical of an external kill"}
    return {"reason": "nonzero_exit", "confidence": "medium",
            "detail": f"exit code {exit_code} with no recognised failure signature"}


def validate_artifacts(run_name: str | None) -> dict:
    """A run is complete only if its expected artifacts actually exist."""
    if not run_name:
        # A command with no --run-name has no training-artifact contract (a
        # probe, an audit, a sweep script). Exit code is then the only
        # evidence available, and the record says so explicitly so this can
        # never be mistaken for a validated training run.
        return {
            "validated": True,
            "artifact_contract": "none_declared",
            "reason": "no --run-name; validated on exit code alone",
        }
    results_dir = ROOT / "results" / run_name
    training_results = results_dir / "training_results.json"
    if not training_results.exists():
        return {"validated": False, "reason": f"missing {training_results}"}
    try:
        payload = json.loads(training_results.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"validated": False, "reason": f"training_results.json unreadable: {exc}"}
    if payload.get("final_test_measurement") is True:
        return {"validated": False, "reason": "artifact claims a sealed-test measurement"}
    monitors = sorted(p.name for p in results_dir.glob("sft_monitor_checkpoint_*.json"))
    return {
        "validated": True,
        "training_results": str(training_results),
        "mode": payload.get("mode"),
        "monitor_checkpoints": monitors,
        "sft_loss": payload.get("sft_loss"),
    }


# --------------------------------------------------------------------------
# supervisor
# --------------------------------------------------------------------------

def supervise(run_dir: Path) -> int:
    """Entry point for the detached supervisor; never fails silently."""
    try:
        return _supervise_inner(run_dir)
    except BaseException as exc:  # noqa: BLE001 - must record every failure
        import traceback as tb
        (run_dir / "status.json").write_text(json.dumps({
            "schema_version": SCHEMA_VERSION,
            "run_id": run_dir.name,
            "state": "failed",
            "exit_code": None,
            "end_utc": _utc(),
            "termination": {
                "reason": "supervisor_error",
                "confidence": "certain",
                "detail": f"{type(exc).__name__}: {exc}",
            },
            "traceback": tb.format_exc()[:4000],
        }, indent=2) + "\n", encoding="utf-8")
        raise


def _supervise_inner(run_dir: Path) -> int:
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    command = manifest["command"]
    run_name = manifest["config_snapshot"].get("run_name")

    stdout_path = run_dir / "stdout.log"
    stderr_path = run_dir / "stderr.log"
    telemetry_path = run_dir / "telemetry.jsonl"
    heartbeat_path = run_dir / "heartbeat.json"
    status_path = run_dir / "status.json"

    started = time.time()
    with open(stdout_path, "w", encoding="utf-8") as out, \
         open(stderr_path, "w", encoding="utf-8") as err:
        child = subprocess.Popen(
            command, cwd=str(ROOT), stdout=out, stderr=err, text=True
        )

        status_path.write_text(json.dumps({
            "schema_version": SCHEMA_VERSION,
            "run_id": manifest["run_id"],
            "state": "running",
            "supervisor_pid": os.getpid(),
            "child_pid": child.pid,
            "start_utc": _utc(),
        }, indent=2) + "\n", encoding="utf-8")

        exit_code = None
        next_sample = 0.0
        while True:
            exit_code = child.poll()
            # Poll the child often so a fast failure is noticed promptly, but
            # only pay for a telemetry sample on its own slower cadence.
            if exit_code is None and time.time() < next_sample:
                time.sleep(CHILD_POLL_SECONDS)
                continue
            next_sample = time.time() + TELEMETRY_INTERVAL_SECONDS
            sample = {
                "utc": _utc(),
                "elapsed_seconds": round(time.time() - started, 1),
                **_gpu_sample(),
                **_host_sample(),
            }
            with open(telemetry_path, "a", encoding="utf-8") as tel:
                tel.write(json.dumps(sample) + "\n")
            heartbeat_path.write_text(json.dumps({
                "utc": _utc(),
                "child_pid": child.pid,
                "child_running": exit_code is None,
                "elapsed_seconds": sample["elapsed_seconds"],
                **_scan_progress(stdout_path),
            }, indent=2) + "\n", encoding="utf-8")
            if exit_code is not None:
                break

    duration = round(time.time() - started, 1)
    classification = classify_termination(
        exit_code, stdout_path, stderr_path, telemetry_path
    )
    validation = validate_artifacts(run_name)
    progress = _scan_progress(stdout_path)

    final_state = (
        "completed"
        if exit_code == 0 and validation.get("validated")
        else "failed"
    )
    status = {
        "schema_version": SCHEMA_VERSION,
        "run_id": manifest["run_id"],
        "name": manifest["name"],
        "state": final_state,
        "exit_code": exit_code,
        "start_utc": manifest["created_utc"],
        "end_utc": _utc(),
        "duration_seconds": duration,
        "termination": classification,
        "artifact_validation": validation,
        "progress": progress,
        "last_checkpoint": _latest_checkpoint(run_name),
        "traceback": _extract_traceback(stderr_path),
    }
    status_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")

    if final_state == "completed":
        (run_dir / ".complete").write_text(
            json.dumps({"run_id": manifest["run_id"], "utc": _utc()}, indent=2) + "\n",
            encoding="utf-8",
        )
    return 0 if final_state == "completed" else 1


# --------------------------------------------------------------------------
# client commands
# --------------------------------------------------------------------------

def _detach_flags() -> dict:
    if os.name == "nt":
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP | 0x00000008}
    return {"start_new_session": True}


def cmd_start(args: argparse.Namespace) -> int:
    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        print("error: no command given after --", file=sys.stderr)
        return 2

    # Windows CreateProcess resolves a relative executable against the calling
    # process's directory and copes badly with forward slashes, so pin the
    # interpreter to an absolute path. It also records exactly which binary ran.
    candidate = (ROOT / command[0]).resolve()
    if candidate.exists():
        command[0] = str(candidate)
    elif shutil.which(command[0]):
        command[0] = shutil.which(command[0])

    run_id = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{args.name}"
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    manifest = build_manifest(run_id, args.name, command)
    if args.resumed_from:
        manifest["resume"] = {
            "resumed": True,
            "resumed_from_checkpoint": args.resumed_from,
            "lineage": (args.lineage or []) + [run_id],
        }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    # The supervisor's own failures must never be silent: a supervisor that
    # dies before writing status.json would otherwise leave a run that looks
    # merely "not started yet" forever.
    supervisor_log = open(run_dir / "supervisor.log", "w", encoding="utf-8")
    supervisor = subprocess.Popen(
        [sys.executable, str(Path(__file__).resolve()), "_supervise", str(run_dir)],
        cwd=str(ROOT),
        stdout=supervisor_log,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        **_detach_flags(),
    )
    print(json.dumps({
        "run_id": run_id,
        "run_dir": str(run_dir),
        "supervisor_pid": supervisor.pid,
        "status_command": f"python scripts/gpu_run.py status --run-id {run_id}",
        "logs_command": f"python scripts/gpu_run.py logs --run-id {run_id} --follow",
        "stop_command": f"python scripts/gpu_run.py stop --run-id {run_id}",
    }, indent=2))
    return 0


def _resolve_run(run_id: str | None) -> Path | None:
    if not RUNS_DIR.exists():
        return None
    if run_id:
        candidate = RUNS_DIR / run_id
        return candidate if candidate.exists() else None
    runs = sorted(RUNS_DIR.iterdir(), key=lambda p: p.name)
    return runs[-1] if runs else None


def cmd_status(args: argparse.Namespace) -> int:
    run_dir = _resolve_run(args.run_id)
    if run_dir is None:
        print("no such run", file=sys.stderr)
        return 1
    status_path = run_dir / "status.json"
    heartbeat_path = run_dir / "heartbeat.json"
    payload: dict = {"run_id": run_dir.name, "complete_marker": (run_dir / ".complete").exists()}
    if status_path.exists():
        payload["status"] = json.loads(status_path.read_text(encoding="utf-8"))
    if heartbeat_path.exists():
        heartbeat = json.loads(heartbeat_path.read_text(encoding="utf-8"))
        payload["heartbeat"] = heartbeat
        age = None
        try:
            beat = datetime.fromisoformat(heartbeat["utc"])
            age = round((datetime.now(timezone.utc) - beat).total_seconds(), 1)
        except Exception:
            pass
        payload["heartbeat_age_seconds"] = age
        if age is not None and age > TELEMETRY_INTERVAL_SECONDS * 3 and \
                payload.get("status", {}).get("state") == "running":
            payload["warning"] = (
                "heartbeat is stale; the supervisor may have been killed "
                "(host restart or external termination)"
            )
    print(json.dumps(payload, indent=2))
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    if not RUNS_DIR.exists():
        print("[]")
        return 0
    rows = []
    for run_dir in sorted(RUNS_DIR.iterdir(), key=lambda p: p.name):
        status_path = run_dir / "status.json"
        state = "unknown"
        if status_path.exists():
            try:
                state = json.loads(status_path.read_text(encoding="utf-8")).get("state", "unknown")
            except Exception:
                pass
        rows.append({
            "run_id": run_dir.name,
            "state": state,
            "complete": (run_dir / ".complete").exists(),
        })
    print(json.dumps(rows, indent=2))
    return 0


def cmd_logs(args: argparse.Namespace) -> int:
    run_dir = _resolve_run(args.run_id)
    if run_dir is None:
        print("no such run", file=sys.stderr)
        return 1
    path = run_dir / ("stderr.log" if args.stderr else "stdout.log")
    if not path.exists():
        print(f"{path} not written yet", file=sys.stderr)
        return 1
    if not args.follow:
        print(path.read_text(encoding="utf-8", errors="replace")[-20_000:])
        return 0
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        handle.seek(0, os.SEEK_END)
        while True:
            line = handle.readline()
            if line:
                sys.stdout.write(line)
                sys.stdout.flush()
            else:
                if (run_dir / ".complete").exists():
                    break
                time.sleep(2)
    return 0


def cmd_stop(args: argparse.Namespace) -> int:
    run_dir = _resolve_run(args.run_id)
    if run_dir is None:
        print("no such run", file=sys.stderr)
        return 1
    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    child_pid = status.get("child_pid")
    if not child_pid:
        print("no child pid recorded", file=sys.stderr)
        return 1
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(child_pid), "/T", "/F"], check=False)
        else:
            os.kill(child_pid, signal.SIGTERM)
        print(json.dumps({"stopped_child_pid": child_pid, "run_id": run_dir.name}, indent=2))
        return 0
    except Exception as exc:
        print(f"failed to stop: {exc}", file=sys.stderr)
        return 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    start = sub.add_parser("start", help="launch a detached, supervised run")
    start.add_argument("--name", required=True, help="short run label")
    start.add_argument("--resumed-from", default=None)
    start.add_argument("--lineage", nargs="*", default=None)
    start.add_argument("command", nargs=argparse.REMAINDER)
    start.set_defaults(func=cmd_start)

    status = sub.add_parser("status")
    status.add_argument("--run-id", default=None)
    status.set_defaults(func=cmd_status)

    listing = sub.add_parser("list")
    listing.set_defaults(func=cmd_list)

    logs = sub.add_parser("logs")
    logs.add_argument("--run-id", default=None)
    logs.add_argument("--follow", action="store_true")
    logs.add_argument("--stderr", action="store_true")
    logs.set_defaults(func=cmd_logs)

    stop = sub.add_parser("stop")
    stop.add_argument("--run-id", default=None)
    stop.set_defaults(func=cmd_stop)

    supervise_parser = sub.add_parser("_supervise", help=argparse.SUPPRESS)
    supervise_parser.add_argument("run_dir")
    supervise_parser.set_defaults(func=lambda a: supervise(Path(a.run_dir)))

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
