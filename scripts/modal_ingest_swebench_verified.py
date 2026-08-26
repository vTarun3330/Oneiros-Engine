"""Durable Oneiros ingestion of SWE-bench Verified through its official harness.

Each instance runs the official repository environment twice in one Modal
Sandbox: once at the base commit (designated F2P tests must fail) and once with
the gold patch (F2P and P2P tests must pass).  Full outputs and condensed
evidence are persisted to a Modal Volume before the function returns.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import modal


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness.corpus import sha256_file, write_json
from harness.swebench_verified import build_repository_record, patch_paths


APP_NAME = "oneiros-v3-swebench-ingestion"
VOLUME_NAME = "oneiros-v3-swebench-volume"
REMOTE_ROOT = Path("/oneiros-state/swebench-verified")
LOCAL_ROOT = ROOT / "data" / "swebench_verified_ingestion"
SOURCE_PARQUET = (
    ROOT / "data" / "swebench_verified_source"
    / "SWE-bench_Verified.test.parquet"
)
PILOT_SELECTION = LOCAL_ROOT / "pilot_selection.json"
SANDBOX_FILESYSTEM_ADAPTER = "modal-filesystem-v1-write-text"

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("swebench==4.1.0", "tenacity", "tokenizers")
    .env({"PYTHONPATH": "/root/oneiros"})
    .add_local_dir("baseline", remote_path="/root/oneiros/baseline")
    .add_local_dir("config", remote_path="/root/oneiros/config")
    .add_local_dir("harness", remote_path="/root/oneiros/harness")
)


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "__", value)


def _json_list(value: Any) -> List[str]:
    return json.loads(value) if isinstance(value, str) else list(value or [])


def _sandbox_write_text(
    sandbox: Any, remote_path: str, content: str, *, optional_cgroup_tuning: bool = False,
) -> bool:
    """Write through Modal's current Sandbox filesystem namespace.

    SWE-bench 4.1.0 still calls the retired ``Sandbox.open`` API. Modal's
    replacement takes ``(content, remote_path)`` and creates parent directories
    as needed. The cgroup write in the upstream runtime is only a best-effort
    pylint scheduling hint, so a current cgroup-v2/read-only filesystem may
    reject it without invalidating test execution.
    """
    try:
        sandbox.filesystem.write_text(content, remote_path)
        return True
    except Exception:
        if optional_cgroup_tuning and remote_path == "/sys/fs/cgroup/cpu/cpu.shares":
            return False
        raise


def _read_files(runner: Any, paths: List[str]) -> Dict[str, str]:
    sources: Dict[str, str] = {}
    for path in paths:
        remote_path = f"/testbed/{path}"
        try:
            sources[path] = runner.sandbox.filesystem.read_text(remote_path)
        except Exception:
            continue
    return sources


def _apply_patch(runner: Any, patch: str, remote_path: str) -> Tuple[bool, str]:
    runner.write_file(remote_path, patch)
    output, returncode = runner.exec(
        f"cd /testbed && git apply -v {remote_path}",
    )
    if returncode == 0:
        return True, output
    fallback, fallback_rc = runner.exec(
        f"cd /testbed && patch --batch --fuzz=5 -p1 -i {remote_path}",
    )
    return fallback_rc == 0, output + "\n" + fallback


def _run_eval(
    runner: Any, test_spec: Any, prediction: Dict[str, str], label: str,
) -> Tuple[Dict[str, Any], str, int, float]:
    from swebench.harness.grading import get_eval_report

    eval_script = test_spec.eval_script.replace("locale-gen", "locale-gen en_US.UTF-8")
    runner.write_file("/root/eval.sh", eval_script)
    command = "cd /testbed"
    if "pylint" in test_spec.instance_id:
        command += " && PYTHONPATH="
    command += " && python3 -c 'import sys; sys.setrecursionlimit(10000)'"
    command += " && /bin/bash /root/eval.sh"
    started = time.time()
    output, returncode = runner.exec(command)
    duration = time.time() - started
    log_path = Path(f"/tmp/{_slug(test_spec.instance_id)}.{label}.test_output.txt")
    log_path.write_text(output, encoding="utf-8")
    report = get_eval_report(
        test_spec=test_spec,
        prediction=prediction,
        test_log_path=str(log_path),
        include_tests_status=True,
    )
    return report, output, returncode, duration


def _variant_counts(report: Dict[str, Any], instance_id: str) -> Dict[str, int]:
    status = (report.get(instance_id) or {}).get("tests_status") or {}
    f2p = status.get("FAIL_TO_PASS") or {}
    p2p = status.get("PASS_TO_PASS") or {}
    return {
        "f2p_success": len(f2p.get("success") or []),
        "f2p_failure": len(f2p.get("failure") or []),
        "p2p_success": len(p2p.get("success") or []),
        "p2p_failure": len(p2p.get("failure") or []),
    }


@app.function(
    image=image,
    volumes={"/oneiros-state": volume},
    timeout=7200,
    cpu=0.25,
    memory=1024,
    max_containers=4,
)
def verify_instance(
    instance: Dict[str, Any], run_id: str, sandbox_timeout: int = 1800,
) -> Dict[str, Any]:
    """Run base and gold variants, persist evidence, and materialize one record."""
    from swebench import __version__ as swebench_version
    from swebench.harness.modal_eval.run_evaluation_modal import (
        LOCAL_SANDBOX_ENTRYPOINT_PATH,
        REMOTE_SANDBOX_ENTRYPOINT_PATH,
        ModalSandboxRuntime,
    )
    from swebench.harness.test_spec.test_spec import make_test_spec

    class CurrentFilesystemModalSandboxRuntime(ModalSandboxRuntime):
        """SWE-bench runtime adapted to Modal's non-legacy filesystem API."""

        def write_file(self, file_path: str, content: str) -> None:
            _sandbox_write_text(
                self.sandbox,
                file_path,
                content,
                optional_cgroup_tuning=True,
            )

    instance_id = instance["instance_id"]
    result_dir = REMOTE_ROOT / run_id / "results"
    evidence_dir = REMOTE_ROOT / run_id / "evidence" / _slug(instance_id)
    result_path = result_dir / f"{_slug(instance_id)}.json"
    volume.reload()
    if result_path.exists():
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        print(json.dumps({"event": "instance_reused", **payload["summary"]}), flush=True)
        return payload["summary"]

    print(json.dumps({
        "event": "instance_started", "run_id": run_id,
        "instance_id": instance_id, "repo": instance["repo"],
    }), flush=True)
    # The official runtime's image builder intentionally reads the entrypoint
    # from its remote destination path.  In the stock ephemeral harness that
    # file is mounted by the outer Modal function.  This durable wrapper uses
    # its own app, so recreate the same mount source before building a Sandbox.
    Path(REMOTE_SANDBOX_ENTRYPOINT_PATH).write_bytes(
        Path(LOCAL_SANDBOX_ENTRYPOINT_PATH).read_bytes()
    )
    test_spec = make_test_spec(instance)
    source_paths = patch_paths(instance.get("patch", ""))
    test_paths = patch_paths(instance.get("test_patch", ""))
    runner: Optional[Any] = None
    base_output = ""
    fixed_output = ""
    outcome: Dict[str, Any]
    record: Optional[Dict[str, Any]] = None
    try:
        runner = CurrentFilesystemModalSandboxRuntime(
            test_spec, sandbox_timeout, verbose=False,
        )
        buggy_sources = _read_files(runner, source_paths)
        base_test_sources = _read_files(runner, test_paths)

        base_prediction = {
            "instance_id": instance_id,
            "model_name_or_path": "oneiros-base-verification",
            "model_patch": "",
        }
        base_report, base_output, base_rc, base_seconds = _run_eval(
            runner, test_spec, base_prediction, "base",
        )

        reset_output, reset_rc = runner.exec(
            f"cd /testbed && git reset --hard {instance['base_commit']} && git clean -fd",
        )
        if reset_rc != 0:
            raise RuntimeError(f"Repository reset failed: {reset_output}")
        patch_ok, patch_output = _apply_patch(
            runner, instance.get("patch", ""), "/tmp/oneiros_gold.patch",
        )
        if not patch_ok:
            raise RuntimeError(f"Gold patch failed to apply: {patch_output}")
        fixed_sources = _read_files(runner, source_paths)
        fixed_prediction = {
            "instance_id": instance_id,
            "model_name_or_path": "oneiros-gold-verification",
            "model_patch": instance.get("patch", ""),
        }
        fixed_report, fixed_output, fixed_rc, fixed_seconds = _run_eval(
            runner, test_spec, fixed_prediction, "fixed",
        )

        test_patch_ok, test_patch_output = _apply_patch(
            runner, instance.get("test_patch", ""), "/tmp/oneiros_test.patch",
        )
        if not test_patch_ok:
            raise RuntimeError(f"Test patch failed to reapply: {test_patch_output}")
        patched_test_sources = _read_files(runner, test_paths)

        expected_f2p = len(_json_list(instance.get("FAIL_TO_PASS")))
        expected_p2p = len(_json_list(instance.get("PASS_TO_PASS")))
        base_counts = _variant_counts(base_report, instance_id)
        fixed_counts = _variant_counts(fixed_report, instance_id)
        base_verified = (
            expected_f2p > 0
            and base_counts["f2p_failure"] == expected_f2p
            and base_counts["f2p_success"] == 0
            and base_counts["p2p_success"] == expected_p2p
            and base_counts["p2p_failure"] == 0
        )
        fixed_verified = (
            (fixed_report.get(instance_id) or {}).get("resolved") is True
            and fixed_counts["f2p_success"] == expected_f2p
            and fixed_counts["f2p_failure"] == 0
            and fixed_counts["p2p_success"] == expected_p2p
            and fixed_counts["p2p_failure"] == 0
        )

        evidence_dir.mkdir(parents=True, exist_ok=True)
        base_log = evidence_dir / "base_test_output.txt"
        fixed_log = evidence_dir / "fixed_test_output.txt"
        base_log.write_text(base_output, encoding="utf-8")
        fixed_log.write_text(fixed_output, encoding="utf-8")
        verification = {
            "harness": "swebench.harness.modal_eval",
            "swebench_version": swebench_version,
            "sandbox_filesystem_adapter": SANDBOX_FILESYSTEM_ADAPTER,
            "base_report": base_report,
            "fixed_report": fixed_report,
            "base_counts": base_counts,
            "fixed_counts": fixed_counts,
            "base_command_returncode": base_rc,
            "fixed_command_returncode": fixed_rc,
            "base_runtime_seconds": round(base_seconds, 3),
            "fixed_runtime_seconds": round(fixed_seconds, 3),
            "base_log_path": str(base_log),
            "fixed_log_path": str(fixed_log),
            "base_log_sha256": sha256_file(base_log),
            "fixed_log_sha256": sha256_file(fixed_log),
            "buggy_fail_verified": base_verified,
            "fixed_pass_verified": fixed_verified,
        }
        if not base_verified:
            outcome = {
                "instance_id": instance_id, "status": "excluded",
                "reason": "base_fail_to_pass_not_reproduced",
                "verification": verification,
            }
        elif not fixed_verified:
            outcome = {
                "instance_id": instance_id, "status": "excluded",
                "reason": "gold_patch_not_fully_resolved",
                "verification": verification,
            }
        else:
            record = build_repository_record(
                instance, buggy_sources, fixed_sources,
                base_test_sources, patched_test_sources, verification,
            )
            if record is None:
                outcome = {
                    "instance_id": instance_id, "status": "excluded",
                    "reason": "no_parseable_python_repository_fragment",
                    "verification": verification,
                }
            else:
                outcome = {
                    "instance_id": instance_id, "status": "accepted",
                    "record_id": record["id"],
                    "verification": verification,
                }
    except Exception as exc:
        outcome = {
            "instance_id": instance_id,
            "status": "excluded",
            "reason": "official_harness_execution_failed",
            "detail": repr(exc),
            "traceback": traceback.format_exc(),
        }
    finally:
        if runner is not None:
            try:
                runner.__exit__(None, None, None)
            except Exception:
                pass

    summary = {
        "run_id": run_id,
        "instance_id": instance_id,
        "repo": instance["repo"],
        "status": outcome["status"],
        "reason": outcome.get("reason"),
        "record_id": (record or {}).get("id"),
    }
    payload = {
        "schema_version": 1,
        "summary": summary,
        "outcome": outcome,
        "record": record,
        "source_identity": {
            "dataset": "SWE-bench/SWE-bench_Verified",
            "split": "test",
            "instance_id": instance_id,
            "source_parquet_sha256": (
                "a45b1fe4e2f0c8390b2b2938ac83e92ed5979000856808f3679c07812e9e6dcd"
            ),
        },
    }
    # The Volume snapshot is committed only after the complete JSON exists, so
    # a direct write is durable here and avoids ghost tempfile entries that the
    # Modal volume downloader cannot retrieve later.
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    volume.commit()
    print(json.dumps({"event": "instance_completed", **summary}), flush=True)
    return summary


@app.function(image=image, volumes={"/oneiros-state": volume}, timeout=300)
def run_status(run_id: str) -> Dict[str, Any]:
    result_dir = REMOTE_ROOT / run_id / "results"
    volume.reload()
    summaries = []
    if result_dir.exists():
        for path in sorted(result_dir.glob("*.json")):
            summaries.append(json.loads(path.read_text(encoding="utf-8"))["summary"])
    accepted = sum(item.get("status") == "accepted" for item in summaries)
    return {
        "run_id": run_id,
        "completed": len(summaries),
        "accepted": accepted,
        "excluded": len(summaries) - accepted,
        "summaries": summaries,
    }


def _load_rows() -> Dict[str, Dict[str, Any]]:
    import pyarrow.parquet as pq
    rows = pq.read_table(SOURCE_PARQUET).to_pylist()
    return {row["instance_id"]: row for row in rows}


def _load_pilot_ids() -> List[str]:
    selection = json.loads(PILOT_SELECTION.read_text(encoding="utf-8"))
    if (
        selection.get("source_sha256") != sha256_file(SOURCE_PARQUET)
        or selection.get("source_rows") != 500
        or selection.get("selected_count") != 12
    ):
        raise RuntimeError("SWE-bench pilot selection identity is invalid")
    return [item["instance_id"] for item in selection["selected"]]


def _ledger_path(run_id: str) -> Path:
    return LOCAL_ROOT / run_id / "submission_checkpoint.json"


def _load_ledger(run_id: str) -> Dict[str, Any]:
    path = _ledger_path(run_id)
    if path.exists():
        ledger = json.loads(path.read_text(encoding="utf-8"))
        if ledger.get("schema_version") != 1 or ledger.get("run_id") != run_id:
            raise RuntimeError("SWE-bench submission checkpoint is malformed")
        return ledger
    return {"schema_version": 1, "run_id": run_id, "submitted": {}}


def _select_instance_ids(
    action: str, ledger: Dict[str, Any], requested: str, pilot_ids: List[str],
    known_ids: set[str],
) -> List[str]:
    """Preserve an explicit durable selection when a run is resumed."""
    if requested:
        selected = list(dict.fromkeys(
            value.strip() for value in requested.split(",") if value.strip()
        ))
    elif action == "resume":
        prior = (ledger.get("selection") or {}).get("instance_ids") or []
        selected = list(dict.fromkeys(prior)) if prior else list(pilot_ids)
    else:
        selected = list(pilot_ids)
    invalid = sorted(set(selected) - known_ids)
    if invalid:
        raise ValueError(f"Unknown SWE-bench instance IDs: {invalid}")
    return selected


def _sync_volume(run_id: str) -> None:
    destination = LOCAL_ROOT / run_id / "remote_results"
    destination.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable, "-X", "utf8", "-m", "modal", "volume", "get", "--force",
        VOLUME_NAME, f"swebench-verified/{run_id}", str(destination),
    ]
    subprocess.run(command, cwd=str(ROOT), check=True)


def _manage(
    action: str, run_id: str, instance_ids: str = "", sandbox_timeout: int = 1800,
) -> None:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", run_id):
        raise ValueError("run_id contains invalid characters")
    deployed_verify = modal.Function.from_name(APP_NAME, "verify_instance")
    deployed_status = modal.Function.from_name(APP_NAME, "run_status")
    ledger = _load_ledger(run_id)
    if action == "status":
        status = deployed_status.remote(run_id)
        write_json(LOCAL_ROOT / run_id / "status_checkpoint.json", status)
        print(json.dumps({k: v for k, v in status.items() if k != "summaries"}, indent=2))
        return
    if action == "sync":
        status = deployed_status.remote(run_id)
        write_json(LOCAL_ROOT / run_id / "status_checkpoint.json", status)
        _sync_volume(run_id)
        print(json.dumps({k: v for k, v in status.items() if k != "summaries"}, indent=2))
        return
    if action == "cancel":
        cancelled = 0
        for item in ledger["submitted"].values():
            call_id = item.get("function_call_id")
            if not call_id:
                continue
            try:
                modal.FunctionCall.from_id(call_id).cancel()
                cancelled += 1
            except Exception as exc:
                item["cancel_error"] = repr(exc)
        ledger["status"] = "cancelled"
        write_json(_ledger_path(run_id), ledger)
        print(json.dumps({"run_id": run_id, "cancelled_calls": cancelled}, indent=2))
        return
    if action not in {"submit", "resume"}:
        raise ValueError("action must be submit, resume, status, sync, or cancel")

    rows = _load_rows()
    selected = _select_instance_ids(
        action, ledger, instance_ids, _load_pilot_ids(), set(rows),
    )
    remote = deployed_status.remote(run_id)
    completed = {item["instance_id"] for item in remote["summaries"]}
    submitted_now = 0
    ledger["selection"] = {
        "instance_ids": selected,
        "sandbox_timeout": sandbox_timeout,
        "source_parquet_sha256": sha256_file(SOURCE_PARQUET),
    }
    ledger["status"] = "submitting"
    write_json(_ledger_path(run_id), ledger)
    for instance_id in selected:
        if instance_id in completed:
            continue
        previous = ledger["submitted"].get(instance_id)
        if previous and previous.get("function_call_id"):
            try:
                modal.FunctionCall.from_id(previous["function_call_id"]).get(timeout=0)
                completed.add(instance_id)
                continue
            except TimeoutError:
                continue
            except Exception as exc:
                previous["prior_call_error"] = repr(exc)
        call = deployed_verify.spawn(rows[instance_id], run_id, sandbox_timeout)
        ledger["submitted"][instance_id] = {
            "function_call_id": call.object_id,
            "dashboard_url": call.get_dashboard_url(),
        }
        submitted_now += 1
        write_json(_ledger_path(run_id), ledger)
    ledger["status"] = "submitted"
    write_json(_ledger_path(run_id), ledger)
    print(json.dumps({
        "run_id": run_id,
        "selected": len(selected),
        "already_completed": len(completed & set(selected)),
        "submitted_now": submitted_now,
        "checkpoint": str(_ledger_path(run_id)),
    }, indent=2))


@app.local_entrypoint()
def main(
    action: str = "status", run_id: str = "v3-swebench-pilot-1",
    instance_ids: str = "", sandbox_timeout: int = 1800,
) -> None:
    _manage(action, run_id, instance_ids, sandbox_timeout)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--action", choices=("submit", "resume", "status", "sync", "cancel"),
        default="status",
    )
    parser.add_argument("--run-id", default="v3-swebench-pilot-1")
    parser.add_argument("--instance-ids", default="")
    parser.add_argument("--sandbox-timeout", type=int, default=1800)
    arguments = parser.parse_args()
    _manage(
        arguments.action, arguments.run_id,
        arguments.instance_ids, arguments.sandbox_timeout,
    )
