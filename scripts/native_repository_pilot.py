"""Pilot native repository execution for a handful of BugsInPy defects.

Part 11 of the research plan.  Until this succeeds, repository records are
VERIFIED SUPERVISION COVERAGE - a defect with a known failing official test -
and must not be described as generated-test real-repository kill evidence.

For each pilot task this script:

1. materializes the buggy revision into an isolated worktree,
2. builds a virtual environment and installs the project,
3. runs the official failing test and requires it to FAIL,
4. materializes the fixed revision,
5. runs the identical test and requires it to PASS,
6. records environment construction, dependency install, timeout, exit code,
   stdout, stderr, and traceback for every step.

Infrastructure failure and test failure are reported as different outcomes.
A task whose environment could not be built proves nothing about the defect and
is never counted as a reproduction.

Intended to run inside WSL/Linux, where the project environments actually
build.  The Windows host cannot install these dependencies.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


PILOT_SCHEMA_VERSION = "oneiros_native_repository_pilot_v1"

#: Distinguishes "we could not build the environment" from "the test behaved
#: unexpectedly".  Only the second is evidence about the defect.
INFRASTRUCTURE_STEPS = ("worktree", "venv", "install")


def _run(
    command: list[str], cwd: Path | None = None, timeout: int = 900,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    started = time.time()
    try:
        completed = subprocess.run(
            command, cwd=str(cwd) if cwd else None, capture_output=True,
            text=True, timeout=timeout, env=env,
        )
        return {
            "command": command,
            "returncode": completed.returncode,
            "timed_out": False,
            "stdout_tail": completed.stdout[-4000:],
            "stderr_tail": completed.stderr[-4000:],
            "seconds": round(time.time() - started, 2),
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "command": command,
            "returncode": None,
            "timed_out": True,
            "timeout_seconds": timeout,
            "stdout_tail": (exc.stdout or b"")[-2000:].decode("utf-8", "replace")
            if isinstance(exc.stdout, bytes) else str(exc.stdout or "")[-2000:],
            "stderr_tail": (exc.stderr or b"")[-2000:].decode("utf-8", "replace")
            if isinstance(exc.stderr, bytes) else str(exc.stderr or "")[-2000:],
            "seconds": round(time.time() - started, 2),
        }
    except OSError as exc:
        return {
            "command": command, "returncode": None, "timed_out": False,
            "os_error": f"{type(exc).__name__}: {exc}",
            "stdout_tail": "", "stderr_tail": "",
            "seconds": round(time.time() - started, 2),
        }


def _checkout(source: str, commit: str, destination: Path) -> dict[str, Any]:
    """Materialize one revision into a fresh worktree.

    The ingestion cache stores each project as a PARTIAL clone
    (``partialclonefilter=blob:none``, promisor remote): it holds commits and
    trees but no file blobs, so cloning from it produces "unable to read sha1
    file" for every file.  The pilot therefore clones from the upstream URL
    recorded in the record's provenance, which is the reproducible source of
    the exact revision anyway.
    """
    if destination.exists():
        shutil.rmtree(destination, ignore_errors=True)
    steps = [_run([
        "git", "clone", "--quiet", "--no-checkout", source, str(destination),
    ], timeout=900)]
    if steps[-1].get("returncode") != 0:
        return {"ok": False, "step": "clone", "steps": steps}
    steps.append(_run(["git", "checkout", "--quiet", "--force", commit],
                      cwd=destination, timeout=300))
    if steps[-1].get("returncode") != 0:
        return {"ok": False, "step": "checkout", "steps": steps}
    return {"ok": True, "steps": steps}


def _build_environment(
    checkout: Path, python: str, install_targets: list[str], timeout: int,
) -> dict[str, Any]:
    environment_dir = checkout / ".oneiros-venv"
    steps = [_run([python, "-m", "venv", str(environment_dir)], timeout=300)]
    if steps[-1].get("returncode") != 0:
        return {"ok": False, "step": "venv", "steps": steps, "python": None}
    venv_python = environment_dir / "bin" / "python"
    # Upgrade pip (old pip cannot talk to the current index), but do NOT
    # upgrade setuptools: modern setuptools no longer ships pkg_resources, and
    # these 2019-era projects import it during their own build. Pinning below
    # that removal is what lets the original build backend run at all.
    steps.append(_run([
        str(venv_python), "-m", "pip", "install", "--quiet", "--upgrade", "pip",
    ], timeout=timeout))
    steps.append(_run([
        str(venv_python), "-m", "pip", "install", "--quiet",
        "setuptools<70", "wheel", "pytest",
    ], timeout=timeout))
    if steps[-1].get("returncode") != 0:
        return {"ok": False, "step": "install", "steps": steps, "python": None}
    for target in install_targets:
        steps.append(_run([
            str(venv_python), "-m", "pip", "install", "--quiet", target,
        ], cwd=checkout, timeout=timeout))
        if steps[-1].get("returncode") != 0:
            return {"ok": False, "step": "install", "steps": steps, "python": None}
    return {"ok": True, "steps": steps, "python": str(venv_python)}


def _resolve_python(requested: str | None, default: str) -> str:
    """Pick the interpreter BugsInPy recorded for this defect.

    These projects are from 2019-2020. On a modern interpreter they fail for
    reasons that have nothing to do with the defect: setuptools no longer ships
    pkg_resources, old flit_core pins do not resolve, and unittest internals
    moved. Running them on the recorded version is what makes the reproduction
    about the bug rather than about the Python release.
    """
    if not requested:
        return default
    parts = str(requested).split(".")
    if len(parts) >= 2:
        candidate = f"python{parts[0]}.{parts[1]}"
        if shutil.which(candidate):
            return candidate
    return default


def run_task(task: dict[str, Any], workdir: Path, python: str, timeout: int) -> dict[str, Any]:
    """Reproduce one defect: official test must fail buggy and pass fixed."""
    started = time.time()
    record: dict[str, Any] = {
        "task_id": task["task_id"],
        "project": task["project"],
        "bug_id": task.get("bug_id"),
        "buggy_commit": task["buggy_commit"],
        "fixed_commit": task["fixed_commit"],
        "test_selector": task["test_selector"],
        "phases": {},
    }
    source = str(task.get("repository_url") or task.get("repository_path") or "")
    if not source:
        record["outcome"] = "infrastructure_failure"
        record["failed_step"] = "worktree"
        record["detail"] = "no repository_url or repository_path on the task"
        return record
    record["repository_source"] = source
    requested_python = task.get("python_version")
    python = _resolve_python(requested_python, python)
    record["python"] = {
        "requested": requested_python,
        "used": python,
        "matched_requested_minor": bool(
            requested_python and python.endswith(
                ".".join(str(requested_python).split(".")[:2])
            )
        ),
    }

    # BugsInPy adds the bug-revealing test IN the fix commit. Checking out the
    # buggy revision therefore yields a test file that does not contain the
    # test at all - pytest reports "no tests ran" (exit 4), or an older test
    # that passes. The defined protocol is buggy SOURCE with the fixed
    # revision's TEST FILE, so the fixed test file is injected below.
    results: dict[str, Any] = {}
    test_file = str(task.get("test_file") or "")
    fixed_test_source: str | None = None
    for label, commit in (("fixed", task["fixed_commit"]), ("buggy", task["buggy_commit"])):
        checkout = workdir / f"{task['project']}-{task.get('bug_id')}-{label}"
        checkout_result = _checkout(source, commit, checkout)
        record["phases"][f"{label}_checkout"] = checkout_result
        if not checkout_result["ok"]:
            record["outcome"] = "infrastructure_failure"
            record["failed_step"] = "worktree"
            return record

        if test_file:
            target = checkout / test_file
            if label == "fixed":
                if target.exists():
                    fixed_test_source = target.read_text(
                        encoding="utf-8", errors="replace",
                    )
                else:
                    record["outcome"] = "infrastructure_failure"
                    record["failed_step"] = "fixed_test_file_missing"
                    record["detail"] = f"{test_file} absent at the fixed revision"
                    return record
            elif fixed_test_source is not None:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(fixed_test_source, encoding="utf-8")
                record["test_injection"] = {
                    "test_file": test_file,
                    "injected_from": "fixed_revision",
                    "characters": len(fixed_test_source),
                }

        environment = _build_environment(
            checkout, python, task.get("install_targets") or ["."], timeout,
        )
        record["phases"][f"{label}_environment"] = {
            "ok": environment["ok"],
            "failed_step": environment.get("step"),
            "steps": environment["steps"],
        }
        if not environment["ok"]:
            record["outcome"] = "infrastructure_failure"
            record["failed_step"] = environment.get("step")
            return record

        test = _run(
            [environment["python"], "-m", "pytest", "-q", task["test_selector"]],
            cwd=checkout, timeout=timeout,
        )
        record["phases"][f"{label}_test"] = test
        results[label] = test
        shutil.rmtree(checkout, ignore_errors=True)

    # pytest exit codes: 0 all passed, 1 tests failed, 2 interrupted,
    # 3 internal error, 4 usage error, 5 no tests collected. Only 1 is a real
    # test failure. Treating 4 or 5 as "the buggy revision failed" would count
    # a collection error - the test never ran - as a reproduced defect.
    buggy_code = results["buggy"].get("returncode")
    fixed_code = results["fixed"].get("returncode")
    buggy_failed = buggy_code == 1
    fixed_passed = fixed_code == 0
    record["pytest_returncodes"] = {"buggy": buggy_code, "fixed": fixed_code}
    harness_error = {
        code for code in (buggy_code, fixed_code) if code in (2, 3, 4, 5)
    }
    if results["buggy"].get("timed_out") or results["fixed"].get("timed_out"):
        record["outcome"] = "infrastructure_failure"
        record["failed_step"] = "test_timeout"
    elif harness_error:
        record["outcome"] = "infrastructure_failure"
        record["failed_step"] = "pytest_could_not_run_the_test"
        record["detail"] = f"pytest exit codes {sorted(harness_error)}"
    elif buggy_failed and fixed_passed:
        record["outcome"] = "reproduced"
    else:
        record["outcome"] = "not_reproduced"
    record["buggy_failed"] = buggy_failed
    record["fixed_passed"] = fixed_passed
    record["seconds"] = round(time.time() - started, 2)
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--python", default="python3.11")
    parser.add_argument("--timeout", type=int, default=1200)
    parser.add_argument("--workdir", type=Path, default=None)
    arguments = parser.parse_args()

    tasks = json.loads(arguments.tasks.read_text(encoding="utf-8"))
    workdir = arguments.workdir or Path(tempfile.mkdtemp(prefix="oneiros-native-"))
    workdir.mkdir(parents=True, exist_ok=True)

    records = []
    for index, task in enumerate(tasks, 1):
        print(f"[{index}/{len(tasks)}] {task['task_id']}", flush=True)
        record = run_task(task, workdir, arguments.python, arguments.timeout)
        print(f"    -> {record['outcome']}", flush=True)
        records.append(record)
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(json.dumps({
            "schema_version": PILOT_SCHEMA_VERSION,
            "python": arguments.python,
            "platform": sys.platform,
            "timeout_seconds": arguments.timeout,
            "tasks_attempted": len(records),
            "tasks_total": len(tasks),
            "outcomes": {
                outcome: sum(1 for item in records if item.get("outcome") == outcome)
                for outcome in sorted({item.get("outcome") for item in records})
            },
            "infrastructure_failure_rate": round(
                sum(
                    1 for item in records
                    if item.get("outcome") == "infrastructure_failure"
                ) / max(1, len(records)), 4,
            ),
            "records": records,
        }, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(arguments.output),
        "outcomes": {
            outcome: sum(1 for item in records if item.get("outcome") == outcome)
            for outcome in sorted({item.get("outcome") for item in records})
        },
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
