"""Native repository execution for *generated* tests (research specification §36).

Repository ingestion already proves that an *official* test fails on the buggy
revision and passes on the fixed one.  That is oracle evidence about the
benchmark, not evidence about the model.  This module answers the different
question the research plan actually asks:

    generated test
         ↓  inject into the buggy project checkout
    buggy revision  → must FAIL
         ↓  same test, fixed project checkout
    fixed revision  → must PASS

Only a test that fails on the buggy revision *and* passes on the fixed one is
``difference_exposing``.  Everything else is reported under its own status.

The module is deliberately fail-closed.  When a project environment cannot be
built, or the task's test command is not a supported pytest invocation, the
record is reported as ``environment_unavailable`` / ``unsupported_task``.  Those
statuses are never counted as either a pass or a kill, and callers must not
fold them into a success rate.
"""

from __future__ import annotations

import hashlib
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from harness.bugsinpy_v2 import (
    BugsInPyTask,
    RepositoryCache,
    _prepare_environment,
    _run,
    _test_environment,
)


#: Statuses that describe the generated test's behaviour across both revisions.
BEHAVIOURAL_STATUSES = (
    "difference_exposing",   # buggy FAIL, fixed PASS -- the only success
    "passes_both",           # no discrimination
    "fails_both",            # reference-invalid: the test is simply wrong
    "inverted",              # buggy PASS, fixed FAIL -- pathological
)

#: Statuses that mean no behavioural evidence was obtained at all.
INCONCLUSIVE_STATUSES = (
    "parse_invalid",
    "unsupported_task",
    "environment_unavailable",
    "timeout",
    "worktree_error",
)


@dataclass
class NativeTestOutcome:
    """One generated test evaluated against both project revisions."""

    record_id: str
    project: str
    bug_id: str
    rank: int
    status: str
    parse_valid: bool = False
    executed: bool = False
    fixed_returncode: Optional[int] = None
    buggy_returncode: Optional[int] = None
    detail: str = ""
    evidence: Dict[str, Any] = field(default_factory=dict)

    @property
    def fixed_pass(self) -> bool:
        return self.fixed_returncode == 0

    @property
    def buggy_fail(self) -> bool:
        return self.buggy_returncode is not None and self.buggy_returncode != 0

    @property
    def difference_exposing(self) -> bool:
        return self.status == "difference_exposing"

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "record_id": self.record_id,
            "project": self.project,
            "bug_id": self.bug_id,
            "rank": self.rank,
            "status": self.status,
            "parse_valid": self.parse_valid,
            "executed": self.executed,
            "fixed_returncode": self.fixed_returncode,
            "buggy_returncode": self.buggy_returncode,
            "fixed_pass": self.fixed_pass,
            "buggy_fail": self.buggy_fail,
            "difference_exposing": self.difference_exposing,
            "evidence": self.evidence,
        }
        if self.detail:
            payload["detail"] = self.detail
        return payload


def generated_test_filename(source: str) -> str:
    """Return a collision-free pytest module name for one generated test."""
    digest = hashlib.sha256(source.encode("utf-8")).hexdigest()[:16]
    return f"test_oneiros_generated_{digest}.py"


def _injection_directory(checkout: Path, task: BugsInPyTask) -> Optional[Path]:
    """Place the generated test beside the official tests.

    Sitting in the official test directory is what makes the project's
    ``conftest.py`` fixtures and relative imports resolve exactly as they do for
    the project's own suite.  Inventing a different location would silently
    change the execution context.
    """
    for test_file in task.test_files:
        candidate = (checkout / test_file).parent
        if candidate.is_dir():
            return candidate
    for name in ("tests", "test"):
        candidate = checkout / name
        if candidate.is_dir():
            return candidate
    return checkout if checkout.is_dir() else None


def _pytest_command(python: Path, test_path: Path) -> List[str]:
    return [str(python), "-m", "pytest", "-x", "-q", "--no-header", str(test_path)]


def _execution_environment(
    checkout: Path, task: BugsInPyTask, prepare_environment: bool
) -> Dict[str, str]:
    """Build the per-revision test environment.

    The ingestion path always installs into an isolated virtualenv, so it sets
    ``PYTHONNOUSERSITE`` to keep the host's packages out.  When the caller has
    explicitly opted out of environment preparation they are asking to use the
    ambient interpreter, and stripping its user site-packages would hide the very
    test runner they intend to use.
    """
    environment = _test_environment(checkout, task)
    if not prepare_environment:
        environment.pop("PYTHONNOUSERSITE", None)
    return environment


def evaluate_generated_repository_tests(
    task: BugsInPyTask,
    repository_root: Path,
    generated_tests: Sequence[str],
    *,
    timeout: int = 900,
    runner_python: str = "python",
    prepare_environment: bool = True,
) -> List[NativeTestOutcome]:
    """Run each generated test in both revisions of one repository defect.

    The worktrees and the isolated environment are built once and reused across
    every candidate for the task, because the expensive part is the checkout and
    the dependency install, not the test run.
    """
    outcomes: List[NativeTestOutcome] = []
    ranked: List[tuple[int, str]] = []
    for rank, source in enumerate(generated_tests, 1):
        try:
            compile(source, "<oneiros-generated-test>", "exec")
        except SyntaxError as exc:
            outcomes.append(NativeTestOutcome(
                record_id=task.id, project=task.project, bug_id=task.bug_id,
                rank=rank, status="parse_invalid", detail=str(exc),
            ))
            continue
        ranked.append((rank, source))

    if not ranked:
        return outcomes

    def fail_all(status: str, detail: str, evidence: Dict[str, Any]) -> List[NativeTestOutcome]:
        for rank, _ in ranked:
            outcomes.append(NativeTestOutcome(
                record_id=task.id, project=task.project, bug_id=task.bug_id,
                rank=rank, status=status, parse_valid=True,
                detail=detail, evidence=evidence,
            ))
        return sorted(outcomes, key=lambda item: item.rank)

    with tempfile.TemporaryDirectory(
        prefix=f"oneiros-native-{task.project}-{task.bug_id}-"
    ) as directory:
        root = Path(directory)
        fixed_dir, buggy_dir = root / "fixed", root / "buggy"
        repository = repository_root
        try:
            cache = RepositoryCache(repository_root.parent)
            repository = cache.ensure_commit(task, task.fixed_commit)
            cache.ensure_commit(task, task.buggy_commit)
            cache.worktree(repository, task.fixed_commit, fixed_dir)
            cache.worktree(repository, task.buggy_commit, buggy_dir)

            # The official test files come from the fixed revision so that
            # shared helpers and conftest fixtures exist identically on both
            # sides.  This mirrors the ingestion gate exactly.
            for test_file in task.test_files:
                source_path, destination = fixed_dir / test_file, buggy_dir / test_file
                if source_path.exists():
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source_path, destination)

            if prepare_environment:
                environment_python, environment_evidence = _prepare_environment(
                    task, root, runner_python
                )
                if environment_python is None:
                    return fail_all(
                        "environment_unavailable",
                        str(environment_evidence.get("reason", "environment_setup_failed")),
                        {"environment": environment_evidence},
                    )
            else:
                environment_python = Path(runner_python)
                environment_evidence = {"runner": "current_python"}

            fixed_target = _injection_directory(fixed_dir, task)
            buggy_target = _injection_directory(buggy_dir, task)
            if fixed_target is None or buggy_target is None:
                return fail_all(
                    "unsupported_task",
                    "no injectable test directory in the project checkout",
                    {"environment": environment_evidence},
                )

            for rank, source in ranked:
                filename = generated_test_filename(source)
                fixed_path = fixed_target / filename
                buggy_path = buggy_target / filename
                fixed_path.write_text(source, encoding="utf-8")
                buggy_path.write_text(source, encoding="utf-8")
                try:
                    fixed_result = _run(
                        _pytest_command(environment_python, fixed_path),
                        cwd=fixed_dir, timeout=timeout,
                        env=_execution_environment(fixed_dir, task, prepare_environment),
                    )
                    buggy_result = _run(
                        _pytest_command(environment_python, buggy_path),
                        cwd=buggy_dir, timeout=timeout,
                        env=_execution_environment(buggy_dir, task, prepare_environment),
                    )
                except subprocess.TimeoutExpired:
                    outcomes.append(NativeTestOutcome(
                        record_id=task.id, project=task.project, bug_id=task.bug_id,
                        rank=rank, status="timeout", parse_valid=True,
                        detail=f"exceeded {timeout}s",
                        evidence={"environment": environment_evidence},
                    ))
                    continue
                finally:
                    fixed_path.unlink(missing_ok=True)
                    buggy_path.unlink(missing_ok=True)

                fixed_ok = fixed_result.returncode == 0
                buggy_ok = buggy_result.returncode == 0
                if fixed_ok and not buggy_ok:
                    status = "difference_exposing"
                elif fixed_ok and buggy_ok:
                    status = "passes_both"
                elif not fixed_ok and not buggy_ok:
                    status = "fails_both"
                else:
                    status = "inverted"

                outcomes.append(NativeTestOutcome(
                    record_id=task.id, project=task.project, bug_id=task.bug_id,
                    rank=rank, status=status, parse_valid=True, executed=True,
                    fixed_returncode=fixed_result.returncode,
                    buggy_returncode=buggy_result.returncode,
                    evidence={
                        "environment": environment_evidence,
                        "injected_as": filename,
                        "fixed_output_tail": fixed_result.stdout[-2000:],
                        "buggy_output_tail": buggy_result.stdout[-2000:],
                    },
                ))
        except RuntimeError as exc:
            return fail_all("worktree_error", str(exc), {})
        except subprocess.TimeoutExpired:
            return fail_all("timeout", f"checkout exceeded {timeout}s", {})
        finally:
            cleanup = RepositoryCache(repository_root.parent)
            for path in (fixed_dir, buggy_dir):
                if path.exists():
                    cleanup.remove_worktree(repository, path)

    return sorted(outcomes, key=lambda item: item.rank)


def summarise_native_outcomes(
    outcomes: Sequence[NativeTestOutcome],
    k_values: Sequence[int] = (1, 2, 4, 8),
) -> Dict[str, Any]:
    """Aggregate the §36 metrics without ever hiding inconclusive records.

    ``difference_exposing_at_k`` uses the original generation order, matching the
    Kill@k definition used for function tasks.
    """
    rows = [outcome.to_dict() for outcome in outcomes]
    total = len(rows)
    executed = [row for row in rows if row["executed"]]
    inconclusive = [row for row in rows if row["status"] in INCONCLUSIVE_STATUSES]

    by_record: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        by_record.setdefault(row["record_id"], []).append(row)
    for candidates in by_record.values():
        candidates.sort(key=lambda item: item["rank"])

    ks = sorted({int(k) for k in k_values if int(k) > 0})
    at_k = {}
    for k in ks:
        hits = sum(
            any(
                row["difference_exposing"] for row in candidates if row["rank"] <= k
            )
            for candidates in by_record.values()
        )
        at_k[str(k)] = {
            "records": hits,
            "rate": round(hits / max(len(by_record), 1), 6),
        }

    projects: Dict[str, Dict[str, int]] = {}
    for row in rows:
        bucket = projects.setdefault(
            row["project"],
            {"candidates": 0, "executed": 0, "fixed_pass": 0,
             "buggy_fail": 0, "difference_exposing": 0, "inconclusive": 0},
        )
        bucket["candidates"] += 1
        bucket["executed"] += int(row["executed"])
        bucket["fixed_pass"] += int(row["fixed_pass"])
        bucket["buggy_fail"] += int(row["buggy_fail"])
        bucket["difference_exposing"] += int(row["difference_exposing"])
        bucket["inconclusive"] += int(row["status"] in INCONCLUSIVE_STATUSES)

    status_counts: Dict[str, int] = {}
    for row in rows:
        status_counts[row["status"]] = status_counts.get(row["status"], 0) + 1

    return {
        "native_repository_schema_version": 1,
        "records": len(by_record),
        "candidates": total,
        "parse_valid_candidates": sum(row["parse_valid"] for row in rows),
        "executed_candidates": len(executed),
        "inconclusive_candidates": len(inconclusive),
        "fixed_pass_candidates": sum(row["fixed_pass"] for row in rows),
        "buggy_fail_candidates": sum(row["buggy_fail"] for row in rows),
        "difference_exposing_candidates": sum(row["difference_exposing"] for row in rows),
        "parse_valid_rate": round(
            sum(row["parse_valid"] for row in rows) / max(total, 1), 6
        ),
        "execution_valid_rate": round(len(executed) / max(total, 1), 6),
        # Rates below are conditioned on candidates that actually executed, so an
        # unavailable environment can never inflate or deflate them.
        "fixed_pass_rate_of_executed": round(
            sum(row["fixed_pass"] for row in executed) / max(len(executed), 1), 6
        ),
        "buggy_fail_rate_of_executed": round(
            sum(row["buggy_fail"] for row in executed) / max(len(executed), 1), 6
        ),
        "difference_exposing_rate_of_executed": round(
            sum(row["difference_exposing"] for row in executed) / max(len(executed), 1), 6
        ),
        "difference_exposing_at_k": at_k,
        "status_counts": status_counts,
        "by_project": projects,
        "outcomes": rows,
    }
