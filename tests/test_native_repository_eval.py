"""End-to-end checks for native generated-test repository execution (§36).

These build a real two-commit git repository in a temporary directory, so the
worktree, injection, and both-revision execution paths are genuinely exercised.
They deliberately do not download BugsInPy: the point is to prove the harness
distinguishes a discriminating test from a non-discriminating one, and that an
unavailable environment can never be counted as a success.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from harness.bugsinpy_v2 import BugsInPyTask
from harness.native_repository_eval import (
    INCONCLUSIVE_STATUSES,
    NativeTestOutcome,
    evaluate_generated_repository_tests,
    generated_test_filename,
    summarise_native_outcomes,
)

pytestmark = pytest.mark.skipif(
    shutil.which("git") is None, reason="git is required for worktree evaluation"
)

BUGGY_SOURCE = "def add(a, b):\n    return a - b\n"
FIXED_SOURCE = "def add(a, b):\n    return a + b\n"
OFFICIAL_TEST = "from calc import add\n\n\ndef test_official():\n    assert add(1, 2) == 3\n"


def _git(args, cwd):
    subprocess.run(
        ["git", *args], cwd=str(cwd), check=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )


@pytest.fixture(scope="module")
def repository(tmp_path_factory) -> tuple[Path, BugsInPyTask]:
    """A minimal project with a buggy commit and a fixed commit."""
    root = tmp_path_factory.mktemp("projects")
    project = root / "calcproject"
    project.mkdir()
    _git(["init", "-q"], project)
    _git(["config", "user.email", "test@example.invalid"], project)
    _git(["config", "user.name", "Oneiros Test"], project)
    _git(["config", "commit.gpgsign", "false"], project)

    (project / "calc.py").write_text(BUGGY_SOURCE, encoding="utf-8")
    tests_dir = project / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_calc.py").write_text(OFFICIAL_TEST, encoding="utf-8")
    _git(["add", "-A"], project)
    _git(["commit", "-q", "-m", "buggy"], project)
    buggy_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=str(project), check=True,
        stdout=subprocess.PIPE, text=True,
    ).stdout.strip()

    (project / "calc.py").write_text(FIXED_SOURCE, encoding="utf-8")
    _git(["add", "-A"], project)
    _git(["commit", "-q", "-m", "fixed"], project)
    fixed_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=str(project), check=True,
        stdout=subprocess.PIPE, text=True,
    ).stdout.strip()

    task = BugsInPyTask(
        project="calcproject",
        bug_id="1",
        repository_url=str(project),
        buggy_commit=buggy_commit,
        fixed_commit=fixed_commit,
        test_files=("tests/test_calc.py",),
        test_command="python -m pytest tests/test_calc.py",
        python_version="3.12",
        bug_dir=project,
    )
    return project, task


def _evaluate(repository, tests):
    project, task = repository
    return evaluate_generated_repository_tests(
        task,
        project.parent / task.project,
        tests,
        timeout=300,
        runner_python=sys.executable,
        prepare_environment=False,
    )


def test_discriminating_and_non_discriminating_tests_are_separated(repository):
    outcomes = _evaluate(repository, [
        # Fails on the buggy revision (1 - 2 == -1), passes on the fixed one.
        "from calc import add\n\n\ndef test_generated():\n    assert add(1, 2) == 3\n",
        # True on both revisions, so it discriminates nothing.
        "from calc import add\n\n\ndef test_generated():\n    assert add(0, 0) == 0\n",
        # Wrong on both revisions: reference-invalid, not a kill.
        "from calc import add\n\n\ndef test_generated():\n    assert add(1, 1) == 99\n",
    ])

    assert [item.rank for item in outcomes] == [1, 2, 3]
    assert outcomes[0].status == "difference_exposing"
    assert outcomes[0].fixed_pass and outcomes[0].buggy_fail
    assert outcomes[1].status == "passes_both"
    assert not outcomes[1].difference_exposing
    assert outcomes[2].status == "fails_both"
    assert all(item.executed for item in outcomes)


def test_syntactically_invalid_test_never_executes(repository):
    outcomes = _evaluate(repository, ["def test_broken(:\n    pass\n"])
    assert len(outcomes) == 1
    assert outcomes[0].status == "parse_invalid"
    assert outcomes[0].parse_valid is False
    assert outcomes[0].executed is False


def test_generated_filename_is_stable_and_distinct():
    first = generated_test_filename("assert True\n")
    assert first == generated_test_filename("assert True\n")
    assert first != generated_test_filename("assert False\n")
    assert first.startswith("test_") and first.endswith(".py")


def test_summary_excludes_inconclusive_records_from_success_rates():
    outcomes = [
        NativeTestOutcome(
            record_id="r1", project="p", bug_id="1", rank=1,
            status="difference_exposing", parse_valid=True, executed=True,
            fixed_returncode=0, buggy_returncode=1,
        ),
        NativeTestOutcome(
            record_id="r1", project="p", bug_id="1", rank=2,
            status="passes_both", parse_valid=True, executed=True,
            fixed_returncode=0, buggy_returncode=0,
        ),
        NativeTestOutcome(
            record_id="r2", project="q", bug_id="2", rank=1,
            status="environment_unavailable", parse_valid=True,
        ),
    ]
    summary = summarise_native_outcomes(outcomes)

    assert summary["records"] == 2
    assert summary["candidates"] == 3
    assert summary["executed_candidates"] == 2
    assert summary["inconclusive_candidates"] == 1
    # Two executed candidates, one of which discriminates.
    assert summary["difference_exposing_rate_of_executed"] == 0.5
    # An unavailable environment must not be scored as a failed kill.
    assert summary["status_counts"]["environment_unavailable"] == 1
    assert "environment_unavailable" in INCONCLUSIVE_STATUSES
    assert summary["difference_exposing_at_k"]["1"]["records"] == 1
    assert summary["difference_exposing_at_k"]["1"]["rate"] == 0.5
    assert summary["by_project"]["q"]["inconclusive"] == 1


def test_ordered_prefix_metric_respects_generation_order():
    outcomes = [
        NativeTestOutcome(
            record_id="r1", project="p", bug_id="1", rank=1,
            status="passes_both", parse_valid=True, executed=True,
            fixed_returncode=0, buggy_returncode=0,
        ),
        NativeTestOutcome(
            record_id="r1", project="p", bug_id="1", rank=4,
            status="difference_exposing", parse_valid=True, executed=True,
            fixed_returncode=0, buggy_returncode=1,
        ),
    ]
    summary = summarise_native_outcomes(outcomes)
    assert summary["difference_exposing_at_k"]["1"]["records"] == 0
    assert summary["difference_exposing_at_k"]["2"]["records"] == 0
    assert summary["difference_exposing_at_k"]["4"]["records"] == 1
