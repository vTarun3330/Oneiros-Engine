"""A deterministic stand-in for the WSL sandbox process (``run_sandboxed``) for CPU tests.

It returns reports in the exact shape the pytest plugin writes, so the real
``execute_candidate``, ``canary_check``, ``classify`` and ``verify_row`` run unchanged. A
candidate containing ``KILL`` fails an assertion on the buggy revision and passes on fixed;
``CRASH`` raises inside the target on buggy; everything else passes on both revisions. The
revision is the view directory's name (``buggy``/``fixed``).
"""
from __future__ import annotations

from pathlib import Path

NODE = "test_candidate.py::test_generated"
EXPECTED = {"buggy": "b", "fixed": "f"}


def _phase(outcome: str, exception=None) -> dict:
    return {"outcome": outcome, "xfail": False, "exception": exception}


def report_for(source: str, label: str) -> dict:
    fails = label == "buggy" and ("KILL" in source or "CRASH" in source)
    exception = None
    if fails:
        crash = "CRASH" in source
        exception = {"type": "ZeroDivisionError" if crash else "AssertionError",
                     "assertion": not crash, "timeout": False,
                     "target_in_traceback": crash, "project_in_traceback": crash}
    return {"attestation": {"module_file": "/target/pkg/core.py",
                            "module_sha256": EXPECTED[label]},
            "target_error": None, "collected": [NODE], "collection_errors": [],
            "nodes": {NODE: {"reached": True, "phases": {
                "setup": _phase("passed"), "call": _phase("failed" if fails else "passed", exception),
                "teardown": _phase("passed")}}},
            "exitstatus": 1 if fails else 0, "uid": 65534, "process_exit": 1 if fails else 0,
            "seconds": 0.25, "wall_timeout": False,
            "tail": "/mnt/c/Users/someone/host-only text that must never be retained",
            "raw_report_sha256": "0" * 64}


def fake_run_sandboxed(python, env_dir, view, spec, candidate, out_dir) -> dict:
    return report_for(candidate, Path(view).name)


def evidence_fields(kill: bool) -> dict:
    """Execution-row fields (class, classification, evidence, fixed_valid, canary_failed)
    produced by the REAL execute_candidate and canary_check over the fake sandbox process."""
    from scripts import native_generated_tests_execute_wsl as ex
    target = {"target_key": "t", "module": "pkg.core", "qualname": "f",
              "python": "/usr/bin/python3.11", "env_dir": "/env",
              "views": {"buggy": "/v/buggy", "fixed": "/v/fixed"}, "module_sha256": EXPECTED}
    source = "def test_generated():\n    assert " + ("'KILL'" if kill else "1 == 1") + "\n"
    original = ex.run_sandboxed
    ex.run_sandboxed = fake_run_sandboxed
    try:
        outcome = ex.execute_candidate(target, source, Path("unused"))
        canary = ex.canary_check(target, Path("unused"))
    finally:
        ex.run_sandboxed = original
    cls = outcome["classification"]
    return {"class": cls["class"], "classification": cls, "canary_failed": False,
            "evidence": {**outcome["evidence"], "canary": canary},
            "fixed_valid": ex.fixed_valid_of(cls)}
