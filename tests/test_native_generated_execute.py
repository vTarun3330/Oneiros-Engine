"""Candidate classification (pure) and static admission for native generated tests."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
ex = pytest.importorskip("native_generated_tests_execute_wsl")

OK_STATIC = {"status": "ok"}


def node(outcome="passed", reached=True, exc=None, xfail=False, setup="passed"):
    phases = {"setup": {"outcome": setup, "xfail": False, "exception": None}}
    if setup == "passed":
        phases["call"] = {"outcome": outcome, "xfail": xfail, "exception": exc}
    return {"reached": reached, "phases": phases}


def report(nodes, collected=None, errors=(), attest=True, wall=False):
    return {"attestation": {"ok": attest}, "collected": list(collected or nodes),
            "collection_errors": list(errors), "nodes": nodes, "wall_timeout": wall}


ASSERT = {"assertion": True, "type": "AssertionError", "target_in_traceback": False,
          "project_in_traceback": False, "timeout": False}
CRASH = {"assertion": False, "type": "ZeroDivisionError", "target_in_traceback": True,
         "project_in_traceback": True, "timeout": False}
TESTERR = {"assertion": False, "type": "NameError", "target_in_traceback": False,
           "project_in_traceback": False, "timeout": False}
TIMEOUT = {"assertion": False, "type": "SandboxTestTimeout", "timeout": True,
           "target_in_traceback": False, "project_in_traceback": False}


@pytest.mark.parametrize("buggy, fixed, expected", [
    (report({"t": node("failed", exc=ASSERT)}), report({"t": node()}), "semantic_kill"),
    (report({"t": node("failed", exc=CRASH)}), report({"t": node()}), "crash_kill"),
    (report({"t": node("failed", exc=TESTERR)}), report({"t": node()}), "buggy_test_error"),
    (report({"t": node()}), report({"t": node()}), "pass_both"),
    (report({"t": node("failed", exc=ASSERT)}), report({"t": node("failed", exc=ASSERT)}),
     "fixed_side_failure"),
    (report({"t": node()}), report({"t": node("failed", exc=ASSERT)}), "fixed_side_failure"),
    (report({"t": node("skipped", setup="skipped")}), report({"t": node("skipped", setup="skipped")}),
     "skipped_or_xfail"),
    (report({"t": node("failed", xfail=True)}), report({"t": node(xfail=True)}), "skipped_or_xfail"),
    (report({"t": node(reached=False)}), report({"t": node(reached=False)}), "target_not_reached"),
    (report({"t": node("failed", exc=TIMEOUT)}), report({"t": node("failed", exc=TIMEOUT)}),
     "timeout"),
    (report({}, collected=[]), report({}, collected=[]), "no_tests_collected"),
    (report({}, collected=[], errors=[{"nodeid": "m", "text": "ImportError: cannot import name 'x'"}]),
     report({}, collected=[]), "fabricated_import"),
    (report({}, collected=[], errors=[{"nodeid": "m", "text": "SyntaxWarning boom"}]),
     report({}, collected=[]), "collection_failure"),
    (report({"t": node()}, collected=["t", "u"]), report({"t": node()}), "collection_failure"),
    (report({"t": node()}, attest=False), report({"t": node()}), "harness_failure"),
    (None, report({"t": node()}), "harness_failure"),
    (report({"t": node()}, wall=True), report({"t": node()}), "timeout"),
])
def test_taxonomy(buggy, fixed, expected):
    assert ex.classify(OK_STATIC, buggy, fixed)["class"] == expected


def test_a_kill_must_reach_the_target_on_both_revisions():
    buggy = report({"t": node("failed", exc=ASSERT)})
    fixed = report({"t": node(reached=False)})
    assert ex.classify(OK_STATIC, buggy, fixed)["class"] == "target_not_reached"


def test_nondeterminism_rerun_disagreement_is_not_a_kill():
    buggy = report({"t": node("failed", exc=ASSERT)})
    fixed = report({"t": node()})
    rerun = (report({"t": node()}), report({"t": node()}))
    result = ex.classify(OK_STATIC, buggy, fixed, rerun)
    assert result["class"] == "nondeterminism" and result["first"] == "semantic_kill"
    agree = ex.classify(OK_STATIC, buggy, fixed, (buggy, fixed))
    assert agree["class"] == "semantic_kill"


def test_static_admission_limits():
    assert ex.static_check("def test_(:\n", "f")["status"] == "syntax_failure"
    many = "".join(f"def test_{i}():\n    assert f(1)\n" for i in range(30))
    result = ex.static_check(many, "f")
    assert result["status"] == "over_limits" and "tests" in result["over"]
    assert ex.static_check("def test_a():\n    assert f(1) == 2\n", "f")["status"] == "ok"


def test_model_failures_are_never_infrastructure():
    model_caused = {"syntax_failure", "fabricated_import", "collection_failure",
                    "no_tests_collected", "skipped_or_xfail", "target_not_reached",
                    "fixed_side_failure", "timeout", "over_limits"}
    assert not model_caused & set(ex.INFRA)


def test_sandbox_inner_script_enforces_the_protocol_controls():
    text = (ROOT / "scripts" / "native_sandbox_inner.sh").read_text(encoding="utf-8")
    for control in ("--reuid=65534", "--no-new-privs", "--bounding-set=-all", "prlimit",
                    "--nproc", "--fsize", "--as", "timeout -k", "env -i",
                    "PYTHONDONTWRITEBYTECODE=1", "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1",
                    "remount,bind,ro", "for d in /root /home /mnt /tmp"):
        assert control in text, control
    source = (ROOT / "scripts" / "native_generated_tests_execute_wsl.py").read_text(encoding="utf-8")
    assert '"--net"' in source and '"--pid"' in source and '"--mount"' in source
