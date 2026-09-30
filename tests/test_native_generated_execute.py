"""Candidate classification (pure), candidate policy and static admission (amendment v2.1)."""
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


def report(nodes, collected=None, errors=(), sha="s", wall=False):
    return {"attestation": {"module_sha256": sha} if sha else None,
            "collected": list(nodes if collected is None else collected),
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
    (report({"t": node()}, sha=None), report({"t": node()}), "harness_failure"),
    (None, report({"t": node()}), "harness_failure"),
    (report({"t": node()}, wall=True), report({"t": node()}), "timeout"),
])
def test_taxonomy(buggy, fixed, expected):
    assert ex.classify(OK_STATIC, buggy, fixed)["class"] == expected


def test_kill_requires_an_agreeing_rerun_and_carries_fixed_evidence():
    buggy = report({"t": node("failed", exc=ASSERT)})
    fixed = report({"t": node()})
    first = ex.classify(OK_STATIC, buggy, fixed)
    assert first["class"] == "semantic_kill" and first["rerun_agrees"] is None
    agreed = ex.classify(OK_STATIC, buggy, fixed, (buggy, fixed))
    assert agreed["rerun_agrees"] is True and agreed["fixed_all_executed_passed_reached"] is True
    flip = ex.classify(OK_STATIC, buggy, fixed, (report({"t": node()}), fixed))
    assert flip["class"] == "nondeterminism"
    assert flip["fixed_all_executed_passed_reached"] is False


def test_whole_module_rule_rejects_mixed_skip_and_partial_reach():
    mixed_b = report({"a": node("failed", exc=ASSERT), "s": node("skipped", setup="skipped")})
    mixed_f = report({"a": node(), "s": node("skipped", setup="skipped")})
    assert ex.classify(OK_STATIC, mixed_b, mixed_f)["class"] == "skipped_or_xfail"
    partial_b = report({"a": node("failed", exc=ASSERT), "b": node(reached=False)})
    partial_f = report({"a": node(), "b": node(reached=False)})
    assert ex.classify(OK_STATIC, partial_b, partial_f)["class"] == "target_not_reached"


def test_revision_attestation_is_checked_by_the_harness():
    buggy, fixed = report({"t": node()}, sha="b"), report({"t": node()}, sha="f")
    assert ex.classify(OK_STATIC, buggy, fixed, None, {"buggy": "b", "fixed": "f"})["class"] == "pass_both"
    swapped = ex.classify(OK_STATIC, buggy, fixed, None, {"buggy": "f", "fixed": "f"})
    assert swapped["class"] == "harness_failure"


@pytest.mark.parametrize("source, violation", [
    ("def test_a():\n    open('x')\n", "name:open"),
    ("import inspect\n", "import:inspect"),
    ("from importlib import util\n", "import:importlib"),
    ("import os.path\n", "import:os.path"),
    ("def test_a(f):\n    f.__code__\n", "attribute:__code__"),
    ("import toy\nX = toy.__file__\n", "attribute:__file__"),
    ("def test_a():\n    eval('1')\n", "name:eval"),
    ("import subprocess\n", "import:subprocess"),
])
def test_candidate_policy_refuses_introspection_and_dangerous_calls(source, violation):
    result = ex.static_check(source, "f")
    assert result["status"] == "policy_refused" and violation in result["violations"]


def test_static_admission_limits_and_clean_candidates():
    assert ex.static_check("def test_(:\n", "f")["status"] == "syntax_failure"
    many = "".join(f"def test_{i}():\n    assert f(1)\n" for i in range(30))
    result = ex.static_check(many, "f")
    assert result["status"] == "over_limits" and "tests" in result["over"]
    assert ex.static_check("import pytest\nfrom pkg.m import f\n\ndef test_a():\n"
                           "    assert f(1) == 2\n", "f")["status"] == "ok"


def test_model_failures_are_never_infrastructure():
    model_caused = {"syntax_failure", "fabricated_import", "collection_failure",
                    "no_tests_collected", "skipped_or_xfail", "target_not_reached",
                    "fixed_side_failure", "timeout", "over_limits", "policy_refused"}
    assert not model_caused & set(ex.INFRA)


def test_sandbox_mounts_one_canonical_target_and_enforces_controls():
    text = (ROOT / "scripts" / "native_sandbox_inner.sh").read_text(encoding="utf-8")
    for control in ("--reuid=65534", "--no-new-privs", "--bounding-set=-all", "prlimit",
                    "--nproc", "--fsize", "--as", "timeout -k", "env -i",
                    "PYTHONDONTWRITEBYTECODE=1", "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1",
                    "remount,bind,ro", "for d in /root /home /mnt /tmp", "mount --bind \"$STAGE/target\" /target"):
        assert control in text, control
    spec_b = ex.spec_for("pkg.mod", "Cls.f")
    assert spec_b == ex.spec_for("pkg.mod", "Cls.f")            # identical for both revisions
    assert not any("buggy" in str(v) or "fixed" in str(v) or "/" in str(v)
                   for v in spec_b.values())


def test_views_remove_tests_metadata_and_keep_packages(tmp_path):
    checkout = tmp_path / "co"
    for rel in ("src/pkg/__init__.py", "src/pkg/core.py", "src/pkg/conftest.py",
                "src/pkg/test_x.py", "src/pkg/x_test.py", "src/pkg/tests/t.py",
                "src/pkg/sub/__init__.py", "src/pkg/sub/testing/h.py", "src/pkg/.gitkeep",
                "src/pkg/data.json", "tests/test_official.py", "setup.py"):
        (checkout / rel).parent.mkdir(parents=True, exist_ok=True)
        (checkout / rel).write_text("x = 1\n")
    view = tmp_path / "view"
    manifest = ex.build_view(checkout, "src", view, {"pkg/_version.py": b"V = 1\n"})
    files = sorted(p.relative_to(view).as_posix() for p in view.rglob("*") if p.is_file())
    assert files == ["pkg/__init__.py", "pkg/_version.py", "pkg/core.py", "pkg/data.json",
                     "pkg/sub/__init__.py"]
    assert manifest["files"] == 5 and ex.build_view_hash(view) == manifest["manifest_sha256"]
