"""v2.6 generic train-side fixes: F2 target ordering, R1/R2 selector normalisation, the
failure-funnel categories (no per-record rules)."""
from __future__ import annotations

from scripts import v26_failure_funnel as ff
from scripts.v25_native_targets_r2 import normalise_selectors
from scripts.v26_funnel_diag_wsl import changed_functions
from scripts.v26_verify_repository_wsl import ordered_targets

BUGGY = '''import re


def js_to_json(code):
    def fix_kv(m):
        return m.group(0)
    return re.sub(r"x", fix_kv, code)


class A:
    @property
    def p(self):
        return 1
'''
FIXED = BUGGY.replace("import re", "import re, json").replace("return m.group(0)",
                                                             "return m.group(1)") \
    .replace('r"x"', 'r"y"')


def test_every_changed_function_is_a_target_and_outer_precedes_inner():
    assert changed_functions(BUGGY, FIXED) == ["js_to_json", "js_to_json.fix_kv"]
    assert ordered_targets(BUGGY, FIXED) == ["js_to_json", "js_to_json.fix_kv"]


def test_a_module_level_only_change_has_no_target():
    assert ordered_targets("X = 1\n\ndef f():\n    return 1\n",
                           "X = 2\n\ndef f():\n    return 1\n") == []


def test_decorated_method_change_is_found():
    fixed = BUGGY.replace("        return 1", "        return 2")
    assert changed_functions(BUGGY, fixed) == ["A.p"]


def test_selector_normalisation_r1_r2():
    patch = ("diff --git a/pkg/tests/test_a.py b/pkg/tests/test_a.py\n+def test_one():\n"
             "diff --git a/pkg/tests/test_b.py b/pkg/tests/test_b.py\n+def test_two():\n"
             "+def test_one():\n")
    out = normalise_selectors(["django:a.B.test_x.test_x", "django:a.B.test_y",
                               "unresolved:test_two", "unresolved:test_one",
                               "unresolved:test_none"], patch)
    assert out == ["django:a.B.test_x", "django:a.B.test_y", "pkg/tests/test_b.py::test_two",
                   "unresolved:test_one", "unresolved:test_none"]     # ambiguous stays


def _diag(cls, **kw):
    ev = {"missing_modules": [], "cannot_import": [], "exceptions": [], "error_head": None}
    ev.update(kw.pop("ev", {}))
    return {"stage": "executed", "class": cls, "project": kw.pop("project", "p"),
            "evidence": {"buggy": ev}, **kw}


def test_funnel_categories_are_rule_based():
    assert ff.categorise(_diag("fabricated_import", ev={"missing_modules": ["django.test"]}))[
        0] == "import:runtime_test_package_stripped_by_view"
    assert ff.categorise(_diag("fabricated_import", ev={"missing_modules": ["tests"]}))[
        0] == "import:project_test_support_required"
    assert ff.categorise(_diag("target_not_reached", localized="f.g",
                               changed_functions=["f", "f.g"]))[0] == \
        "reach:nested_localization_outer_also_changed"
    assert ff.categorise({"stage": "localization", "changed_functions": ["f"]})[1] == "harness"
    assert ff.categorise({"stage": "localization", "changed_functions": []})[1] == "candidate"
    assert ff.categorise(_diag("crash_kill"))[0] == "positive:crash_kill"
    assert ff.categorise(_diag("semantic_kill"))[0] == "positive:semantic_kill"
    assert ff.runtime_test_package("django.test") and not ff.runtime_test_package("tests.utils")
