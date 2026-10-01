"""v2.5 deterministic conversion of verified completions into pytest_module_v1."""
from __future__ import annotations

import ast

import pytest

from scripts import v25_module_conversion as mc

REF = "import math\n\ndef helper(x):\n    return x\n\ndef area(r):\n    return math.pi * r * r\n"
CUT = "import math\n\ndef helper(x):\n    return x\n\ndef area(r):\n    return math.pi * r\n"


def test_assertion_becomes_one_test_function_with_interface_imports():
    out = mc.convert_synthetic("assert area(2) == 4 * math.pi", "area", REF, CUT)
    assert out["accepted"]
    assert out["module"] == ("import math\nfrom oneiros_target import area\n\n\n"
                             "def test_area_defect():\n    assert area(2) == 4 * math.pi\n")
    tree = ast.parse(out["module"])
    assert sum(isinstance(n, ast.FunctionDef) for n in tree.body) == 1
    assert "```" not in out["module"] and "def area" not in out["module"]


def test_helpers_defined_in_both_revisions_are_imported():
    out = mc.convert_synthetic("assert area(helper(1)) == math.pi", "area", REF, CUT)
    assert "from oneiros_target import area, helper" in out["module"]


@pytest.mark.parametrize("completion, reason", [
    ("assert area(2) == np.pi", "unresolvable_name:np"),
    ("assert helper(1) == 1", "target_not_referenced"),
    ("def test_x():\n    assert area(1)", "not_an_assertion_only_completion"),
    ("assert area(", "completion_does_not_parse"),
    ("assert area(1) == json.loads('1')", "unresolvable_name:json"),
])
def test_unresolvable_completions_are_rejected_not_guessed(completion, reason):
    out = mc.convert_synthetic(completion, "area", REF, CUT)
    assert out == {"accepted": False, "reason": reason}


def test_conversion_is_deterministic():
    a = mc.convert_synthetic("assert area(2) == 4 * math.pi", "area", REF, CUT)
    assert a == mc.convert_synthetic("assert area(2) == 4 * math.pi", "area", REF, CUT)


def test_repository_fragment_gains_only_the_public_header():
    ctx = ("Test framework: pytest.\nNon-gold test-module environment:\n"
           "from __future__ import annotations\n\nimport pytest\n"
           "from pkg.mod import (\n    Thing,\n)\n\nOther text here.\n")
    out = mc.convert_repository("def test_t():\n    assert Thing()\n", ctx)
    assert out["accepted"] and out["verification"] == "pending_native_environment"
    assert out["module"].startswith("from __future__ import annotations\nimport pytest\n"
                                    "from pkg.mod import (\n    Thing,\n)")
    ast.parse(out["module"])
    assert mc.convert_repository("def test_t():\n    pass\n", "no header")["reason"] \
        == "no_public_import_header"


def _ctx(header: str) -> str:
    return f"Test framework: pytest.\nNon-gold test-module environment:\n{header}\n\nProse.\n"


def test_missing_pytest_is_inserted_after_future_imports():
    ctx = _ctx("from __future__ import annotations\nfrom pkg.mod import Thing")
    out = mc.convert_repository("def test_t():\n    with pytest.raises(ValueError):\n"
                                "        Thing()\n", ctx)
    assert out["accepted"] and out["inserted_imports"] == ["import pytest"]
    assert out["module"].startswith("from __future__ import annotations\nimport pytest\n"
                                    "from pkg.mod import Thing\n")
    assert "import pytest" in out["imports_added"]
    ast.parse(out["module"])


def test_multiline_header_imports_are_kept_whole():
    ctx = _ctx("from pkg.mod import (\n    A,\n    B,\n)\nimport os")
    out = mc.convert_repository("def test_t():\n    assert A() != B()\n", ctx)
    assert out["accepted"] and "from pkg.mod import (\n    A,\n    B,\n)" in out["module"]
    assert out["inserted_imports"] == []


def test_unittest_fragment_gets_unittest_only_when_unbound():
    frag = "class TCase(unittest.TestCase):\n    def test_a(self):\n        self.assertTrue(1)\n"
    assert mc.convert_repository(frag, _ctx("from pkg.mod import A"))["inserted_imports"] \
        == ["import unittest"]
    bound = mc.convert_repository(frag, _ctx("from units.compat import unittest"))
    assert bound["inserted_imports"] == []


@pytest.mark.parametrize("fragment, ctx, reason", [
    ("def test_t():\n    pass\n", _ctx("from pkg import (\n"), "no_public_import_header"),
    ("def helper():\n    return 1\n", _ctx("from pkg import A"), "no_test_defined"),
    ("def test_t(:\n", _ctx("from pkg import A"), "fragment_does_not_parse"),
])
def test_malformed_or_testless_inputs_are_rejected(fragment, ctx, reason):
    assert mc.convert_repository(fragment, ctx)["reason"] == reason


@pytest.mark.parametrize("fragment", [
    "class FinderTests(SimpleTestCase):\n    def test_find(self):\n        self.assertTrue(A)\n",
    "class X(unittest.TestCase):\n    def test_a(self):\n        self.assertTrue(A)\n",
    "class TestThing:\n    def test_a(self):\n        assert A\n",
])
def test_test_classes_imported_by_name_or_attribute_are_recognised(fragment):
    out = mc.convert_repository(fragment, _ctx("import unittest\nfrom django.test import "
                                               "SimpleTestCase\nfrom pkg import A"))
    assert out["accepted"], out


def test_a_class_without_test_methods_is_not_a_test():
    out = mc.convert_repository("class Helper(SimpleTestCase):\n    def setUp(self):\n"
                                "        pass\n", _ctx("from django.test import SimpleTestCase"))
    assert out["reason"] == "no_test_defined"


def test_fabricated_header_import_stays_pending_never_verified_here():
    out = mc.convert_repository("def test_t():\n    assert Ghost()\n",
                                _ctx("from pkg.nowhere import Ghost"))
    assert out["accepted"] and out["verification"] == "pending_native_environment"


def test_forbidden_operations_are_refused_by_the_candidate_policy():
    from scripts import v25_converted_corpus as cc
    out = mc.convert_repository("def test_t():\n    os.system('id')\n    assert A()\n",
                                _ctx("from pkg.mod import A"))
    assert out["accepted"] and out["inserted_imports"] == ["import os"]
    assert cc.policy_status(out["module"], "A") == "policy_refused"
    ok = mc.convert_synthetic("assert area(2) == 4 * math.pi", "area", REF, CUT)
    assert cc.policy_status(ok["module"], "area") == "ok"
