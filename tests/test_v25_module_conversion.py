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
