"""v2.5 pytest_module_v1 prompt: same information as v2.4, only the output contract changes."""
from __future__ import annotations

from harness import native_generated_test_prompt as v24
from harness import native_generated_test_prompt_v25 as v25

BUGGY = '''"""Mod."""
import math

LIMIT = 10


def helper(x):
    return x + 1


def area(r: float) -> float:
    """Area of a circle of radius r."""
    if r > LIMIT:
        return helper(r)
    return math.pi * r
'''
DTO = {"target_key": "cand:o/r@abc", "repository": "o/r", "buggy_commit": "abc",
       "target_file": "src/pkg/geo.py", "qualname": "area"}


def test_label_and_instruction_are_pytest_module_v1():
    p = v25.build_prompt(DTO, BUGGY)["prompt"]
    assert "Expected test format: pytest_module_v1" in p and p.endswith("pytest_module_v1")
    assert "pytest_fragment" not in p and "assert_statement" not in p
    assert "Import the target with: `from pkg.geo import area`" in p
    assert "from pkg.geo import area\n\n\ndef test_area():\n    ..." in p


def test_same_model_visible_information_as_v24():
    old = v24.build_prompt(DTO, BUGGY)["prompt"]
    new = v25.build_prompt(DTO, BUGGY)["prompt"]
    for section in ("### Behavioral specification", "### Available execution context",
                    "### Code under test"):
        a = old.split(section, 1)[1].split("###", 1)[0]
        b = new.split(section, 1)[1].split("###", 1)[0]
        assert a == b, section


def test_skeleton_carries_no_oracle_or_reference():
    p = v25.build_prompt(DTO, BUGGY)
    body = p["prompt"].split("Module structure", 1)[1].split("Output only", 1)[0]
    assert "assert" not in body and "==" not in body and "math.pi" not in body
    assert p["builder_version"].endswith("pytest_module_v1")
