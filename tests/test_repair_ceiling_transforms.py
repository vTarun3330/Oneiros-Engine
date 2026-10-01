"""The post-hoc mechanical-repair transforms: deterministic, idempotent, outcome-free,
ambiguity-refusing, and never touching the original artifacts."""
from __future__ import annotations

import ast
import inspect
import json
from pathlib import Path

import pytest

from scripts import repair_ceiling_transforms as rt

PROMPT = ("### Available execution context\n\nModule: `pkg.mod`\n\nTarget: `Box.f`\n\n"
          "Import the target with: `from pkg.mod import Box`\n\nMore text\n")
FENCED = "```python\nimport pytest\n\ndef test_x():\n    assert Box().f() == 1\n```\nThis test shows it.\n"


def test_r1_extracts_exactly_one_python_fence():
    out, why = rt.r1_single_fence(FENCED)
    assert why is None and out == "import pytest\n\ndef test_x():\n    assert Box().f() == 1\n"
    ast.parse(out)


@pytest.mark.parametrize("module, reason", [
    ("def test_a():\n    assert 1\n", "no_fence"),
    ("```python\ndef test_a():\n    assert 1\n", "unclosed_fence"),
    ("```python\na = 1\n```\n\n```python\nb = 2\n```\n", "multiple_or_unbalanced_fences"),
    ("```bash\npip install x\n```\n", "non_python_fence"),
])
def test_r1_refuses_ambiguity_and_leaves_the_module_unchanged(module, reason):
    out, why = rt.r1_single_fence(module)
    assert out == module and why.startswith(reason)


def test_r2_adds_pytest_import_only_when_referenced_and_missing():
    mod = "def test_a():\n    with pytest.raises(ValueError):\n        f()\n"
    out, why = rt.r2_pytest_import(mod)
    assert why is None and out == "import pytest\n" + mod
    assert rt.r2_pytest_import("import pytest\n" + mod) == ("import pytest\n" + mod,
                                                            "already_imported")
    assert rt.r2_pytest_import("def test_a():\n    assert 1\n")[1] == "pytest_not_referenced"


def test_r2_respects_leading_future_imports():
    mod = "from __future__ import annotations\n\ndef test_a():\n    pytest.fail()\n"
    out, _ = rt.r2_pytest_import(mod)
    assert out.startswith("from __future__ import annotations\nimport pytest\n")
    ast.parse(out)


def test_r3_uses_only_the_prompt_import_line():
    mod = "def test_a():\n    assert Box().f() == 1\n"
    out, why = rt.r3_target_import(mod, PROMPT)
    assert why is None and out == "from pkg.mod import Box\n" + mod
    assert rt.r3_target_import(mod, "no import line here")[1] == "no_target_import_in_prompt"
    already = "from pkg.mod import Box as Box\n" + mod
    assert rt.r3_target_import(already, PROMPT) == (already, "already_imported")


def test_r3_refuses_an_ambiguous_prompt():
    twice = PROMPT + "Import the target with: `from other import Box`\n"
    assert rt.target_import_statement(twice) is None


def test_r4_is_r1_then_r2_then_r3_in_fixed_order():
    module = "```python\ndef test_x():\n    with pytest.raises(KeyError):\n        Box().f()\n```\nWhy.\n"
    out = rt.apply("R4", module, PROMPT)
    assert out["applied"] == ["R1", "R2", "R3"]
    assert out["module"].startswith("from pkg.mod import Box\nimport pytest\ndef test_x")
    ast.parse(out["module"])


@pytest.mark.parametrize("condition", rt.CONDITIONS)
@pytest.mark.parametrize("module", [FENCED, "def test_a():\n    pytest.fail()\n",
                                    "assert Box().f() == 2\n", "```python\nbroken(\n"])
def test_every_condition_is_deterministic_and_idempotent(condition, module):
    first = rt.apply(condition, module, PROMPT)
    assert rt.apply(condition, module, PROMPT) == first
    again = rt.apply(condition, first["module"], PROMPT)
    assert again["module"] == first["module"] and again["changed"] is False


def test_r0_is_the_identity():
    out = rt.apply("R0", FENCED, PROMPT)
    assert out["module"] == FENCED and not out["changed"] and out["applied"] == []


def test_transforms_take_no_outcome_or_protected_input():
    for fn in (rt.apply, rt.r1_single_fence, rt.r2_pytest_import, rt.r3_target_import):
        params = set(inspect.signature(fn).parameters)
        assert params <= {"condition", "module", "prompt"}, (fn.__name__, params)
    tree = ast.parse(Path(rt.__file__).read_text(encoding="utf-8"))
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | \
        {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    imported = {a.name for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))
                for a in n.names}
    assert not names & {"open", "read_text", "read_bytes", "Path", "subprocess", "fixed",
                        "patch", "classification", "outcome"}
    assert imported <= {"annotations", "ast", "hashlib", "re", "Dict", "List", "Optional",
                        "Tuple"}


def test_original_artifacts_are_not_modified(tmp_path):
    original = tmp_path / "generations.jsonl"
    row = {"candidates": [{"module": FENCED}]}
    original.write_text(json.dumps(row) + "\n")
    before = original.read_bytes()
    loaded = json.loads(original.read_text())
    for c in rt.CONDITIONS:
        rt.apply(c, loaded["candidates"][0]["module"], PROMPT)
    assert original.read_bytes() == before and loaded == row


def test_definition_declares_the_exploratory_study():
    d = rt.definition()
    assert d["study"] == "posthoc_exploratory_mechanical_repair_ceiling"
    assert d["primary_metric_replacement"] is False and d["confirmation_claims"] is False
    assert d["conditions"] == ["R0", "R1", "R2", "R3", "R4"]
