"""Native generated-test prompt builder and leakage scanner (protocol v2)."""
from __future__ import annotations

import hashlib
import inspect
import textwrap

import pytest

from harness import native_generated_test_leakage as leak
from harness import native_generated_test_prompt as prompt

BUGGY = textwrap.dedent('''
    """Utilities."""
    import math
    from typing import List

    LIMIT = 10

    class Box:
        """A small box."""
        SIZE = 3

        def __init__(self, items: List[int]):
            self.items = items

        def total(self, scale: int = 1) -> int:
            """Sum the items, scaled."""
            return sum(self.items) + scale

        def helper(self):
            return 1

    def clamp(x: int) -> int:
        """Clamp to LIMIT."""
        return x if x < LIMIT else LIMIT - 1
''')
FIXED = BUGGY.replace("return sum(self.items) + scale", "return sum(self.items) * scale") \
             .replace("return x if x < LIMIT else LIMIT - 1", "return x if x < LIMIT else LIMIT")
DTO = {"target_key": "cand:o/r@abc", "repository": "o/r", "buggy_commit": "abc",
       "target_file": "src/pkg/util.py", "qualname": "Box.total"}


def test_dto_is_allowlisted_and_extra_fields_refuse():
    with pytest.raises(prompt.PromptRefused, match="extra"):
        prompt.build_prompt({**DTO, "fixed_commit": "def"}, BUGGY)
    with pytest.raises(prompt.PromptRefused, match="missing"):
        prompt.build_prompt({k: v for k, v in DTO.items() if k != "qualname"}, BUGGY)
    params = inspect.signature(prompt.build_prompt).parameters
    assert list(params) == ["dto", "buggy_source"]


def test_method_prompt_contains_only_the_permitted_buggy_view():
    sealed = prompt.build_prompt(DTO, BUGGY)
    text = sealed["prompt"]
    assert "return sum(self.items) + scale" in text          # buggy target body
    assert "from pkg.util import Box" in text                 # src/ stripped import path
    assert "def helper(self):" in text and "return 1" not in text   # other bodies elided
    assert "LIMIT = 10" in text and "import math" in text
    assert "return x if x < LIMIT" not in text                # other functions not included
    assert "Expected test format: pytest_fragment" in text
    assert sealed["prompt_sha256"] == hashlib.sha256(text.encode()).hexdigest()
    assert sealed["condition"] == "whole_module"


def test_function_prompt_and_deterministic_seal():
    dto = {**DTO, "qualname": "clamp"}
    first, second = prompt.build_prompt(dto, BUGGY), prompt.build_prompt(dto, BUGGY)
    assert first == second
    assert "Clamp to LIMIT." in first["prompt"]
    with pytest.raises(prompt.PromptRefused, match="not found"):
        prompt.build_prompt({**DTO, "qualname": "missing"}, BUGGY)


def test_nested_class_targets_are_supported():
    source = BUGGY + textwrap.dedent('''
        class URL:
            class Memo:
                def _gen(self, relative: bool):
                    return relative
    ''')
    sealed = prompt.build_prompt({**DTO, "qualname": "URL.Memo._gen"}, source)
    assert "from pkg.util import URL" in sealed["prompt"]
    assert "return relative" in sealed["prompt"]
    with pytest.raises(prompt.PromptRefused, match="class 'Nope'"):
        prompt.build_prompt({**DTO, "qualname": "Nope.x"}, source)


def test_scaffolded_diagnostic_was_removed_by_the_amendment():
    assert not hasattr(prompt, "build_scaffold_prompt")
    assert not hasattr(prompt, "compose_scaffolded")


def _verifier(**extra):
    return {"buggy_source": BUGGY, "fixed_source": FIXED, "patch": "", "official_tests": [],
            "issue_text": "", "commit_message": "", **extra}


def test_clean_prompt_passes_the_scan():
    sealed = prompt.build_prompt(DTO, BUGGY)
    assert leak.scan(sealed, _verifier())["ok"] is True


@pytest.mark.parametrize("injected, verifier_extra, reason", [
    ("return sum(self.items) * scale", {}, "fixed_line"),
    ("assert Box([1, 2, 3]).total(2) == 12", {"official_tests": [
        "from pkg.util import Box\ndef test_total():\n    assert Box([1, 2, 3]).total(2) == 12\n"]},
     "official_test_line"),
    ("the value 'expected-sentinel' matters", {"official_tests": [
        "def test_x():\n    assert f() == 'expected-sentinel'\n"]}, "expected_literal"),
    ("Total should multiply by the scale factor rather than add it to the sum.",
     {"issue_text": "Total should multiply by the scale factor rather than add it to the sum."},
     "issue_text"),
    ("Fix Box.total to multiply by scale", {"commit_message": "Fix Box.total to multiply by scale"},
     "commit_message"),
    ("return sum(self.items) * scale", {"patch": "+        return sum(self.items) * scale\n"},
     "patch_line"),
])
def test_leakage_canaries_are_refused(injected, verifier_extra, reason):
    sealed = prompt.build_prompt(DTO, BUGGY)
    tampered = dict(sealed, prompt=sealed["prompt"] + "\n" + injected + "\n")
    tampered["prompt_sha256"] = hashlib.sha256(tampered["prompt"].encode()).hexdigest()
    result = leak.scan(tampered, _verifier(**verifier_extra))
    assert result["ok"] is False and any(r.startswith(reason) for r in result["reasons"])


def test_unsealed_prompt_is_refused():
    sealed = prompt.build_prompt(DTO, BUGGY)
    assert leak.scan(dict(sealed, prompt=sealed["prompt"] + "x"), _verifier())["ok"] is False


def test_prompt_module_never_touches_verifier_material():
    source = inspect.getsource(prompt)
    for forbidden in ("open(", "read_text", "git", "fixed_commit", "official", "sidecar",
                      "manifest"):
        assert forbidden not in source.split('"""', 2)[2], forbidden
