from engine.test_generation_prompt import (
    PROMPT_SCHEMA_VERSION,
    SYSTEM_PROMPT,
    build_unified_user_prompt,
    sanitize_behavioral_specification,
)
from scripts.build_corpus_v4 import (
    _mbpp_target_symbol,
    _normalize_record,
    changed_target_symbols,
    partition_repository_source,
)
from scripts.train_on_dataset import _record_to_pair, build_pair_prompt


def _function_record():
    record = {
        "id": "mutation::mbpp_89_mut_02475",
        "task_type": "hidden_mutation_reproduction",
        "source": {"name": "oneiros_clean_mutations", "upstream": "mbpp"},
        "group_id": "function:test",
        "code_under_test": "def closest_num(n):\n    return n\n",
        "reference_code": "def closest_num(n):\n    return n - 1\n",
        "entry_point": "closest_num",
        "specification": "",
        "tests": [
            {"code": "assert closest_num(11) == 10", "oracle": "stale"},
            {"code": "assert closest_num(0) == closest_num(0)", "oracle": "stale"},
        ],
        "provenance": {"upstream_record_id": "mbpp_89_mut_02475"},
        "quality": {"execution_mode": "function_assertion"},
    }
    return record


def test_unified_prompt_has_the_same_sections_for_function_and_repository():
    function_prompt = build_unified_user_prompt(
        code_under_test="def f(x): return x",
        execution_mode="function_assertion",
        specification="Return the successor.",
        target_symbols=["f"],
    )
    repository_prompt = build_unified_user_prompt(
        code_under_test="def f(x): return x",
        execution_mode="repository_pytest_fragment",
        specification="Return the successor.",
        support_context="import pytest",
        target_symbols=["f"],
    )

    headings = (
        "### TEST GENERATION TASK",
        "### Behavioral specification",
        "### Available execution context",
        "### Code under test",
        "### Task",
        "### Output",
    )
    assert all(heading in function_prompt for heading in headings)
    assert all(heading in repository_prompt for heading in headings)
    assert "Task mode: function" in function_prompt
    assert "Expected test format: assert_statement" in function_prompt
    assert "Task mode: repository" in repository_prompt
    assert "Expected test format: pytest_fragment" in repository_prompt


def test_prompt_contract_cannot_receive_oracle_only_fields():
    prompt = build_unified_user_prompt(
        code_under_test="def f(x): return x",
        execution_mode="function_assertion",
        specification="Return x plus one.",
        target_symbols=["f"],
    )

    assert PROMPT_SCHEMA_VERSION in "oneiros_unified_test_generation_v1"
    assert "reference_code" not in prompt
    assert "mutation_type" not in prompt
    assert "gold_patch" not in prompt
    assert "dataset:" not in prompt.lower()
    assert "reference implementation is intentionally hidden" in SYSTEM_PROMPT


def test_specification_sanitizer_removes_patch_leakage_but_keeps_behavior():
    raw = """Nested models produce an incorrect matrix.
diff --git a/model.py b/model.py
@@ -1 +1 @@
Replace value 1 with right.
Expected output is diagonal for the nested case."""

    cleaned = sanitize_behavioral_specification(raw)

    assert "incorrect matrix" in cleaned
    assert "Expected output" in cleaned
    assert "diff --git" not in cleaned
    assert "Replace value" not in cleaned


def test_v4_mbpp_normalization_restores_description_and_executes_each_oracle():
    normalized = _normalize_record(
        _function_record(), {
            89: {
                "specification": "Find the closest smaller number than n.",
                "entry_point": "closest_num",
            }
        }
    )

    assert normalized["specification"] == "Find the closest smaller number than n."
    assert normalized["task_mode"] == "function"
    assert normalized["test_format"] == "assert_statement"
    assert normalized["target_symbols"] == ["closest_num"]
    assert normalized["quality"]["test_oracle_labels_execution_derived"] is True
    assert normalized["tests"][0]["distinguishing"] is True
    assert normalized["tests"][0]["oracle"] == "passes_reference_fails_target"
    assert normalized["tests"][1]["distinguishing"] is False
    assert normalized["tests"][1]["oracle"] == "passes_reference_passes_target"


def test_mbpp_target_derivation_prefers_outer_public_call_over_constructor_helpers():
    item = {
        "task_id": 601,
        "code": "class Pair:\n    pass\ndef max_chain_length(items, n):\n    return n\n",
        "test_list": [
            "assert max_chain_length([Pair(), Pair(), Pair()], 3) == 3",
        ],
    }

    assert _mbpp_target_symbol(item) == "max_chain_length"


def test_repository_target_is_separated_from_support_context():
    buggy = """# File: pkg/rule.py
import re
CONST = 1
def helper(x):
    return x
def match(command):
    return 'php -s' in command.script
"""
    fixed = buggy.replace("'php -s'", "' -s '")

    targets = changed_target_symbols(buggy, fixed)
    support, target = partition_repository_source(buggy, targets)

    assert targets == ["match"]
    assert "def match" in target
    assert "def helper" not in target
    assert "import re" in support
    assert "def helper" in support


def test_adapted_record_uses_only_model_visible_unified_fields():
    record = _normalize_record(
        _function_record(), {
            89: {
                "specification": "Find the closest smaller number than n.",
                "entry_point": "closest_num",
            }
        }
    )
    pair = _record_to_pair(record)
    prompt = build_pair_prompt(pair)

    assert "Find the closest smaller number than n." in prompt
    assert "def closest_num(n):\n    return n" in prompt
    assert "return n - 1" not in prompt
    assert "mbpp" not in prompt.lower()
