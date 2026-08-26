import time

from harness.safe_execution import classify_assertions, execute_code


def test_executes_in_worker_and_returns_json_safe_result():
    ok, result, error = execute_code(
        "def add(a, b): return a + b",
        "result = add(2, 3)",
    )
    assert ok is True
    assert result == 5
    assert error == ""


def test_classifies_reference_valid_mutant_kill():
    row = classify_assertions(
        ["assert add(2, 3) == 5"],
        "def add(a, b): return a + b",
        "def add(a, b): return a - b",
    )[0]
    assert row["valid"] is True
    assert row["killed"] is True


def test_reference_failure_is_invalid_not_a_kill():
    row = classify_assertions(
        ["assert add(2, 3) == 99"],
        "def add(a, b): return a + b",
        "def add(a, b): return a - b",
    )[0]
    assert row["valid"] is False
    assert row["killed"] is False


def test_parent_stops_infinite_loop():
    started = time.perf_counter()
    ok, _, error = execute_code(
        "def hang():\n    while True:\n        pass",
        "result = hang()",
        timeout_seconds=0.1,
    )
    assert ok is False
    assert "TIMEOUT" in error
    assert time.perf_counter() - started < 3.0


def test_restricted_builtins_block_file_access():
    ok, _, error = execute_code(
        "def read_secret(): return open('secret.txt').read()",
        "result = read_secret()",
    )
    assert ok is False
    assert "source_call_not_allowed:open" in error


def test_source_policy_blocks_import_and_module_internal_escape():
    ok, _, error = execute_code(
        "import os\ndef dangerous(): return os.getcwd()",
        "result = dangerous()",
    )
    assert ok is False
    assert "source_import_not_allowed:os" in error

    ok, _, error = execute_code(
        "import random\ndef dangerous(): return random._os.getcwd()",
        "result = dangerous()",
    )
    assert ok is False
    assert "source_attribute_not_allowed:_os" in error


def test_source_policy_allows_whitelisted_from_imports():
    ok, result, error = execute_code(
        "from typing import List\ndef total(values: List[int]): return sum(values)",
        "result = total([1, 2, 3])",
    )
    assert (ok, result, error) == (True, 6, "")
