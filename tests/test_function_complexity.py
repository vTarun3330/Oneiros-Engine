import pytest

from harness.function_complexity import (
    COMPLEXITY_POLICY_VERSION,
    analyze_function_complexity,
)


def test_simple_function_is_classified_from_buggy_source() -> None:
    result = analyze_function_complexity("def add(a, b):\n    return a + b\n", "add")

    assert result.policy_version == COMPLEXITY_POLICY_VERSION
    assert result.tier == "simple"
    assert result.cyclomatic_complexity == 1
    assert result.parameter_count == 2


def test_nested_branching_function_is_complex() -> None:
    source = """
def route(values, fallback=None):
    total = 0
    for value in values:
        if value is not None:
            while value > 0:
                if value % 2 == 0 and value > 4:
                    total += value
                value -= 1
    return total if total else fallback
"""

    result = analyze_function_complexity(source, "route")

    assert result.tier == "complex"
    assert result.cyclomatic_complexity >= 6
    assert result.max_control_nesting >= 4


def test_docstring_does_not_inflate_logical_line_count() -> None:
    source = '''
def identity(value):
    """A long
    multi-line
    explanation.
    """
    return value
'''
    result = analyze_function_complexity(source, "identity")

    assert result.logical_lines == 2


def test_missing_entry_point_fails_closed() -> None:
    with pytest.raises(ValueError, match="not defined"):
        analyze_function_complexity("def present():\n    return 1\n", "missing")
