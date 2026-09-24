from __future__ import annotations

from scripts.evaluate_execution_supervision_pilot import score_assertion, shown_actual_prompt


def _item():
    return {
        "focused_prompt": (
            "intro\n\n### Function (may be defective)\n\ndef f(x):\n    return x + 1"
            "\n\n### Call\n\nf(2)\n\n### Assertion\n"
        ),
        "call_expression": "f(2)",
    }


def test_actual_prompt_omits_intended_behaviour_and_preserves_call():
    prompt = shown_actual_prompt(_item())
    assert "Intended behaviour" not in prompt
    assert "def f(x):" in prompt
    assert "f(2)" in prompt


def test_strict_scorer_requires_exact_call_and_type():
    expected = {"type": "int", "literal": "3"}
    assert score_assertion("assert f(2) == 3", "f(2)", expected)["verdict"] == "correct"
    assert score_assertion("assert g(2) == 3", "f(2)", expected)["verdict"] == "wrong_call"
    assert score_assertion("assert f(2) == 3.0", "f(2)", expected)["verdict"] == "wrong_type"


def test_strict_and_lenient_format_are_reported_separately():
    raw = "```python\nassert f(2) == 3\n```"
    expected = {"type": "int", "literal": "3"}
    assert score_assertion(raw, "f(2)", expected)["verdict"] == "invalid_syntax"
    assert score_assertion(raw, "f(2)", expected, lenient=True)["verdict"] == "correct"


def test_scorer_never_accepts_extra_statement():
    result = score_assertion(
        "x = f(2)\nassert f(2) == 3", "f(2)", {"type": "int", "literal": "3"}
    )
    assert result["verdict"] == "multiple_or_missing_statements"
