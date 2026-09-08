"""Pin the prompt-borne-answer audit.

46% of ablation_dev humaneval records state, in their own prompt, an input and
the correct output for it that the mutant does not produce. Asserting the
stated pair kills the mutant without determining anything, and 44-46% of
humaneval kills on that panel are exactly that assertion.

These are claims about the corpus we report, so the parsing they rest on has
to be pinned. A parser that silently missed examples would understate the
leakage and make the benchmark look cleaner than it is.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_native_example_leakage import audit, examples_in
from scripts.measure_example_copying import _assertion_parts, measure
from scripts.measure_value_prediction_headroom import call_expression
from scripts.synthesize_worked_examples import parameters_of


def test_doctest_examples_are_parsed():
    specification = "Return a thing.\n>>> rounded_avg(1, 5)\n'0b11'\n"
    found = examples_in(specification, "rounded_avg")
    assert found == [{"call": "rounded_avg(1, 5)", "output": "'0b11'"}]


def test_arrow_examples_are_parsed():
    specification = 'For example:\nexchange([1, 2], [3, 4]) => "YES"'
    found = examples_in(specification, "exchange")
    assert found and found[0]["output"] == '"YES"'


def test_an_example_for_a_different_function_is_ignored():
    assert examples_in(">>> helper(1)\n2\n", "target") == []


def test_unparseable_output_is_not_treated_as_an_example():
    """Prose after >>> is not a stated value and must not be counted as one."""
    assert examples_in(">>> f(1)\nreturns the answer\n", "f") == []


def test_an_equality_assertion_is_decomposed():
    assert _assertion_parts("assert f(1, 2) == 'x'", "f") == ("f(1, 2)", "'x'")


def test_a_reversed_equality_assertion_is_decomposed():
    assert _assertion_parts("assert 'x' == f(1, 2)", "f") == ("f(1, 2)", "'x'")


def test_a_non_equality_assertion_is_not_copying():
    """assert f(x) > 3 states no value and cannot be copied from a prompt."""
    assert _assertion_parts("assert f(1) > 3", "f") is None


def test_a_call_is_extracted_for_the_headroom_probe():
    assert call_expression("assert f([1, 2]) == 3", "f") == "f([1, 2])"


def test_parameters_come_from_the_reference_signature():
    assert parameters_of("def f(a, b):\n    return a\n", "f") == ["a", "b"]


def test_varargs_are_refused_rather_than_guessed():
    assert parameters_of("def f(*args):\n    return args\n", "f") is None


def _artifact(tmp_path: Path, sealed: bool = False, split: str = "val") -> Path:
    path = tmp_path / "artifact.json"
    path.write_text(json.dumps({
        "evaluation_split": split,
        "final_test_measurement": sealed,
        "function_results": [],
    }), encoding="utf-8")
    return path


def test_the_copying_measurement_refuses_sealed_data(tmp_path):
    with pytest.raises(SystemExit):
        measure(_artifact(tmp_path, sealed=True), tmp_path, "humaneval")


def test_the_copying_measurement_refuses_the_test_split(tmp_path):
    with pytest.raises(SystemExit):
        measure(_artifact(tmp_path, split="test"), tmp_path, "humaneval")


def test_the_native_audit_refuses_the_sealed_split(tmp_path):
    with pytest.raises(SystemExit):
        audit(tmp_path, "test", "humaneval", 1.0, 1)


REPORT = ROOT / "results" / "v4_2_native_example_leakage_adev.json"
COPYING = ROOT / "results" / "v4_2_example_copying_relearn_val.json"


def test_the_committed_leakage_finding_still_holds():
    """The caveat every humaneval number now carries."""
    if not REPORT.exists():
        return
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    assert report["records_whose_prompt_states_a_killing_value"] > 0, (
        "humaneval prompts no longer state killing values; every humaneval "
        "comparison in the writeup depends on this number and it must be "
        "re-derived rather than assumed unchanged"
    )
    assert report["sealed_final_test_accessed"] is False


def test_copying_is_reported_as_a_floor_not_a_ceiling():
    if not COPYING.exists():
        return
    report = json.loads(COPYING.read_text(encoding="utf-8"))
    assert "floor, not a ceiling" in report["interpretation"]
    assert report["killed_by_asserting_a_stated_example"] <= \
        report["killed_functions_whose_prompt_states_an_example"]
