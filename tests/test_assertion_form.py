"""Assertion forms must be told apart, especially the value-free ones.

This classifier exists to stop a prompt change being credited with a mechanism
it did not use. It only does that job if "needs an exact value" and "does not"
are actually separated - so the cases below are chosen where a careless
classifier would collapse them.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.classify_assertion_form import FORMS, build, classify

ROOT = Path(__file__).resolve().parent.parent
CONTROL = ROOT / "results" / "v4_2_assertion_form_control.json"


@pytest.mark.parametrize("code,expected", [
    ("assert f(1, 2) == 3", "literal_oracle"),
    ("assert f(f(3)) == f(3)", "relational"),
    ("assert f([1, 2]) == f([2, 1])", "relational"),
    ("assert len(f([1, 2])) == 2", "property"),
    ("assert type(f(1)) == type(1)", "property"),
    ("assert f(3) > 0", "bounded"),
    ("assert f(3) != None", "bounded"),
    ("assert f(", "other"),
    ("x = 1", "other"),
])
def test_forms_are_classified(code, expected):
    assert classify(code, "f") == expected


def test_an_inequality_against_a_constant_is_not_an_exact_oracle():
    """The distinction the whole measurement turns on.

    `f(3) > 0` mentions a constant but never requires knowing which positive
    number f(3) is. Bucketing it with `f(3) == 7` would hide exactly the
    value-free behaviour being looked for.
    """
    assert classify("assert f(3) > 0", "f") != classify("assert f(3) == 7", "f")
    assert classify("assert f(3) > 0", "f") == "bounded"


def test_the_most_informative_assertion_decides_a_multi_assertion_output():
    code = "assert f(1) == 2\nassert f(f(1)) == f(1)"
    assert classify(code, "f") == "relational"


def test_a_call_to_a_different_function_is_not_a_relation():
    """Two calls, one of them not the entry point, is not a relation."""
    assert classify("assert f(1) == g(g(1))", "f") == "other"


def test_every_form_is_declared():
    produced = {classify(code, "f") for code in (
        "assert f(1) == 2", "assert f(f(1)) == f(1)", "assert len(f(1)) == 2",
        "assert f(1) > 0", "x = 1")}
    assert produced <= set(FORMS)


def test_build_refuses_sealed_artifacts(tmp_path):
    artifact = tmp_path / "sealed.json"
    artifact.write_text(json.dumps(
        {"evaluation_split": "test", "function_results": []}), encoding="utf-8")
    with pytest.raises(SystemExit, match="sealed"):
        build(artifact, ROOT / "data" / "corpus" / "v4_1_research_hardened_candidate")


@pytest.mark.skipif(not CONTROL.exists(), reason="control not built here")
def test_the_control_shows_the_base_model_almost_never_writes_relations():
    """The premise of the intervention, pinned.

    If the base model already wrote relations, a metamorphic prompt would have
    nothing to add and the measured gain would need another explanation.
    """
    report = json.loads(CONTROL.read_text(encoding="utf-8"))["reports"][0]
    shares = report["form_shares"]
    assert shares["relational"] < 0.01
    assert shares["literal_oracle"] > 0.5
