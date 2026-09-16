"""The consumed split must be unreachable, by every path, forever.

Background. The sealed final test was attempted once, on 2026-09-16, and failed
after authorization. Afterwards the suite itself reached the split a second
time: ``sealed_records()`` gated on ``authorization_granted()``, which is
permanently true once a grant is recorded, so an obsolete test walked straight
through it. Authorization answers "was this permitted once". It cannot answer
"is there anything left to measure".

These tests pin the repair. None of them reads the corpus, names an identifier,
or asserts anything about the split's contents - a test that had to look would
be the same defect wearing a different hat.
"""
from __future__ import annotations

import ast
import inspect
import json
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from harness import sealed_final_loader as loader  # noqa: E402
from harness.sealed_final import SEALED_SPLIT, SealedAccessError  # noqa: E402

STATE = ROOT / "results" / "sealed_final_state.json"
RUN_STATE = ROOT / "results" / "sealed_final_run_state.json"
OLD_OUTPUT = ROOT / "results" / "sealed_final_base_qwen_s42"


def _function_record(identifier="synthetic::function::1"):
    return {
        "id": identifier,
        "task_type": "hidden_mutation_reproduction",
        "entry_point": "add_two",
        "reference_code": "def add_two(a, b):\n    return a + b\n",
        "code_under_test": "def add_two(a, b):\n    return a - b\n",
        "specification": "Return the sum.",
        "tests": [{"code": "assert add_two(1, 2) == 3"}],
        "quality": {"execution_mode": "function_assertion"},
    }


# --------------------------------------------------------------------------
# The refusal itself.
# --------------------------------------------------------------------------

def test_sealed_records_refuses_unconditionally():
    with pytest.raises(SealedAccessError, match="consumed"):
        loader.sealed_records()


def test_sealed_records_refuses_before_opening_any_corpus_file(monkeypatch):
    """Proof by sabotage: make every file read explode, then call it.

    If the refusal were not first, one of these would fire instead and the test
    would fail with the wrong exception.
    """
    def explode(*args, **kwargs):
        raise AssertionError("a corpus file was opened on a refused path")

    monkeypatch.setattr(Path, "read_text", explode)
    monkeypatch.setattr(Path, "read_bytes", explode)
    monkeypatch.setattr(json, "loads", explode)

    with pytest.raises(SealedAccessError, match="consumed"):
        loader.sealed_records()


def _function_ast(function):
    """The function's parsed body, with its docstring dropped.

    Parsed rather than grepped. An earlier version of these tests matched
    source text and failed against the module's own docstrings, which is the
    same error as asserting on prose instead of behaviour.
    """
    tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
    body = tree.body[0].body
    if (body and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)):
        body = body[1:]
    return body


def _called_names(nodes):
    """Every function name actually called anywhere in these nodes."""
    called = set()
    for node in nodes:
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                func = child.func
                if isinstance(func, ast.Name):
                    called.add(func.id)
                elif isinstance(func, ast.Attribute):
                    called.add(func.attr)
    return called


def test_the_refusal_is_the_first_statement_executed():
    body = _function_ast(loader.sealed_records)

    first = body[0]
    assert isinstance(first, ast.Expr)
    assert isinstance(first.value, ast.Call)
    assert first.value.func.id == "refuse_consumed_split"


@pytest.mark.parametrize("corpus_version", [
    "v4_1_research_hardened_candidate", "some_other_corpus", "",
])
def test_no_argument_can_reach_the_split(corpus_version):
    with pytest.raises(SealedAccessError, match="consumed"):
        loader.sealed_records(corpus_version)


def test_select_split_records_refuses_the_consumed_split_before_indexing():
    """Refused even when the split mapping does not contain it at all.

    If the name check came first, this would raise SplitSchemaError("absent")
    and the caller would learn something about the split's presence.
    """
    with pytest.raises(SealedAccessError, match="consumed"):
        loader.select_split_records({}, [], SEALED_SPLIT)


def test_select_split_records_refuses_before_iterating_ids():
    """Sabotage the id sequence; the refusal must beat it."""
    class Exploding(list):
        def __iter__(self):
            raise AssertionError("split ids were iterated on a refused path")

    splits = {SEALED_SPLIT: Exploding(["never-read"])}
    with pytest.raises(SealedAccessError, match="consumed"):
        loader.select_split_records(splits, [], SEALED_SPLIT)


def test_permitted_splits_still_work_through_the_same_function():
    """The refusal is targeted, not a blanket disable of split parsing."""
    record = _function_record()
    splits = {"rehearsal": [record["id"]]}

    selected = loader.select_split_records(splits, [record], "rehearsal")

    assert [r["id"] for r in selected] == [record["id"]]


# --------------------------------------------------------------------------
# The refusal does not depend on - and cannot be undone by - guard state.
# --------------------------------------------------------------------------

def test_the_refusal_ignores_guard_state(monkeypatch, tmp_path):
    """True, False or missing: the answer is the same."""
    for granted in (True, False):
        monkeypatch.setattr(loader, "authorization_granted", lambda: granted)
        with pytest.raises(SealedAccessError, match="consumed"):
            loader.sealed_records()

    monkeypatch.setattr(loader, "STATE_PATH", tmp_path / "absent.json")
    with pytest.raises(SealedAccessError, match="consumed"):
        loader.sealed_records()


def test_authorization_granted_still_reports_the_truth():
    """History is not falsified to obtain safety.

    One authorization was granted and spent. The state file says so, and this
    function must keep saying so - the safety guarantee lives elsewhere.
    """
    if not STATE.is_file():
        pytest.skip("guard state absent")

    assert loader.authorization_granted() is True


def test_authorization_is_no_longer_what_gates_access():
    """The gate is the refusal, and the old gate is not called at all."""
    called = _called_names(_function_ast(loader.sealed_records))

    assert "refuse_consumed_split" in called
    assert "_require_authorization" not in called


def test_there_is_no_override_that_re_enables_the_loader():
    """No flag, no keyword, no environment lookup anywhere in the module."""
    signature = inspect.signature(loader.refuse_consumed_split)
    assert list(signature.parameters) == ["split_name"]

    module = ast.parse(
        (ROOT / "harness" / "sealed_final_loader.py").read_text(encoding="utf-8"))
    assert "getenv" not in _called_names(module.body)
    for node in ast.walk(module):
        if isinstance(node, ast.Attribute) and node.attr == "environ":
            raise AssertionError("the module reads the environment")
        if isinstance(node, ast.arg) and node.arg in {
                "force", "override", "allow_consumed", "ignore_consumed"}:
            raise AssertionError(f"a reactivation parameter appeared: {node.arg}")


def test_a_future_final_set_is_directed_elsewhere():
    assert "separately authorized protocol" in loader.CONSUMED_SPLIT_REFUSAL
    assert "SEALED_FINAL_INCIDENT" in loader.CONSUMED_SPLIT_REFUSAL


# --------------------------------------------------------------------------
# The historical record, asserted as history rather than as absence.
# --------------------------------------------------------------------------

def test_the_state_records_exactly_one_consumed_attempt():
    if not STATE.is_file():
        pytest.skip("guard state absent")
    state = json.loads(STATE.read_text(encoding="utf-8"))

    assert len(state.get("spent_tokens", [])) == 1
    assert len(state.get("runs", [])) == 1
    assert len(state.get("issued", [])) == 1


def test_a_run_state_blocks_all_re_execution():
    if not RUN_STATE.is_file():
        pytest.skip("run state absent")
    run = json.loads(RUN_STATE.read_text(encoding="utf-8"))

    assert run["status"] == "failed_after_authorization"
    assert run["authorization_spent"] is True
    assert run["may_never_run_again"] is True
    assert run["result"] == {}


def test_the_old_output_directory_exists_and_is_empty():
    """It was created, then nothing was ever written into it."""
    if not OLD_OUTPUT.exists():
        pytest.skip("old output directory absent")

    assert OLD_OUTPUT.is_dir()
    assert list(OLD_OUTPUT.iterdir()) == []


# --------------------------------------------------------------------------
# These tests themselves leak nothing.
# --------------------------------------------------------------------------

def test_this_test_file_reads_no_corpus_and_leaks_nothing():
    """Checked over string *literals*, not over the source text.

    Matching raw text would match this test's own forbidden-token list, which
    is how the first version of this check failed itself.
    """
    module = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    literals = [
        node.value for node in ast.walk(module)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    ]
    # This test's own list is a literal too, so compare against everything
    # except the tokens it is built from.
    forbidden = ("records" ".json", "splits" ".json", "official-" "repository::")
    offenders = [
        text for text in literals
        if any(token in text for token in forbidden)
        and text not in forbidden
    ]

    assert offenders == []
