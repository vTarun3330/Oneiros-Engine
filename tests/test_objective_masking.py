"""Phase 4 V2 design: control and treatment differ ONLY in labels (real tokenizer, toy data).

Uses the cached Qwen2.5-Coder-1.5B tokenizer at its pinned revision (local files only;
no model weights, no generation, no network).  All prompts and values are synthetic.
"""
from __future__ import annotations

import pytest

from harness.objective_masking import (
    IGNORE, AlignmentError, build_example, decode_supervised, sequence_hash, validate_manifest,
    verify_pair,
)

transformers = pytest.importorskip("transformers")


@pytest.fixture(scope="module")
def tokenizer():
    return transformers.AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-Coder-1.5B-Instruct", revision="2e1fd397ee46e1388853d2af2c993145b0f1098a",
        local_files_only=True)


VALUES = [
    ("f(3)", "42"), ("f(3)", "-7"), ("f(3)", "3.25"), ("f(3)", "True"), ("f(3)", "None"),
    ("f('a')", "'it\\'s \\n ok'"), ("f('a')", "'naïve – 東京'"), ("f('a')", "''"),
    ("f([1])", "[1, 2, 3]"), ("f((1,))", "(1, 'b')"), ("f({})", "{'k': [1, {'z': None}]}"),
    ("f(9)", "[[1, 2], [3, [4, 5]], (6, 7)]"),
    ("f(123456789)", "123456789123456789"),
    ("g(42, '42')", "42"),                       # the value also appears inside the call
    ("h([1, 2, 3])", "[1, 2, 3]"),               # the call argument equals the value
]
USER = "Write a test for f. Example in the prompt: f(3) == 42 and h([1, 2, 3])."


@pytest.mark.parametrize("call, value", VALUES)
def test_only_the_labels_differ_and_treatment_supervises_exactly_the_value(tokenizer, call,
                                                                           value):
    ex = build_example(tokenizer, USER, call, value)
    control, treatment = ex["labels"]["control"], ex["labels"]["treatment"]
    assert len(control) == len(treatment) == len(ex["input_ids"]) == len(ex["attention_mask"])
    assert set(ex["attention_mask"]) == {1}
    start = ex["completion_start"]
    # control: every completion token, nothing from the prompt
    assert control[:start] == [IGNORE] * start
    assert control[start:] == ex["input_ids"][start:]
    # treatment: exactly the value tokens and EOS
    supervised = [i for i, t in enumerate(treatment) if t != IGNORE]
    assert supervised == ex["value_token_positions"] + [len(ex["input_ids"]) - 1], \
        f"supervised: {decode_supervised(tokenizer, ex, 'treatment')!r}"
    assert all(treatment[i] == ex["input_ids"][i] for i in supervised)
    decoded = decode_supervised(tokenizer, ex, "treatment")
    assert decoded == ex["supervised_value_text"] + tokenizer.eos_token, \
        f"decoded treatment labels {decoded!r} for value {value!r}"
    assert ex["supervised_value_text"].strip() == value, \
        f"supervised span {ex['supervised_value_text']!r} != value {value!r}"
    assert decode_supervised(tokenizer, ex, "control") == \
        f"assert {call} == {value}" + tokenizer.eos_token
    assert sequence_hash(control) != sequence_hash(treatment)


def test_two_arms_built_independently_have_identical_inputs(tokenizer):
    a = build_example(tokenizer, USER, "f(3)", "42")
    b = build_example(tokenizer, USER, "f(3)", "42")
    assert a["input_ids"] == b["input_ids"] and a["attention_mask"] == b["attention_mask"]
    assert a["completion_text"] == b["completion_text"] == "assert f(3) == 42"


def test_non_canonical_values_are_refused(tokenizer):
    for bad in (" 42", "42 ", ""):
        with pytest.raises(AlignmentError):
            build_example(tokenizer, USER, "f(3)", bad)


# --- C4 manifest validation on toy rows -----------------------------------------------------

def _row(i, group="g1", **over):
    row = {"record_id": f"r{i}", "group_id": group, "call": f"f({i})", "value": str(i),
           "execution_mode": "function", "representation": None, "passes_reference": True,
           "discriminates": True, "dataset": "toy", "bug_family": "arith",
           "complexity_tier": "simple", "input_ids_sha256": "x",
           "control_input_ids_sha256": f"in{i}", "treatment_input_ids_sha256": f"in{i}",
           "control_labels_sha256": f"c{i}", "treatment_labels_sha256": f"t{i}",
           "order_index_control": i, "order_index_treatment": i,
           "tokens": {"input": 10, "control_supervised": 6, "treatment_supervised": 2}}
    row.update(over)
    return row


@pytest.mark.parametrize("bad, message", [
    (dict(passes_reference=False), "fails on the reference"),
    (dict(discriminates=False), "does not discriminate"),
    (dict(execution_mode="repository_pytest_fragment"), "non-function"),
    (dict(group_id="gate1"), "gate/confirmation"),
    (dict(treatment_input_ids_sha256="other"), "input sequences differ"),
    (dict(treatment_labels_sha256="c1"), "label masks are identical"),
    (dict(order_index_treatment=99), "order differs"),
])
def test_manifest_rules_refuse(bad, message):
    with pytest.raises(ValueError, match=message):
        validate_manifest([_row(0), _row(1, **bad)], gate_groups={"gate1"},
                          confirmation_groups={"conf1"}, group_cap=5)


def test_manifest_refuses_repeats_and_group_cap_and_reports_composition():
    with pytest.raises(ValueError, match="repeated row"):
        validate_manifest([_row(0), _row(0)], gate_groups=(), confirmation_groups=(),
                          group_cap=5)
    with pytest.raises(ValueError, match="cap"):
        validate_manifest([_row(i) for i in range(3)], gate_groups=(), confirmation_groups=(),
                          group_cap=2)
    report = validate_manifest([_row(0), _row(1, group="g2")], gate_groups=(),
                               confirmation_groups=(), group_cap=2)
    assert report["rows"] == 2 and report["tokens"]["treatment_supervised"] == 4


def test_verify_pair_executes_on_toy_functions():
    ref = "def f(x):\n    return x + 1 if x > 2 else x\n"
    mut = "def f(x):\n    return x + 1 if x >= 2 else x\n"
    assert verify_pair(ref, mut, "f(2)", "2") == {"passes_reference": True, "discriminates": True}
    assert verify_pair(ref, mut, "f(5)", "6") == {"passes_reference": True, "discriminates": False}
    assert verify_pair(ref, mut, "f(2)", "3")["passes_reference"] is False
