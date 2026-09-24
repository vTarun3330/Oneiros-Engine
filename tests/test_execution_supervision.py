"""Safety and semantic tests for isolated execution supervision."""

from __future__ import annotations

import inspect
import json
import time

import pytest

import harness.execution_supervision as execution_supervision
from harness.execution_supervision import (
    ExecutionSupervisionRejected,
    MAX_DISTINCT_TRACE_TRANSITIONS,
    MAX_TRACE_EVENT_LIMIT,
    MAX_TRACE_PAYLOAD_BYTES,
    build_output_prediction_completion,
    build_output_prediction_prompt,
    build_execution_supervision_completion,
    build_execution_supervision_example,
    build_execution_supervision_prompt,
    collect_execution_evidence,
)


BRANCHING = """def choose(value):
    if value > 0:
        answer = value + 1
    else:
        answer = value - 1
    return answer
"""


def test_collects_only_bounded_shown_code_lines_and_typed_actual() -> None:
    evidence = collect_execution_evidence(
        shown_code=BRANCHING,
        entry_point="choose",
        call_expression="choose(4)",
        trace_event_limit=10,
    )
    assert evidence.actual == {"type": "int", "literal": "5"}
    assert evidence.determinism_runs == 2
    assert [event["line"] for event in evidence.trace] == [2, 3, 6]
    assert all(event["source"] in BRANCHING.splitlines() for event in evidence.trace)


def test_fixed_completion_has_exact_fields_and_computes_type_sensitive_difference() -> None:
    evidence = collect_execution_evidence(
        shown_code="def identity(value):\n    return value\n",
        entry_point="identity",
        call_expression="identity(1)",
    )
    completion = json.loads(
        build_execution_supervision_completion(evidence, intended_output=True)
    )
    assert list(completion) == ["trace", "actual", "intended", "differs"]
    assert completion["actual"] == {"type": "int", "literal": "1"}
    assert completion["intended"] == {"type": "bool", "literal": "True"}
    assert completion["differs"] is True
    first = build_execution_supervision_completion(evidence, intended_output=True)
    second = build_execution_supervision_completion(evidence, intended_output=True)
    assert first == second
    assert "\n" not in first and ": " not in first


def test_prompt_api_is_closed_and_contains_only_shown_inputs() -> None:
    parameters = set(inspect.signature(build_execution_supervision_prompt).parameters)
    assert parameters == {"specification", "shown_code", "entry_point", "call_expression"}
    prompt = build_execution_supervision_prompt(
        specification="Increment the input.",
        shown_code="def inc(value):\n    return value + 1\n",
        entry_point="inc",
        call_expression="inc(2)",
    )
    assert "Increment the input." in prompt
    assert "def inc" in prompt
    assert "reference_code" not in prompt
    assert "gold_assertion" not in prompt
    assert "patch_diff" not in prompt
    with pytest.raises(TypeError):
        build_execution_supervision_prompt(  # type: ignore[call-arg]
            specification="x",
            shown_code="def f(): return 1",
            entry_point="f",
            call_expression="f()",
            reference_code="SENTINEL",
        )


def test_high_level_example_never_places_intended_value_in_prompt() -> None:
    example = build_execution_supervision_example(
        specification="Return the secret token associated with the integer.",
        shown_code="def token(value):\n    return value + 10\n",
        entry_point="token",
        call_expression="token(7)",
        intended_output="UNIQUE_INTENDED_SENTINEL",
    )
    assert "UNIQUE_INTENDED_SENTINEL" not in example["prompt"]
    assert "UNIQUE_INTENDED_SENTINEL" in example["completion"]


def test_output_prediction_changes_prompt_only_and_keeps_one_assert_shape() -> None:
    shown = collect_execution_evidence(
        shown_code="def inc(value):\n    return value - 1\n",
        entry_point="inc",
        call_expression="inc(2)",
    )
    intended = collect_execution_evidence(
        shown_code="def inc(value):\n    return value + 1\n",
        entry_point="inc",
        call_expression="inc(2)",
    )
    prompt = build_output_prediction_prompt(
        specification="Return the input incremented by one.",
        shown_code="def inc(value):\n    return value - 1\n",
        entry_point="inc",
        call_expression="inc(2)",
    )
    completion, evidence = build_output_prediction_completion(
        call_expression="inc(2)",
        shown_evidence=shown,
        intended_evidence=intended,
    )
    assert completion == "assert inc(2) == 3"
    assert evidence["actual"] == {"type": "int", "literal": "1"}
    assert evidence["intended"] == {"type": "int", "literal": "3"}
    assert evidence["differs"] is True
    assert "== 3" not in prompt
    assert "return value + 1" not in prompt


def test_output_prediction_prompt_api_cannot_accept_hidden_fields() -> None:
    parameters = set(inspect.signature(build_output_prediction_prompt).parameters)
    assert parameters == {
        "specification", "shown_code", "entry_point", "call_expression",
    }
    with pytest.raises(TypeError):
        build_output_prediction_prompt(  # type: ignore[call-arg]
            specification="Return one.",
            shown_code="def one(): return 0",
            entry_point="one",
            call_expression="one()",
            intended_output=1,
        )


@pytest.mark.parametrize(
    ("code", "call", "reason"),
    [
        (
            "def hang():\n    return sum(range(10**10))\n",
            "hang()",
            "timeout",
        ),
        (
            "def fail():\n    raise ValueError('no')\n",
            "fail()",
            "execution_error",
        ),
        (
            "def unsupported():\n    return range(3)\n",
            "unsupported()",
            "unsupported_value",
        ),
    ],
)
def test_rejects_timeouts_execution_failures_and_unsupported_values(
    code: str, call: str, reason: str,
) -> None:
    started = time.perf_counter()
    with pytest.raises(ExecutionSupervisionRejected) as caught:
        collect_execution_evidence(
            shown_code=code,
            entry_point=call.split("(", 1)[0],
            call_expression=call,
            timeout_seconds=0.05,
        )
    assert caught.value.reason == reason
    assert time.perf_counter() - started < 3


def test_rejects_trace_cap_overflow() -> None:
    with pytest.raises(ExecutionSupervisionRejected) as caught:
        collect_execution_evidence(
            shown_code="def total():\n    x = 0\n    for value in range(20):\n        x += value\n    return x\n",
            entry_point="total",
            call_expression="total()",
            trace_event_limit=4,
        )
    assert caught.value.reason == "trace_cap_overflow"


def test_v1_trace_ceilings_are_frozen() -> None:
    assert MAX_TRACE_EVENT_LIMIT == 128
    assert MAX_DISTINCT_TRACE_TRANSITIONS == 64
    assert MAX_TRACE_PAYLOAD_BYTES == 4096


def test_rejects_serialized_trace_payload_overflow_instead_of_truncating() -> None:
    long_line = "    result = value + 1  # " + ("x" * 900)
    code = "\n".join([
        "def verbose(value):",
        long_line,
        long_line.replace("result = value + 1", "result += 1"),
        long_line.replace("result = value + 1", "result += 2"),
        long_line.replace("result = value + 1", "result += 3"),
        long_line.replace("result = value + 1", "result += 4"),
        "    return result",
    ])
    with pytest.raises(ExecutionSupervisionRejected) as caught:
        collect_execution_evidence(
            shown_code=code,
            entry_point="verbose",
            call_expression="verbose(1)",
        )
    assert caught.value.reason == "trace_cap_overflow"
    assert "serialized trace payload cap exceeded" in caught.value.detail


def test_rejects_more_than_64_distinct_consecutive_line_transitions() -> None:
    body = ["def many_lines():", "    value = 0"]
    body.extend(f"    value = {index}" for index in range(1, 68))
    body.append("    return value")
    with pytest.raises(ExecutionSupervisionRejected) as caught:
        collect_execution_evidence(
            shown_code="\n".join(body),
            entry_point="many_lines",
            call_expression="many_lines()",
        )
    assert caught.value.reason == "trace_cap_overflow"
    assert "distinct trace transition cap exceeded" in caught.value.detail


def test_rejects_nondeterminism_across_fresh_workers() -> None:
    with pytest.raises(ExecutionSupervisionRejected) as caught:
        collect_execution_evidence(
            shown_code=(
                "import random\n"
                "def changing():\n"
                "    return random.SystemRandom().randbytes(16)\n"
            ),
            entry_point="changing",
            call_expression="changing()",
        )
    assert caught.value.reason == "nondeterminism"


def test_rejects_malformed_worker_evidence_as_harness_failure(monkeypatch) -> None:
    monkeypatch.setattr(
        execution_supervision,
        "_invoke_worker",
        lambda payload: {
            "status": "ok",
            "trace": "not-a-list",
            "actual": {"type": "int", "literal": "1"},
        },
    )
    with pytest.raises(ExecutionSupervisionRejected) as caught:
        collect_execution_evidence(
            shown_code="def value():\n    return 1\n",
            entry_point="value",
            call_expression="value()",
        )
    assert caught.value.reason == "harness_failure"


@pytest.mark.parametrize(
    "call",
    [
        "other(1)",
        "target(helper())",
        "target(value)",
        "target(*[1])",
        "target(**{'value': 1})",
    ],
)
def test_call_accepts_only_direct_entry_point_with_literal_arguments(call: str) -> None:
    with pytest.raises(ExecutionSupervisionRejected) as caught:
        collect_execution_evidence(
            shown_code="def target(value=0):\n    return value\n",
            entry_point="target",
            call_expression=call,
        )
    assert caught.value.reason == "call_policy_error"


def test_rejects_filesystem_process_network_and_private_module_escape() -> None:
    dangerous = [
        "import os\ndef f(): return os.getcwd()",
        "def f(): return open('secret').read()",
        "import random\ndef f(): return random._os.getcwd()",
    ]
    for code in dangerous:
        with pytest.raises(ExecutionSupervisionRejected) as caught:
            collect_execution_evidence(
                shown_code=code,
                entry_point="f",
                call_expression="f()",
            )
        assert caught.value.reason == "source_policy_error"


def test_trace_excludes_import_and_standard_library_frames() -> None:
    code = (
        "from statistics import mean\n"
        "def average(values):\n"
        "    return mean(values)\n"
    )
    evidence = collect_execution_evidence(
        shown_code=code,
        entry_point="average",
        call_expression="average([1, 2, 6])",
    )
    assert evidence.actual == {"type": "int", "literal": "3"}
    assert evidence.trace == ({"line": 3, "source": "    return mean(values)"},)


def test_supported_nested_literals_round_trip_canonically() -> None:
    evidence = collect_execution_evidence(
        shown_code="def structure():\n    return {'b': [2, 1], 'a': (True, None)}\n",
        entry_point="structure",
        call_expression="structure()",
    )
    assert evidence.actual == {
        "type": "dict",
        "literal": "{'a': (True, None), 'b': [2, 1]}",
    }
