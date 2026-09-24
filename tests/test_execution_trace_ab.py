from __future__ import annotations

import json

from scripts.build_execution_trace_ab import build_pair_prompt, event_completion


def _row():
    return {
        "focused_prompt": (
            "prefix\n\n### Intended behaviour\n\nadd one"
            "\n\n### Function (may be defective)\n\ndef f(x):\n    return x + 1"
            "\n\n### Call\n\nf(1)\n\n### Assertion\n"
        ),
        "call_expression": "f(1)",
        "trace_completion": json.dumps({
            "trace": [
                {"line": 2, "source": "    return x + 1"},
                {"line": 1, "source": "def f(x):"},
            ],
            "actual": {"type": "int", "literal": "2"},
            "intended": {"type": "int", "literal": "2"},
            "differs": False,
        }, separators=(",", ":")),
        "execution_evidence": {
            "actual": {"type": "int", "literal": "2"},
            "intended": {"type": "int", "literal": "2"},
        },
    }


def test_ordered_and_unordered_targets_share_event_multiset_and_values():
    row = _row()
    ordered = json.loads(event_completion(row, ordered=True))
    unordered = json.loads(event_completion(row, ordered=False))
    assert ordered["events"] != unordered["events"]
    assert sorted(ordered["events"], key=lambda event: event["line"]) == unordered["events"]
    assert ordered["actual"] == unordered["actual"]
    assert ordered["intended"] == unordered["intended"]
    assert ordered["differs"] == unordered["differs"]


def test_prompts_differ_only_in_declared_order_instruction():
    unordered = build_pair_prompt(_row(), ordered=False)
    ordered = build_pair_prompt(_row(), ordered=True)
    assert "temporal order is intentionally removed" in unordered
    assert "Preserve the temporal order" in ordered
    for text in ("add one", "def f(x):", "f(1)"):
        assert text in unordered and text in ordered
