"""The single, immovable fact the sealed-final tests assert against.

Before 2026-09-16 these tests asserted absence: no guard state, no run state,
no output directory, no granted authorization. That was a fine invariant while
it held, and it stopped holding the moment the sealed final test was attempted.
Seventeen tests then failed - not because anything regressed, but because they
encoded a world that no longer exists.

Absence was never the property worth protecting. The property is **exactly one
attempt, and no way to make another**. This module states that once, so a
command under test is checked against what actually happened rather than
against a past that cannot come back.

Nothing here reads the corpus, and nothing here names an identifier or a token.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

STATE_PATH = ROOT / "results" / "sealed_final_state.json"
RUN_STATE_PATH = ROOT / "results" / "sealed_final_run_state.json"
OLD_OUTPUT_DIR = ROOT / "results" / "sealed_final_base_qwen_s42"

#: The attempt happened once. Any command under test must leave this alone.
EXPECTED_ATTEMPTS = 1


def guard_fingerprint():
    """How many issuances, spends and runs the guard has recorded.

    Returned as a tuple so a test can capture it before a command and compare
    after: the question is never "is there state" but "did this command add
    any", and the difference is the whole point.
    """
    if not STATE_PATH.is_file():
        return (0, 0, 0)
    state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    return (
        len(state.get("issued", [])),
        len(state.get("spent_tokens", [])),
        len(state.get("runs", [])),
    )


def assert_no_new_authorization(before=None):
    """The guard recorded nothing new.

    With ``before``, this is a true no-change check around a command. Without
    it, it asserts the standing fact: one attempt, never more.
    """
    now = guard_fingerprint()
    if before is not None:
        assert now == before, (
            f"a command changed guard state: {before} -> {now}. "
            "Nothing may add an issuance, a spend or a run.")
        return
    if STATE_PATH.is_file():
        assert now == (EXPECTED_ATTEMPTS,) * 3, (
            f"guard state records {now}, expected exactly one attempt")


def assert_re_execution_is_blocked():
    """A run-state file exists and permanently forbids a second run."""
    if not RUN_STATE_PATH.is_file():
        return
    run = json.loads(RUN_STATE_PATH.read_text(encoding="utf-8"))
    assert run["status"] == "failed_after_authorization"
    assert run["authorization_spent"] is True
    assert run["may_never_run_again"] is True
    assert run["result"] == {}, "the attempt produced no measurement"


def assert_old_output_is_empty():
    """The output directory was created and nothing was ever written into it."""
    if not OLD_OUTPUT_DIR.exists():
        return
    assert OLD_OUTPUT_DIR.is_dir()
    assert list(OLD_OUTPUT_DIR.iterdir()) == [], \
        "the failed attempt must not have produced result files"
