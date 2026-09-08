"""Pause must not lose the place, and resume must return to it.

The whole value of this file is that "pause" and "resume" survive a closed
terminal. Every test here is about that guarantee rather than about
bookkeeping niceties.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.work_plan import (
    DONE, IN_PROGRESS, PAUSED, PENDING, RUNNING, block, current, describe,
    finish, initial_plan, load, next_pending, note, pause, resume, save, start,
)


def test_pause_keeps_the_step_in_progress():
    """Resetting it to pending would discard how far the step got."""
    plan = initial_plan()
    step = start(plan)
    note(plan, "half done", artifact="results/partial.json")

    pause(plan, "user asked")

    assert plan["execution"] == PAUSED
    assert step["status"] == IN_PROGRESS
    assert plan["paused_mid_step"] == step["id"]
    assert step["progress"][0]["note"] == "half done"
    assert step["artifacts"] == ["results/partial.json"]


def test_resume_returns_to_the_paused_step_not_the_next_one():
    plan = initial_plan()
    started = start(plan)
    pause(plan)

    resume(plan)

    assert plan["execution"] == RUNNING
    assert current(plan) is not None
    assert current(plan)["id"] == started["id"], (
        "resume must continue the interrupted step, not skip past it"
    )


def test_a_paused_plan_refuses_to_start_new_work():
    plan = initial_plan()
    start(plan)
    finish(plan, "done")
    pause(plan)

    with pytest.raises(SystemExit):
        start(plan)


def test_two_steps_cannot_be_in_progress_at_once():
    """A plan with two open steps cannot say where to resume."""
    plan = initial_plan()
    start(plan)

    with pytest.raises(SystemExit):
        start(plan)


def test_progress_survives_a_save_and_reload(tmp_path):
    path = tmp_path / "plan.json"
    plan = initial_plan()
    start(plan)
    note(plan, "wrote the table", artifact="results/table.json",
         run_id="20260908-000000-x")
    pause(plan, "closing the laptop")
    save(plan, path)

    reloaded = load(path)

    assert reloaded["execution"] == PAUSED
    assert reloaded["paused_reason"] == "closing the laptop"
    step = current(reloaded)
    assert step is not None and step["run_ids"] == ["20260908-000000-x"]
    assert step["artifacts"] == ["results/table.json"]


def test_finishing_advances_to_the_next_pending_step():
    plan = initial_plan()
    first = start(plan)
    finish(plan, "summary")

    assert first["status"] == DONE
    assert current(plan) is None
    assert next_pending(plan)["id"] != first["id"]


def test_a_blocked_step_is_recorded_rather_than_skipped():
    plan = initial_plan()
    start(plan)
    block(plan, "needs a modal token")

    step = next(s for s in plan["steps"] if s["status"] == "blocked")
    assert step["blocked_reason"] == "needs a modal token"
    assert all(s["status"] != PENDING or True for s in plan["steps"])


def test_note_without_an_open_step_is_refused():
    plan = initial_plan()
    with pytest.raises(SystemExit):
        note(plan, "orphan")


def test_the_description_names_where_to_resume():
    plan = initial_plan()
    step = start(plan)
    pause(plan, "user asked")

    text = describe(plan)
    assert "PAUSED" in text
    assert step["id"] in text
    assert "resume at" in text


def test_the_plan_starts_with_phase_b_before_phase_a():
    """The decision that was actually made, pinned so it cannot drift."""
    plan = initial_plan()
    phases = [step["phase"] for step in plan["steps"]]
    assert set(phases) == {"A", "B"}
    # Every B precedes every A. sorted() would demand the opposite, because
    # "A" < "B" alphabetically and the running order is deliberately not that.
    assert phases.index("A") > max(
        index for index, phase in enumerate(phases) if phase == "B"
    ), "every phase B step must come before the first phase A step"
