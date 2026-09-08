"""Durable, resumable work plan: pause here, resume exactly there.

Long multi-phase work on this project has repeatedly lost its place - not
because a step failed, but because the ordering lived only in a conversation.
This keeps the ordering on disk, so "pause" and "resume" mean something after
a laptop shutdown, a closed terminal or a lost session.

Design rules that follow from how this has actually gone wrong:

* one step is in progress at a time, and starting a second while one is open
  is refused. A plan that silently allows two in-flight steps cannot tell you
  where to resume.
* pausing NEVER discards the in-progress step. It records that work stopped
  mid-step, and resume returns to that same step rather than the next one.
* a step records the artifacts it produced. Resuming a half-done step should
  not redo the parts that already wrote their output, and the only way to know
  which those are is to have written it down.
* GPU evaluations already checkpoint every generation batch and resume from
  completed functions, so pausing between steps loses at most one batch. This
  file records the run id so the resume knows what to look at.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness.corpus import write_json

STATE = ROOT / "results" / "v4_2_work_plan_state.json"

PENDING, IN_PROGRESS, DONE, BLOCKED = "pending", "in_progress", "done", "blocked"
RUNNING, PAUSED = "running", "paused"


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def load(path: Path = STATE) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"no work plan at {path}; create one with --init")
    return json.loads(path.read_text(encoding="utf-8"))


def save(plan: dict[str, Any], path: Path = STATE) -> None:
    plan["updated_utc"] = _now()
    write_json(path, plan)


def current(plan: dict[str, Any]) -> dict[str, Any] | None:
    for step in plan["steps"]:
        if step["status"] == IN_PROGRESS:
            return step
    return None


def next_pending(plan: dict[str, Any]) -> dict[str, Any] | None:
    for step in plan["steps"]:
        if step["status"] == PENDING:
            return step
    return None


def start(plan: dict[str, Any], step_id: str | None = None) -> dict[str, Any]:
    if plan["execution"] == PAUSED:
        raise SystemExit("the plan is paused; resume before starting a step")
    open_step = current(plan)
    if open_step is not None:
        raise SystemExit(
            f"step {open_step['id']!r} is already in progress; finish, block "
            "or pause it rather than opening a second one"
        )
    step = (
        next(s for s in plan["steps"] if s["id"] == step_id) if step_id
        else next_pending(plan)
    )
    if step is None:
        raise SystemExit("no pending step remains")
    if step["status"] != PENDING:
        raise SystemExit(f"step {step['id']!r} is {step['status']}, not pending")
    step["status"] = IN_PROGRESS
    step["started_utc"] = _now()
    return step


def note(plan: dict[str, Any], text: str, artifact: str | None = None,
         run_id: str | None = None) -> dict[str, Any]:
    """Record progress WITHIN a step, which is what makes a resume precise."""
    step = current(plan)
    if step is None:
        raise SystemExit("no step is in progress; nothing to record against")
    step.setdefault("progress", []).append({"at": _now(), "note": text})
    if artifact:
        step.setdefault("artifacts", []).append(artifact)
    if run_id:
        step.setdefault("run_ids", []).append(run_id)
    return step


def finish(plan: dict[str, Any], summary: str) -> dict[str, Any]:
    step = current(plan)
    if step is None:
        raise SystemExit("no step is in progress")
    step["status"] = DONE
    step["finished_utc"] = _now()
    step["summary"] = summary
    return step


def block(plan: dict[str, Any], reason: str) -> dict[str, Any]:
    """A step that cannot proceed is recorded, never silently skipped."""
    step = current(plan)
    if step is None:
        raise SystemExit("no step is in progress")
    step["status"] = BLOCKED
    step["blocked_utc"] = _now()
    step["blocked_reason"] = reason
    return step


def pause(plan: dict[str, Any], reason: str = "") -> dict[str, Any]:
    plan["execution"] = PAUSED
    plan["paused_utc"] = _now()
    plan["paused_reason"] = reason
    step = current(plan)
    # The in-progress step stays in progress. Resetting it to pending would
    # discard the record of how far it got, which is the one thing a resume
    # needs.
    plan["paused_mid_step"] = step["id"] if step else None
    return plan


def resume(plan: dict[str, Any]) -> dict[str, Any]:
    plan["execution"] = RUNNING
    plan["resumed_utc"] = _now()
    plan["paused_reason"] = ""
    return plan


def describe(plan: dict[str, Any]) -> str:
    lines = [
        f"plan       : {plan['name']}",
        f"execution  : {plan['execution'].upper()}",
    ]
    if plan["execution"] == PAUSED:
        mid = plan.get("paused_mid_step")
        lines.append(f"paused at  : {mid or 'between steps'}")
        if plan.get("paused_reason"):
            lines.append(f"reason     : {plan['paused_reason']}")
    lines.append("")
    marks = {DONE: "[x]", IN_PROGRESS: "[>]", PENDING: "[ ]", BLOCKED: "[!]"}
    for step in plan["steps"]:
        lines.append(f"{marks[step['status']]} {step['id']:<28} {step['title']}")
        if step["status"] == IN_PROGRESS:
            for entry in step.get("progress", []):
                lines.append(f"      - {entry['note']}")
        if step["status"] == BLOCKED:
            lines.append(f"      blocked: {step.get('blocked_reason', '')}")
    done = sum(1 for s in plan["steps"] if s["status"] == DONE)
    lines.append("")
    lines.append(f"{done}/{len(plan['steps'])} steps complete")
    step = current(plan)
    if step is not None:
        verb = "resume at" if plan["execution"] == PAUSED else "in progress"
        lines.append(f"{verb}: {step['id']} - {step['title']}")
    elif plan["execution"] == RUNNING:
        nxt = next_pending(plan)
        lines.append(f"next: {nxt['id']} - {nxt['title']}" if nxt else "nothing pending")
    return "\n".join(lines)


def initial_plan() -> dict[str, Any]:
    def step(identifier: str, title: str, phase: str, detail: str,
             gpu: bool = False) -> dict[str, Any]:
        return {
            "id": identifier, "title": title, "phase": phase, "detail": detail,
            "uses_gpu": gpu, "status": PENDING, "progress": [],
            "artifacts": [], "run_ids": [],
        }

    return {
        "schema_version": "oneiros_work_plan_v1",
        "name": "Phase B writeup, then Phase A value-prediction training",
        "created_utc": _now(),
        "execution": RUNNING,
        "paused_mid_step": None,
        "paused_reason": "",
        "decided": (
            "B before A: the 80% target is not reachable on this panel at 1.5B "
            "without contaminating it, and the negative results plus the "
            "prompt-copying finding are the substantive output."
        ),
        "steps": [
            step("B1-adev-seeds", "Finish ablation_dev seeds 43/44 and report the "
                 "selection-panel inflation", "B",
                 "Compare the ablation_dev gain against the locked val gain across "
                 "three seeds; the single-seed figures were +12.2 and +3.8.",
                 gpu=True),
            step("B2-results-table", "Build one derived results table for every arm", "B",
                 "Every arm, both panels, per benchmark, rebuilt from artifacts so "
                 "no figure is retyped."),
            step("B3-leakage-section", "Write the prompt-copying finding up in full", "B",
                 "The 46% of humaneval records stating a killing value, the 44-46% "
                 "copied kills, and SFT doubling the copy rate."),
            step("B4-report", "Write the research report", "B",
                 "Pipeline, corpus, method, every measured arm including the "
                 "negative ones, the honest 80% arithmetic, threats to validity."),
            step("B5-publish", "Publish the report for the mentor meeting", "B",
                 "A shareable page carrying the same content as the repo document."),
            step("A1-design", "Design the auxiliary value-prediction task", "A",
                 "Predict what reference code returns for a given input. Train "
                 "split only; no val or sealed data; no prompt-borne answers."),
            step("A2-build", "Build the auxiliary dataset and trainer path", "A",
                 "Derived from train-split references, with leakage gates and a "
                 "manifest, mirroring the existing corpus discipline."),
            step("A3-preflight", "Preflight: tests, leakage audit, dry run", "A",
                 "Must pass before any GPU time, per the standing rule."),
            step("A4-train", "Train the auxiliary arm", "A",
                 "Durable GPU run, checkpointed.", gpu=True),
            step("A5-evaluate", "Evaluate on ablation_dev, then locked val", "A",
                 "Report against base and relearning on the same panels and seeds.",
                 gpu=True),
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, default=STATE)
    parser.add_argument("--init", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--start", nargs="?", const="", default=None,
                        metavar="STEP_ID")
    parser.add_argument("--note", default=None)
    parser.add_argument("--artifact", default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--finish", default=None, metavar="SUMMARY")
    parser.add_argument("--block", default=None, metavar="REASON")
    parser.add_argument("--pause", nargs="?", const="", default=None,
                        metavar="REASON")
    parser.add_argument("--resume", action="store_true")
    arguments = parser.parse_args()

    if arguments.init:
        if arguments.state.exists():
            raise SystemExit(f"{arguments.state} exists; refusing to overwrite a plan")
        plan = initial_plan()
        save(plan, arguments.state)
        print(describe(plan))
        return 0

    plan = load(arguments.state)
    changed = False
    if arguments.pause is not None:
        pause(plan, arguments.pause)
        changed = True
    if arguments.resume:
        resume(plan)
        changed = True
    if arguments.start is not None:
        start(plan, arguments.start or None)
        changed = True
    if arguments.note or arguments.artifact or arguments.run_id:
        note(plan, arguments.note or "", arguments.artifact, arguments.run_id)
        changed = True
    if arguments.block:
        block(plan, arguments.block)
        changed = True
    if arguments.finish:
        finish(plan, arguments.finish)
        changed = True
    if changed:
        save(plan, arguments.state)
    print(describe(plan))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
