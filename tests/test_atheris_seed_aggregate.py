"""Pin the Atheris seed aggregate, which feeds the headline baseline claim."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.aggregate_atheris_seeds import aggregate, score_directory


def _write(directory: Path, index: int, payload: dict) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"task_{index:05d}.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )


def _task(seed: int, outcome: str, kind: str | None = None) -> dict:
    return {
        "seed": seed, "max_runs": 8,
        "results": [{"outcome": outcome, "kill_kind": kind}],
    }


def test_kills_and_survivors_are_counted_separately(tmp_path):
    directory = tmp_path / "seed42"
    _write(directory, 0, _task(42, "killed", "semantic_kill"))
    _write(directory, 1, _task(42, "killed", "crash_kill"))
    _write(directory, 2, _task(42, "survived"))
    _write(directory, 3, _task(42, "survived"))

    row = score_directory(directory)
    assert row["targets"] == 4
    assert row["killed"] == 2
    assert row["kill_rate"] == 0.5
    assert row["kill_kinds"] == {"semantic_kill": 1, "crash_kill": 1}
    assert row["seed"] == 42
    assert row["max_runs"] == 8


def test_an_unreadable_task_file_is_incomplete_not_a_survivor(tmp_path):
    """A target that never ran is not a target that survived.

    Counting it as a survivor would leave it in the denominator and understate
    Atheris; skipping it entirely would drop it from the denominator and
    flatter Atheris. It is counted as incomplete and reported.
    """
    directory = tmp_path / "seed43"
    _write(directory, 0, _task(43, "killed", "semantic_kill"))
    directory.joinpath("task_00001.json").write_text("{ truncated", encoding="utf-8")

    row = score_directory(directory)
    assert row["targets"] == 1
    assert row["killed"] == 1
    assert row["incomplete_task_files"] == 1


def test_the_aggregate_reports_spread_not_only_a_mean(tmp_path):
    """A mean alone hides exactly what the sweep was run to expose."""
    for seed, outcomes in ((42, ["killed", "survived"]),
                           (43, ["killed", "killed"]),
                           (44, ["survived", "survived"])):
        directory = tmp_path / f"seed{seed}"
        for index, outcome in enumerate(outcomes):
            _write(directory, index, _task(
                seed, outcome, "semantic_kill" if outcome == "killed" else None
            ))

    report = aggregate({"matched_8": [
        tmp_path / "seed42", tmp_path / "seed43", tmp_path / "seed44",
    ]})
    arm = report["arms"]["matched_8"]
    assert arm["seeds"] == [42, 43, 44]
    assert arm["mean_kill_rate"] == 0.5
    assert arm["range_across_seeds"] == 1.0
    assert arm["min_kill_rate"] == 0.0
    assert arm["max_kill_rate"] == 1.0
    assert report["sealed_final_test_accessed"] is False


def test_the_artifact_never_calls_this_the_simulated_fuzzer(tmp_path):
    """Rule 9: the simulated coverage fuzzer is never labelled Atheris.

    The inverse matters too - this artifact holds real Atheris runs and must
    say so, because the two live side by side in the baseline tables.
    """
    report = aggregate({"matched_8": []})
    assert "actual Atheris" in report["system"]
    assert "simulated" in report["system"]
