from __future__ import annotations

import json
from pathlib import Path

from baseline.atheris_harness import finalize_result, infer_parameter_kinds


def test_infer_parameter_kinds_uses_annotations_and_examples() -> None:
    code = "def target(x: int, words):\n    return x, words\n"
    assertions = ["assert target(3, ['a', 'b']) == (3, ['a', 'b'])"]
    assert infer_parameter_kinds(code, "target", assertions) == ["int", "list_str"]


def _checkpoint(path: Path) -> None:
    path.write_text(
        json.dumps({
            "harness_version": "test",
            "results": [{"task_id": "x", "outcome": "incomplete"}],
        }),
        encoding="utf-8",
    )


def test_finalize_clean_libfuzzer_exit_as_survived(tmp_path: Path) -> None:
    output = tmp_path / "result.json"
    _checkpoint(output)
    payload = finalize_result(output, 0, "30s", 0.25)
    assert payload["results"][0]["outcome"] == "survived"
    assert payload["results"][0]["elapsed_seconds"] == 0.25
    assert payload["runner_finalized"] is True


def test_finalize_outer_timeout_is_not_reported_as_survival(tmp_path: Path) -> None:
    output = tmp_path / "result.json"
    _checkpoint(output)
    payload = finalize_result(output, 124, "30s")
    row = payload["results"][0]
    assert row["outcome"] == "outer_wall_timeout"
    assert "30s" in row["harness_error"]


def test_finalize_libfuzzer_unit_timeout_is_distinct(tmp_path: Path) -> None:
    output = tmp_path / "result.json"
    _checkpoint(output)
    payload = finalize_result(output, 70, "30s")
    assert payload["results"][0]["outcome"] == "unit_timeout"


def test_finalize_preserves_kill_verdict(tmp_path: Path) -> None:
    output = tmp_path / "result.json"
    output.write_text(
        json.dumps({"results": [{"outcome": "killed", "kill_kind": "semantic_kill"}]}),
        encoding="utf-8",
    )
    payload = finalize_result(output, 0, "30s")
    assert payload["results"][0]["outcome"] == "killed"
    assert payload["results"][0]["process_returncode"] == 0


# --- identical implementations must never disagree ------------------------
#
# The reference and the buggy function were called with the SAME argument
# objects. A function that mutates its input therefore handed the second call
# a different value than the first received, so two identical implementations
# disagreed and the run was recorded as a semantic_kill. That inflates the
# baseline's kill count on precisely the targets where mutation is common -
# list and dict arguments - and any Oneiros-versus-Atheris comparison drawn
# from those runs understates the gap in the baseline's favour.

def test_mutating_the_argument_does_not_make_identical_code_disagree():
    import copy

    source = "def target(xs):\n    xs.append(7)\n    return len(xs)\n"
    namespace: dict = {}
    exec(compile(source, "<reference>", "exec"), namespace)
    reference = namespace["target"]
    namespace_two: dict = {}
    exec(compile(source, "<buggy>", "exec"), namespace_two)
    buggy = namespace_two["target"]

    arguments = [[1, 2, 3]]

    shared = list(arguments)
    expected_shared = reference(*shared)
    actual_shared = buggy(*shared)
    assert expected_shared != actual_shared, (
        "this fixture only demonstrates anything if sharing the argument does "
        "cause a spurious disagreement"
    )

    expected = reference(*copy.deepcopy(arguments))
    actual = buggy(*copy.deepcopy(arguments))
    assert expected == actual, (
        "identical implementations disagreed; each call must receive its own "
        "copy of the arguments"
    )


def test_the_harness_deep_copies_arguments_for_each_call():
    """Pins the fix in the source, so it cannot be quietly reverted."""
    from pathlib import Path

    source = (Path(__file__).resolve().parent.parent
              / "baseline" / "atheris_harness.py").read_text(encoding="utf-8")
    assert "reference_arguments = copy.deepcopy(arguments)" in source
    assert "buggy_arguments = copy.deepcopy(arguments)" in source
    assert "reference(*reference_arguments)" in source
    assert "buggy(*buggy_arguments)" in source


# --- the baseline must actually be coverage-guided -------------------------
#
# enable_python_coverage=True was set, so the harness LOOKED instrumented. It
# was not: libFuzzer's counters come from Atheris bytecode instrumentation,
# which a bare exec() of the target source never receives. The evidence was
# sitting in the run logs the whole time - 740 of 757 warned "no interesting
# inputs were found. Is the code instrumented for coverage?" and 697 finished
# with the corpus still at 1 input of 1 byte.

def test_the_target_function_is_instrumented():
    """Pins the call, so the baseline cannot silently go blind again."""
    from pathlib import Path
    source = (Path(__file__).resolve().parent.parent
              / "baseline" / "atheris_harness.py").read_text(encoding="utf-8")
    assert "atheris.instrument_func(buggy)" in source, (
        "without instrument_func the fuzzer is a blind random searcher and "
        "must not be reported as coverage-guided Atheris"
    )


def test_instrumentation_status_is_recorded_in_every_result():
    """A blind run must be visible in the artifact, not only in a log line."""
    from pathlib import Path
    source = (Path(__file__).resolve().parent.parent
              / "baseline" / "atheris_harness.py").read_text(encoding="utf-8")
    assert '"coverage_instrumented": state.get("coverage_instrumented")' in source
    assert '"instrumentation_error"' in source


def test_the_historical_logs_show_the_blind_runs():
    """Keeps the evidence attached to the claim rather than to a memory."""
    from pathlib import Path
    directory = (Path(__file__).resolve().parent.parent
                 / "results" / "atheris_val_20000_seed42")
    if not directory.is_dir():
        return
    logs = sorted(directory.glob("*.log"))
    if not logs:
        return
    warned = sum(
        1 for log in logs
        if "no interesting inputs were found" in log.read_text(
            encoding="utf-8", errors="ignore")
    )
    assert warned > len(logs) // 2, (
        "the historical Atheris sweep is the evidence that those runs were "
        "not coverage-guided; if this no longer holds the claim needs "
        "re-deriving before it is repeated"
    )


# --- the driver must not depend on the caller's working directory ---------
#
# Each target now runs inside its own directory so concurrent libFuzzer
# processes cannot overwrite one another's crash artifacts. That silently
# broke relative OUTDIR and TASKS paths: every one of 4542 targets failed with
# "no checkpoint (return code 1)" while the driver printed "complete" for all
# six sets. The smoke test had used an absolute path and missed it entirely.

def test_the_driver_resolves_paths_before_changing_directory():
    from pathlib import Path

    script = (Path(__file__).resolve().parent.parent
              / "scripts" / "run_atheris_wsl.sh").read_text(encoding="utf-8")
    resolve_at = script.index('OUTDIR="$(cd "$OUTDIR" && pwd)"')
    first_cd = script.index('cd "$WORKDIR"')
    assert resolve_at < first_cd, (
        "OUTDIR and TASKS must be made absolute before any cd, or a relative "
        "path stops resolving inside the per-task working directory"
    )
    assert 'TASKS="$(cd "$(dirname "$TASKS")" && pwd)/$(basename "$TASKS")"' in script


def test_each_target_gets_its_own_working_directory():
    """Concurrent libFuzzer runs must not share a crash-artifact directory."""
    from pathlib import Path

    script = (Path(__file__).resolve().parent.parent
              / "scripts" / "run_atheris_wsl.sh").read_text(encoding="utf-8")
    assert 'TASKDIR="$(mktemp -d "$WORKDIR/task_XXXXXX")"' in script
    assert 'cd "$TASKDIR"' in script


def test_the_driver_has_no_carriage_returns():
    """A CRLF write made bash refuse to parse the function definition."""
    from pathlib import Path

    data = (Path(__file__).resolve().parent.parent
            / "scripts" / "run_atheris_wsl.sh").read_bytes()
    assert b"\r\n" not in data
