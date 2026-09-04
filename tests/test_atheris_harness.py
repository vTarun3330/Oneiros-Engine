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
