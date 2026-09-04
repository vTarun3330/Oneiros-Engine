from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.build_atheris_panel import build_panel
from scripts.compare_atheris_and_oneiros import compare


def _view(tmp_path: Path) -> Path:
    view = tmp_path / "development_view"
    view.mkdir()
    (view / "manifest.json").write_text(
        json.dumps({"sealed_splits_excluded": ["test"]}), encoding="utf-8",
    )
    records = [{
        "id": "mutation::x",
        "task_mode": "function",
        "entry_point": "f",
        "code_under_test": "def f(x):\n    return x + 1\n",
        "reference_code": "def f(x):\n    return x\n",
        "tests": [{"code": "assert f(1) == 1"}],
        "source": {"upstream": "mbpp"},
        "provenance": {"mutation_type": "arithmetic"},
        "group_id": "function:x",
    }]
    (view / "ablation_dev.records.json").write_text(
        json.dumps(records), encoding="utf-8",
    )
    return view


def test_build_panel_is_function_only_and_sealed_safe(tmp_path: Path) -> None:
    output = tmp_path / "tasks.json"
    manifest = build_panel(_view(tmp_path), "ablation_dev", output)
    tasks = json.loads(output.read_text(encoding="utf-8"))
    assert manifest["sealed_final_test_accessed"] is False
    assert manifest["function_task_count"] == 1
    assert tasks[0]["task_id"] == "mutation::x"
    assert tasks[0]["bug_family"] == "arithmetic"


def test_build_panel_refuses_final_split(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="sealed final split"):
        build_panel(_view(tmp_path), "test", tmp_path / "tasks.json")


def test_comparison_does_not_call_incomplete_result_complete(tmp_path: Path) -> None:
    results = tmp_path / "atheris"
    results.mkdir()
    (results / "task_00000.json").write_text(
        json.dumps({"results": [{
            "task_id": "mutation::x", "outcome": "incomplete",
            "runs": 4, "kill_kind": None,
        }]}),
        encoding="utf-8",
    )
    report = compare(results, {})
    assert report["atheris"]["comparison_complete"] is False
    assert report["atheris"]["incomplete_results"] == 1
