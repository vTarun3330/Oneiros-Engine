"""Build a sealed-test-free function panel for the actual Atheris harness."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness.corpus import sha256_file, write_json
from utils.reproducibility import source_tree_sha256


DEFAULT_VIEW = (
    ROOT / "data" / "corpus" / "v4_1_research_hardened_candidate"
    / "development_view"
)
ALLOWED_SPLITS = {"train", "ablation_dev", "val"}


def _load_panel_ids(path: Path | None) -> list[str] | None:
    if path is None:
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("final_test_measurement") is True:
        raise ValueError(f"refusing sealed final-test artifact {path}")
    split = payload.get("evaluation_split")
    if split == "test":
        raise ValueError(f"refusing sealed final-test artifact {path}")
    rows = payload.get("function_results")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"panel artifact {path} has no function_results")
    record_ids = [str(row.get("record_id") or "") for row in rows]
    if any(not record_id for record_id in record_ids):
        raise ValueError(f"panel artifact {path} has a missing record_id")
    if len(set(record_ids)) != len(record_ids):
        raise ValueError(f"panel artifact {path} repeats a record_id")
    return record_ids


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    if resolved.is_relative_to(ROOT):
        return str(resolved.relative_to(ROOT)).replace("\\", "/")
    return str(resolved)


def build_panel(
    view_dir: Path, split: str, output_path: Path,
    evaluation_panel: Path | None = None,
) -> dict[str, Any]:
    if split not in ALLOWED_SPLITS:
        raise ValueError(
            f"split {split!r} is not available to development tooling; "
            "the sealed final split is deliberately unreachable"
        )
    records_path = view_dir / f"{split}.records.json"
    manifest_path = view_dir / "manifest.json"
    view_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if "test" not in set(view_manifest.get("sealed_splits_excluded") or []):
        raise ValueError("development view does not attest that test is excluded")
    records = json.loads(records_path.read_text(encoding="utf-8"))
    by_id = {str(record["id"]): record for record in records}
    panel_ids = _load_panel_ids(evaluation_panel)
    selected_ids = panel_ids if panel_ids is not None else list(by_id)

    missing = [record_id for record_id in selected_ids if record_id not in by_id]
    if missing:
        raise ValueError(
            f"evaluation panel contains {len(missing)} records outside split {split}: "
            f"{missing[:3]}"
        )

    tasks: list[dict[str, Any]] = []
    skipped_non_function: list[str] = []
    for record_id in selected_ids:
        record = by_id[record_id]
        if record.get("task_mode") != "function":
            skipped_non_function.append(record_id)
            continue
        provenance = record.get("provenance") or {}
        source = record.get("source") or {}
        tasks.append({
            "task_id": record_id,
            "entry_point": record["entry_point"],
            "buggy_code": record["code_under_test"],
            "reference_code": record["reference_code"],
            "example_assertions": [
                str(test.get("code") or "") for test in record.get("tests") or []
                if test.get("code")
            ],
            "source_dataset": (
                source.get("upstream") or source.get("name") or "unknown"
            ),
            "bug_family": (
                provenance.get("mutation_family")
                or provenance.get("mutation_type")
                or provenance.get("category")
                or record.get("task_type")
                or "unknown"
            ),
            "semantic_group": record.get("group_id", record_id),
        })

    if not tasks:
        raise ValueError("no function targets were eligible for Atheris")
    write_json(output_path, tasks)
    manifest = {
        "schema_version": "oneiros_atheris_panel_v1",
        "source_tree_sha256": source_tree_sha256(ROOT),
        "development_view": _display_path(view_dir),
        "development_view_manifest_sha256": sha256_file(manifest_path),
        "records_file_sha256": sha256_file(records_path),
        "split": split,
        "sealed_final_test_accessed": False,
        "selection_artifact": (
            _display_path(evaluation_panel) if evaluation_panel is not None else None
        ),
        "selected_record_count": len(selected_ids),
        "function_task_count": len(tasks),
        "skipped_non_function_count": len(skipped_non_function),
        "skipped_non_function_ids": skipped_non_function,
        "tasks_sha256": sha256_file(output_path),
    }
    manifest_output = output_path.with_suffix(".manifest.json")
    write_json(manifest_output, manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--view-dir", type=Path, default=DEFAULT_VIEW)
    parser.add_argument("--split", choices=sorted(ALLOWED_SPLITS), default="ablation_dev")
    parser.add_argument("--evaluation-panel", type=Path)
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "results" / "v4_2_atheris_tasks.json",
    )
    arguments = parser.parse_args()
    manifest = build_panel(
        arguments.view_dir, arguments.split, arguments.output,
        arguments.evaluation_panel,
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
