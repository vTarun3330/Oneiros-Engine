"""State what the measured panel actually contains, per split.

Every headline number in this project - base 0.621, relearning 0.654, actual
Atheris at both budgets, all five non-LLM baselines - is quoted over "757
held-out functions". That phrasing invites the reading that the balanced
synthetic/repository corpus is being measured. It is not.

The val corpus holds 757 synthetic function records AND 24 repository records.
The evaluation panel is the 757. No repository record is evaluated at any
seed, by any arm. The repository half of the corpus contributes supervision
during training and is never measured.

Nothing about the numbers is wrong, and nothing was hidden deliberately - the
reports already say repository data is supervision coverage rather than
end-to-end evidence. But "supervision coverage" is a weaker statement than
"the measured panel contains no repository record at all", and the second is
the one a reader needs to evaluate the claim. The reporting rules for this
project require that an aggregate must not hide weak repository performance
and that a high synthetic score is insufficient evidence of real-world
superiority; neither can be checked by a reader who believes the panel is
mixed.

This audit derives the composition from the artifacts themselves rather than
from any recorded intent, so it cannot drift from what was actually measured.
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness.corpus import write_json
from utils.reproducibility import source_tree_sha256

DEFAULT_VIEW = (
    ROOT / "data" / "corpus" / "v4_1_research_hardened_candidate" / "development_view"
)
#: The sealed split is never opened, including to count it.
SPLITS = ("train", "ablation_dev", "val")


def corpus_composition(view_dir: Path, split: str) -> dict[str, Any]:
    path = view_dir / f"{split}.records.json"
    if not path.exists():
        return {}
    records = json.loads(path.read_text(encoding="utf-8"))
    modes = collections.Counter(
        "repository" if record.get("task_mode") == "repository" else "function"
        for record in records
    )
    return {
        "records": len(records),
        "function_records": modes.get("function", 0),
        "repository_records": modes.get("repository", 0),
        "repository_record_ids": sorted(
            str(record["id"]) for record in records
            if record.get("task_mode") == "repository"
        ),
    }


def panel_composition(
    artifact: Path, repository_ids: set[str],
) -> dict[str, Any] | None:
    try:
        payload = json.loads(artifact.read_text(encoding="utf-8"))
    except Exception:
        return None
    rows = payload.get("function_results") or []
    if not rows:
        return None
    evaluated = {str(row.get("record_id") or "") for row in rows}
    repository_evaluated = sorted(evaluated & repository_ids)
    return {
        "artifact": artifact.as_posix().replace(str(ROOT).replace("\\", "/"), "."),
        "split": payload.get("evaluation_split"),
        "seed": payload.get("seed"),
        "evaluated_targets": len(rows),
        "repository_targets_evaluated": len(repository_evaluated),
        "repository_target_ids": repository_evaluated[:10],
    }


def audit(view_dir: Path) -> dict[str, Any]:
    splits: dict[str, Any] = {}
    repository_ids: dict[str, set[str]] = {}
    for split in SPLITS:
        composition = corpus_composition(view_dir, split)
        if not composition:
            continue
        repository_ids[split] = set(composition.pop("repository_record_ids"))
        splits[split] = composition

    panels: list[dict[str, Any]] = []
    for pattern in ("results/*/sft_validation_*.json",
                    "results/*/base_validation_*.json"):
        for path in sorted(glob.glob(str(ROOT / pattern))):
            if ".progress." in path:
                continue
            payload_path = Path(path)
            try:
                split = json.loads(
                    payload_path.read_text(encoding="utf-8")
                ).get("evaluation_split")
            except Exception:
                continue
            row = panel_composition(
                payload_path, repository_ids.get(str(split), set())
            )
            if row:
                panels.append(row)

    any_repository = sum(row["repository_targets_evaluated"] for row in panels)
    by_split = collections.Counter(str(row["split"]) for row in panels)

    return {
        "schema_version": "oneiros_evaluation_panel_composition_v1",
        "source_tree_sha256": source_tree_sha256(ROOT),
        "sealed_final_test_accessed": False,
        "question": (
            "do the reported Kill@8 numbers measure the balanced "
            "synthetic/repository corpus, or only its synthetic half?"
        ),
        "answer": (
            "Only the synthetic half. The evaluation panel is built from "
            "function-mode records; repository records are supervision during "
            "training and are never evaluated."
            if any_repository == 0 else
            f"{any_repository} repository targets appear across the evaluated "
            "panels; see per-artifact rows."
        ),
        "corpus_by_split": splits,
        "artifacts_audited": len(panels),
        "artifacts_by_split": dict(by_split),
        "total_repository_targets_evaluated": any_repository,
        "panels": panels,
        "reporting_consequence": (
            "Report these results as measured on held-out SYNTHETIC mutation "
            "targets, not as 'held-out functions' unqualified. Per-dataset and "
            "synthetic-versus-repository breakdowns cannot be produced for "
            "validation because the repository side has no measurements to "
            "break down. The comparison against actual Atheris and the non-LLM "
            "baselines is sound - every arm ran the identical panel - but it "
            "is a synthetic-target comparison."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--view-dir", type=Path, default=DEFAULT_VIEW)
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "results" / "v4_2_evaluation_panel_composition.json",
    )
    arguments = parser.parse_args()

    report = audit(arguments.view_dir)
    write_json(arguments.output, report)
    print(json.dumps({
        "corpus_by_split": report["corpus_by_split"],
        "artifacts_audited": report["artifacts_audited"],
        "total_repository_targets_evaluated":
            report["total_repository_targets_evaluated"],
        "answer": report["answer"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
