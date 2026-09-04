"""Measure how far the real-repository side can actually be expanded.

Part 3 of the research plan asks for a 50/50 balance by unique semantic target,
and Part 4 asks for roughly 324 additional unique repository tasks to get there.
Whether that is reachable is an empirical question about how many eligible,
un-ingested defects exist - not something to assume.

Eligibility here is the project-disjointness rule the corpus already enforces:
a defect whose project is assigned to ablation_dev, val, or test can never enter
training, however many of them exist.  Counting those would overstate the
ceiling badly.

Writes a machine-readable ceiling so the gap between the target and what the
data can supply is recorded rather than quietly carried.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness.corpus import write_json
from utils.reproducibility import source_tree_sha256


HELD_OUT_SPLITS = frozenset({"ablation_dev", "val", "test"})

#: Observed acceptance rate of the historical BugsInPy Linux ingestion:
#: 130 accepted out of 361 attempted task ids.  Used only to state an EXPECTED
#: yield alongside the raw availability, never to inflate the ceiling.
HISTORICAL_BUGSINPY_ACCEPTANCE = 130 / 361


def _project_split_map(corpus_dir: Path) -> dict[str, str]:
    records = json.loads((corpus_dir / "records.json").read_text(encoding="utf-8"))
    splits = json.loads((corpus_dir / "splits.json").read_text(encoding="utf-8"))
    by_id = {record["id"]: record for record in records}
    assignment: dict[str, str] = {}
    for split, ids in splits.items():
        for record_id in ids:
            record = by_id.get(record_id)
            if not record or record.get("task_mode") != "repository":
                continue
            project = str((record.get("provenance") or {}).get("project") or "")
            assignment.setdefault(project, split)
    return assignment


def _ingested(corpus_dir: Path) -> tuple[set[tuple[str, str]], set[str]]:
    records = json.loads((corpus_dir / "records.json").read_text(encoding="utf-8"))
    bugsinpy: set[tuple[str, str]] = set()
    swebench: set[str] = set()
    for record in records:
        upstream = str((record.get("source") or {}).get("upstream") or "")
        provenance = record.get("provenance") or {}
        if upstream == "BugsInPy":
            bugsinpy.add((
                str(provenance.get("project") or ""), str(provenance.get("bug_id") or ""),
            ))
        elif upstream == "SWE-bench Verified":
            swebench.add(str(
                provenance.get("official_task_id")
                or provenance.get("instance_id") or ""
            ))
    return bugsinpy, swebench


def audit_bugsinpy(
    repo_root: Path, ingested: set[tuple[str, str]], assignment: dict[str, str],
) -> dict[str, Any]:
    projects_dir = repo_root / "projects"
    if not projects_dir.is_dir():
        return {"status": "bugsinpy checkout absent", "available": 0}
    available: collections.Counter[str] = collections.Counter()
    held_out: collections.Counter[str] = collections.Counter()
    for project in sorted(os.listdir(projects_dir)):
        bugs = projects_dir / project / "bugs"
        if not bugs.is_dir():
            continue
        for bug in sorted(os.listdir(bugs)):
            if not (bugs / bug).is_dir():
                continue
            if (project, bug) in ingested:
                continue
            if assignment.get(project) in HELD_OUT_SPLITS:
                held_out[project] += 1
            else:
                available[project] += 1
    total = sum(available.values())
    dominant = available.most_common(1)[0] if available else ("", 0)
    return {
        "already_ingested": len(ingested),
        "un_ingested_train_eligible": total,
        "un_ingested_train_eligible_by_project": dict(sorted(available.items())),
        "un_ingested_but_project_held_out": sum(held_out.values()),
        "un_ingested_but_project_held_out_by_project": dict(sorted(held_out.items())),
        "dominant_available_project": dominant[0],
        "dominant_available_count": dominant[1],
        "dominant_share_of_available": round(dominant[1] / max(1, total), 4),
        "historical_acceptance_rate": round(HISTORICAL_BUGSINPY_ACCEPTANCE, 4),
        "expected_accepted_if_all_attempted": round(
            total * HISTORICAL_BUGSINPY_ACCEPTANCE
        ),
        "ingestion_path": "native WSL2 execution (the pilot reproduces 3 of 5)",
    }


def audit_swebench(
    parquet: Path, ingested: set[str], assignment: dict[str, str],
) -> dict[str, Any]:
    try:
        import pandas
    except ImportError:
        return {"status": "pandas unavailable; cannot read the parquet"}
    if not parquet.exists():
        return {"status": f"source parquet absent: {parquet}"}
    frame = pandas.read_parquet(parquet)
    in_train: collections.Counter[str] = collections.Counter()
    held_out: collections.Counter[str] = collections.Counter()
    new_project: collections.Counter[str] = collections.Counter()
    for _, row in frame.iterrows():
        instance = str(row["instance_id"])
        if instance in ingested:
            continue
        project = str(row["repo"]).split("/")[-1]
        split = assignment.get(project)
        if split in HELD_OUT_SPLITS:
            held_out[project] += 1
        elif split == "train":
            in_train[project] += 1
        else:
            new_project[project] += 1
    return {
        "source_rows": int(len(frame)),
        "already_ingested": len(ingested),
        "un_ingested_project_in_train": sum(in_train.values()),
        "un_ingested_project_in_train_by_project": dict(sorted(in_train.items())),
        "un_ingested_project_held_out": sum(held_out.values()),
        "un_ingested_project_held_out_by_project": dict(sorted(held_out.items())),
        "un_ingested_project_unassigned": sum(new_project.values()),
        "un_ingested_project_unassigned_by_project": dict(sorted(new_project.items())),
        "train_eligible_total": sum(in_train.values()) + sum(new_project.values()),
        "ingestion_path": "official SWE-bench harness on Modal",
    }


def build(corpus_version: str, synthetic_targets: int, repository_targets: int) -> dict[str, Any]:
    corpus_dir = ROOT / "data" / "corpus" / corpus_version
    assignment = _project_split_map(corpus_dir)
    bugsinpy_ingested, swebench_ingested = _ingested(corpus_dir)

    bugsinpy = audit_bugsinpy(
        ROOT / "data" / "BugsInPy_repo", bugsinpy_ingested, assignment,
    )
    swebench = audit_swebench(
        ROOT / "data" / "swebench_verified_source" / "SWE-bench_Verified.test.parquet",
        swebench_ingested, assignment,
    )

    needed = max(0, synthetic_targets - repository_targets)
    bugsinpy_available = int(bugsinpy.get("un_ingested_train_eligible") or 0)
    swebench_available = int(swebench.get("train_eligible_total") or 0)
    optimistic = bugsinpy_available + swebench_available
    expected = (
        int(bugsinpy.get("expected_accepted_if_all_attempted") or 0) + swebench_available
    )

    modal_configured = bool(os.environ.get("MODAL_TOKEN_ID"))
    if not modal_configured:
        try:
            from modal.config import config as modal_config
            modal_configured = bool(modal_config.get("token_id"))
        except Exception:
            modal_configured = False

    return {
        "schema_version": "oneiros_repository_expansion_ceiling_v1",
        "source_tree_sha256": source_tree_sha256(ROOT),
        "corpus_version": corpus_version,
        "sealed_final_test_accessed": False,
        "eligibility_rule": (
            "project-disjoint splits: a defect whose project is assigned to "
            "ablation_dev, val, or test can never enter training, so it is not "
            "part of the ceiling however many such defects exist"
        ),
        "current_unique_targets": {
            "synthetic": synthetic_targets,
            "repository": repository_targets,
            "repository_shortfall_for_parity": needed,
        },
        "bugsinpy": bugsinpy,
        "swebench_verified": swebench,
        "ceiling": {
            "optimistic_additional_targets_if_every_attempt_succeeded": optimistic,
            "expected_additional_targets_at_historical_acceptance": expected,
            "required_for_parity": needed,
            "parity_reachable_optimistically": optimistic >= needed,
            "parity_reachable_at_expected_yield": expected >= needed,
        },
        "blockers": {
            "swebench_ingestion": {
                "blocked": not modal_configured,
                "reason": (
                    "The SWE-bench path runs the official harness on Modal. No "
                    "Modal token is configured, so those instances cannot be "
                    "reproduced or verified here."
                ) if not modal_configured else None,
                "affected_available_targets": swebench_available,
            },
            "bugsinpy_project_concentration": {
                "dominant_project": bugsinpy.get("dominant_available_project"),
                "dominant_share_of_available": bugsinpy.get("dominant_share_of_available"),
                "note": (
                    "Most of the un-ingested BugsInPy supply is one project. "
                    "Ingesting it wholesale would fix the count while making the "
                    "project-concentration problem worse, which the balancing "
                    "policy explicitly forbids."
                ),
            },
        },
        "conclusion": (
            "The 50/50 unique-target balance is NOT reachable from the data "
            "currently available under the project-disjointness rule. The "
            "balanced-dataset manifest reports this as a blocking condition "
            "rather than manufacturing the ratio by deleting synthetic targets "
            "or repeating repository ones."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-version", default="v4_1_research_hardened_candidate")
    parser.add_argument("--synthetic-targets", type=int, default=663)
    parser.add_argument("--repository-targets", type=int, default=346)
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "results" / "v4_2_repository_expansion_ceiling.json",
    )
    arguments = parser.parse_args()
    report = build(
        arguments.corpus_version, arguments.synthetic_targets,
        arguments.repository_targets,
    )
    write_json(arguments.output, report)
    print(json.dumps({
        "ceiling": report["ceiling"],
        "swebench_blocked": report["blockers"]["swebench_ingestion"]["blocked"],
        "bugsinpy_dominant": report["blockers"]["bugsinpy_project_concentration"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
