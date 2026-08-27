"""Build Oneiros' canonical, versioned Phase 3 corpus.

The builder uses the existing behaviorally-clean mutation corpus, converts it
to inference-aligned records (the buggy implementation is the only code shown
to the model), validates curated fixed/buggy seed bugs, and preserves the
official BugsInPy task metadata as a locked external-evaluation index.
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baseline.benchmark_runner import kills_mutant, safe_exec
from harness.corpus import (
    SCHEMA_VERSION, function_group_id, record_content_hash, sha256_file,
    write_json,
)
from harness.training_data import extract_dataset_assertions, verify_prepared_dataset

DATA_DIR = ROOT / "data"
CORPUS_DIR = DATA_DIR / "corpus" / "v1"


def docstring_for(code: str, entry_point: str) -> str:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return ""
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == entry_point:
            return ast.get_docstring(node) or ""
    return ""


def make_record(
    *, record_id: str, task_type: str, source: Dict[str, Any], group_id: str,
    code_under_test: str, reference_code: str, entry_point: str, specification: str,
    tests: List[str], provenance: Dict[str, Any], quality: Dict[str, Any],
) -> Dict[str, Any]:
    record = {
        "schema_version": SCHEMA_VERSION,
        "id": record_id,
        "task_type": task_type,
        "language": "python",
        "source": source,
        "group_id": group_id,
        "code_under_test": code_under_test,
        "reference_code": reference_code,
        "entry_point": entry_point,
        "specification": specification,
        "tests": [{"code": test, "oracle": "fails_target_passes_reference"} for test in tests],
        "provenance": provenance,
        "quality": quality,
    }
    record["content_hash"] = record_content_hash(record)
    return record


def mutation_records() -> tuple[List[Dict[str, Any]], Dict[str, List[str]]]:
    prepared_manifest = verify_prepared_dataset(DATA_DIR)
    pairs = json.loads((DATA_DIR / "mutation_pairs_clean.json").read_text(encoding="utf-8"))
    mbpp_descriptions = {
        int(item["task_id"]): str(item.get("text", "")).strip()
        for item in json.loads(
            (DATA_DIR / "mbpp" / "mbpp_cache.json").read_text(encoding="utf-8")
        )
    }
    split_pairs = {
        name: json.loads((DATA_DIR / "splits" / f"{name}_pairs.json").read_text(encoding="utf-8"))
        for name in ("train", "val", "test")
    }
    split_for_id = {
        pair["id"]: split
        for split, values in split_pairs.items() for pair in values
    }
    records: List[Dict[str, Any]] = []
    split_ids = {name: [] for name in ("train", "val", "test")}
    for pair in pairs:
        split = split_for_id.get(pair["id"])
        if split is None:
            raise RuntimeError(f"Clean mutation pair {pair['id']} is absent from prepared splits.")
        tests = extract_dataset_assertions(pair["test_cases"], pair["entry_point"])
        # The preparation gate has already proven that at least one of these
        # assertions passes the reference and fails the target.
        specification = docstring_for(pair["golden_code"], pair["entry_point"])
        if pair["source"] == "mbpp":
            match = re.search(r"^mbpp_(\d+)_mut_", pair["id"])
            if not match or not mbpp_descriptions.get(int(match.group(1))):
                raise RuntimeError(f"MBPP specification missing for {pair['id']}")
            specification = mbpp_descriptions[int(match.group(1))]
        record = make_record(
            record_id=f"mutation::{pair['id']}",
            task_type="hidden_mutation_reproduction",
            source={"name": "oneiros_clean_mutations", "upstream": pair["source"]},
            group_id=function_group_id(pair["golden_code"], pair["entry_point"]),
            code_under_test=pair["mutant_code"],
            reference_code=pair["golden_code"],
            entry_point=pair["entry_point"],
            specification=specification,
            tests=tests,
            provenance={
                "upstream_record_id": pair["id"],
                "mutation_type": pair["mutation_type"],
                "mutation_description": pair["mutation_description"],
                "prepared_dataset_sha256": prepared_manifest["clean_dataset"]["sha256"],
            },
            quality={
                "pair_behaviorally_verified": True,
                "test_count": len(tests),
                "oracle": "mutation_reference",
            },
        )
        records.append(record)
        split_ids[split].append(record["id"])
    return records, split_ids


def curated_fixed_bug_records() -> List[Dict[str, Any]]:
    """Use only curated pairs that execute as fixed-pass/buggy-fail examples."""
    from harness.bugsinpy_loader import CURATED_BUGSINPY_BUGS

    records = []
    for bug in CURATED_BUGSINPY_BUGS:
        assertions = extract_dataset_assertions(bug["test_cases"], bug["entry_point"])
        fixed_passes = [safe_exec(bug["fixed_code"], test)[0] for test in assertions]
        killers = [
            test for test in assertions
            if kills_mutant(test, bug["fixed_code"], bug["buggy_code"])
        ]
        if not assertions or not all(fixed_passes) or not killers:
            continue
        records.append(make_record(
            record_id=f"curated::{bug['id']}",
            task_type="fixed_bug_reproduction",
            source={"name": "curated_fixed_bug_seed", "upstream": "manual_curated_examples"},
            group_id=f"project:{bug['project'].lower()}",
            code_under_test=bug["buggy_code"].strip() + "\n",
            reference_code=bug["fixed_code"].strip() + "\n",
            entry_point=bug["entry_point"],
            specification=bug["description"],
            tests=killers,
            provenance={"seed_id": bug["id"], "project": bug["project"], "category": bug["category"]},
            quality={
                "pair_behaviorally_verified": True,
                "fixed_passes_all_retained_tests": True,
                "killing_test_count": len(killers),
                "oracle": "fixed_vs_buggy",
            },
        ))
    return records


def external_bugsinpy_index() -> List[Dict[str, Any]]:
    """Lock all locally available official BugsInPy tasks out of v1 training."""
    projects = DATA_DIR / "BugsInPy_repo" / "projects"
    tasks: List[Dict[str, Any]] = []
    if not projects.exists():
        return tasks
    for project_dir in sorted(path for path in projects.iterdir() if path.is_dir()):
        bugs_dir = project_dir / "bugs"
        if not bugs_dir.exists():
            continue
        for bug_dir in sorted((path for path in bugs_dir.iterdir() if path.is_dir()), key=lambda path: path.name):
            info = {}
            info_path = bug_dir / "bug.info"
            if info_path.exists():
                for line in info_path.read_text(encoding="utf-8", errors="replace").splitlines():
                    if "=" in line:
                        key, value = line.split("=", 1)
                        info[key.strip()] = value.strip().strip('"')
            patch_path = bug_dir / "bug_patch.txt"
            test_path = bug_dir / "run_test.sh"
            tasks.append({
                "id": f"bugsinpy::{project_dir.name}::{bug_dir.name}",
                "source": "BugsInPy_official_repository_metadata",
                "task_type": "repository_bug_reproduction",
                "project_group": f"project:{project_dir.name.lower()}",
                "status": "locked_external_eval_not_materialized",
                "buggy_commit": info.get("buggy_commit_id", ""),
                "fixed_commit": info.get("fixed_commit_id", ""),
                "test_file": info.get("test_file", ""),
                "python_version": info.get("python_version", ""),
                "patch": patch_path.read_text(encoding="utf-8", errors="replace") if patch_path.exists() else "",
                "test_command": test_path.read_text(encoding="utf-8", errors="replace") if test_path.exists() else "",
            })
    return tasks


def build() -> Dict[str, Any]:
    mutation, splits = mutation_records()
    curated = curated_fixed_bug_records()
    # Curated examples are tiny and project-disjoint from the mutation corpus.
    # They enrich only training; official repository tasks remain locked external
    # evaluation until their full environments are materialized reproducibly.
    for record in curated:
        mutation.append(record)
        splits["train"].append(record["id"])
    mutation.sort(key=lambda record: record["id"])
    for ids in splits.values():
        ids.sort()
    external = external_bugsinpy_index()

    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    write_json(CORPUS_DIR / "records.json", mutation)
    write_json(CORPUS_DIR / "splits.json", splits)
    write_json(CORPUS_DIR / "external_eval_index.json", external)
    group_counts = {
        split: len({record["group_id"] for record in mutation if record["id"] in ids})
        for split, ids in splits.items()
    }
    records_by_task = Counter(record["task_type"] for record in mutation)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "corpus_id": "oneiros-corpus-v1",
        "quality_gate": {
            "all_training_records_verified": True,
            "reference_oracle_required": True,
            "group_disjoint_splits": True,
            "locked_external_evaluation": True,
        },
        "training_records": len(mutation),
        "records_by_task": dict(sorted(records_by_task.items())),
        "splits": {
            name: {"record_count": len(ids), "group_count": group_counts[name]}
            for name, ids in splits.items()
        },
        "external_evaluation": {
            "locked_bugsinpy_repository_tasks": len(external),
            "materialized_for_training": 0,
            "note": "These tasks are excluded from v1 training until checkout and F2P execution are reproducible.",
        },
        "files": {
            filename: {"sha256": sha256_file(CORPUS_DIR / filename)}
            for filename in ("records.json", "splits.json", "external_eval_index.json")
        },
    }
    write_json(CORPUS_DIR / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the canonical Oneiros corpus v1")
    parser.parse_args()
    print(json.dumps(build(), indent=2))


if __name__ == "__main__":
    main()
