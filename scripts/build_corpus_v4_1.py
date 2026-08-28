"""Build the immutable V4.1 research-hardened corpus from frozen V4.

The build preserves V4, removes gold-test-driven context selection, replaces
reference-diff target selection with public/buggy-side derivation, records
field lineage, freezes a training-only ablation-dev split, and emits the
required leakage/exclusion artifacts.
"""

from __future__ import annotations

import argparse
import ast
import builtins
import csv
import difflib
import hashlib
import io
import json
import subprocess
import sys
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.prompt_provenance import build_repository_prompt_context
from engine.test_generation_prompt import (
    PROMPT_SCHEMA_VERSION,
    sanitize_behavioral_specification,
)
from harness.corpus import record_content_hash, sha256_file, verify_corpus, write_json
from scripts.audit_prompt_lineage import audit_records


SOURCE_DIR = ROOT / "data" / "corpus" / "v4_unified_prompt_candidate"
V3_DIR = ROOT / "data" / "corpus" / "v3_final_candidate"
OUTPUT_DIR = ROOT / "data" / "corpus" / "v4_1_research_hardened_candidate"
CACHE_DIR = ROOT / "data" / "repository_context_v4_1_cache"
ABLATION_FRACTION = 0.10


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
        capture_output=True, check=True,
    )
    return result.stdout.strip()


def _dataset(record: dict[str, Any]) -> str:
    source = record.get("source", {})
    return str(source.get("upstream") or source.get("name") or "unknown")


def _mutation_family(record: dict[str, Any]) -> str:
    provenance = record.get("provenance", {})
    quality = record.get("quality", {})
    return str(
        provenance.get("mutation_family")
        or provenance.get("mutation_type")
        or provenance.get("category")
        or quality.get("mutation_family")
        or ("real_repository_defect" if record.get("task_mode") == "repository" else "unknown")
    )


def _test_module_path(record: dict[str, Any]) -> str:
    provenance = record.get("provenance", {})
    return str(
        provenance.get("test_file")
        or ((provenance.get("test_paths") or [""])[0])
        or ""
    )


def _cache_path(record: dict[str, Any], test_path: str) -> Path:
    provenance = record.get("provenance", {})
    identity = "|".join((
        str(provenance.get("repository") or provenance.get("repository_url") or ""),
        str(provenance.get("base_commit") or provenance.get("buggy_commit") or ""),
        test_path,
    ))
    return CACHE_DIR / f"{hashlib.sha256(identity.encode()).hexdigest()}.py"


def _local_bugsinpy_test_source(record: dict[str, Any], test_path: str) -> str:
    provenance = record.get("provenance", {})
    project = str(provenance.get("project", ""))
    commit = str(provenance.get("buggy_commit", ""))
    repository = ROOT / "data" / "bugsinpy_v2_ingestion" / "repositories" / project
    if not (repository.exists() and commit and test_path):
        return ""
    result = subprocess.run(
        ["git", "-C", str(repository), "show", f"{commit}:{test_path}"],
        text=True, capture_output=True, encoding="utf-8", errors="replace",
    )
    return result.stdout if result.returncode == 0 else ""


def _raw_github_source(record: dict[str, Any], test_path: str) -> str:
    provenance = record.get("provenance", {})
    repository = str(provenance.get("repository", ""))
    if not repository:
        repository_url = str(provenance.get("repository_url", ""))
        marker = "github.com/"
        if marker in repository_url:
            repository = repository_url.split(marker, 1)[1].removesuffix(".git").strip("/")
    commit = str(provenance.get("base_commit") or provenance.get("buggy_commit") or "")
    if not repository or not commit or not test_path:
        return ""
    url = f"https://raw.githubusercontent.com/{repository}/{commit}/{test_path}"
    request = urllib.request.Request(url, headers={"User-Agent": "Oneiros-V4.1-corpus-builder"})
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            return response.read().decode("utf-8", errors="replace")
    except (OSError, urllib.error.URLError, urllib.error.HTTPError):
        return ""


def repository_test_environment_source(
    record: dict[str, Any], *, allow_network: bool,
) -> tuple[str, str]:
    """Load an immutable buggy-revision module; gold test bodies are removed later."""
    test_path = _test_module_path(record)
    if not test_path:
        return "", "unavailable"
    cache = _cache_path(record, test_path)
    if cache.exists():
        return cache.read_text(encoding="utf-8"), "immutable_local_cache"
    source = _local_bugsinpy_test_source(record, test_path)
    origin = "buggy_git_object"
    if not source and allow_network:
        source = _raw_github_source(record, test_path)
        origin = "buggy_revision_public_source"
    if source:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(source, encoding="utf-8")
        return source, origin
    return "", "unavailable"


def _bound_names(source: str) -> set[str]:
    try:
        tree = ast.parse(str(source or ""))
    except SyntaxError:
        return set()
    result: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            result.add(node.name)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                result.update(arg.arg for arg in (
                    *node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs
                ))
        elif isinstance(node, ast.Import):
            result.update(alias.asname or alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            result.update(alias.asname or alias.name for alias in node.names)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Param)):
            result.add(node.id)
    return result


def _loaded_names(source: str) -> set[str]:
    try:
        tree = ast.parse(str(source or ""))
    except SyntaxError:
        return set()
    return {
        node.id for node in ast.walk(tree)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
    } - _bound_names(source)


def _normalise_fragment(value: str) -> str:
    return " ".join(str(value or "").split()).strip()


def _remove_reference_only_fix_lines(
    specification: str, buggy_source: str, reference_source: str,
) -> tuple[str, list[str]]:
    """Remove exact fixed-only code statements while retaining behavior prose."""
    normal_buggy = _normalise_fragment(buggy_source)
    fixed_only = []
    for line in difflib.ndiff(buggy_source.splitlines(), reference_source.splitlines()):
        if not line.startswith("+ "):
            continue
        fragment = _normalise_fragment(line[2:])
        if (
            len(fragment) >= 8
            and not fragment.startswith("#")
            and fragment not in normal_buggy
        ):
            fixed_only.append(fragment)
    retained: list[str] = []
    removed_hashes: list[str] = []
    for line in str(specification or "").splitlines():
        normalized = _normalise_fragment(line)
        matches = [
            fragment for fragment in fixed_only
            if fragment in normalized or (normalized and normalized in fragment)
        ]
        if matches:
            removed_hashes.extend(
                hashlib.sha256(fragment.encode("utf-8")).hexdigest()
                for fragment in matches
            )
            continue
        retained.append(line.rstrip())
    return "\n".join(retained).strip(), list(dict.fromkeys(removed_hashes))


def _normalise_record(
    source_record: dict[str, Any], *, allow_network: bool,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    record = json.loads(json.dumps(source_record))
    sanitized = sanitize_behavioral_specification(record.get("specification", ""))
    sanitized, removed_fix_hashes = _remove_reference_only_fix_lines(
        sanitized, record["code_under_test"], record["reference_code"]
    )
    record["specification"] = sanitized
    record["quality"]["unified_prompt_schema"] = PROMPT_SCHEMA_VERSION
    record["quality"]["fault_localization_assumed"] = True
    record["quality"]["fault_localization_scope"] = "outside_test_generation"
    record["quality"]["reference_only_specification_lines_removed"] = len(
        removed_fix_hashes
    )
    record["quality"]["removed_specification_fragment_hashes"] = removed_fix_hashes
    exclusion = None
    if record.get("task_mode") == "function":
        record["field_lineage"] = {
            "task_mode": ["static_config:execution_mode"],
            "test_format": ["static_config:execution_mode"],
            "specification": ["upstream_public:behavioral_specification"],
            "prompt_code_under_test": ["buggy_revision:function_mutant"],
            "target_symbols": ["explicit_public_entry_point:target_symbols"],
            "support_context": ["static_config:no_additional_context"],
        }
        record["quality"]["localization_source"] = "explicit_public_entry_point"
    else:
        module_source, module_origin = repository_test_environment_source(
            record, allow_network=allow_network
        )
        context = build_repository_prompt_context(
            buggy_localized_source=record["code_under_test"],
            specification=record.get("specification", ""),
            execution_mode=record["quality"]["execution_mode"],
            declared_entry_point=record.get("entry_point", ""),
            public_test_module_path=_test_module_path(record),
            buggy_test_environment_source=module_source,
        )
        record["target_symbols"] = list(context.target_symbols)
        record["prompt_code_under_test"] = context.prompt_code_under_test
        record["support_context"] = context.support_context
        record["field_lineage"] = {
            field: list(values) for field, values in context.field_lineage.items()
        }
        available = _bound_names(record["code_under_test"]) | _bound_names(module_source)
        available |= set(dir(builtins)) | {"self", "pytest", "unittest"}
        unresolved: set[str] = set()
        for test in record.get("tests", []):
            unresolved |= _loaded_names(test.get("code", "")) - available
        context_complete = not unresolved and bool(module_source)
        record["quality"].update({
            "localization_source": context.localization_source,
            "localized_region_source": "benchmark_declared_affected_region",
            "support_context_gold_test_independent": True,
            "support_context_source": module_origin,
            "support_context_complete_for_verified_tests": context_complete,
            "completion_context_unresolved_symbols": sorted(unresolved),
            "native_test_symbols_declared": [],
        })
        if not context_complete:
            exclusion = {
                "record_id": record["id"],
                "reason": "independent_repository_context_incomplete",
                "dataset": _dataset(record),
                "mutation_family": _mutation_family(record),
                "unresolved_symbols": sorted(unresolved),
                "test_environment_source": module_origin,
            }
    record["content_hash"] = record_content_hash(record)
    return record, exclusion


def _fixed_ablation_dev(
    records_by_id: dict[str, dict[str, Any]], train_ids: list[str],
) -> tuple[list[str], list[str], dict[str, Any]]:
    group_ids: dict[str, list[str]] = defaultdict(list)
    for record_id in train_ids:
        group_ids[records_by_id[record_id]["group_id"]].append(record_id)
    selected_groups = {
        group for group in group_ids
        if int(hashlib.sha256(f"oneiros-v4.1-ablation-dev|{group}".encode()).hexdigest()[:8], 16)
        / 0xFFFFFFFF < ABLATION_FRACTION
    }
    # Guarantee every training dataset appears without selecting individual
    # variants from a semantic group.
    groups_by_dataset: dict[str, set[str]] = defaultdict(set)
    for group, ids in group_ids.items():
        for record_id in ids:
            groups_by_dataset[_dataset(records_by_id[record_id])].add(group)
    for dataset, groups in groups_by_dataset.items():
        if not (selected_groups & groups):
            selected_groups.add(min(groups, key=lambda value: hashlib.sha256(
                f"oneiros-v4.1-ablation-dev|{dataset}|{value}".encode()
            ).hexdigest()))
    dev_ids = [record_id for record_id in train_ids if records_by_id[record_id]["group_id"] in selected_groups]
    retained_train = [record_id for record_id in train_ids if record_id not in set(dev_ids)]
    manifest = {
        "schema_version": "oneiros_ablation_dev_v1",
        "selection_policy": "training-groups-hash-sampled-10pct-with-dataset-minimum",
        "selection_salt": "oneiros-v4.1-ablation-dev",
        "source_split": "v4_train_only",
        "target_fraction": ABLATION_FRACTION,
        "record_count": len(dev_ids),
        "semantic_group_count": len(selected_groups),
        "record_ids_sha256": _sha256_json(dev_ids),
        "semantic_group_ids_sha256": _sha256_json(sorted(selected_groups)),
        "records_by_dataset": dict(sorted(Counter(
            _dataset(records_by_id[item]) for item in dev_ids
        ).items())),
        "records_by_mutation_family": dict(sorted(Counter(
            _mutation_family(records_by_id[item]) for item in dev_ids
        ).items())),
        "validation_overlap": 0,
        "test_overlap": 0,
        "frozen_before_results": True,
    }
    return retained_train, dev_ids, manifest


def _exclusion_artifacts(output_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    exclusions = _load(SOURCE_DIR / "reverification_exclusions.json")
    v3_by_id = {
        record["id"]: record for record in _load(V3_DIR / "records.json")
    }
    rows: list[dict[str, Any]] = []
    for item in exclusions:
        record = v3_by_id.get(item.get("record_id"), {})
        rows.append({
            "record_id": item.get("record_id", ""),
            "dataset": _dataset(record) if record else "unknown",
            "mutation_family": _mutation_family(record) if record else "unknown",
            "split": next((
                name for name, ids in _load(V3_DIR / "splits.json").items()
                if item.get("record_id") in set(ids)
            ), "unknown"),
            "reason": item.get("reason", "unknown"),
            "source_task": record.get("provenance", {}).get("upstream_record_id", ""),
            "execution_failure_type": (
                "reference_invalid" if item.get("reference_invalid_tests_excluded") else "no_killing_test"
            ),
        })
    analysis = {
        "schema_version": "oneiros_v4_1_exclusion_analysis_v1",
        "excluded_records": len(rows),
        "by_dataset": dict(sorted(Counter(row["dataset"] for row in rows).items())),
        "by_mutation_family": dict(sorted(Counter(row["mutation_family"] for row in rows).items())),
        "by_split": dict(sorted(Counter(row["split"] for row in rows).items())),
        "by_reason": dict(sorted(Counter(row["reason"] for row in rows).items())),
        "by_execution_failure_type": dict(sorted(Counter(
            row["execution_failure_type"] for row in rows
        ).items())),
        "bias_statement": (
            "Exclusions are not assumed unbiased; dataset and mutation-family distributions "
            "must be interpreted using the counts in this artifact."
        ),
        "records": rows,
    }
    write_json(output_dir / "exclusion_analysis.json", analysis)
    with output_dir.joinpath("exclusion_analysis.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["record_id"])
        writer.writeheader()
        writer.writerows(rows)
    summary = (
        "# V4.1 exclusion analysis\n\n"
        f"V4.1 retains the V4 policy-valid corpus after {len(rows)} V3 records were "
        "removed because they did not satisfy the current candidate/reference oracle policy.\n\n"
        "The exclusions are not assumed to be uniformly distributed. See "
        "`exclusion_analysis.json` and `exclusion_analysis.csv` for dataset, family, "
        "split, source-task, and execution-failure counts.\n"
    )
    output_dir.joinpath("EXCLUSION_ANALYSIS.md").write_text(summary, encoding="utf-8")
    return rows, analysis


def build(
    source_dir: Path = SOURCE_DIR,
    output_dir: Path = OUTPUT_DIR,
    *,
    workers: int = 16,
    allow_network: bool = True,
) -> dict[str, Any]:
    if workers <= 0:
        raise ValueError("workers must be positive")
    parent = verify_corpus(source_dir)
    source_records = _load(source_dir / "records.json")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        outcomes = list(pool.map(
            lambda record: _normalise_record(record, allow_network=allow_network),
            source_records,
        ))
    records = sorted((item[0] for item in outcomes), key=lambda item: item["id"])
    new_exclusions = [item[1] for item in outcomes if item[1] is not None]
    records_by_id = {record["id"]: record for record in records}

    parent_splits = _load(source_dir / "splits.json")
    train_ids, dev_ids, ablation_manifest = _fixed_ablation_dev(
        records_by_id, list(parent_splits["train"])
    )
    splits = {
        "train": train_ids,
        "ablation_dev": dev_ids,
        "val": list(parent_splits["val"]),
        "test": list(parent_splits["test"]),
    }
    ablation_manifest["validation_overlap"] = len(set(dev_ids) & set(splits["val"]))
    ablation_manifest["test_overlap"] = len(set(dev_ids) & set(splits["test"]))

    parent_training_exclusions = _load(source_dir / "training_exclusions.json")
    exclusions_by_id = {item["record_id"]: item for item in parent_training_exclusions}
    exclusions_by_id.update({item["record_id"]: item for item in new_exclusions})
    training_exclusions = [exclusions_by_id[key] for key in sorted(exclusions_by_id)]

    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "records.json", records)
    write_json(output_dir / "splits.json", splits)
    write_json(output_dir / "external_eval_index.json", _load(source_dir / "external_eval_index.json"))
    write_json(output_dir / "training_exclusions.json", training_exclusions)
    write_json(output_dir / "ablation_dev_manifest.json", ablation_manifest)
    leakage = audit_records(records)
    write_json(output_dir / "leakage_audit.json", leakage)
    reverification_rows, exclusion_analysis = _exclusion_artifacts(output_dir)

    excluded_ids = {item["record_id"] for item in training_exclusions}
    context_eligible_train = [item for item in train_ids if item not in excluded_ids]
    mode_counts = Counter(record["task_mode"] for record in records)
    source_counts = Counter(_dataset(record) for record in records)
    family_counts = Counter(_mutation_family(record) for record in records)
    manifest = {
        "schema_version": parent["schema_version"],
        "corpus_id": "oneiros-corpus-v4.1-research-hardened-candidate",
        "version": "v4_1_research_hardened_candidate",
        "source_commit": _git_commit(),
        "parent_corpus": {
            "corpus_id": parent["corpus_id"],
            "version": "v4_unified_prompt_candidate",
            "records_sha256": parent["files"]["records.json"]["sha256"],
            "splits_sha256": parent["files"]["splits.json"]["sha256"],
        },
        "prompt_schema_version": PROMPT_SCHEMA_VERSION,
        "prompt_budgets": {
            "function": 512,
            "repository": 1024,
            "sequence": 2048,
            "function_completion": 128,
            "repository_completion": 1024,
            "compaction": "section_aware_ast_units_before_chat_v4_1",
        },
        "quality_gate": {
            **parent["quality_gate"],
            "unified_prompt_schema": True,
            "prompt_field_lineage_recorded": True,
            "gold_test_independent_support_context": True,
            "buggy_side_target_selection": True,
            "fixed_ablation_dev_split": True,
            "repository_context_incomplete_records_excluded_from_training": True,
        },
        "record_count": len(records),
        "training_records": len(records),
        "context_eligible_train_count": len(context_eligible_train),
        "records_by_source": dict(sorted(source_counts.items())),
        "records_by_mutation_family": dict(sorted(family_counts.items())),
        "records_by_mode": dict(sorted(mode_counts.items())),
        "splits": {
            name: {
                "record_count": len(ids),
                "group_count": len({records_by_id[item]["group_id"] for item in ids}),
                "record_ids_sha256": _sha256_json(ids),
            }
            for name, ids in splits.items()
        },
        "excluded_record_count": len(reverification_rows),
        "exclusion_reasons": exclusion_analysis["by_reason"],
        "training_exclusions": {
            "record_count": len(training_exclusions),
            "reasons": dict(sorted(Counter(
                item.get("reason", "unknown") for item in training_exclusions
            ).items())),
        },
        "leakage_summary": {
            key: leakage[key] for key in (
                "records_scanned", "schema_failures", "verbatim_reference_leaks",
                "partial_reference_overlap_flags", "pending_manual_review_flags",
                "reviewed_manual_flags", "gold_test_lineage_failures",
                "gold_patch_lineage_failures", "oracle_lineage_failures",
            )
        },
        "external_evaluation": parent.get("external_evaluation", {}),
        "files": {},
    }
    filenames = (
        "records.json", "splits.json", "external_eval_index.json",
        "training_exclusions.json", "ablation_dev_manifest.json",
        "leakage_audit.json", "exclusion_analysis.json", "exclusion_analysis.csv",
        "EXCLUSION_ANALYSIS.md",
    )
    for filename in filenames:
        manifest["files"][filename] = {"sha256": sha256_file(output_dir / filename)}
    manifest["corpus_sha256"] = manifest["files"]["records.json"]["sha256"]
    manifest["split_sha256"] = manifest["files"]["splits.json"]["sha256"]
    manifest["leakage_audit_sha256"] = manifest["files"]["leakage_audit.json"]["sha256"]
    manifest["exclusion_ledger_sha256"] = manifest["files"]["training_exclusions.json"]["sha256"]
    write_json(output_dir / "manifest.json", manifest)
    verify_corpus(output_dir)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=SOURCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    print(json.dumps(build(
        args.source_dir, args.output_dir,
        workers=args.workers, allow_network=not args.offline,
    ), indent=2))


if __name__ == "__main__":
    main()
