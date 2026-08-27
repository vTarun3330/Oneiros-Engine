"""Build the V4 unified-prompt corpus from the immutable V3 candidate.

V4 keeps every oracle-only field and split assignment from V3, repairs MBPP
specifications, derives one dataset-agnostic model-input schema, declares the
native symbols required by verified repository fragments, sanitizes behavioral
specifications, and replaces inherited function-test oracle labels with fresh
per-assertion execution evidence.
"""

from __future__ import annotations

import argparse
import ast
import builtins
import copy
import json
import re
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.test_generation_prompt import (
    PROMPT_SCHEMA_VERSION,
    sanitize_behavioral_specification,
    task_mode_for_execution_mode,
    test_format_for_execution_mode,
)
from harness.candidate_policy import validate_function_assertion
from harness.corpus import (
    function_group_id,
    record_content_hash,
    sha256_file,
    verify_corpus,
    write_json,
)
from harness.safe_execution import classify_assertions


SOURCE_DIR = ROOT / "data" / "corpus" / "v3_final_candidate"
OUTPUT_DIR = ROOT / "data" / "corpus" / "v4_unified_prompt_candidate"
MBPP_CACHE = ROOT / "data" / "mbpp" / "mbpp_cache.json"


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _node_name(node: ast.AST) -> str:
    return getattr(node, "name", "") if isinstance(
        node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
    ) else ""


def changed_target_symbols(code_under_test: str, reference_code: str) -> list[str]:
    """Find top-level functions/classes whose AST differs across the real pair."""
    try:
        buggy = ast.parse(code_under_test)
        fixed = ast.parse(reference_code)
    except SyntaxError:
        return []
    buggy_nodes = {_node_name(node): node for node in buggy.body if _node_name(node)}
    fixed_nodes = {_node_name(node): node for node in fixed.body if _node_name(node)}
    changed = [
        name for name in buggy_nodes.keys() | fixed_nodes.keys()
        if name not in buggy_nodes
        or name not in fixed_nodes
        or ast.dump(buggy_nodes[name], include_attributes=False)
        != ast.dump(fixed_nodes[name], include_attributes=False)
    ]
    return sorted(changed)


def partition_repository_source(
    code_under_test: str, target_symbols: Iterable[str],
) -> tuple[str, str]:
    """Separate target definitions from imports/helpers without changing the oracle pair."""
    target_symbols = set(target_symbols)
    if not target_symbols:
        return "", code_under_test.strip()
    try:
        tree = ast.parse(code_under_test)
    except SyntaxError:
        return "", code_under_test.strip()
    lines = code_under_test.splitlines()
    target_ranges: list[tuple[int, int]] = []
    for node in tree.body:
        if _node_name(node) in target_symbols:
            target_ranges.append((node.lineno - 1, getattr(node, "end_lineno", node.lineno)))
    if not target_ranges:
        return "", code_under_test.strip()
    target_indices = {
        index for start, end in target_ranges for index in range(start, end)
    }
    target = "\n\n".join(
        "\n".join(lines[start:end]).strip() for start, end in target_ranges
    ).strip()
    support = "\n".join(
        line for index, line in enumerate(lines) if index not in target_indices
    ).strip()
    file_markers = [line for line in lines if line.lstrip().startswith("# File:")]
    if file_markers and not target.startswith("# File:"):
        target = f"{file_markers[0]}\n{target}"
    return support, target


def _bound_names(tree: ast.AST) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                names.update(argument.arg for argument in (
                    [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]
                ))
                if node.args.vararg:
                    names.add(node.args.vararg.arg)
                if node.args.kwarg:
                    names.add(node.args.kwarg.arg)
        elif isinstance(node, ast.Import):
            names.update(alias.asname or alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            names.update(alias.asname or alias.name for alias in node.names)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Param)):
            names.add(node.id)
    return names


def _loaded_names(source: str) -> set[str]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()
    return {
        node.id for node in ast.walk(tree)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
    } - _bound_names(tree)


def repository_native_symbols(record: dict[str, Any]) -> list[str]:
    """Declare test-module globals proven available by official native execution."""
    source_symbols = _bound_names(ast.parse(record["code_under_test"]))
    unresolved: set[str] = set()
    for test in record["tests"]:
        unresolved.update(_loaded_names(test["code"]))
    unresolved -= source_symbols
    unresolved -= set(dir(builtins))
    # These are framework-provided rather than repository-specific globals.
    unresolved -= {"self"}
    return sorted(unresolved)


def repository_execution_context(record: dict[str, Any], source_context: str) -> str:
    provenance = record.get("provenance", {})
    native_symbols = repository_native_symbols(record)
    lines = [
        "The test executes in the verified native repository test module and environment.",
    ]
    test_module = provenance.get("test_file") or (
        (provenance.get("test_paths") or [""])[0]
    )
    if test_module:
        lines.append(f"Native test module: {test_module}")
    if native_symbols:
        lines.append(
            "Native test-module/framework symbols available: " + ", ".join(native_symbols)
        )
    if source_context.strip():
        lines.extend(["", "Relevant imports, constants, and helper definitions:", source_context.strip()])
    return "\n".join(lines).strip()


def _mbpp_target_symbol(item: dict[str, Any]) -> str:
    """Derive the public task target from official MBPP tests, not helper order.

    Some MBPP programs define helpers before the public function.  V3 inherited
    the first definition as ``entry_point`` for those tasks, which made otherwise
    valid official assertions fail the generation-time candidate policy.
    """
    try:
        defined = {
            _node_name(node) for node in ast.parse(item.get("code", "")).body
            if _node_name(node)
        }
    except SyntaxError as exc:
        raise RuntimeError(f"Invalid cached MBPP program for {item.get('task_id')}") from exc
    calls: Counter[str] = Counter()

    def visit(node: ast.AST, inside_defined_call: bool = False) -> None:
        is_defined_call = (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in defined
        )
        # A public task call can contain constructor/helper calls in its
        # arguments. Count only the outermost program-defined call so helpers
        # such as Pair(...) do not outweigh max_chain_length(...).
        if is_defined_call and not inside_defined_call:
            calls[node.func.id] += 1
        for child in ast.iter_child_nodes(node):
            visit(child, inside_defined_call or is_defined_call)

    for test_code in item.get("test_list", []):
        try:
            tree = ast.parse(test_code)
        except SyntaxError as exc:
            raise RuntimeError(
                f"Invalid cached MBPP test for {item.get('task_id')}: {test_code!r}"
            ) from exc
        visit(tree)
    if not calls:
        raise RuntimeError(f"Cannot derive MBPP target symbol for {item.get('task_id')}")
    return calls.most_common(1)[0][0]


def _mbpp_metadata() -> dict[int, dict[str, str]]:
    return {
        int(item["task_id"]): {
            "specification": str(item.get("text", "")).strip(),
            "entry_point": _mbpp_target_symbol(item),
        }
        for item in _load(MBPP_CACHE)
    }


def _mbpp_task_id(record: dict[str, Any]) -> int | None:
    if record.get("source", {}).get("upstream") != "mbpp":
        return None
    match = re.search(
        r"(?:^|_)mbpp_(\d+)_mut_",
        str(record.get("provenance", {}).get("upstream_record_id", "")),
    )
    return int(match.group(1)) if match else None


def _function_oracle_updates(record: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    policy_valid_tests: list[dict[str, Any]] = []
    policy_rejections: Counter[str] = Counter()
    for test in record["tests"]:
        decision = validate_function_assertion(test["code"], record["entry_point"])
        if decision.valid:
            policy_valid_tests.append(test)
        else:
            policy_rejections[decision.reason] += 1
    tests = [test["code"] for test in policy_valid_tests]
    outcomes = classify_assertions(
        tests, record["reference_code"], record["code_under_test"]
    )
    updated: list[dict[str, Any]] = []
    killing = 0
    invalid_reference_tests = 0
    for prior, outcome in zip(policy_valid_tests, outcomes):
        golden = outcome.get("golden") or {}
        mutant = outcome.get("mutant") or {}
        reference_passed = bool(golden.get("ok"))
        target_passed = bool(mutant.get("ok")) if reference_passed else False
        distinguishing = reference_passed and not target_passed
        killing += int(distinguishing)
        if not reference_passed:
            invalid_reference_tests += 1
            continue
        if distinguishing:
            oracle = "passes_reference_fails_target"
        elif target_passed:
            oracle = "passes_reference_passes_target"
        test = copy.deepcopy(prior)
        test["oracle"] = oracle
        test["distinguishing"] = distinguishing
        test["reference_execution_status"] = golden.get("status", "unknown")
        test["target_execution_status"] = mutant.get("status", "not_executed")
        updated.append(test)
    return updated, {
        "test_oracle_labels_execution_derived": True,
        "killing_test_count": killing,
        "non_killing_test_count": len(updated) - killing,
        "reference_invalid_tests_excluded": invalid_reference_tests,
        "candidate_policy_invalid_tests_excluded": sum(policy_rejections.values()),
        "candidate_policy_rejection_reasons": dict(sorted(policy_rejections.items())),
        "current_execution_policy_reverified": bool(updated),
    }


def _normalize_record(
    source_record: dict[str, Any], mbpp_metadata: dict[int, dict[str, str]],
) -> dict[str, Any]:
    record = copy.deepcopy(source_record)
    mode = record.get("quality", {}).get("execution_mode", "function_assertion")
    task_mode = task_mode_for_execution_mode(mode)
    specification = record.get("specification", "")
    task_id = _mbpp_task_id(record)
    if task_id is not None:
        metadata = mbpp_metadata.get(task_id, {})
        specification = metadata.get("specification", specification)
        if not specification:
            raise RuntimeError(f"MBPP specification unavailable for {record['id']}")
        entry_point = metadata.get("entry_point", "")
        if not entry_point:
            raise RuntimeError(f"MBPP target symbol unavailable for {record['id']}")
        record["entry_point"] = entry_point
    record["specification"] = sanitize_behavioral_specification(specification)
    record["task_mode"] = task_mode
    record["test_format"] = test_format_for_execution_mode(mode)

    if task_mode == "function":
        record["group_id"] = function_group_id(
            record["reference_code"], record["entry_point"]
        )
        record["target_symbols"] = [record["entry_point"]]
        record["support_context"] = ""
        record["prompt_code_under_test"] = record["code_under_test"]
        tests, quality_updates = _function_oracle_updates(record)
        record["tests"] = tests
        record["quality"].update(quality_updates)
    else:
        targets = changed_target_symbols(
            record["code_under_test"], record["reference_code"]
        )
        source_context, target_code = partition_repository_source(
            record["code_under_test"], targets
        )
        native_symbols = repository_native_symbols(record)
        record["target_symbols"] = targets
        record["support_context"] = repository_execution_context(record, source_context)
        record["prompt_code_under_test"] = target_code
        record["quality"].update({
            "support_context_complete_for_verified_tests": True,
            "native_test_symbols_declared": native_symbols,
            "generated_test_native_execution_required": True,
        })

    record["quality"]["unified_prompt_schema"] = PROMPT_SCHEMA_VERSION
    record["content_hash"] = record_content_hash(record)
    return record


def build(
    source_dir: Path = SOURCE_DIR,
    output_dir: Path = OUTPUT_DIR,
    workers: int = 4,
) -> dict[str, Any]:
    if workers <= 0:
        raise ValueError("workers must be positive")
    parent = verify_corpus(source_dir)
    source_records = _load(source_dir / "records.json")
    mbpp_metadata = _mbpp_metadata()

    with ThreadPoolExecutor(max_workers=workers) as pool:
        normalized_records = list(pool.map(
            lambda item: _normalize_record(item, mbpp_metadata), source_records
        ))
    excluded_records = [
        {
            "record_id": record["id"],
            "reason": "no_current-policy-reference-valid-mutation-killing-test",
            "reference_invalid_tests_excluded": record.get("quality", {}).get(
                "reference_invalid_tests_excluded", 0
            ),
        }
        for record in normalized_records
        if record.get("task_mode") == "function"
        and record.get("quality", {}).get("killing_test_count", 0) == 0
    ]
    excluded_ids = {item["record_id"] for item in excluded_records}
    records = [record for record in normalized_records if record["id"] not in excluded_ids]
    records.sort(key=lambda item: item["id"])

    output_dir.mkdir(parents=True, exist_ok=True)
    source_splits = _load(source_dir / "splits.json")
    splits = {
        split: [record_id for record_id in record_ids if record_id not in excluded_ids]
        for split, record_ids in source_splits.items()
    }
    write_json(output_dir / "splits.json", splits)
    write_json(
        output_dir / "external_eval_index.json",
        _load(source_dir / "external_eval_index.json"),
    )
    source_exclusions = source_dir / "training_exclusions.json"
    retained_training_exclusions = []
    if source_exclusions.exists():
        retained_training_exclusions = [
            item for item in _load(source_exclusions)
            if item.get("record_id") not in excluded_ids
        ]
        write_json(
            output_dir / "training_exclusions.json", retained_training_exclusions
        )
    write_json(output_dir / "reverification_exclusions.json", excluded_records)
    write_json(output_dir / "records.json", records)

    mode_counts = Counter(record["task_mode"] for record in records)
    source_counts = Counter(
        record.get("source", {}).get("upstream")
        or record.get("source", {}).get("name", "unknown")
        for record in records
    )
    manifest = {
        "schema_version": parent["schema_version"],
        "corpus_id": "oneiros-corpus-v4-unified-prompt-candidate",
        "parent_corpus": {
            "corpus_id": parent["corpus_id"],
            "records_sha256": parent["files"]["records.json"]["sha256"],
            "splits_sha256": parent["files"]["splits.json"]["sha256"],
        },
        "prompt_schema_version": PROMPT_SCHEMA_VERSION,
        "quality_gate": {
            **parent["quality_gate"],
            "unified_prompt_schema": True,
            "mbpp_specifications_retained": True,
            "behavioral_specifications_sanitized": True,
            "support_context_first_class": True,
            "repository_verified_test_symbols_declared": True,
            "function_test_oracles_execution_derived": True,
            "reference_fields_excluded_from_prompt_contract": True,
        },
        "training_records": len(records),
        "records_by_task": dict(sorted(Counter(
            record["task_type"] for record in records
        ).items())),
        "records_by_mode": dict(sorted(mode_counts.items())),
        "records_by_source": dict(sorted(source_counts.items())),
        "splits": {
            split: {
                "record_count": len(record_ids),
                "group_count": len({
                    record["group_id"] for record in records
                    if record["id"] in set(record_ids)
                }),
            }
            for split, record_ids in splits.items()
        },
        "reverification": {
            "retained_records": len(records),
            "excluded_records": len(excluded_records),
            "exclusion_reason": "no_current-policy-reference-valid-mutation-killing-test",
        },
        "external_evaluation": parent.get("external_evaluation", {}),
        "files": {},
    }
    for filename in (
        "records.json", "splits.json", "external_eval_index.json",
        "reverification_exclusions.json",
    ):
        manifest["files"][filename] = {"sha256": sha256_file(output_dir / filename)}
    exclusions = output_dir / "training_exclusions.json"
    if exclusions.exists():
        manifest["files"]["training_exclusions.json"] = {
            "sha256": sha256_file(exclusions)
        }
        manifest["training_exclusions"] = {
            "record_count": len(retained_training_exclusions),
            "reason": "canonical_retained_training_excluded",
        }
    write_json(output_dir / "manifest.json", manifest)
    verify_corpus(output_dir)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=SOURCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    print(json.dumps(build(args.source_dir, args.output_dir, args.workers), indent=2))


if __name__ == "__main__":
    main()
