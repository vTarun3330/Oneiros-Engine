"""Versioned canonical corpus helpers for Oneiros SFT/DPO."""
from __future__ import annotations

import ast
import functools
import hashlib
import json
import os
import re
import tempfile
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple


SCHEMA_VERSION = 1
REQUIRED_RECORD_FIELDS = {
    "schema_version", "id", "task_type", "language", "source", "group_id",
    "code_under_test", "reference_code", "entry_point", "specification", "tests",
    "provenance", "quality", "content_hash",
}
REPOSITORY_EXECUTION_MODES = {
    "repository_pytest_fragment": ({
        "official_repository_pytest_reproduction",
        "official_repository_swebench_reproduction",
    }, "pytest_fragment"),
    "repository_unittest_fragment": ({
        "official_repository_unittest_reproduction",
    }, "unittest_fragment"),
}
CORPUS_VERSION_PATTERN = re.compile(
    r"v[1-9][0-9]*(?:_[a-z0-9][a-z0-9_-]*)?"
)


def valid_corpus_version(value: str) -> bool:
    """Accept safe versioned corpus directory names without allowing paths."""
    return bool(CORPUS_VERSION_PATTERN.fullmatch(value))


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@functools.lru_cache(maxsize=512)
def semantic_python(source: str) -> str:
    """Return a formatting/comment-insensitive identity for valid Python."""
    tree = _parse_python(source, "<semantic-python>")
    return ast.dump(tree, annotate_fields=True, include_attributes=False)


def normalize_text(value: str) -> str:
    """Normalize descriptive text without changing Python string literals."""
    return re.sub(r"\s+", " ", value.replace("\r\n", "\n").replace("\r", "\n")).strip()


def function_group_id(code: str, entry_point: str = "") -> str:
    """Group semantically identical functions despite formatting differences."""
    payload = {"entry_point": entry_point, "reference_ast": semantic_python(code)}
    return f"function:semantic:{sha256_text(canonical_json(payload))}"


def semantic_supervision_key(record: Dict[str, Any]) -> Tuple[str, str]:
    """Return stable prompt/supervision identities for duplicate/conflict gates."""
    mode = record.get("quality", {}).get("execution_mode", "function_assertion")
    test_asts = sorted(semantic_python(test["code"]) for test in record["tests"])
    prompt_payload = {
        "execution_mode": mode,
        "target_ast": semantic_python(
            record.get("prompt_code_under_test", record["code_under_test"])
        ),
        "entry_point": record.get("entry_point", ""),
        "specification": normalize_text(record.get("specification", "")),
        "task_mode": record.get("task_mode", ""),
        "test_format": record.get("test_format", ""),
        "target_symbols": list(record.get("target_symbols", [])),
        "support_context": normalize_text(record.get("support_context", "")),
    }
    supervision_payload = {
        **prompt_payload,
        "reference_ast": semantic_python(record["reference_code"]),
        "test_asts": test_asts,
    }
    return (
        sha256_text(canonical_json(prompt_payload)),
        sha256_text(canonical_json(supervision_payload)),
    )


def record_content_hash(record: Dict[str, Any]) -> str:
    payload = {key: value for key, value in record.items() if key != "content_hash"}
    return sha256_text(canonical_json(payload))


def official_evidence_verifies_pair(evidence: Dict[str, Any]) -> bool:
    """Accept native return-code or SWE-bench structured report evidence."""
    if (
        evidence.get("fixed_returncode") == 0
        and isinstance(evidence.get("buggy_returncode"), int)
        and evidence["buggy_returncode"] != 0
    ):
        return True
    if not (
        evidence.get("fixed_pass_verified") is True
        and evidence.get("buggy_fail_verified") is True
    ):
        return False
    base = evidence.get("base_counts") or {}
    fixed = evidence.get("fixed_counts") or {}
    return (
        base.get("f2p_failure", 0) > 0
        and base.get("f2p_success", 0) == 0
        and base.get("p2p_failure", 0) == 0
        and fixed.get("f2p_success", 0) > 0
        and fixed.get("f2p_failure", 0) == 0
        and fixed.get("p2p_failure", 0) == 0
    )


def write_json(path: Path, value: Any) -> None:
    """Atomically publish JSON so interruption cannot leave a partial checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    temporary_name = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="\n", delete=False,
            dir=path.parent, prefix=f".{path.name}.", suffix=".tmp",
        ) as handle:
            temporary_name = handle.name
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if temporary_name and os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _parse_python(source: str, filename: str) -> ast.AST:
    """Parse legacy-valid code without flooding corpus validation with warnings."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        return ast.parse(source, filename=filename)


def _compile_test(source: str) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        compile(source, "<canonical-test>", "exec")


def load_corpus_split(corpus_dir: Path, split: str) -> List[Dict[str, Any]]:
    verify_corpus(corpus_dir)
    records = json.loads((corpus_dir / "records.json").read_text(encoding="utf-8"))
    split_ids = json.loads((corpus_dir / "splits.json").read_text(encoding="utf-8"))[split]
    by_id = {record["id"]: record for record in records}
    return [by_id[record_id] for record_id in split_ids]


def verify_corpus(corpus_dir: Path) -> Dict[str, Any]:
    """Fail closed unless canonical records, splits, and hashes all agree."""
    manifest_path = corpus_dir / "manifest.json"
    if not manifest_path.exists():
        raise RuntimeError("Canonical corpus manifest is missing. Run build_corpus_v1.py first.")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("Canonical corpus manifest is unreadable.") from exc
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError("Unsupported or missing canonical corpus schema version.")
    if not manifest.get("quality_gate", {}).get("all_training_records_verified"):
        raise RuntimeError("Canonical corpus lacks a verified training quality gate.")

    files = manifest.get("files", {})
    for filename in ("records.json", "splits.json", "external_eval_index.json"):
        path = corpus_dir / filename
        if not path.exists() or sha256_file(path) != files.get(filename, {}).get("sha256"):
            raise RuntimeError(f"Canonical corpus file '{filename}' does not match the manifest.")

    exclusions_path = corpus_dir / "training_exclusions.json"
    if "training_exclusions.json" in files:
        if (
            not exclusions_path.exists()
            or sha256_file(exclusions_path) != files["training_exclusions.json"].get("sha256")
        ):
            raise RuntimeError("Canonical training exclusions do not match the manifest.")

    records = json.loads((corpus_dir / "records.json").read_text(encoding="utf-8"))
    split_ids = json.loads((corpus_dir / "splits.json").read_text(encoding="utf-8"))
    expected_splits = (
        {"train", "ablation_dev", "val", "test"}
        if manifest.get("quality_gate", {}).get("fixed_ablation_dev_split")
        else {"train", "val", "test"}
    )
    if not isinstance(records, list) or set(split_ids) != expected_splits:
        raise RuntimeError("Canonical corpus records or split definitions are malformed.")
    exclusions = []
    excluded_ids: set[str] = set()
    if exclusions_path.exists():
        exclusions = json.loads(exclusions_path.read_text(encoding="utf-8"))
        if not isinstance(exclusions, list) or not all(isinstance(item, dict) for item in exclusions):
            raise RuntimeError("Canonical training exclusions are malformed.")
        excluded_ids = {item.get("record_id") for item in exclusions}
    by_id: Dict[str, Dict[str, Any]] = {}
    groups: Dict[str, set[str]] = {name: set() for name in split_ids}
    repository_projects: Dict[str, set[str]] = {name: set() for name in split_ids}
    semantic_gate = manifest.get("quality_gate", {}).get("semantic_group_disjoint_splits", False)
    dedup_gate = manifest.get("quality_gate", {}).get("semantic_supervision_deduplicated", False)
    conflict_gate = manifest.get("quality_gate", {}).get("conflicting_supervision_rejected", False)
    unified_prompt_gate = manifest.get("quality_gate", {}).get("unified_prompt_schema", False)
    semantic_supervision_ids: Dict[str, str] = {}
    prompt_references: Dict[str, set[str]] = {}
    for record in records:
        if not isinstance(record, dict) or REQUIRED_RECORD_FIELDS - record.keys():
            raise RuntimeError("Canonical record is missing required fields.")
        if record.get("schema_version") != SCHEMA_VERSION or record.get("language") != "python":
            raise RuntimeError("Canonical record has an unsupported schema or language.")
        if record["id"] in by_id or record_content_hash(record) != record["content_hash"]:
            raise RuntimeError("Canonical record ID or content hash is invalid.")
        try:
            target_tree = _parse_python(record["code_under_test"], "<canonical-target>")
            _parse_python(record["reference_code"], "<canonical-reference>")
        except SyntaxError as exc:
            raise RuntimeError(f"Canonical record {record['id']} has invalid Python.") from exc
        execution_mode = record.get("quality", {}).get("execution_mode", "function_assertion")
        if unified_prompt_gate:
            model_fields = {
                "task_mode", "test_format", "target_symbols", "support_context",
                "prompt_code_under_test",
            }
            if model_fields - record.keys():
                raise RuntimeError(
                    f"Canonical unified-prompt record {record['id']} is missing model fields."
                )
            if record.get("task_mode") not in {"function", "repository"}:
                raise RuntimeError(f"Canonical record {record['id']} has an invalid task mode.")
            if record.get("test_format") not in {
                "assert_statement", "pytest_fragment", "unittest_fragment",
            }:
                raise RuntimeError(f"Canonical record {record['id']} has an invalid test format.")
            if not isinstance(record.get("target_symbols"), list):
                raise RuntimeError(f"Canonical record {record['id']} has malformed target symbols.")
            if not isinstance(record.get("support_context"), str):
                raise RuntimeError(f"Canonical record {record['id']} has malformed support context.")
            try:
                _parse_python(record["prompt_code_under_test"], "<model-visible-target>")
            except SyntaxError as exc:
                raise RuntimeError(
                    f"Canonical record {record['id']} has invalid model-visible target code."
                ) from exc
        if execution_mode in REPOSITORY_EXECUTION_MODES:
            expected_task_types, expected_test_format = REPOSITORY_EXECUTION_MODES[execution_mode]
            if record.get("task_type") not in expected_task_types:
                raise RuntimeError(f"Canonical repository record {record['id']} has an invalid task type.")
            if record.get("entry_point"):
                raise RuntimeError(f"Canonical repository record {record['id']} must not claim a standalone entry point.")
            if not record.get("quality", {}).get("official_targeted_test_fixed_pass_buggy_fail"):
                raise RuntimeError(f"Canonical repository record {record['id']} lacks official F2P evidence.")
            context_complete = record.get("quality", {}).get(
                "support_context_complete_for_verified_tests"
            )
            if unified_prompt_gate and (
                record.get("task_mode") != "repository"
                or record.get("test_format") != expected_test_format
                or (not context_complete and record["id"] not in excluded_ids)
            ):
                raise RuntimeError(
                    f"Canonical repository record {record['id']} lacks unified context evidence."
                )
            if semantic_gate:
                project = record.get("provenance", {}).get("project", "").lower()
                if record.get("task_type") == "official_repository_swebench_reproduction":
                    repository = record.get("provenance", {}).get("repository", "").lower()
                    if not repository or project != repository.rsplit("/", 1)[-1]:
                        raise RuntimeError(
                            f"Canonical SWE-bench record {record['id']} lacks canonical project identity."
                        )
                # V3's historical prefix is retained for compatibility, but
                # the group itself is source-agnostic: BugsInPy and SWE-bench
                # records from the same repository share this exact identity.
                expected_group = f"project:bugsinpy:{project}"
                if not project or record.get("group_id") != expected_group:
                    raise RuntimeError(
                        f"Canonical repository record {record['id']} has a non-project semantic group."
                    )
        elif execution_mode == "function_assertion":
            if record["entry_point"] not in {
                node.name for node in ast.walk(target_tree)
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            }:
                raise RuntimeError(f"Canonical record {record['id']} lacks its entry point.")
            if unified_prompt_gate and (
                record.get("task_mode") != "function"
                or record.get("test_format") != "assert_statement"
                or record.get("target_symbols") != [record["entry_point"]]
                or not record.get("quality", {}).get(
                    "test_oracle_labels_execution_derived"
                )
            ):
                raise RuntimeError(
                    f"Canonical function record {record['id']} lacks executed oracle labels."
                )
            if semantic_gate:
                expected_group = function_group_id(record["reference_code"], record["entry_point"])
                if record.get("group_id") != expected_group:
                    raise RuntimeError(
                        f"Canonical function record {record['id']} lacks its AST-normalized group ID."
                    )
        else:
            raise RuntimeError(f"Canonical record {record['id']} has an unknown execution mode.")
        if not isinstance(record["tests"], list) or not record["tests"]:
            raise RuntimeError(f"Canonical record {record['id']} has no tests.")
        for test in record["tests"]:
            if not isinstance(test, dict) or not isinstance(test.get("code"), str):
                raise RuntimeError(f"Canonical record {record['id']} has a malformed test.")
            if execution_mode in REPOSITORY_EXECUTION_MODES:
                _, expected_test_format = REPOSITORY_EXECUTION_MODES[execution_mode]
                if test.get("format") != expected_test_format or test.get("oracle") != "fixed_passes_buggy_fails_repository":
                    raise RuntimeError(f"Canonical repository record {record['id']} has an invalid test oracle.")
            elif unified_prompt_gate:
                if test.get("oracle") not in {
                    "passes_reference_fails_target",
                    "passes_reference_passes_target",
                    "fails_reference",
                } or not isinstance(test.get("distinguishing"), bool):
                    raise RuntimeError(
                        f"Canonical function record {record['id']} has stale test oracle metadata."
                    )
            _compile_test(test["code"])
        if semantic_gate or dedup_gate or conflict_gate:
            prompt_key, supervision_key = semantic_supervision_key(record)
            reference_key = sha256_text(semantic_python(record["reference_code"]))
            if dedup_gate and supervision_key in semantic_supervision_ids:
                raise RuntimeError(
                    "Canonical corpus has semantically duplicate supervision: "
                    f"{semantic_supervision_ids[supervision_key]} and {record['id']}."
                )
            semantic_supervision_ids[supervision_key] = record["id"]
            prompt_references.setdefault(prompt_key, set()).add(reference_key)
        by_id[record["id"]] = record

    if conflict_gate:
        conflicts = [key for key, references in prompt_references.items() if len(references) > 1]
        if conflicts:
            raise RuntimeError("Canonical corpus maps a semantic prompt to conflicting references.")

    assigned_ids = set()
    for split, ids in split_ids.items():
        if not isinstance(ids, list):
            raise RuntimeError(f"Canonical {split} split is malformed.")
        for record_id in ids:
            if record_id not in by_id or record_id in assigned_ids:
                raise RuntimeError("Canonical split has a missing or duplicate record ID.")
            assigned_ids.add(record_id)
            groups[split].add(by_id[record_id]["group_id"])
            if by_id[record_id].get("quality", {}).get("execution_mode") in REPOSITORY_EXECUTION_MODES:
                repository_projects[split].add(
                    by_id[record_id].get("provenance", {}).get("project", "").lower()
                )
    if assigned_ids != set(by_id):
        raise RuntimeError("Canonical records are not assigned to exactly one split.")
    if exclusions_path.exists():
        if None in excluded_ids or not excluded_ids.issubset(by_id):
            raise RuntimeError("Canonical training exclusions reference unknown records.")
    split_names = sorted(split_ids)
    for index, left in enumerate(split_names):
        for right in split_names[index + 1:]:
            if groups[left] & groups[right]:
                raise RuntimeError("Canonical corpus has group leakage across splits.")
            if repository_projects[left] & repository_projects[right]:
                raise RuntimeError("Canonical corpus has repository-project leakage across splits.")

    if manifest.get("quality_gate", {}).get("locked_external_evaluation"):
        external_index = json.loads((corpus_dir / "external_eval_index.json").read_text(encoding="utf-8"))
        if not isinstance(external_index, list):
            raise RuntimeError("Canonical external evaluation index is malformed.")
        training_tasks: Dict[str, set[str]] = {}
        for record in records:
            task_id = record.get("provenance", {}).get("official_task_id")
            if task_id:
                training_tasks.setdefault(task_id, set()).add(record["id"])
        for item in external_index:
            task_id = item.get("id")
            status = item.get("status")
            materialized = training_tasks.get(task_id, set())
            if status in {"locked_external_eval_not_materialized", "verified_repository_eval_only"} and materialized:
                raise RuntimeError(f"Locked external task {task_id} overlaps canonical training data.")
            if status == "materialized_for_training":
                declared = set(item.get("training_record_ids", []))
                if not materialized or (declared and declared != materialized):
                    raise RuntimeError(f"Materialized external task {task_id} has inconsistent training IDs.")
    return manifest
