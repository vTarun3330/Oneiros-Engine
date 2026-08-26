"""Stage the audited Linux expansion as an immutable V3 build input.

The original 501-task ingestion and both Modal run directories remain
unchanged.  Eight corrected retry payloads supersede their mismatched originals
only in this effective staging view.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness.corpus import (
    REQUIRED_RECORD_FIELDS,
    record_content_hash,
    semantic_python,
    semantic_supervision_key,
    sha256_file,
    verify_corpus,
    write_json,
)

ORIGINAL_DIR = ROOT / "data" / "bugsinpy_v3_ingestion"
MAIN_RESULTS = (
    ROOT / "data" / "bugsinpy_v3_linux_ingestion"
    / "v3-linux-expansion-final-1" / "remote_results" / "results"
)
RETRY_RESULTS = (
    ROOT / "data" / "bugsinpy_v3_linux_ingestion"
    / "v3-linux-selector-retry-1" / "remote_results" / "results"
)
BASELINE_CORPUS = ROOT / "data" / "corpus" / "v3_semantic_candidate"
OUTPUT_DIR = ROOT / "data" / "bugsinpy_v3_expansion_staging"
AUDIT_PATH = ROOT / "results" / "v3_linux_expansion_effective_audit.json"
RETRY_TASKS = {
    "bugsinpy::scrapy::8",
    "bugsinpy::scrapy::14",
    "bugsinpy::scrapy::16",
    "bugsinpy::scrapy::19",
    "bugsinpy::scrapy::23",
    "bugsinpy::scrapy::33",
    "bugsinpy::scrapy::34",
    "bugsinpy::tornado::4",
}


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _payloads(directory: Path) -> Dict[str, Tuple[Path, Dict[str, Any]]]:
    payloads: Dict[str, Tuple[Path, Dict[str, Any]]] = {}
    for path in sorted(directory.glob("*.json")):
        payload = _load(path)
        task_id = (payload.get("summary") or {}).get("task_id")
        if not task_id or task_id in payloads:
            raise RuntimeError(f"Missing or duplicate task identity in {path}")
        payloads[task_id] = (path, payload)
    return payloads


def _tokenizer() -> Tuple[Any, Path]:
    from tokenizers import Tokenizer

    cached = sorted((Path.home() / ".cache" / "huggingface" / "hub").glob(
        "models--microsoft--Phi-3-mini-4k-instruct/snapshots/*/tokenizer.json"
    ))
    if not cached:
        raise RuntimeError("The locked Phi-3 tokenizer is unavailable.")
    return Tokenizer.from_file(str(cached[-1])), cached[-1]


def _validate_record(
    task_id: str, record: Dict[str, Any], tokenizer: Any, completion_limit: int,
) -> List[int]:
    missing = REQUIRED_RECORD_FIELDS - record.keys()
    if missing or record_content_hash(record) != record.get("content_hash"):
        raise RuntimeError(f"Invalid record schema/hash for {task_id}: {sorted(missing)}")
    ast.parse(record["code_under_test"])
    ast.parse(record["reference_code"])
    for test in record["tests"]:
        compile(test["code"], f"<{task_id}-test>", "exec")
    quality = record.get("quality") or {}
    if (
        quality.get("pair_behaviorally_verified") is not True
        or quality.get("official_targeted_test_fixed_pass_buggy_fail") is not True
    ):
        raise RuntimeError(f"Missing behavioral quality gates for {task_id}")
    provenance = record.get("provenance") or {}
    evidence = provenance.get("official_test_evidence") or {}
    fixed_rc = evidence.get("fixed_returncode")
    buggy_rc = evidence.get("buggy_returncode")
    if fixed_rc != 0 or not isinstance(buggy_rc, int) or buggy_rc == 0:
        raise RuntimeError(f"Missing fixed-pass/buggy-fail evidence for {task_id}")
    selector = provenance.get("test_selector", "")
    if quality.get("execution_mode", "").startswith("repository_"):
        selected_test = selector.split("::")[-1]
        failing_commands = [
            command for command, returncode in zip(
                evidence.get("commands", []),
                evidence.get("buggy_command_returncodes", []),
            )
            if returncode
        ]
        if not selected_test or not any(
            selected_test in " ".join(command) for command in failing_commands
        ):
            raise RuntimeError(
                f"Stored selector is not the observed buggy failure for {task_id}"
            )
    counts = [
        len(tokenizer.encode(
            test["code"].strip() + "<|endoftext|>", add_special_tokens=False,
        ).ids)
        for test in record["tests"]
    ]
    if not counts or any(count >= completion_limit for count in counts):
        raise RuntimeError(f"Completion exceeds context gate for {task_id}: {counts}")
    return counts


def _unique_records(records: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_id: Dict[str, Dict[str, Any]] = {}
    for record in records:
        prior = by_id.get(record["id"])
        if prior is not None and prior["content_hash"] != record["content_hash"]:
            raise RuntimeError(f"Conflicting staged record identity: {record['id']}")
        by_id[record["id"]] = record
    return sorted(by_id.values(), key=lambda item: item["id"])


def stage(
    output_dir: Path = OUTPUT_DIR,
    audit_path: Path = AUDIT_PATH,
    completion_limit: int = 1536,
) -> Dict[str, Any]:
    tokenizer, tokenizer_path = _tokenizer()
    original_report = _load(ORIGINAL_DIR / "ingestion_report.json")
    original_by_task = {item["task_id"]: item for item in original_report}
    expected = {
        task_id for task_id, item in original_by_task.items()
        if item.get("status") != "accepted"
    }
    main = _payloads(MAIN_RESULTS)
    retry = _payloads(RETRY_RESULTS)
    if set(main) != expected:
        raise RuntimeError("The 357-result expansion task set is incomplete or unexpected.")
    if set(retry) != RETRY_TASKS:
        raise RuntimeError("The selector retry task set is not the exact audited set of eight.")
    effective = dict(main)
    effective.update(retry)

    new_function_records: List[Dict[str, Any]] = []
    new_repository_records: List[Dict[str, Any]] = []
    effective_report = list(original_report)
    report_index = {item["task_id"]: index for index, item in enumerate(effective_report)}
    accepted_details = []
    exclusions = Counter()
    projects = Counter()
    completion_counts: List[int] = []
    for task_id, (path, payload) in sorted(effective.items()):
        summary = payload.get("summary") or {}
        outcome = payload.get("outcome") or {}
        status = summary.get("status")
        if outcome.get("status") != status:
            raise RuntimeError(f"Summary/outcome mismatch for {task_id}")
        if status == "excluded":
            reason = summary.get("reason") or outcome.get("reason")
            if not reason:
                raise RuntimeError(f"Excluded task lacks a reason: {task_id}")
            exclusions[reason] += 1
            effective_report[report_index[task_id]] = {
                **outcome,
                "task_id": task_id,
                "status": "excluded",
                "reason": reason,
                "linux_result_file": str(path.resolve()),
            }
            continue
        if status != "accepted":
            raise RuntimeError(f"Unsupported effective status for {task_id}: {status}")
        function_record = payload.get("function_record")
        repository_record = payload.get("repository_record")
        record = function_record or repository_record
        if not isinstance(record, dict):
            raise RuntimeError(f"Accepted task lacks a record: {task_id}")
        counts = _validate_record(task_id, record, tokenizer, completion_limit)
        completion_counts.extend(counts)
        if function_record:
            new_function_records.append(record)
        else:
            new_repository_records.append(record)
        project = (record.get("provenance") or {}).get("project")
        projects[project] += 1
        accepted_details.append({
            "task_id": task_id,
            "record_id": record["id"],
            "project": project,
            "record_mode": (record.get("quality") or {}).get("execution_mode"),
            "completion_tokens": counts,
            "source_result_file": str(path.resolve()),
            "supersedes_original_result": task_id in retry,
        })
        effective_report[report_index[task_id]] = {
            **outcome,
            "task_id": task_id,
            "status": "accepted",
            "record_id": record["id"],
            "linux_result_file": str(path.resolve()),
            "supersedes_original_result": task_id in retry,
        }

    original_function = _load(ORIGINAL_DIR / "materialized_records.json")
    original_repository = _load(ORIGINAL_DIR / "repository_fragment_records.json")
    staged_function = _unique_records([*original_function, *new_function_records])
    staged_repository = _unique_records([*original_repository, *new_repository_records])

    verify_corpus(BASELINE_CORPUS)
    baseline_records = _load(BASELINE_CORPUS / "records.json")
    supervision: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    prompt_references: Dict[str, set[str]] = defaultdict(set)
    new_records = [*new_function_records, *new_repository_records]
    for origin, records in (("baseline", baseline_records), ("expansion", new_records)):
        for record in records:
            prompt_key, supervision_key = semantic_supervision_key(record)
            reference_key = hashlib.sha256(
                semantic_python(record["reference_code"]).encode("utf-8")
            ).hexdigest()
            supervision[supervision_key].append((origin, record["id"]))
            prompt_references[prompt_key].add(reference_key)
    duplicate_buckets = [items for items in supervision.values() if len(items) > 1]
    prompt_conflicts = sum(len(references) > 1 for references in prompt_references.values())
    if prompt_conflicts:
        raise RuntimeError("Conflicting semantic supervision in expansion staging.")

    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "materialized_records.json", staged_function)
    write_json(output_dir / "repository_fragment_records.json", staged_repository)
    write_json(output_dir / "ingestion_report.json", effective_report)
    source_hashes = {
        "original_manifest": sha256_file(ORIGINAL_DIR / "manifest.json"),
        "main_status": sha256_file(
            MAIN_RESULTS.parent.parent / "status_checkpoint.json"
        ),
        "retry_status": sha256_file(
            RETRY_RESULTS.parent.parent / "status_checkpoint.json"
        ),
    }
    manifest = {
        "schema_version": 1,
        "ingestion_complete": True,
        "official_task_count": len(effective_report),
        "processed_task_count": len(effective_report),
        "accepted_task_count": sum(item.get("status") == "accepted" for item in effective_report),
        "excluded_task_count": sum(item.get("status") == "excluded" for item in effective_report),
        "original_verified_record_count": len(original_function) + len(original_repository),
        "new_verified_record_count": len(new_records),
        "staged_verified_record_count": len(staged_function) + len(staged_repository),
        "superseded_selector_payloads": sorted(RETRY_TASKS),
        "source_hashes": source_hashes,
        "files": {
            name: {"sha256": sha256_file(output_dir / name)}
            for name in (
                "materialized_records.json",
                "repository_fragment_records.json",
                "ingestion_report.json",
            )
        },
    }
    write_json(output_dir / "manifest.json", manifest)
    audit = {
        "schema_version": 1,
        "ready_for_merge": True,
        "effective_task_count": len(effective),
        "accepted_expansion_records": len(new_records),
        "excluded_expansion_tasks": sum(exclusions.values()),
        "exclusion_reasons": dict(sorted(exclusions.items())),
        "projects": dict(sorted(projects.items())),
        "record_ids_unique": len({record["id"] for record in new_records}) == len(new_records),
        "semantic_duplicate_buckets_with_baseline": len(duplicate_buckets),
        "semantic_duplicate_record_excess": sum(len(items) - 1 for items in duplicate_buckets),
        "semantic_prompt_conflicts": prompt_conflicts,
        "completion_token_limit": completion_limit,
        "completion_token_max": max(completion_counts),
        "overlong_completions": 0,
        "tokenizer_json": str(tokenizer_path.resolve()),
        "tokenizer_sha256": hashlib.sha256(tokenizer_path.read_bytes()).hexdigest(),
        "accepted_records": accepted_details,
        "staging_manifest": str((output_dir / "manifest.json").resolve()),
        "staging_manifest_sha256": sha256_file(output_dir / "manifest.json"),
    }
    write_json(audit_path, audit)
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--audit-path", type=Path, default=AUDIT_PATH)
    parser.add_argument("--completion-token-limit", type=int, default=1536)
    arguments = parser.parse_args()
    print(json.dumps(stage(
        arguments.output_dir, arguments.audit_path, arguments.completion_token_limit,
    ), indent=2))


if __name__ == "__main__":
    main()
