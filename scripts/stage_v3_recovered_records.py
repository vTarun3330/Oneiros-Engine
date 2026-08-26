"""Stage only audited recovery records into a new immutable V3 build input."""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness.corpus import (
    REQUIRED_RECORD_FIELDS,
    official_evidence_verifies_pair,
    record_content_hash,
    sha256_file,
    write_json,
)

BASE_STAGING = ROOT / "data" / "bugsinpy_v3_expansion_staging"
OUTPUT_DIR = ROOT / "data" / "v3_recovered_staging_candidate"
OUTPUT_AUDIT = ROOT / "results" / "v3_recovered_staging_audit.json"
OVERLONG_AUDIT = ROOT / "results" / "v3_overlong_reverification_audit.json"


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _tokenizer() -> tuple[Any, Path]:
    from tokenizers import Tokenizer

    cached = sorted((Path.home() / ".cache" / "huggingface" / "hub").glob(
        "models--microsoft--Phi-3-mini-4k-instruct/snapshots/*/tokenizer.json"
    ))
    if not cached:
        raise RuntimeError("The locked Phi-3 tokenizer is unavailable.")
    return Tokenizer.from_file(str(cached[-1])), cached[-1]


def _audited_result_files(audit_paths: Iterable[Path]) -> List[Path]:
    result_files: List[Path] = []
    for audit_path in audit_paths:
        audit = _load(audit_path)
        if audit.get("ready") is not True:
            raise RuntimeError(f"Recovery audit is not ready: {audit_path}")
        for item in audit.get("records", []):
            result_files.append(Path(item["result_file"]).resolve())
    if len(result_files) != len(set(result_files)):
        raise RuntimeError("The selected recovery audits contain duplicate result files.")
    return sorted(result_files)


def _canonicalize_record(payload: Dict[str, Any]) -> tuple[Dict[str, Any], str, str]:
    summary = payload.get("summary") or {}
    if summary.get("status") != "accepted":
        raise RuntimeError(f"Audited result is no longer accepted: {summary}")
    record = payload.get("function_record") or payload.get("repository_record") or payload.get("record")
    if not isinstance(record, dict):
        raise RuntimeError(f"Accepted result lacks a record: {summary}")
    record = copy.deepcopy(record)
    provenance = record.get("provenance") or {}
    task_id = provenance.get("official_task_id")
    source = "swebench" if record.get("task_type") == "official_repository_swebench_reproduction" else "bugsinpy"
    if source == "swebench":
        repository = provenance.get("repository", "").lower()
        if not repository:
            raise RuntimeError(f"SWE-bench record lacks repository identity: {record.get('id')}")
        provenance["project"] = repository.rsplit("/", 1)[-1]
        record["provenance"] = provenance
        record["content_hash"] = record_content_hash(record)
    if not task_id:
        raise RuntimeError(f"Record lacks official task identity: {record.get('id')}")
    return record, task_id, source


def _validate_record(record: Dict[str, Any], tokenizer: Any, limit: int) -> List[int]:
    missing = REQUIRED_RECORD_FIELDS - record.keys()
    if missing or record_content_hash(record) != record.get("content_hash"):
        raise RuntimeError(f"Invalid recovered record schema/hash: {record.get('id')}")
    ast.parse(record["code_under_test"])
    ast.parse(record["reference_code"])
    counts: List[int] = []
    for test in record["tests"]:
        compile(test["code"], f"<{record['id']}-test>", "exec")
        counts.append(len(tokenizer.encode(
            test["code"].strip() + "<|endoftext|>", add_special_tokens=False,
        ).ids))
    evidence = (record.get("provenance") or {}).get("official_test_evidence") or {}
    if (
        not counts
        or not official_evidence_verifies_pair(evidence)
        or record.get("quality", {}).get("pair_behaviorally_verified") is not True
        or record.get("quality", {}).get("official_targeted_test_fixed_pass_buggy_fail") is not True
    ):
        raise RuntimeError(f"Recovered record fails evidence/context gates: {record.get('id')}")
    return counts


def _unique(records: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_id: Dict[str, Dict[str, Any]] = {}
    for record in records:
        prior = by_id.get(record["id"])
        if prior is not None and prior["content_hash"] != record["content_hash"]:
            raise RuntimeError(f"Conflicting recovered record identity: {record['id']}")
        by_id[record["id"]] = record
    return sorted(by_id.values(), key=lambda item: item["id"])


def stage(
    audit_paths: List[Path], output_dir: Path = OUTPUT_DIR,
    output_audit: Path = OUTPUT_AUDIT, completion_limit: int = 1536,
) -> Dict[str, Any]:
    tokenizer, tokenizer_path = _tokenizer()
    base_function = _load(BASE_STAGING / "materialized_records.json")
    base_repository = _load(BASE_STAGING / "repository_fragment_records.json")
    report = _load(BASE_STAGING / "ingestion_report.json")
    report_index = {item["task_id"]: index for index, item in enumerate(report)}

    new_function: List[Dict[str, Any]] = []
    new_repository: List[Dict[str, Any]] = []
    details: List[Dict[str, Any]] = []
    for result_file in _audited_result_files(audit_paths):
        payload = _load(result_file)
        record, task_id, source = _canonicalize_record(payload)
        completion_tokens = _validate_record(record, tokenizer, completion_limit)
        if payload.get("function_record"):
            new_function.append(record)
        else:
            new_repository.append(record)
        outcome = payload.get("outcome") or {}
        report_item = {
            **outcome,
            "task_id": task_id,
            "status": "accepted",
            "record_id": record["id"],
            "verified_source": source,
            "result_file": str(result_file),
        }
        if task_id in report_index:
            report[report_index[task_id]] = report_item
        else:
            report_index[task_id] = len(report)
            report.append(report_item)
        details.append({
            "task_id": task_id,
            "record_id": record["id"],
            "source": source,
            "project": record["provenance"]["project"],
            "completion_tokens": completion_tokens,
            "result_file": str(result_file),
        })

    staged_function = _unique([*base_function, *new_function])
    staged_repository = _unique([*base_repository, *new_repository])
    if len(staged_function) + len(staged_repository) != (
        len(base_function) + len(base_repository) + len(new_function) + len(new_repository)
    ):
        raise RuntimeError("A recovered record duplicates the immutable base staging input.")

    overlong = _load(OVERLONG_AUDIT)
    training_exclusions = [{
        "record_id": item["record_id"],
        "reason": item["decision"],
        "completion_tokens": item["completion_tokens"],
        "completion_token_limit": item["completion_token_limit"],
    } for item in overlong["details"]]
    existing_exclusions = {item["record_id"] for item in training_exclusions}
    for item in details:
        if (
            any(count >= completion_limit for count in item["completion_tokens"])
            and item["record_id"] not in existing_exclusions
        ):
            training_exclusions.append({
                "record_id": item["record_id"],
                "reason": "canonical_retained_training_excluded",
                "completion_tokens": item["completion_tokens"],
                "completion_token_limit": completion_limit,
            })
            existing_exclusions.add(item["record_id"])

    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "materialized_records.json", staged_function)
    write_json(output_dir / "repository_fragment_records.json", staged_repository)
    write_json(output_dir / "ingestion_report.json", report)
    write_json(output_dir / "training_exclusions.json", training_exclusions)
    files = {
        name: {"sha256": sha256_file(output_dir / name)}
        for name in (
            "materialized_records.json", "repository_fragment_records.json",
            "ingestion_report.json", "training_exclusions.json",
        )
    }
    manifest = {
        "schema_version": 1,
        "ingestion_complete": True,
        "base_staging_manifest_sha256": sha256_file(BASE_STAGING / "manifest.json"),
        "audit_files": {
            str(path.resolve()): sha256_file(path) for path in sorted(audit_paths)
        },
        "base_verified_record_count": len(base_function) + len(base_repository),
        "new_verified_record_count": len(new_function) + len(new_repository),
        "staged_verified_record_count": len(staged_function) + len(staged_repository),
        "canonical_training_exclusion_count": len(training_exclusions),
        "files": files,
    }
    write_json(output_dir / "manifest.json", manifest)
    audit = {
        "schema_version": 1,
        "ready_for_build": True,
        "audited_result_count": len(details),
        "sources": dict(sorted(Counter(item["source"] for item in details).items())),
        "projects": dict(sorted(Counter(item["project"] for item in details).items())),
        "completion_token_limit": completion_limit,
        "completion_token_max": max(
            count for item in details for count in item["completion_tokens"]
        ),
        "canonical_training_exclusions": training_exclusions,
        "tokenizer_json": str(tokenizer_path.resolve()),
        "tokenizer_sha256": hashlib.sha256(tokenizer_path.read_bytes()).hexdigest(),
        "records": details,
        "staging_manifest": str((output_dir / "manifest.json").resolve()),
        "staging_manifest_sha256": sha256_file(output_dir / "manifest.json"),
    }
    write_json(output_audit, audit)
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", action="append", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--output-audit", type=Path, default=OUTPUT_AUDIT)
    parser.add_argument("--completion-token-limit", type=int, default=1536)
    arguments = parser.parse_args()
    print(json.dumps(stage(
        arguments.audit, arguments.output_dir, arguments.output_audit,
        arguments.completion_token_limit,
    ), indent=2))


if __name__ == "__main__":
    main()
