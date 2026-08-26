"""Audit synced SWE-bench Verified results before they may enter Oneiros V3."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness.corpus import (
    REQUIRED_RECORD_FIELDS,
    record_content_hash,
    semantic_python,
    semantic_supervision_key,
    verify_corpus,
    write_json,
)

SOURCE_PARQUET = (
    ROOT / "data" / "swebench_verified_source"
    / "SWE-bench_Verified.test.parquet"
)


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "__", value)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(
    result_files: List[Path], corpus_dir: Path, tokenizer_json: Path,
    completion_token_limit: int, require_all_accepted: bool = False,
) -> Dict[str, Any]:
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(tokenizer_json))
    verify_corpus(corpus_dir)
    baseline_records = _load(corpus_dir / "records.json")
    source_sha256 = _sha256(SOURCE_PARQUET)

    baseline_supervision: Dict[str, str] = {}
    baseline_prompts: Dict[str, set[str]] = {}
    baseline_groups = {record["group_id"] for record in baseline_records}
    for record in baseline_records:
        prompt_key, supervision_key = semantic_supervision_key(record)
        reference_key = hashlib.sha256(
            semantic_python(record["reference_code"]).encode("utf-8")
        ).hexdigest()
        baseline_supervision[supervision_key] = record["id"]
        baseline_prompts.setdefault(prompt_key, set()).add(reference_key)

    seen_instances: set[str] = set()
    new_supervision: set[str] = set()
    new_prompts: Dict[str, set[str]] = {}
    accepted = []
    excluded = []
    audit_excluded = []
    training_exclusions = []
    details = []

    for result_file in result_files:
        payload = _load(result_file)
        if payload.get("schema_version") != 1:
            raise RuntimeError(f"Unsupported result schema: {result_file}")
        summary = payload.get("summary") or {}
        instance_id = summary.get("instance_id")
        if not instance_id or instance_id in seen_instances:
            raise RuntimeError("Missing or duplicate SWE-bench instance identity")
        seen_instances.add(instance_id)
        identity = payload.get("source_identity") or {}
        if (
            identity.get("dataset") != "SWE-bench/SWE-bench_Verified"
            or identity.get("split") != "test"
            or identity.get("instance_id") != instance_id
            or identity.get("source_parquet_sha256") != source_sha256
        ):
            raise RuntimeError(f"Invalid source identity: {instance_id}")

        if summary.get("status") != "accepted":
            reason = summary.get("reason")
            if not reason or payload.get("record") is not None:
                raise RuntimeError(f"Invalid exclusion payload: {instance_id}")
            excluded.append({"instance_id": instance_id, "reason": reason})
            continue

        record = payload.get("record")
        if not isinstance(record, dict):
            raise RuntimeError(f"Accepted result has no record: {instance_id}")
        missing = REQUIRED_RECORD_FIELDS - record.keys()
        if missing or record_content_hash(record) != record.get("content_hash"):
            raise RuntimeError(f"Invalid record schema/hash: {instance_id}")
        ast.parse(record["code_under_test"])
        ast.parse(record["reference_code"])
        completion_tokens = []
        for test in record["tests"]:
            compile(test["code"], f"<{instance_id}-test>", "exec")
            completion_tokens.append(len(tokenizer.encode(
                test["code"].strip() + "<|endoftext|>",
                add_special_tokens=False,
            ).ids))
        if not completion_tokens:
            raise RuntimeError(f"Accepted result has no test completion: {instance_id}")
        overlong = any(count >= completion_token_limit for count in completion_tokens)

        verification = record.get("provenance", {}).get("official_test_evidence") or {}
        expected_f2p = len(record.get("provenance", {}).get("fail_to_pass") or [])
        expected_p2p = len(record.get("provenance", {}).get("pass_to_pass") or [])
        base = verification.get("base_counts") or {}
        fixed = verification.get("fixed_counts") or {}
        if not (
            expected_f2p > 0
            and verification.get("buggy_fail_verified") is True
            and verification.get("fixed_pass_verified") is True
            and base.get("f2p_failure") == expected_f2p
            and base.get("f2p_success") == 0
            and base.get("p2p_success") == expected_p2p
            and base.get("p2p_failure") == 0
            and fixed.get("f2p_success") == expected_f2p
            and fixed.get("f2p_failure") == 0
            and fixed.get("p2p_success") == expected_p2p
            and fixed.get("p2p_failure") == 0
        ):
            raise RuntimeError(f"Invalid base/gold behavioral evidence: {instance_id}")
        if verification.get("sandbox_filesystem_adapter") != "modal-filesystem-v1-write-text":
            raise RuntimeError(f"Untracked Modal filesystem adapter: {instance_id}")

        run_root = result_file.parent.parent
        evidence_dir = run_root / "evidence" / _slug(instance_id)
        base_log = evidence_dir / "base_test_output.txt"
        fixed_log = evidence_dir / "fixed_test_output.txt"
        if (
            not base_log.exists() or not fixed_log.exists()
            or _sha256(base_log) != verification.get("base_log_sha256")
            or _sha256(fixed_log) != verification.get("fixed_log_sha256")
        ):
            raise RuntimeError(f"Missing or invalid evidence logs: {instance_id}")

        prompt_key, supervision_key = semantic_supervision_key(record)
        reference_key = hashlib.sha256(
            semantic_python(record["reference_code"]).encode("utf-8")
        ).hexdigest()
        if supervision_key in baseline_supervision:
            audit_excluded.append({
                "instance_id": instance_id,
                "reason": "duplicate_semantic_supervision_baseline",
                "duplicate_of": baseline_supervision[supervision_key],
            })
            continue
        if supervision_key in new_supervision:
            audit_excluded.append({
                "instance_id": instance_id,
                "reason": "duplicate_semantic_supervision_selected_results",
            })
            continue
        conflicting = baseline_prompts.get(prompt_key, set()) | new_prompts.get(prompt_key, set())
        if conflicting and conflicting != {reference_key}:
            audit_excluded.append({
                "instance_id": instance_id,
                "reason": "conflicting_semantic_supervision",
            })
            continue
        new_supervision.add(supervision_key)
        new_prompts.setdefault(prompt_key, set()).add(reference_key)

        if overlong:
            training_exclusions.append({
                "record_id": record["id"],
                "instance_id": instance_id,
                "reason": "canonical_retained_training_excluded",
                "completion_tokens": completion_tokens,
                "completion_token_limit": completion_token_limit,
            })

        details.append({
            "instance_id": instance_id,
            "repo": record.get("provenance", {}).get("repository"),
            "record_id": record["id"],
            "group_id": record["group_id"],
            "expected_f2p": expected_f2p,
            "expected_p2p": expected_p2p,
            "base_counts": base,
            "fixed_counts": fixed,
            "completion_tokens": completion_tokens,
            "training_eligible": not overlong,
            "training_exclusion_reason": (
                "canonical_retained_training_excluded" if overlong else None
            ),
            "project_group_already_in_baseline": record["group_id"] in baseline_groups,
            "base_runtime_seconds": verification.get("base_runtime_seconds"),
            "fixed_runtime_seconds": verification.get("fixed_runtime_seconds"),
            "result_file": str(result_file.resolve()),
        })
        accepted.append(record)

    all_exclusions = [*excluded, *audit_excluded]
    if require_all_accepted and all_exclusions:
        raise RuntimeError(f"Selected SWE-bench gate has exclusions: {all_exclusions}")
    if not accepted:
        raise RuntimeError("No accepted SWE-bench records passed the audit")

    return {
        "schema_version": 1,
        "ready": True,
        "baseline_corpus": str(corpus_dir.resolve()),
        "baseline_records_checked": len(baseline_records),
        "source_parquet_sha256": source_sha256,
        "results_checked": len(result_files),
        "accepted": len(accepted),
        "excluded": len(all_exclusions),
        "remote_excluded": len(excluded),
        "audit_excluded": len(audit_excluded),
        "exclusions": all_exclusions,
        "canonical_training_exclusion_count": len(training_exclusions),
        "canonical_training_exclusions": training_exclusions,
        "training_eligible": len(accepted) - len(training_exclusions),
        "projects": dict(sorted(Counter(item["repo"] for item in details).items())),
        "tokenizer_json": str(tokenizer_json.resolve()),
        "tokenizer_sha256": _sha256(tokenizer_json),
        "completion_token_limit": completion_token_limit,
        "quality_gates": {
            "source_identity_valid": True,
            "record_schema_hash_valid": True,
            "python_and_tests_parse": True,
            "official_base_f2p_failure_and_p2p_maintenance": True,
            "official_gold_f2p_and_p2p_success": True,
            "evidence_log_hashes_valid": True,
            "all_training_completions_fit_sft_context": True,
            "overlong_records_canonically_retained_and_explicitly_excluded": True,
            "semantic_supervision_unique": True,
            "semantic_prompt_conflicts": 0,
            "repository_projects_may_extend_existing_canonical_groups": True,
        },
        "records": details,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-file", action="append", type=Path, default=[])
    parser.add_argument("--result-dir", action="append", type=Path, default=[])
    parser.add_argument(
        "--corpus-dir", type=Path,
        default=ROOT / "data" / "corpus" / "v3_expanded_candidate",
    )
    parser.add_argument("--output", type=Path, required=True)
    cached = sorted((Path.home() / ".cache" / "huggingface" / "hub").glob(
        "models--microsoft--Phi-3-mini-4k-instruct/snapshots/*/tokenizer.json"
    ))
    parser.add_argument("--tokenizer-json", type=Path, default=cached[-1] if cached else None)
    parser.add_argument("--completion-token-limit", type=int, default=1536)
    parser.add_argument("--require-all-accepted", action="store_true")
    arguments = parser.parse_args()
    if arguments.tokenizer_json is None or not arguments.tokenizer_json.exists():
        raise RuntimeError("Locked Phi-3 tokenizer is unavailable")
    result_files = list(arguments.result_file)
    for result_dir in arguments.result_dir:
        result_files.extend(sorted(result_dir.glob("*.json")))
    if not result_files:
        raise RuntimeError("No SWE-bench result payloads were selected for audit.")
    report = audit(
        result_files,
        arguments.corpus_dir,
        arguments.tokenizer_json,
        arguments.completion_token_limit,
        arguments.require_all_accepted,
    )
    write_json(arguments.output, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
