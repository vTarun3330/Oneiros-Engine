"""Audit selected checkpointed Linux BugsInPy results before corpus ingestion."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
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


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def audit(
    result_files: List[Path], corpus_dir: Path, tokenizer_json: Path,
    completion_token_limit: int,
) -> Dict[str, Any]:
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(tokenizer_json))
    verify_corpus(corpus_dir)
    corpus_records = _load(corpus_dir / "records.json")
    splits = _load(corpus_dir / "splits.json")
    split_by_id = {
        record_id: split for split, record_ids in splits.items() for record_id in record_ids
    }
    corpus_supervision: Dict[str, str] = {}
    corpus_prompts: Dict[str, set[str]] = {}
    for record in corpus_records:
        prompt_key, supervision_key = semantic_supervision_key(record)
        reference_key = hashlib.sha256(
            semantic_python(record["reference_code"]).encode("utf-8")
        ).hexdigest()
        corpus_supervision[supervision_key] = record["id"]
        corpus_prompts.setdefault(prompt_key, set()).add(reference_key)

    accepted: List[Dict[str, Any]] = []
    seen_tasks: set[str] = set()
    details = []
    overlong_records = []
    semantic_duplicates = []
    new_supervision: Dict[str, str] = {}
    new_prompts: Dict[str, set[str]] = {}
    for path in result_files:
        payload = _load(path)
        summary = payload.get("summary", {})
        task_id = summary.get("task_id")
        if not task_id or task_id in seen_tasks:
            raise RuntimeError("Pilot results have a missing or duplicate task identity.")
        seen_tasks.add(task_id)
        if summary.get("status") != "accepted":
            raise RuntimeError(f"Selected pilot result is not accepted: {task_id}")
        record = payload.get("function_record") or payload.get("repository_record")
        if not isinstance(record, dict):
            raise RuntimeError(f"Accepted pilot result lacks a record: {task_id}")
        missing = REQUIRED_RECORD_FIELDS - record.keys()
        if missing or record_content_hash(record) != record.get("content_hash"):
            raise RuntimeError(f"Accepted pilot record has invalid schema/hash: {task_id}")
        ast.parse(record["code_under_test"])
        ast.parse(record["reference_code"])
        for test in record["tests"]:
            compile(test["code"], f"<{task_id}-test>", "exec")
        completion_tokens = [
            len(tokenizer.encode(
                test["code"].strip() + "<|endoftext|>",
                add_special_tokens=False,
            ).ids)
            for test in record["tests"]
        ]
        if not completion_tokens:
            raise RuntimeError(f"Pilot result has no completion: {task_id}")
        fits_training_context = all(
            count < completion_token_limit for count in completion_tokens
        )

        evidence = record.get("provenance", {}).get("official_test_evidence", {})
        fixed_rc = evidence.get("fixed_returncode")
        buggy_rc = evidence.get("buggy_returncode")
        if fixed_rc != 0 or not isinstance(buggy_rc, int) or buggy_rc == 0:
            raise RuntimeError(f"Accepted pilot record lacks fixed-pass/buggy-fail evidence: {task_id}")
        commands = evidence.get("commands", [])
        buggy_rcs = evidence.get("buggy_command_returncodes", [])
        failing_commands = [
            command for command, returncode in zip(commands, buggy_rcs) if returncode
        ]
        selector = record.get("provenance", {}).get("test_selector", "")
        selected_test = selector.split("::")[-1]
        if not failing_commands or not any(
            selected_test in " ".join(command) for command in failing_commands
        ):
            raise RuntimeError(
                f"Record selector was not the observed buggy failure: {task_id}"
            )

        prompt_key, supervision_key = semantic_supervision_key(record)
        reference_key = hashlib.sha256(
            semantic_python(record["reference_code"]).encode("utf-8")
        ).hexdigest()
        duplicate_of = corpus_supervision.get(supervision_key) or new_supervision.get(supervision_key)
        if duplicate_of:
            semantic_duplicates.append({
                "task_id": task_id,
                "record_id": record["id"],
                "duplicate_of": duplicate_of,
                "decision": "equivalent_supervision_excluded",
            })
            continue
        conflicting_references = corpus_prompts.get(prompt_key, set()) | new_prompts.get(prompt_key, set())
        if conflicting_references and conflicting_references != {reference_key}:
            raise RuntimeError(f"Pilot record conflicts with existing supervision: {task_id}")
        new_supervision[supervision_key] = record["id"]
        new_prompts.setdefault(prompt_key, set()).add(reference_key)

        group_splits = {
            split_by_id[item["id"]]
            for item in corpus_records if item["group_id"] == record["group_id"]
        }
        if len(group_splits) > 1:
            raise RuntimeError(f"Existing project group already leaks across splits: {task_id}")
        details.append({
            "task_id": task_id,
            "record_id": record["id"],
            "project": record.get("provenance", {}).get("project"),
            "group_id": record["group_id"],
            "existing_group_split": next(iter(group_splits), None),
            "requested_python_version": evidence.get("environment", {}).get("requested_python_version"),
            "runner_python_version": evidence.get("environment", {}).get("runner_python_version"),
            "fixed_returncode": fixed_rc,
            "buggy_returncode": buggy_rc,
            "selected_test": selector,
            "completion_tokens": completion_tokens,
            "fits_training_context": fits_training_context,
            "record_characters": len(json.dumps(record, ensure_ascii=False)),
            "result_file": str(path.resolve()),
        })
        if not fits_training_context:
            overlong_records.append({
                "task_id": task_id,
                "record_id": record["id"],
                "completion_tokens": completion_tokens,
                "decision": "canonical_retained_training_excluded",
            })
        accepted.append(record)

    return {
        "schema_version": 1,
        "ready": True,
        "baseline_corpus": str(corpus_dir.resolve()),
        "baseline_records_checked": len(corpus_records),
        "result_files_checked": len(result_files),
        "behaviorally_accepted_records": len(result_files),
        "accepted_records": len(accepted),
        "semantic_duplicate_records_excluded": len(semantic_duplicates),
        "semantic_duplicates": semantic_duplicates,
        "training_retained_records": len(accepted) - len(overlong_records),
        "canonical_training_excluded_records": len(overlong_records),
        "overlong_records": overlong_records,
        "tokenizer_json": str(tokenizer_json.resolve()),
        "tokenizer_json_sha256": hashlib.sha256(tokenizer_json.read_bytes()).hexdigest(),
        "completion_token_limit": completion_token_limit,
        "projects": dict(sorted(Counter(item["project"] for item in details).items())),
        "quality_gates": {
            "all_selected_results_accepted": True,
            "record_schema_and_hash_valid": True,
            "python_and_tests_compile": True,
            "fixed_pass_buggy_fail": True,
            "selected_test_observed_buggy_fail": True,
            "all_training_retained_completions_fit_sft_context": True,
            "overlong_completions_explicitly_excluded": True,
            "semantic_supervision_unique": True,
            "equivalent_supervision_deduplicated": True,
            "semantic_prompt_conflicts": 0,
            "project_group_split_safe": True,
        },
        "records": details,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-file", action="append", type=Path, default=[])
    parser.add_argument("--result-dir", action="append", type=Path, default=[])
    parser.add_argument(
        "--accepted-only", action="store_true",
        help="Audit only accepted payloads discovered through --result-dir.",
    )
    parser.add_argument(
        "--corpus-dir", type=Path,
        default=ROOT / "data" / "corpus" / "v3_semantic_candidate",
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "results" / "v3_linux_pilot_audit.json",
    )
    cached_tokenizers = sorted((Path.home() / ".cache" / "huggingface" / "hub").glob(
        "models--microsoft--Phi-3-mini-4k-instruct/snapshots/*/tokenizer.json"
    ))
    parser.add_argument(
        "--tokenizer-json", type=Path,
        default=cached_tokenizers[-1] if cached_tokenizers else None,
    )
    parser.add_argument("--completion-token-limit", type=int, default=1536)
    arguments = parser.parse_args()
    if arguments.tokenizer_json is None or not arguments.tokenizer_json.exists():
        raise RuntimeError("The locked Phi-3 tokenizer.json is unavailable for the context gate.")
    result_files = list(arguments.result_file)
    for result_dir in arguments.result_dir:
        discovered = sorted(result_dir.glob("*.json"))
        if arguments.accepted_only:
            discovered = [
                path for path in discovered
                if (_load(path).get("summary") or {}).get("status") == "accepted"
            ]
        result_files.extend(discovered)
    if not result_files:
        raise RuntimeError("No BugsInPy result payloads were selected for audit.")
    report = audit(
        result_files,
        arguments.corpus_dir,
        arguments.tokenizer_json,
        arguments.completion_token_limit,
    )
    write_json(arguments.output, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
