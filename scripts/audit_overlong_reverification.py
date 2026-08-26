"""Audit the targeted BugsInPy overlong-fragment reverification run.

The canonical corpus is not modified here.  Records remain trainable only when
their complete verified test fragment fits the locked Phi-3 completion gate;
otherwise the report records the explicit canonical-retain/training-exclude
decision required by the V3 build plan.
"""
from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path
from typing import Any, Dict

from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness.corpus import record_content_hash, write_json


EXPECTED_TASKS = {
    "bugsinpy::tqdm::2",
    "bugsinpy::youtube-dl::2",
    "bugsinpy::youtube-dl::14",
    "bugsinpy::youtube-dl::16",
}


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def audit(result_dir: Path, tokenizer_json: Path, token_limit: int) -> Dict[str, Any]:
    tokenizer = Tokenizer.from_file(str(tokenizer_json))
    originals = {
        record["id"]: record
        for record in _load(ROOT / "data" / "bugsinpy_v2_ingestion" / "repository_fragment_records.json")
    }
    result_files = sorted(result_dir.glob("*.json"))
    task_ids = set()
    details = []
    for path in result_files:
        payload = _load(path)
        summary = payload.get("summary") or {}
        task_id = summary.get("task_id")
        if not task_id or task_id in task_ids:
            raise RuntimeError(f"Missing or duplicate task identity in {path}")
        task_ids.add(task_id)
        if summary.get("status") != "accepted":
            raise RuntimeError(f"Targeted reverification was not accepted: {task_id}")
        record = payload.get("function_record") or payload.get("repository_record")
        if not isinstance(record, dict):
            raise RuntimeError(f"Accepted payload lacks a record: {task_id}")
        if record_content_hash(record) != record.get("content_hash"):
            raise RuntimeError(f"Record hash mismatch: {task_id}")
        ast.parse(record["code_under_test"])
        ast.parse(record["reference_code"])
        for test in record.get("tests", []):
            compile(test["code"], f"<{task_id}-test>", "exec")
        evidence = (record.get("provenance") or {}).get("official_test_evidence") or {}
        fixed_rc = evidence.get("fixed_returncode")
        buggy_rc = evidence.get("buggy_returncode")
        if fixed_rc != 0 or not isinstance(buggy_rc, int) or buggy_rc == 0:
            raise RuntimeError(f"Missing fixed-pass/buggy-fail evidence: {task_id}")
        completion_tokens = [
            len(tokenizer.encode(
                test["code"].strip() + "<|endoftext|>", add_special_tokens=False,
            ).ids)
            for test in record.get("tests", [])
        ]
        if not completion_tokens:
            raise RuntimeError(f"Record has no test completion: {task_id}")
        fits = all(count < token_limit for count in completion_tokens)
        original = originals.get(record["id"])
        details.append({
            "task_id": task_id,
            "record_id": record["id"],
            "fixed_returncode": fixed_rc,
            "buggy_returncode": buggy_rc,
            "completion_tokens": completion_tokens,
            "completion_token_limit": token_limit,
            "fits_training_context": fits,
            "original_buggy_source_chars": len((original or {}).get("code_under_test", "")),
            "regenerated_buggy_source_chars": len(record["code_under_test"]),
            "decision": (
                "trainable_after_compaction"
                if fits else "canonical_retained_training_excluded"
            ),
            "source_result_file": str(path.resolve()),
        })
    if task_ids != EXPECTED_TASKS:
        raise RuntimeError(
            f"Expected exactly {sorted(EXPECTED_TASKS)}, got {sorted(task_ids)}"
        )
    return {
        "schema_version": 1,
        "run_id": "v3-overlong-reverify-1",
        "expected_tasks": len(EXPECTED_TASKS),
        "completed": len(details),
        "behaviorally_reverified": len(details),
        "trainable_after_compaction": sum(
            item["fits_training_context"] for item in details
        ),
        "canonical_retained_training_excluded": sum(
            not item["fits_training_context"] for item in details
        ),
        "ready_for_next_stage": len(details) == len(EXPECTED_TASKS),
        "details": details,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result-dir", type=Path,
        default=ROOT / "data" / "bugsinpy_v3_linux_ingestion"
        / "v3-overlong-reverify-1" / "remote_results" / "results",
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "results" / "v3_overlong_reverification_audit.json",
    )
    parser.add_argument("--completion-token-limit", type=int, default=1536)
    cached = sorted((Path.home() / ".cache" / "huggingface" / "hub").glob(
        "models--microsoft--Phi-3-mini-4k-instruct/snapshots/*/tokenizer.json"
    ))
    parser.add_argument(
        "--tokenizer-json", type=Path,
        default=cached[-1] if cached else None,
    )
    arguments = parser.parse_args()
    if arguments.tokenizer_json is None or not arguments.tokenizer_json.exists():
        raise RuntimeError("Locked Phi-3 tokenizer.json is unavailable")
    report = audit(
        arguments.result_dir, arguments.tokenizer_json,
        arguments.completion_token_limit,
    )
    write_json(arguments.output, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
