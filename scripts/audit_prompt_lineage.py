"""Provenance-aware V4.1 prompt leakage audit.

The auditor renders only the declared model-visible contract, then combines
lineage validation with structural reference-only overlap checks.  It never
changes records and never executes the sealed final test.
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.prompt_provenance import prohibited_lineage_entries
from engine.test_generation_prompt import build_unified_user_prompt
from harness.corpus import write_json

_DISPOSITION_PATH = (
    ROOT / "research" / "leakage_reviews" / "V4_1_DISPOSITIONS.json"
)


def _reviewed_dispositions() -> set[tuple[str, str]]:
    if not _DISPOSITION_PATH.exists():
        return set()
    payload = json.loads(_DISPOSITION_PATH.read_text(encoding="utf-8"))
    return {
        (str(item["record_id"]), str(item["reference_only_fragment_sha256"]))
        for item in payload.get("dispositions", [])
    }


def _normalise_fragment(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _reference_only_lines(buggy: str, reference: str) -> list[str]:
    lines = []
    normal_buggy = _normalise_fragment(buggy)
    for line in difflib.ndiff(buggy.splitlines(), reference.splitlines()):
        if not line.startswith("+ "):
            continue
        value = _normalise_fragment(line[2:])
        # SequenceMatcher may mark a repeated line as locally added even when
        # the exact line already exists elsewhere in the buggy program. Such a
        # line is not reference-only and therefore is not leakage evidence.
        if (
            len(value) >= 8
            and not value.startswith("#")
            and value not in normal_buggy
        ):
            lines.append(value)
    return list(dict.fromkeys(lines))


def _prompt(record: Mapping[str, Any]) -> str:
    return build_unified_user_prompt(
        code_under_test=record.get("prompt_code_under_test") or record["code_under_test"],
        execution_mode=record.get("quality", {}).get("execution_mode", "function_assertion"),
        specification=record.get("specification", ""),
        support_context=record.get("support_context", ""),
        target_symbols=record.get("target_symbols", []),
        entry_point=record.get("entry_point", ""),
    )


def audit_records(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    records = list(records)
    reviewed = _reviewed_dispositions()
    verbatim: list[str] = []
    partial: list[dict[str, Any]] = []
    lineage_failures: dict[str, list[str]] = {
        "gold_test": [],
        "gold_patch": [],
        "oracle": [],
        "other": [],
    }
    schema_failures: list[dict[str, str]] = []
    for record in records:
        record_id = str(record.get("id", "<missing>"))
        try:
            prompt = _prompt(record)
        except Exception as exc:
            schema_failures.append({"record_id": record_id, "error": str(exc)})
            continue
        reference = str(record.get("reference_code", "")).strip()
        if reference and reference in prompt:
            verbatim.append(record_id)
        normal_prompt = _normalise_fragment(prompt)
        for changed_line in _reference_only_lines(
            str(record.get("code_under_test", "")), reference
        ):
            if changed_line in normal_prompt:
                fragment_hash = hashlib.sha256(
                    changed_line.encode("utf-8")
                ).hexdigest()
                is_reviewed = (record_id, fragment_hash) in reviewed
                partial.append({
                    "record_id": record_id,
                    "reference_only_fragment_sha256": fragment_hash,
                    "source_field": (
                        "specification"
                        if changed_line in _normalise_fragment(record.get("specification", ""))
                        else "other_model_visible_field"
                    ),
                    "review_status": (
                        "reviewed_allowed_independent_buggy_context"
                        if is_reviewed else "pending_manual_review"
                    ),
                    "disposition": (
                        "Independent immutable buggy-revision non-gold test-module context; "
                        "not derived from the fixed source."
                        if is_reviewed else ""
                    ),
                })
        failures = prohibited_lineage_entries(record.get("field_lineage", {}))
        for failure in failures:
            lowered = failure.lower()
            if "gold_test" in lowered or "official_test_body" in lowered:
                lineage_failures["gold_test"].append(f"{record_id}:{failure}")
            elif "gold_patch" in lowered:
                lineage_failures["gold_patch"].append(f"{record_id}:{failure}")
            elif "oracle" in lowered or "reference_code" in lowered:
                lineage_failures["oracle"].append(f"{record_id}:{failure}")
            else:
                lineage_failures["other"].append(f"{record_id}:{failure}")

    return {
        "audit_schema_version": "oneiros_prompt_leakage_audit_v4_1",
        "records_scanned": len(records),
        "schema_failures": len(schema_failures),
        "schema_failure_details": schema_failures,
        "verbatim_reference_leaks": len(verbatim),
        "verbatim_reference_leak_record_ids": verbatim,
        "partial_reference_overlap_flags": len(partial),
        "pending_manual_review_flags": sum(
            item["review_status"] == "pending_manual_review" for item in partial
        ),
        "reviewed_manual_flags": sum(
            item["review_status"] != "pending_manual_review" for item in partial
        ),
        "gold_test_lineage_failures": len(lineage_failures["gold_test"]),
        "gold_patch_lineage_failures": len(lineage_failures["gold_patch"]),
        "oracle_lineage_failures": len(lineage_failures["oracle"]),
        "other_prohibited_lineage_failures": len(lineage_failures["other"]),
        "lineage_failure_details": lineage_failures,
        "manual_review_flags": partial,
        "claim_zero_oracle_leakage_supported": not any(
            (
                verbatim,
                lineage_failures["gold_test"],
                lineage_failures["gold_patch"],
                lineage_failures["oracle"],
                lineage_failures["other"],
            )
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus-dir",
        type=Path,
        default=ROOT / "data" / "corpus" / "v4_1_research_hardened_candidate",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    records = json.loads(args.corpus_dir.joinpath("records.json").read_text(encoding="utf-8"))
    report = audit_records(records)
    output = args.output or args.corpus_dir / "leakage_audit.json"
    write_json(output, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
