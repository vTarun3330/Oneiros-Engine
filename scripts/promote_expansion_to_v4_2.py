"""Build the v4.2 successor corpus by adding verified expansion records.

Rule 3 of this project is that the V4.1 corpus is immutable. This creates a
SUCCESSOR with its own directory, its own hashes and its own manifest, and it
copies V4.1 rather than editing it. If anything here is wrong, V4.1 is still
the corpus every reported result was produced against.

Three properties this enforces rather than assumes:

* The sealed test split is carried across byte-for-byte and never inspected.
  Its record ids move with splits.json and its payloads move with records.json,
  and nothing in this script reads a test record's fields.
* Every added record is train-only, and its id must be absent from all four
  splits before it is accepted. A duplicate id is refused outright rather than
  overwriting an existing record.
* Added records must already carry the corpus schema. This script does not
  repair or infer fields, because a record that needs repairing here is a
  record the expansion should not have emitted.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness.corpus import (
    canonical_json, sha256_file, sha256_text, verify_corpus, write_json,
)
from harness.corpus_view import materialize_development_view

SOURCE_VERSION = "v4_1_research_hardened_candidate"
TARGET_VERSION = "v4_2_balanced_expansion_candidate"
STAGING = ROOT / "data" / "humaneval_expansion_staging" / "records.json"

#: Every field an existing corpus record carries. A staged record missing one
#: is rejected, because the annotator and multi-mutant builder read them all
#: and skip silently rather than failing when one is absent.
REQUIRED_FIELDS = (
    "id", "group_id", "language", "task_mode", "task_type", "test_format",
    "entry_point", "target_symbols", "specification", "support_context",
    "reference_code", "code_under_test", "prompt_code_under_test",
    "content_hash", "source", "provenance", "quality", "field_lineage",
    "tests", "schema_version",
)


def promote(source_dir: Path, target_dir: Path, staging: Path) -> dict[str, Any]:
    verify_corpus(source_dir)

    new_records = json.loads(staging.read_text(encoding="utf-8"))["records"]
    for record in new_records:
        missing = [field for field in REQUIRED_FIELDS if field not in record]
        if missing:
            raise SystemExit(
                f"staged record {record.get('id')} is missing {missing}; the "
                "expansion must emit corpus-schema records, not this script"
            )
        if record.get("task_mode") == "repository":
            raise SystemExit("repository records need their own promotion path")

    if target_dir.exists():
        raise SystemExit(
            f"{target_dir.name} already exists; delete it deliberately rather "
            "than letting a rebuild silently replace a hashed corpus"
        )
    shutil.copytree(source_dir, target_dir, ignore=shutil.ignore_patterns(
        "development_view"
    ))

    records = json.loads((target_dir / "records.json").read_text(encoding="utf-8"))
    splits = json.loads((target_dir / "splits.json").read_text(encoding="utf-8"))

    existing_ids = {str(record["id"]) for record in records}
    assigned = {
        record_id for ids in splits.values() for record_id in ids
    }
    for record in new_records:
        record_id = str(record["id"])
        if record_id in existing_ids or record_id in assigned:
            raise SystemExit(
                f"{record_id} already exists in the corpus; refusing to "
                "overwrite a verified record"
            )

    records.extend(new_records)
    splits["train"] = list(splits["train"]) + [str(r["id"]) for r in new_records]

    write_json(target_dir / "records.json", records)
    write_json(target_dir / "splits.json", splits)

    manifest = json.loads((target_dir / "manifest.json").read_text(encoding="utf-8"))
    manifest["corpus_id"] = "oneiros-corpus-v4.2-balanced-expansion-candidate"
    manifest["version"] = TARGET_VERSION
    manifest["parent_corpus"] = {
        "corpus_id": "oneiros-corpus-v4.1-research-hardened-candidate",
        "version": SOURCE_VERSION,
        "records_sha256": sha256_file(source_dir / "records.json"),
        "splits_sha256": sha256_file(source_dir / "splits.json"),
    }
    manifest["expansion"] = {
        "added_records": len(new_records),
        "added_to_split": "train",
        "source": "scripts/expand_humaneval_coverage.py",
        "rationale": (
            "grow the scarce sources rather than trim the dominant one; mbpp "
            "supplied 555 of 1009 unique targets before this"
        ),
        "sealed_test_split_untouched": True,
    }
    # The manifest carries a per-split summary of its own, and the development
    # view checks its shards against THAT rather than against splits.json. An
    # updated splits.json with a stale manifest summary produces a corpus that
    # verifies and a view that cannot be built.
    train_ids = list(splits["train"])
    train_groups = {
        str(record["group_id"]) for record in records
        if str(record["id"]) in set(train_ids)
    }
    manifest["splits"]["train"] = {
        "record_count": len(train_ids),
        "group_count": len(train_groups),
        "record_ids_sha256": sha256_text(canonical_json(train_ids)),
    }

    for filename in ("records.json", "splits.json"):
        entry = dict(manifest["files"].get(filename) or {})
        entry["sha256"] = sha256_file(target_dir / filename)
        entry["record_count"] = len(
            records if filename == "records.json" else splits
        )
        manifest["files"][filename] = entry
    write_json(target_dir / "manifest.json", manifest)

    verify_corpus(target_dir)
    view = materialize_development_view(target_dir)

    return {
        "schema_version": "oneiros_corpus_promotion_v1",
        "source_corpus": SOURCE_VERSION,
        "target_corpus": TARGET_VERSION,
        "source_corpus_modified": False,
        "sealed_test_records_inspected": False,
        "added_records": len(new_records),
        "added_unique_lineages": len({str(r["group_id"]) for r in new_records}),
        "train_records_before": len(splits["train"]) - len(new_records),
        "train_records_after": len(splits["train"]),
        "sealed_test_record_count_carried": len(splits.get("test") or []),
        "view_split_counts": {
            name: entry["record_count"]
            for name, entry in (view.get("splits") or {}).items()
        },
        "records_sha256": sha256_file(target_dir / "records.json"),
        "splits_sha256": sha256_file(target_dir / "splits.json"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staging", type=Path, default=STAGING)
    parser.add_argument(
        "--target-version", default=TARGET_VERSION,
        help="successor corpus directory name under data/corpus",
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "results" / "v4_2_corpus_promotion.json",
    )
    arguments = parser.parse_args()

    report = promote(
        ROOT / "data" / "corpus" / SOURCE_VERSION,
        ROOT / "data" / "corpus" / arguments.target_version,
        arguments.staging,
    )
    write_json(arguments.output, report)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
