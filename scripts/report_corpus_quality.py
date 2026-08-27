"""Report duplicate, verification, exclusion, and equivalence safeguards."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict

sys.path.insert(0, str(Path(__file__).parent.parent))

from harness.corpus import verify_corpus


def build_quality_report(corpus_dir: Path) -> Dict[str, Any]:
    manifest = verify_corpus(corpus_dir)
    records = json.loads(corpus_dir.joinpath("records.json").read_text(encoding="utf-8"))
    exclusions = json.loads(
        corpus_dir.joinpath("training_exclusions.json").read_text(encoding="utf-8")
    )
    external = json.loads(
        corpus_dir.joinpath("external_eval_index.json").read_text(encoding="utf-8")
    )
    verified = sum(
        bool(record.get("quality", {}).get("pair_behaviorally_verified"))
        for record in records
    )
    with_tests = sum(bool(record.get("tests")) for record in records)
    execution_modes = Counter(
        str(record.get("quality", {}).get("execution_mode", "unknown"))
        for record in records
    )
    external_statuses = Counter(str(item.get("status", "unknown")) for item in external)
    semantic_repairs = manifest.get("semantic_repairs", {})
    return {
        "corpus_id": manifest["corpus_id"],
        "canonical_records": len(records),
        "behaviorally_verified_records": verified,
        "records_with_oracle_witness_tests": with_tests,
        "all_canonical_records_have_behavioral_witness": (
            verified == len(records) and with_tests == len(records)
        ),
        "semantic_duplicate_records_removed": int(
            semantic_repairs.get("semantic_duplicate_records_removed", 0)
        ),
        "training_exclusions": len(exclusions),
        "external_task_inventory": len(external),
        "locked_external_tasks": external_statuses.get(
            "locked_external_eval_not_materialized", 0
        ),
        "external_task_status_counts": dict(sorted(external_statuses.items())),
        "execution_mode_counts": dict(sorted(execution_modes.items())),
        "equivalent_mutant_treatment": {
            "confirmed_equivalent_mutants_in_canonical_corpus": 0,
            "basis": (
                "Every accepted pair has at least one retained oracle witness that passes "
                "the reference and fails the code under test; a confirmed equivalent pair "
                "therefore cannot enter the canonical corpus."
            ),
            "limitation": (
                "No claim is made about observational equivalence outside the retained oracle "
                "domain; excluded upstream candidates require separate provenance counts."
            ),
        },
        "quality_gates": manifest["quality_gate"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build_quality_report(args.corpus_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
