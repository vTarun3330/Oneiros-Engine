"""Phase 1 of the execution-dose experiment: census the verified train pool.

Re-materialises every execution-supervision candidate the first pilot could
have drawn from, under the same eligibility rules, so the dose designs are
chosen from measured supply rather than assumed supply:

* the O1 candidate artifact and the development view are hash-verified by the
  existing loader, which also re-runs the HumanEval giveaway audit and refuses
  unless the audited 585-lineage universe is reproduced;
* the frozen lineage split is reused, never re-drawn: its file hash must match
  the source manifest and a fresh deterministic re-derivation must equal it;
* only the 385 train lineages are materialised.  Pilot-development and
  unopened-confirmation lineages are never executed here;
* only the train shard of the development view is opened, and the corpus
  manifest must certify group-, semantic-group- and project-disjoint splits.

Each surviving row is executed afresh (shown and intended code) through the
sandboxed worker, token-audited with the frozen tokenizer, and kept with its
rejection reason if it fails.  CPU only; no model weights are loaded.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness.execution_supervision_sidecar import (
    freeze_lineage_split, sha256_file, stable_rank,
)
from harness.parallel_execution import progress_map
from scripts.build_execution_supervision_dataset import (
    MODEL_NAME, MODEL_REVISION, _load_verified_inputs, _materialize_row,
)

SCHEMA = "oneiros_execution_dose_census_v1"
SOURCE_DIR = ROOT / "results" / "v4_3_execution_supervision_v1"
CORPUS = ROOT / "data" / "corpus" / "v4_1_research_hardened_candidate"


def is_supervision_candidate(row: dict[str, Any], train_lineages: set[str]) -> bool:
    """The first pilot's replacement rule, unchanged (see _select_replacements)."""
    return (
        str(row["function_lineage"]) in train_lineages
        and row.get("label") == "wrong_oracle"
        and (row.get("call_replay") or {}).get("distinguishing") is True
        and (row.get("verified_correction") or {}).get("verified") is True
    )


def corpus_disjointness_problems(corpus: Path) -> list[str]:
    manifest = json.loads((corpus / "manifest.json").read_text(encoding="utf-8"))
    gate = manifest.get("quality_gate") or {}
    return [f"corpus manifest does not certify {flag}" for flag in (
        "group_disjoint_splits", "semantic_group_disjoint_splits",
        "repository_project_disjoint_splits",
    ) if gate.get(flag) is not True]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, default=ROOT / "results"
                        / "v4_2_oracle_dataset_full" / "candidates.json")
    parser.add_argument("--oracle-manifest", type=Path, default=ROOT / "results"
                        / "v4_2_oracle_dataset_full" / "manifest.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results"
                        / "v4_3_execution_dose_v1")
    parser.add_argument("--receipt", type=Path, default=ROOT / "results"
                        / "v4_3_execution_dose_census.json")
    parser.add_argument("--workers", type=int, default=24)
    args = parser.parse_args(argv)
    started = time.time()

    problems = corpus_disjointness_problems(CORPUS)
    source_manifest_path = SOURCE_DIR / "manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    split_path = SOURCE_DIR / "lineage_split.json"
    if sha256_file(split_path) != source_manifest["lineage_split_sha256"]:
        problems.append("frozen lineage split hash differs from its manifest")
    if problems:
        print("REFUSED: " + "; ".join(problems))
        return 2

    eligible, records, input_audit = _load_verified_inputs(
        args.candidates, args.oracle_manifest, CORPUS, args.workers,
    )
    frozen = json.loads(split_path.read_text(encoding="utf-8"))
    if freeze_lineage_split(eligible) != frozen:
        print("REFUSED: lineage split does not re-derive identically")
        return 2
    train = set(frozen["train_lineages"])
    held_out = set(frozen["pilot_development_lineages"]) | set(
        frozen["unopened_confirmation_lineages"])
    if train & held_out:
        print("REFUSED: train lineages overlap held-out lineages")
        return 2

    candidates = [row for row in eligible if is_supervision_candidate(row, train)]
    candidates.sort(key=lambda row: stable_rank(
        "dose_census", row.get("function_lineage"), row.get("record_id"),
        row.get("rank"), row.get("candidate_position"),
    ))

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, revision=MODEL_REVISION, trust_remote_code=True)

    def job(row: dict[str, Any]) -> dict[str, Any]:
        materialized, reason = _materialize_row(
            row, records[str(row["record_id"])], tokenizer)
        return {"row": materialized, "reason": reason,
                "record_id": str(row["record_id"]),
                "function_lineage": str(row["function_lineage"]),
                "source_dataset": str(row.get("source_dataset") or "unknown")}

    outcomes = progress_map(job, candidates, "dose census", every=500,
                            workers=args.workers)
    pool = [item["row"] for item in outcomes if item["row"] is not None]
    rejections = Counter(item["reason"] for item in outcomes if item["row"] is None)
    if any(row["function_lineage"] in held_out for row in pool):
        print("REFUSED: a held-out lineage reached the materialised pool")
        return 2
    if any(row.get("evaluation_split") != "train" for row in pool):
        print("REFUSED: a materialised row is not labelled train")
        return 2

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pool_path = args.output_dir / "pool.execution.json"
    pool_path.write_bytes((json.dumps(pool, indent=2, ensure_ascii=False) + "\n")
                          .encode("utf-8"))
    eligible_pool = [row for row in pool if row["trace_training_eligible"]]

    def counts(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
        return dict(sorted(Counter(str(row[field]) for row in rows).items()))

    receipt = {
        "schema_version": SCHEMA,
        "label": "train-only supply census; no model was loaded or evaluated",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "input_audit": input_audit,
        "candidate_rule": ("train lineage, label wrong_oracle, distinguishing call, "
                           "verified correction (identical to the first pilot)"),
        "candidate_rows": len(candidates),
        "materialised_rows": len(pool),
        "trace_training_eligible_rows": len(eligible_pool),
        "rejections": dict(rejections.most_common()),
        "unique": {
            "records": len({row["record_id"] for row in eligible_pool}),
            "lineages": len({row["function_lineage"] for row in eligible_pool}),
            "trace_completions": len({row["trace_completion_sha256"]
                                      for row in eligible_pool}),
            "record_call_pairs": len({(row["record_id"], row["call_expression"])
                                      for row in eligible_pool}),
        },
        "eligible_by_source": counts(eligible_pool, "source_dataset"),
        "eligible_by_complexity_tier": counts(eligible_pool, "complexity_tier"),
        "eligible_by_bug_family": counts(eligible_pool, "bug_family"),
        "eligible_lineages_by_source": dict(sorted(Counter(
            source for source, _ in {(row["source_dataset"], row["function_lineage"])
                                     for row in eligible_pool}).items())),
        "held_out_lineages_materialised": 0,
        "inputs": {
            "candidates_sha256": sha256_file(args.candidates),
            "oracle_manifest_sha256": sha256_file(args.oracle_manifest),
            "lineage_split_sha256": sha256_file(split_path),
            "source_manifest_sha256": sha256_file(source_manifest_path),
            "development_view_manifest_sha256": sha256_file(
                CORPUS / "development_view" / "manifest.json"),
            "corpus_manifest_sha256": sha256_file(CORPUS / "manifest.json"),
        },
        "outputs": {"pool": {"path": pool_path.relative_to(ROOT).as_posix(),
                             "sha256": sha256_file(pool_path), "rows": len(pool)}},
        "source_files_sha256": {
            relative: sha256_file(ROOT / relative) for relative in (
                "scripts/census_execution_dose_pool.py",
                "scripts/build_execution_supervision_dataset.py",
                "harness/execution_supervision.py",
                "harness/execution_supervision_sidecar.py",
            )
        },
        "leakage": {"splits_opened": ["train"], "validation_accessed": False,
                    "ablation_dev_accessed": False, "test_accessed": False,
                    "sealed_final_test_accessed": False, "confirmation_opened": False,
                    "pilot_development_lineages_materialised": False},
        "elapsed_seconds": round(time.time() - started, 1),
    }
    args.receipt.write_bytes((json.dumps(receipt, indent=2) + "\n").encode("utf-8"))
    print(json.dumps({key: receipt[key] for key in (
        "candidate_rows", "materialised_rows", "trace_training_eligible_rows",
        "rejections", "unique", "eligible_by_source", "eligible_by_complexity_tier",
        "eligible_lineages_by_source", "elapsed_seconds")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
