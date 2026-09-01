"""CPU-only upstream-label audit and deterministic sampling-control diagnostic.

Only the training split is selected. No model, locked validation, final-test
performance, Modal job, corpus rewrite or checkpoint is involved.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import CANONICAL_CORPUS_VERSION
from harness.corpus import verify_corpus, write_json
from scripts.train_on_dataset import balanced_repeat_examples, load_phase3_pairs, make_sft_data_point
from utils.dataset_identity import DATASET_IDENTITY_POLICY, dataset_name_for_pair
from utils.reproducibility import source_tree_sha256
from utils.sampling_audit import summarize_sampling_weights


def saturated_sampler_diagnostic() -> dict:
    """A fixture, not corpus-derived test supervision or a model result."""
    examples = [make_sft_data_point({
        "id": f"fixture-{index}", "source": {
            "name": "shared_ingestion", "upstream": "mbpp" if index < 8 else "humaneval",
        }, "bug_family": "boundary" if index % 2 else "arithmetic",
    }, "fixture prompt", f"fixture completion {index}") for index in range(10)]
    reports = {}
    for mode in ("dataset", "dataset_family"):
        doubled, stats = balanced_repeat_examples(examples, 20, 2, mode)
        reports[mode] = {
            "sampler": stats,
            "weights": summarize_sampling_weights(examples, doubled),
        }
    return {"scope": "deterministic_10_example_fixture_not_model_performance", "treatments": reports}


def build_audit(corpus_dir: Path) -> dict:
    manifest = verify_corpus(corpus_dir)
    pairs = load_phase3_pairs(corpus_dir, "train")
    sources = Counter(str(pair.get("source_name") or "unknown") for pair in pairs)
    datasets = Counter(dataset_name_for_pair(pair) for pair in pairs)
    by_source = defaultdict(Counter)
    for pair in pairs:
        by_source[str(pair.get("source_name") or "unknown")][dataset_name_for_pair(pair)] += 1
    return {
        "schema_version": "oneiros_dataset_sampling_audit_v1",
        "scope": "eligible_training_records_only",
        "corpus_id": manifest["corpus_id"],
        "records_sha256": manifest["files"]["records.json"]["sha256"],
        "source_tree_sha256": source_tree_sha256(ROOT),
        "dataset_identity_policy": DATASET_IDENTITY_POLICY,
        "training_records": len(pairs),
        "records_by_ingestion_source": dict(sorted(sources.items())),
        "records_by_upstream_dataset": dict(sorted(datasets.items())),
        "upstream_datasets_within_each_source": {
            key: dict(sorted(value.items())) for key, value in sorted(by_source.items())
        },
        "sources_that_would_collapse_multiple_datasets": sorted(
            key for key, value in by_source.items() if len(value) > 1
        ),
        "unknown_dataset_records": datasets.get("unknown", 0),
        "labels_complete": bool(pairs) and not datasets.get("unknown", 0),
        "sampling_diagnostic": saturated_sampler_diagnostic(),
        "G_group_status": "blocked_unmatched_sampling_controls",
        "modal_used": False,
        "gpu_model_loaded": False,
        "final_test_evaluated": False,
        "corpus_modified": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-dir", type=Path, default=ROOT / "data" / "corpus" / CANONICAL_CORPUS_VERSION)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "v4_1_dataset_sampling_audit.json")
    args = parser.parse_args()
    report = build_audit(args.corpus_dir)
    write_json(args.output, report)
    print(f"Training records: {report['training_records']}; upstream datasets: {report['records_by_upstream_dataset']}")
    print(f"Labels complete: {report['labels_complete']}; G controls remain blocked. Audit: {args.output}")
    return 0 if report["labels_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
