"""Materialize sealed-test-free local shards from a verified canonical corpus."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import CANONICAL_CORPUS_VERSION
from harness.corpus import valid_corpus_version
from harness.corpus_view import materialize_development_view


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-version", default=CANONICAL_CORPUS_VERSION)
    args = parser.parse_args()
    if not valid_corpus_version(args.corpus_version):
        parser.error("invalid canonical corpus version")
    manifest = materialize_development_view(
        ROOT / "data" / "corpus" / args.corpus_version
    )
    summary = {
        "schema_version": manifest["schema_version"],
        "source_corpus_id": manifest["source_corpus_id"],
        "included_splits": manifest["included_splits"],
        "sealed_splits_excluded": manifest["sealed_splits_excluded"],
        "split_record_counts": {
            split: item["record_count"] for split, item in manifest["splits"].items()
        },
        "complexity_policy_version": manifest["complexity_manifest"]["policy_version"],
        "payload_logged": False,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
