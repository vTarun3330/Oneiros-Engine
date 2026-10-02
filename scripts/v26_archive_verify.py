"""v2.6 Phase 3: portable read-only verification of every tracked external-archive manifest.

    python scripts/v26_archive_verify.py --archive-root "<external archive directory>" \
        [--receipt results/sft_root_cause_v26_archive_verification.json]

The archive root is required (no personal path in tracked source); it may contain spaces. It is
resolved at run time and printed to the local terminal only. A TRACKED receipt never contains the
absolute path: it records ``absolute_path_redacted: true``, a SHA-256 fingerprint of the
normalised root, the manifest hashes, archive names, item counts and the content verification.
Fails closed on a missing root, manifest or directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MANIFESTS = ("results/sft_root_cause_v25_stage1_archive_manifest.json",
             "results/sft_root_cause_v25_stage1_r2_archive_manifest.json",
             "results/sft_root_cause_v25_cpu2_archive_manifest.json",
             "results/sft_root_cause_v25_cpu2_archive_manifest_r2.json",
             "results/sft_root_cause_v26_archive_manifest.json")


def main(argv=None) -> int:
    from harness.archive_verify import verify
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--archive-root", required=True)
    parser.add_argument("--receipt", default=None)
    args = parser.parse_args(argv)
    result = verify(args.archive_root, [ROOT / m for m in MANIFESTS])
    result["manifests_sha256"] = {m: hashlib.sha256((ROOT / m).read_bytes()).hexdigest()
                                  for m in MANIFESTS}
    if args.receipt:
        from scripts.native_rehearsal_rebuild_v22 import publish_once
        payload = {"schema_version": "oneiros_v26_archive_verification_v1", **result}
        root = payload.pop("archive_root")  # machine-specific: printed, never tracked
        payload.update(archive_root_supplied_by="--archive-root (resolved at run time)",
                       absolute_path_redacted=True,
                       archive_root_fingerprint_sha256=hashlib.sha256(
                           str(Path(root)).replace("\\", "/").lower().encode()).hexdigest())
        print(publish_once(args.receipt, payload))
    print(json.dumps({k: v for k, v in result.items() if k != "manifests_sha256"}, indent=1))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
