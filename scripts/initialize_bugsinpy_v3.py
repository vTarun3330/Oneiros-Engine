"""Create independent V3 BugsInPy staging from the completed V2 evidence.

V2's corpus must stay immutable.  This utility copies only the verified
materialization/report JSON into a new V3 staging directory; fetched Git
history is reused separately through the ingestion runner's cache option.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
V2_STAGING = ROOT / "data" / "bugsinpy_v2_ingestion"
V3_STAGING = ROOT / "data" / "bugsinpy_v3_ingestion"
SEED_FILES = (
    "materialized_records.json",
    "repository_fragment_records.json",
    "ingestion_report.json",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed isolated BugsInPy V3 ingestion staging")
    parser.add_argument("--output-dir", default=str(V3_STAGING))
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    copied = {}
    for filename in SEED_FILES:
        source = V2_STAGING / filename
        destination = output_dir / filename
        if not source.exists():
            raise RuntimeError(f"Missing completed V2 staging artifact: {source}")
        if destination.exists():
            if _sha256(source) != _sha256(destination):
                raise RuntimeError(
                    f"Refusing to overwrite non-identical V3 staging artifact: {destination}"
                )
        else:
            shutil.copy2(source, destination)
        copied[filename] = _sha256(destination)

    report = json.loads((output_dir / "ingestion_report.json").read_text(encoding="utf-8"))
    seed_manifest = {
        "seeded_from": str(V2_STAGING),
        "seed_files_sha256": copied,
        "seed_report_count": len(report),
        "seed_accepted_count": sum(item.get("status") == "accepted" for item in report),
    }
    (output_dir / "seed_manifest.json").write_text(
        json.dumps(seed_manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(seed_manifest, indent=2))


if __name__ == "__main__":
    main()
