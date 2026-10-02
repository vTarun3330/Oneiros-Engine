"""v2.7: qualification manifest = rehearsal manifest targets that were natively qualified, with
their recorded difference-exposing official tests (input of native_rehearsal_prepare_wsl.py,
which requalifies 3/3 and exports the model-visible and verifier-only bundles separately).

    python scripts/v27_qualification_manifest.py --manifest M --records R --out OUT
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent


def sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--records", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    out = ROOT / args.out
    if out.exists():
        raise SystemExit(f"REFUSED: {args.out} exists")
    manifest = json.loads((ROOT / args.manifest).read_text(encoding="utf-8"))
    records = {}
    for line in (ROOT / args.records).read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            records[r["key"]] = r                         # last record per key wins (resume)
    targets, outcomes = [], Counter()
    for t in manifest["targets"]:
        r = records.get(t["key"])
        cat = r["category"] if r else "not_rehearsed"
        outcomes[cat] += 1
        if cat == "natively_qualified" and r.get("difference_exposing_tests"):
            targets.append({**t, "difference_exposing_tests": r["difference_exposing_tests"]})
    payload = {"schema_version": "oneiros_v27_confirmation_qualification_manifest_v1",
               "role": manifest["role"], "training_prohibited": True,
               "source": {"manifest": args.manifest, "manifest_sha256": sha(ROOT / args.manifest),
                          "records": args.records, "records_sha256": sha(ROOT / args.records)},
               "rehearsal_outcomes": dict(outcomes), "targets": targets}
    out.write_bytes((json.dumps(payload, indent=1, sort_keys=True) + "\n").encode("utf-8"))
    print(json.dumps({"targets": len(targets), "outcomes": dict(outcomes), "sha256": sha(out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
