"""Run the complete test suite alone and record a source-bound receipt.

The receipt lives in ignored local storage and names the commit and the canonical
executable-source identity it ran at (amendment v2.2 section D). An existing receipt is
never overwritten (the v2.1 receipt is historical evidence); pass a new --out instead.

    python scripts/run_full_suite_receipt.py [--out results/sft_root_cause/native_v22_full_suite.json]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
OUT = "results/sft_root_cause/native_v24_full_suite.json"


def receipt_bytes(receipt: dict) -> bytes:
    """UTF-8 JSON with LF line endings and exactly one final LF, on every platform: the
    preflight hashes exactly the bytes a checkout (``*.json text eol=lf``) reproduces."""
    return (json.dumps(receipt, indent=1) + "\n").encode("utf-8")


def write_receipt(out: Path, receipt: dict) -> dict:
    from scripts.receipt_sanitize import scrub_json     # tracked receipt: no user paths
    receipt = scrub_json(receipt)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(receipt_bytes(receipt))
    return receipt


def main(argv=None) -> int:
    from harness.native_launch_gate import source_identity
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=OUT)
    out = ROOT / parser.parse_args(argv).out
    if out.exists():
        raise SystemExit(f"REFUSED: {out} exists; receipts are never overwritten")
    identity = source_identity(ROOT)
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                          text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=all"], cwd=ROOT,
                           capture_output=True, text=True).stdout.strip()
    started = time.time()
    done = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                           "-rfE"], cwd=ROOT, capture_output=True, text=True)
    tail = done.stdout[-3000:]
    counts = {k: int(v) for v, k in re.findall(r"(\d+) (passed|failed|skipped|error)", tail)}
    receipt = {"schema_version": "oneiros_native_full_suite_receipt_v2",
               "source_commit": head, "tree_clean_at_start": not dirty,
               "executable_tree_sha256": identity["executable_tree_sha256"],
               "protocol_sha256": identity["protocol_sha256"],
               "exit": done.returncode, "passed": counts.get("passed", 0),
               "failed": counts.get("failed", 0) + counts.get("error", 0),
               "skipped": counts.get("skipped", 0), "seconds": round(time.time() - started, 1),
               "tail": tail[-1500:]}
    if dirty:
        receipt["exit"] = receipt["exit"] or 99
    receipt = write_receipt(out, receipt)
    print(json.dumps({k: receipt[k] for k in ("source_commit", "exit", "passed", "failed",
                                              "skipped", "seconds")}))
    return 0 if receipt["exit"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
