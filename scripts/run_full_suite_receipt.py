"""Run the complete test suite alone and record a source-bound receipt (for preflight v2).

The receipt lives in ignored local storage and names the commit it ran at; preflight v2
accepts it only if that commit is the current clean HEAD.
"""
from __future__ import annotations

import json
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "sft_root_cause" / "native_v2_full_suite.json"


def main() -> int:
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                          text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=all"], cwd=ROOT,
                           capture_output=True, text=True).stdout.strip()
    started = time.time()
    done = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                           "-rfE"], cwd=ROOT, capture_output=True, text=True)
    tail = done.stdout[-3000:]
    counts = {k: int(v) for v, k in re.findall(r"(\d+) (passed|failed|skipped|error)", tail)}
    receipt = {"source_commit": head, "tree_clean_at_start": not dirty,
               "exit": done.returncode, "passed": counts.get("passed", 0),
               "failed": counts.get("failed", 0) + counts.get("error", 0),
               "skipped": counts.get("skipped", 0), "seconds": round(time.time() - started, 1),
               "tail": tail[-1500:]}
    if dirty:
        receipt["exit"] = receipt["exit"] or 99
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(receipt, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: receipt[k] for k in ("source_commit", "exit", "passed", "failed",
                                              "skipped", "seconds")}))
    return 0 if receipt["exit"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
