"""v2.6 full-suite receipt: run the complete suite ALONE and bind the result to the exact
executable tree. Records the source commit, the canonical executable-tree hash, the full
command, counts, elapsed time, interpreter/platform/library versions and every skip reason.
Refuses to run on a dirty tree; never overwrites an existing receipt.

    python scripts/v26_full_suite_receipt.py --out results/sft_root_cause/<name>.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main(argv=None) -> int:
    from harness.atomic_publish import publish_file_atomically
    from harness.native_launch_gate import source_identity
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    out = ROOT / args.out
    if out.exists():
        raise SystemExit(f"REFUSED: {args.out} exists")
    git = lambda *a: subprocess.run(["git", *a], cwd=ROOT, capture_output=True,
                                    text=True).stdout.strip()
    if git("status", "--porcelain", "--untracked-files=all"):
        raise SystemExit("REFUSED: the tree is not clean")
    identity = source_identity(ROOT)
    pre_commit = git("rev-parse", "HEAD")
    command = [sys.executable, "-m", "pytest", "-q", "-rs", "-p", "no:cacheprovider"]
    started = time.time()
    done = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    tail = done.stdout[-4000:]
    post_identity = source_identity(ROOT)["executable_tree_sha256"]
    post_commit = git("rev-parse", "HEAD")
    post_dirty = git("status", "--porcelain", "--untracked-files=all")
    heavy = subprocess.run(["wsl", "-u", "root", "--", "bash", "-c",
                            "pgrep -af 'v25_|v26_|atheris|native_env' | grep -v pgrep || true"],
                           capture_output=True, text=True).stdout.strip()
    stable = (post_identity == identity["executable_tree_sha256"] and post_commit == pre_commit
              and not post_dirty and not heavy)
    counts = {k: int(v) for v, k in re.findall(r"(\d+) (passed|failed|skipped|error)", tail)}
    skips = re.findall(r"^SKIPPED \[\d+\] (.+)$", done.stdout, re.M)
    import importlib.metadata as md
    receipt = {
        "schema_version": "oneiros_v26_full_suite_receipt_v1",
        "source_commit": pre_commit, "tree_clean_at_start": True,
        "executable_tree_sha256": identity["executable_tree_sha256"],
        "post_run": {"source_commit": post_commit, "executable_tree_sha256": post_identity,
                     "tree_clean": not post_dirty, "concurrent_heavy_processes": heavy or None,
                     "identity_stable": stable},
        "command": ["<python>", *command[1:]], "exit": done.returncode,
        "passed": counts.get("passed", 0), "failed": counts.get("failed", 0),
        "skipped": counts.get("skipped", 0), "errors": counts.get("error", 0),
        "seconds": round(time.time() - started, 1),
        "skip_reasons": skips,
        "environment": {"python": sys.version.split()[0], "platform": platform.platform(),
                        "pytest": md.version("pytest"),
                        "torch": md.version("torch"), "transformers": md.version("transformers")},
        "concurrency": "run alone: no acquisition, native execution, Atheris or build process"}
    publish_file_atomically(out, (json.dumps(receipt, indent=1, sort_keys=True) + "\n")
                            .encode("utf-8"))
    print(json.dumps({k: receipt[k] for k in ("source_commit", "exit", "passed", "failed",
                                              "skipped", "seconds")}))
    return 0 if done.returncode == 0 and stable else 1


if __name__ == "__main__":
    raise SystemExit(main())
