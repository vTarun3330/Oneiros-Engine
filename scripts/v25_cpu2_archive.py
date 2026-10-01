"""v2.5 CPU program 2, Phase 1: byte-verified, read-only external archive of every local-only v2.5
native / pilot / panel / canary artifact created after Stage 1-r2 (copy, never move).

Multi-gigabyte WSL environments are preserved by RECIPE (``v25_native/env_recipes.jsonl``:
install inputs, full freeze text, recomputed locks equal to the recorded ones), which is itself
archived. Only the portable manifest is tracked.

    python scripts/v25_cpu2_archive.py --archive-root <existing external archive directory>
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ARCHIVE = Path("native_v25_cpu_a5be32e")            # name only; the root is --archive-root
MANIFEST = "results/sft_root_cause_v25_cpu2_archive_manifest.json"
BASE = "results/sft_root_cause"
SOURCES = ("v25_native", "v25_pilot", "v25_panel", "native_v25_canaries/quarantine_attempt1")


def role(rel: str) -> str:
    name = rel.rsplit("/", 1)[-1]
    if "/patches/" in rel:
        return "SWE-bench gold+test patch, hash-verified against train metadata"
    if name == "targets.jsonl":
        return "train-only native target manifest"
    if name == "env_recipes.jsonl":
        return "environment rebuild recipes (requirements, freeze, locks)"
    if name.startswith("envs_"):
        return "native environment results (build/qualification run)"
    if name.startswith("canary_"):
        return "native environment canary run"
    if "repository_verification" in name:
        return "repository-candidate verification run"
    if name == "export.jsonl":
        return "30-target train-only pilot export (not run)"
    if name == "training_function_fingerprints.json":
        return "confirmation-panel near-duplicate fingerprints"
    if "quarantine" in rel:
        return "quarantined Atheris canary attempt 1 (v4 list plans)"
    return "other"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main(argv=None) -> int:
    import argparse
    global ARCHIVE, MANIFEST, SOURCES
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", required=True,
                        help="existing external archive directory (never created here)")
    parser.add_argument("--archive-name", default=ARCHIVE.name)
    parser.add_argument("--manifest", default=MANIFEST)
    parser.add_argument("--sources", nargs="*", default=list(SOURCES))
    parser.add_argument("--recipes", default="v25_native/env_recipes.jsonl")
    args = parser.parse_args(argv)
    from harness.archive_verify import resolve_root
    ARCHIVE = resolve_root(args.archive_root) / args.archive_name
    MANIFEST, SOURCES = args.manifest, tuple(args.sources)
    from scripts.native_rehearsal_rebuild_v22 import publish_once
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                          text=True, check=True).stdout.strip()
    files = sorted(p for s in SOURCES for p in (ROOT / BASE / s).rglob("*") if p.is_file())
    items, issues = [], []
    for src in files:
        rel = src.relative_to(ROOT / BASE).as_posix()
        dest = ARCHIVE / rel
        digest = sha(src)
        if dest.exists():
            if sha(dest) != digest:
                issues.append(f"{rel}: archived copy differs (not overwritten)")
        else:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)
        if sha(dest) != digest:
            issues.append(f"{rel}: copy not byte-identical")
        rows = None
        if src.suffix == ".jsonl":
            rows = sum(1 for line in src.read_bytes().splitlines() if line.strip())
        items.append({"file": rel, "role": role(rel), "bytes": src.stat().st_size, "rows": rows,
                      "sha256": digest})
    for p in ARCHIVE.rglob("*"):
        if p.is_file():
            os.chmod(p, stat.S_IREAD)
    recipes = ROOT / BASE / args.recipes
    rec_rows = [json.loads(l) for l in recipes.read_text(encoding="utf-8").splitlines()]
    manifest = {
        "schema_version": "oneiros_v25_cpu2_archive_manifest_v1",
        "archive_directory_name": ARCHIVE.name, "originating_commit": head,
        "items": items, "read_only": True,
        "byte_identical_to_originals": not issues, "issues": issues,
        "wsl_environments_by_recipe": {
            "recipes_file": args.recipes,
            "environments": sum(r["task"] != "_extra" for r in rec_rows),
            "present_and_lock_recomputed": sum("recomputed_lock" in r for r in rec_rows),
            "lock_matches_recorded": sum(r.get("lock_matches_recorded") is True
                                         for r in rec_rows),
            "rebuild": "scripts/v25_native_env_wsl.py with targets.jsonl, the cached patches and "
                       "each recipe's interpreter, cutoff and requirements; identity = recorded "
                       "freeze_sha256 + site_packages_manifest_sha256 + view manifests",
            "not_copied": ["/root/oneiros_v25_native (9.2 GB)", "/root/oneiros_v25_native_r2",
                           "/opt/oneiros_atheris_v5 "
                           "(461 MB; RECORD-verified wheels, spec in overlays.json)"]}}
    status = publish_once(MANIFEST, manifest)
    print(json.dumps({"status": status, "items": len(items), "issues": issues,
                      "bytes": sum(i["bytes"] for i in items),
                      "sha256": sha(ROOT / MANIFEST)}, indent=1))
    return 1 if issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
