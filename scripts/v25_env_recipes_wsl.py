"""v2.5 CPU program 2, Phase 1 (WSL, stdlib only): rebuild recipes of the prepared native
environments, so the multi-gigabyte environments are preserved by recipe, not by copy.

For every prepared target directory it records the install inputs (``env_requirements.txt``
text), the FULL sanitised freeze text, and a recomputed environment lock compared with the lock
recorded in the environment-result rows. It also records the Atheris v5 overlay specification
and the synthetic verification environment. Read-only on the environments.

    wsl -u root -- bash scripts/wsl_native_python.sh scripts/v25_env_recipes_wsl.py \
        --envs <env rows>... --out results/sft_root_cause/v25_native/env_recipes.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
import native_generated_tests_execute_wsl as ex  # noqa: E402

NATIVE = Path("/root/oneiros_v25_native")
EXTRA = {"atheris_overlays": Path("/opt/oneiros_atheris_v5/overlays.json"),
         "synthetic_env": Path("/root/oneiros_v25_synthetic_env")}


def freeze_text(python: str) -> str:
    out = subprocess.run(["uv", "pip", "freeze", "--python", python], capture_output=True,
                         text=True).stdout
    return "\n".join(l for l in out.splitlines() if not l.startswith("-e "))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--envs", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    recorded = {}
    for rel in args.envs:
        for row in map(json.loads, (REPO / rel).read_text(encoding="utf-8").splitlines()):
            recorded[row["task"]] = row
    out = REPO / args.out
    if out.exists():
        raise SystemExit(f"REFUSED: {out} exists")
    rows, mismatched = [], []
    for task, row in sorted(recorded.items()):
        prep = Path(row["env_dir"]).parent if row.get("env_dir") else None
        python = row.get("python_path")
        rec = {"task": task, "project": row["project"], "category": row["category"],
               "interpreter_version": row.get("interpreter_version"),
               "resolution_cutoff": row.get("resolution_cutoff"),
               "install_command": row.get("install_command"),
               "dropped_self_requirements": row.get("dropped_self_requirements"),
               "recorded_lock": row.get("environment_lock"),
               "view_manifest_sha256": row.get("view_manifest_sha256")}
        req = prep / "env_requirements.txt" if prep else None
        if req is not None and req.is_file():
            rec["env_requirements"] = req.read_text(encoding="utf-8")
        if python and Path(python).exists():
            rec["freeze"] = freeze_text(python)
            rec["recomputed_lock"] = ex.env_lock(python)
            same = rec["recomputed_lock"] == row.get("environment_lock")
            rec["lock_matches_recorded"] = same
            if row.get("environment_lock") and not same:
                mismatched.append(task)
        else:
            rec["environment_present"] = False
        rows.append(rec)
    extra = {"atheris_overlays": json.loads(EXTRA["atheris_overlays"].read_text())
             if EXTRA["atheris_overlays"].is_file() else None}
    syn = EXTRA["synthetic_env"] / "bin" / "python"
    if syn.exists():
        extra["synthetic_env"] = {"freeze": freeze_text(str(syn)), "lock": ex.env_lock(str(syn))}
    rows.append({"task": "_extra", **extra})
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(("\n".join(json.dumps(r, sort_keys=True) for r in rows) + "\n")
                    .encode("utf-8"))
    print(json.dumps({"environments": len(rows) - 1,
                      "with_python": sum("recomputed_lock" in r for r in rows),
                      "lock_mismatch": mismatched,
                      "sha256": hashlib.sha256(out.read_bytes()).hexdigest()}, indent=1))
    return 1 if mismatched else 0


if __name__ == "__main__":
    raise SystemExit(main())
