"""v2.5 CPU program 2, Phase 13 (WSL, stdlib only): Atheris v5 ABI / import / overlay readiness
for every qualified train environment. NO fuzzing, no search, no comparison.

Per qualified environment: the overlay for its CPython minor version exists (else the
infrastructure category ``atheris_abi_unavailable``); ``import atheris`` from the read-only
overlay succeeds with the target's own interpreter and the view on PYTHONPATH; the
environment lock is identical before and after. Structural eligibility is frozen here.

    wsl -u root -- bash scripts/wsl_native_python.sh scripts/v25_atheris_readiness_wsl.py \
        --envs <env rows>... --out results/sft_root_cause/v25_atheris_readiness.jsonl
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
import native_generated_tests_execute_wsl as ex  # noqa: E402

OVERLAYS = Path("/opt/oneiros_atheris_v5")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--envs", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    out = REPO / args.out
    if out.exists():
        raise SystemExit(f"REFUSED: {out} exists")
    rows = []
    for rel in args.envs:
        for env in map(json.loads, (REPO / rel).read_text(encoding="utf-8").splitlines()):
            if env.get("category") != "qualified":
                continue
            version = env.get("interpreter_version")
            overlay = OVERLAYS / f"cp{str(version).replace('.', '')}"
            row = {"task": env["task"], "project": env["project"], "interpreter": version}
            if not (overlay / "atheris").exists():
                rows.append({**row, "status": "atheris_abi_unavailable",
                             "category": "infrastructure"})
                continue
            python = env["python_path"]
            before = ex.env_lock(python)
            done = subprocess.run(
                [python, "-B", "-c", "import atheris, os; print(os.path.realpath("
                                     "atheris.__file__))"],
                capture_output=True, text=True, cwd="/",
                env={"PATH": "/usr/bin:/bin", "PYTHONNOUSERSITE": "1",
                     "PYTHONDONTWRITEBYTECODE": "1",
                     "PYTHONPATH": f"{env['views']['buggy']}:{overlay}"})
            after = ex.env_lock(python)
            ok = done.returncode == 0 and done.stdout.strip().startswith(str(overlay))
            rows.append({**row, "status": "eligible_abi" if ok and before == after
                         else "infrastructure_failure",
                         "import_ok": ok, "environment_unchanged": before == after,
                         "error": None if ok else (done.stderr or done.stdout)[-200:]})
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(("\n".join(json.dumps(r, sort_keys=True) for r in rows) + "\n")
                    .encode("utf-8"))
    print(json.dumps({"qualified_environments": len(rows),
                      "status": dict(Counter(r["status"] for r in rows)),
                      "by_interpreter": {f"{k[0]}:{k[1]}": v for k, v in Counter(
                          (r["interpreter"], r["status"]) for r in rows).items()},
                      "sha256": hashlib.sha256(out.read_bytes()).hexdigest()}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
