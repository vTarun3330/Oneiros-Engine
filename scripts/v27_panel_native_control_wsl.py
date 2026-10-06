"""v2.7 Phase 6a: model-free native control of every frozen panel target (WSL, root, stdlib
only; run inside scripts/wsl_isolated.sh).

For each target of the panel cohort, the executor's own known-good environment canary
(scripts/native_generated_tests_execute_wsl.py:canary_check) runs in the generated-test sandbox
on BOTH revision views, twice. A target passes when every run imported the target, executed and
passed on both revisions, the revision attestation matched, the sandbox ran as nobody, and both
repetitions agree (determinism). No model, no generated test, no oracle: this proves the
environment and harness only. Views are re-hashed against their preparation manifests first.

    python scripts/v27_panel_native_control_wsl.py --cohort <cohort json> --out <jsonl>
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))

import native_generated_tests_execute_wsl as ex  # noqa: E402
import native_generation_io as gio  # noqa: E402

REPEATS = 2


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cohort", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    out = REPO / args.out
    if out.exists():
        raise SystemExit(f"REFUSED: {out} exists")
    cohort_path = REPO / args.cohort
    cohort = json.loads(cohort_path.read_text(encoding="utf-8"))
    prepared = gio.resolve_prep(REPO / cohort["requalification_records"]["path"], cohort_path,
                                REPO)
    keys = list(cohort["qualified_targets"])
    gio.verify_live_views(prepared["rows"], keys)
    rows = []
    base = Path(tempfile.mkdtemp(prefix="oneiros_v27_control_", dir="/root"))
    try:
        for i, key in enumerate(keys):
            row = prepared["rows"][key]
            target = {"target_key": key, "module": row["module"], "qualname": row["qualname"],
                      "python": row["python_path"], "env_dir": row["env_dir"],
                      "views": row["views"], "module_sha256": row["module_sha256"]}
            runs = [ex.canary_check(target, base / f"t{i:02d}_{r}") for r in range(REPEATS)]
            uids = {rep.get("uid") for run in runs for rep in run["reports"].values() if rep}
            outcome = [run["ok"] for run in runs]
            rows.append({"key": key, "repository": key.split(":", 1)[1].split("@")[0],
                         "control_passed": all(outcome) and uids <= {65534},
                         "repetitions": outcome, "deterministic": len(set(outcome)) == 1,
                         "uids": sorted(u for u in uids if u is not None),
                         "attestation_expected": target["module_sha256"],
                         "evidence_sha256": hashlib.sha256(json.dumps(
                             runs, sort_keys=True, default=str).encode()).hexdigest(),
                         "evidence": runs})      # sandbox-relative, sanitised by the executor
            print(json.dumps({"key": key[-48:], "control_passed": rows[-1]["control_passed"],
                              "deterministic": rows[-1]["deterministic"]}), flush=True)
    finally:
        shutil.rmtree(base, ignore_errors=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows), encoding="utf-8")
    summary = {"targets": len(rows), "passed": sum(r["control_passed"] for r in rows),
               "deterministic": sum(r["deterministic"] for r in rows),
               "executor_sha256": hashlib.sha256((HERE / "native_generated_tests_execute_wsl.py")
                                                 .read_bytes()).hexdigest(),
               "sha256": hashlib.sha256(out.read_bytes()).hexdigest()}
    print(json.dumps(summary, indent=1))
    return 0 if summary["passed"] == summary["targets"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
