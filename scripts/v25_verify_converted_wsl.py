"""Verify v2.5 converted synthetic pytest_module_v1 candidates with the FROZEN v2.4 executor
(static policy, sandboxed buggy/fixed runs with reach tracking, kill reruns, classification).

Each record gets a two-revision view holding only ``oneiros_target.py``: the mutant (buggy) or
the reference (fixed). The candidate test never contains either. Accepted = a confirmed kill
(semantic or crash) that is fixed-valid (fixed passes and reaches the target; the rerun agrees).
Only deterministic verdict fields are written, in input order, so two runs can be compared
byte for byte. Repository fragments are reported as pending (no native environment).

    wsl -u root -- bash scripts/wsl_native_python.sh scripts/v25_verify_converted_wsl.py \
        --dir results/sft_root_cause/v25_corpus_stage1 --run 1
"""
from __future__ import annotations

import argparse
import hashlib
import json
from multiprocessing import Pool
from pathlib import Path
import shutil
import sys
import tempfile

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
import native_generated_tests_execute_wsl as ex  # noqa: E402
import v25_module_conversion as mc  # noqa: E402

ENV = Path("/root/oneiros_v25_synthetic_env")
VIEWS = Path("/root/oneiros_v25_synthetic_views")
WORKERS = 16


def view_for(row: dict) -> dict:
    key = hashlib.sha256((row["reference_code"] + "\x00" + row["code_under_test"])
                         .encode()).hexdigest()[:24]
    views, shas = {}, {}
    for label, code in (("buggy", row["code_under_test"]), ("fixed", row["reference_code"])):
        d = VIEWS / key / label
        f = d / f"{mc.INTERFACE_MODULE}.py"
        if not f.exists():
            d.mkdir(parents=True, exist_ok=True)
            tmp = d / ".tmp"
            tmp.write_text(code, encoding="utf-8")
            tmp.replace(f)
        views[label], shas[label] = str(d), hashlib.sha256(f.read_bytes()).hexdigest()
    return {"module": mc.INTERFACE_MODULE, "qualname": row["entry_point"],
            "python": str(ENV / "bin" / "python"), "env_dir": str(ENV), "views": views,
            "module_sha256": shas}


def verify(row: dict) -> dict:
    out = {"index": row["index"], "module_sha256": row["conversion"].get("module_sha256")}
    if not row["conversion"]["accepted"]:
        return {**out, "status": "conversion_rejected", "accepted": False}
    if row["execution_mode"] != "function_assertion":
        return {**out, "status": "pending_native_environment", "accepted": False}
    scratch = Path(tempfile.mkdtemp(prefix="oneiros_v25v_"))
    try:
        outcome = ex.execute_candidate(view_for(row), row["conversion"]["module"], scratch)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    cls = outcome["classification"]
    fixed_valid = ex.fixed_valid_of(cls)
    seconds = {label: (outcome.get("runs") or {}).get(label, {}).get("seconds")
               for label in ("buggy", "fixed")}
    return {**out, "status": "executed", "class": cls["class"], "fixed_valid": fixed_valid,
            "rerun_agrees": cls.get("rerun_agrees"),
            "accepted": cls["class"] in ex.KILLS and fixed_valid,
            "_seconds": round(sum(s or 0 for s in seconds.values()), 2)}


def env_manifest() -> dict:
    """Deterministic identity of the pinned synthetic verification environment."""
    import subprocess
    site = sorted(p for p in (ENV / "lib").rglob("*") if p.is_file()
                  and "__pycache__" not in p.parts)
    files = {p.relative_to(ENV).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in site}
    version = subprocess.run([str(ENV / "bin" / "python"), "-c",
                              "import sys, pytest; print(sys.version.split()[0], "
                              "pytest.__version__)"], capture_output=True, text=True
                             ).stdout.split()
    dists = sorted(p.name for p in (ENV / "lib").rglob("*.dist-info"))
    return {"path_role": "pinned synthetic verification venv", "python": version[0],
            "pytest": version[1], "distributions": dists,
            "site_packages_sha256": hashlib.sha256(json.dumps(files, sort_keys=True)
                                                   .encode()).hexdigest(),
            "executor_limits": ex.LIMITS, "module_limits": ex.MODULE_LIMITS,
            "workers": WORKERS}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dir", required=True)
    parser.add_argument("--run", type=int, required=True)
    args = parser.parse_args(argv)
    directory = REPO / args.dir
    out = directory / f"verification_run{args.run}.jsonl"
    if out.exists():
        raise SystemExit(f"REFUSED: {out.name} exists")
    rows = [json.loads(l) for l in (directory / "converted_candidates.jsonl")
            .read_text(encoding="utf-8").splitlines()]
    for row in rows:                       # views are created once, before any worker starts
        if row["execution_mode"] == "function_assertion" and row["conversion"]["accepted"]:
            view_for(row)
    manifest = env_manifest()
    mpath = directory / "synthetic_env_manifest.json"
    if mpath.exists() and json.loads(mpath.read_text(encoding="utf-8")) != manifest:
        raise SystemExit("REFUSED: the verification environment changed between runs")
    mpath.write_bytes((json.dumps(manifest, indent=1, sort_keys=True) + "\n").encode())
    with Pool(WORKERS) as pool:
        verdicts = pool.map(verify, rows, chunksize=8)
    # deterministic verdicts (compared byte for byte) and a separate, non-deterministic
    # timing sidecar used only by the exclusion audit
    timing = [{"index": v["index"], "seconds": v.pop("_seconds", None)} for v in verdicts]
    out.write_bytes(("\n".join(json.dumps(v, sort_keys=True) for v in verdicts) + "\n")
                    .encode("utf-8"))
    (directory / f"verification_run{args.run}.timing.jsonl").write_bytes(
        ("\n".join(json.dumps(t, sort_keys=True) for t in timing) + "\n").encode("utf-8"))
    accepted = sum(v["accepted"] for v in verdicts)
    print(json.dumps({"rows": len(verdicts), "accepted": accepted,
                      "sha256": hashlib.sha256(out.read_bytes()).hexdigest()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
