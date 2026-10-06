"""v2.7 Phase 6b: Atheris ELIGIBILITY probes for the native-runnable panel targets (WSL, root,
stdlib only; run inside scripts/wsl_isolated.sh). NO fuzzing campaign, no model, no GPU.

For every native-runnable target (model-free control passed) of the frozen panel cohort, using
the Atheris v5 harness's own functions unchanged (scripts/native_generated_tests_atheris_wsl.py)
inside its sandbox:

1. runtime: the target's prepared interpreter + locked environment and the read-only Atheris
   2.3.0 overlay for its CPython minor version (absent overlay -> atheris_abi_unavailable);
2. live views of BOTH revisions re-hashed against the preparation manifest;
3. adapter-eligibility probe (the harness PROBE child) on a fresh copy of EACH revision: the
   target must import, resolve and receive the SAME adapter plan on both;
4. instrumentation: the target imports under ``atheris.instrument_imports()`` (no Fuzz call);
5. one deterministic seed input (fixed bytes, identical for every target) replayed through the
   harness REPLAY child on each revision, twice: ``deterministic`` iff each revision's two
   outcomes are identical. Buggy and fixed outcomes are NOT compared (no kill is computed).

Every failure is classified (adapter / dependency / environment / harness / target) and kept;
nothing is dropped. Five panel targets excluded by the native control are listed, not probed.

    python scripts/v27_atheris_eligibility_wsl.py --cohort <cohort> --control <control jsonl>
        --out <dir>
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

import native_generated_tests_atheris_wsl as ath  # noqa: E402
import native_generation_io as gio  # noqa: E402

SEED_INPUT = hashlib.sha256(b"oneiros-v27-atheris-eligibility-seed").digest() * 8   # 256 bytes
INSTRUMENT = ath.COMMON + r'''
import atheris
spec = json.load(open(sys.argv[1]))
try:
    with atheris.instrument_imports():
        func = resolve(spec["module"], spec["qualname"])
    print(json.dumps({"instrumented": True, "atheris": atheris.__version__
                      if hasattr(atheris, "__version__") else "2.3.0"}))
except Exception as exc:
    print(json.dumps({"instrumented": False, "reason": type(exc).__name__}))
'''


def failure_class(reason: str | None) -> str | None:
    if not reason:
        return None
    if reason.startswith("adapter_unsupported"):
        return "adapter"
    if reason.startswith("import_failure"):
        return "dependency"
    if reason.startswith(("atheris_abi_unavailable", "runtime_mismatch")):
        return "environment"
    if reason.startswith(("revision_plan_mismatch", "revision_load_mismatch")):
        return "target"
    return "harness"


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def probe_target(row: dict, scratch: Path) -> dict:
    runtime = ath.runtime_for(row["python_path"], row["env_dir"])
    out = {"runtime": ath.runtime_identity(runtime)}
    if not runtime.get("available"):
        return {**out, "eligible": False, "reason": runtime.get("reason")}
    ath.set_runtime(runtime)
    views = {label: ath.fresh_view(Path(row["views"][label]), scratch / f"probe_{label}")
             for label in ("buggy", "fixed")}
    probes = {label: ath.probe(views[label], row["module"], row["qualname"])
              for label in ("buggy", "fixed")}
    out["probe"] = probes
    if not probes["buggy"].get("eligible"):
        return {**out, "eligible": False, "reason": probes["buggy"].get("reason")
                or "probe_failure"}
    if not probes["fixed"].get("eligible"):
        return {**out, "eligible": False,
                "reason": "revision_load_mismatch:" + str(probes["fixed"].get("reason"))}
    if probes["buggy"].get("plan") != probes["fixed"].get("plan"):
        return {**out, "eligible": False, "reason": "revision_plan_mismatch"}
    inst = ath.run_once(INSTRUMENT, {"module": row["module"], "qualname": row["qualname"]},
                        ath.fresh_view(Path(row["views"]["buggy"]), scratch / "instrument"),
                        cpu=60, wall=120)
    out["instrumentation"] = inst
    if not inst.get("instrumented"):
        return {**out, "eligible": False,
                "reason": "instrumentation_failure:" + str(inst.get("reason") or inst.get("error"))}
    info = {"plan": probes["buggy"]["plan"], "returns": probes["buggy"].get("returns")}
    target = {"module": row["module"], "qualname": row["qualname"]}
    seed = scratch / "seed.bin"
    seed.write_bytes(SEED_INPUT)
    reps = {}
    for label in ("buggy", "fixed"):
        reps[label] = []
        for r in range(2):
            view = ath.fresh_view(Path(row["views"][label]), scratch / f"seed_{label}_{r}")
            reps[label].append(ath.replay(view, target, info, [seed]))
    errors = [r["error"] for runs in reps.values() for r in runs if "error" in r]
    deterministic = not errors and all(runs[0] == runs[1] for runs in reps.values())
    out["seed_invocation"] = {"seed_sha256": hashlib.sha256(SEED_INPUT).hexdigest(),
                              "repetitions_per_revision": 2,
                              "outcome_sha256": {label: [hashlib.sha256(json.dumps(
                                  r, sort_keys=True).encode()).hexdigest() for r in runs]
                                  for label, runs in reps.items()},
                              "errors": errors, "deterministic": deterministic}
    if errors:
        return {**out, "eligible": False, "reason": "seed_invocation_failure:" + errors[0]}
    if not deterministic:
        return {**out, "eligible": False, "reason": "nondeterministic_seed_invocation"}
    return {**out, "eligible": True, "reason": None, "plan": info["plan"],
            "returns": info["returns"]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cohort", required=True)
    ap.add_argument("--control", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    out_dir = REPO / args.out
    if out_dir.exists():
        raise SystemExit(f"REFUSED: {out_dir} exists")
    cohort_path = REPO / args.cohort
    cohort = json.loads(cohort_path.read_text(encoding="utf-8"))
    prepared = gio.resolve_prep(REPO / cohort["requalification_records"]["path"], cohort_path,
                                REPO)
    keys = list(cohort["qualified_targets"])
    live = gio.verify_live_views(prepared["rows"], keys)
    control = {json.loads(l)["key"]: json.loads(l) for l in
               (REPO / args.control).read_text(encoding="utf-8").splitlines() if l.strip()}
    if set(control) != set(keys):
        raise SystemExit("REFUSED: control rows do not cover exactly the cohort")
    (out_dir / "evidence").mkdir(parents=True)
    rows = []
    base = Path(tempfile.mkdtemp(prefix="oneiros_v27_ath_elig_", dir="/root"))
    try:
        for i, key in enumerate(keys):
            row = prepared["rows"][key]
            ctl = control[key]
            rec = {"target_id": key, "native_runnable": bool(ctl["control_passed"]),
                   "live_views_sha256": live[key]}
            if not ctl["control_passed"]:
                rec.update(atheris_status="not_probed", eligible=False,
                           reason="native_control_failed:sandbox_policy_blocks_ctypes_dlopen",
                           failure_class="policy")
            else:
                try:
                    result = probe_target(row, base / f"t{i:02d}")
                except Exception as exc:                     # noqa: BLE001 - recorded
                    result = {"eligible": False,
                              "reason": f"harness_exception:{type(exc).__name__}:{exc}"[:300]}
                finally:
                    ath.RUNTIME.clear()
                evidence = out_dir / "evidence" / f"{row['tag']}.json"
                evidence.write_text(json.dumps(result, indent=1, sort_keys=True, default=str)
                                    + "\n", encoding="utf-8")
                rec.update(atheris_status="eligible" if result["eligible"] else "ineligible",
                           eligible=bool(result["eligible"]), reason=result.get("reason"),
                           failure_class=failure_class(result.get("reason")),
                           deterministic=(result.get("seed_invocation") or {}).get(
                               "deterministic"),
                           runtime=result.get("runtime"),
                           evidence={"path": evidence.relative_to(REPO).as_posix(),
                                     "sha256": sha(evidence)})
            rows.append(rec)
            print(json.dumps({"target": key[-50:], "status": rec["atheris_status"],
                              "reason": rec["reason"]}), flush=True)
    finally:
        shutil.rmtree(base, ignore_errors=True)
        ath.RUNTIME.clear()
    results = out_dir / "eligibility.jsonl"
    results.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows),
                       encoding="utf-8")
    contract = {"schema_version": "oneiros_v27_atheris_eligibility_v1",
                "design_version": ath.DESIGN_VERSION,
                "harness_script_sha256": sha(HERE / "native_generated_tests_atheris_wsl.py"),
                "probe_script_sha256": sha(Path(__file__)),
                "inner_sha256": sha(ath.INNER),
                "cohort": {"path": args.cohort, "sha256": sha(cohort_path)},
                "control": {"path": args.control, "sha256": sha(REPO / args.control)},
                "prep": {"path": prepared["path"], "sha256": prepared["sha256"]},
                "seed_sha256": hashlib.sha256(SEED_INPUT).hexdigest(),
                "fuzzing": False, "model": False,
                "results": {"path": results.relative_to(REPO).as_posix(), "sha256": sha(results)},
                "counts": {"cohort": len(rows),
                           "native_runnable": sum(r["native_runnable"] for r in rows),
                           "eligible": sum(r["eligible"] for r in rows)}}
    (out_dir / "contract.json").write_text(json.dumps(contract, indent=1, sort_keys=True) + "\n",
                                           encoding="utf-8")
    print(json.dumps(contract["counts"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
