"""v2.7 Phase 6b, probe v2: Atheris adapter-coverage and MEANINGFUL-FUZZABILITY probes for the
sandbox-compatible panel targets (WSL, root, stdlib only; run inside scripts/wsl_isolated.sh).
NO fuzzing campaign, no model, no GPU. Supersedes probe v1 (scripts/v27_atheris_eligibility_wsl.py,
results preserved) which (a) lost child exit evidence (``replay_process_failure`` without return
code or stderr) and (b) could not tell whether a target consumed any fuzz input.

Every sandboxed child (the v5 harness's own PROBE / instrumentation / invocation code, run with
the harness's own launcher ``_launch`` in the target's runtime) records its return code, signal,
bounded path-scrubbed stdout/stderr tails and full-stream SHA-256. A process failure is reported
as exactly that, with its evidence; it is never reinterpreted.

The invocation child replays FIXED seed inputs (8, identical for every target) through the v5
adapter plan and records per seed: ``receiver_built``, ``arguments_built``, ``consumed_bytes``
(fuzz bytes taken while building the receiver and arguments), ``target_invoked`` (the target's
own code object entered, observed by a profile hook) and the outcome, with ``SystemExit`` and
any other ``BaseException`` captured as structured outcomes (the v5 fuzz/replay children catch
``Exception`` only, so such outcomes are also flagged ``v5_terminating``). Each revision is
invoked twice; ``deterministic`` iff both invocations agree for every seed. Buggy and fixed
outcomes are never compared (no kill is computed).

``eligible_for_fuzzing`` (meaningful): adapter plan identical on both revisions, instrumentation
succeeds, every child exited normally with a structured result, deterministic, and on BOTH
revisions at least one seed built receiver + arguments, entered the target and consumed fuzz
bytes, with no v5-terminating outcome. A target whose plan takes no fuzz-derived value is listed
as ``zero_input_invocable`` and never counted as fuzzable.

    python scripts/v27_atheris_eligibility_v2_wsl.py --cohort <cohort> --control <control jsonl>
        --out <dir>
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))

import native_generated_tests_atheris_wsl as ath  # noqa: E402
import native_generation_io as gio  # noqa: E402

PROBE_VERSION = "oneiros_v27_atheris_eligibility_v2"
SEEDS = [hashlib.sha256(b"oneiros-v27-eligibility-v2-seed-%d" % i).digest() * 8 for i in range(8)]
TAIL = 600

INSTRUMENT = ath.COMMON + r'''
import atheris
spec = json.load(open(sys.argv[1]))
try:
    with atheris.instrument_imports():
        func = resolve(spec["module"], spec["qualname"])
    print(json.dumps({"instrumented": True}))
except BaseException as exc:
    print(json.dumps({"instrumented": False, "reason": type(exc).__name__}))
'''

INVOKE = ath.COMMON + r'''
import atheris
spec = json.load(open(sys.argv[1]))
plan = spec["plan"]
func = resolve(spec["module"], spec["qualname"])
if plan.get("receiver"):
    raw = inspect.getattr_static(resolve(plan["receiver"]["module"], plan["receiver"]["qualname"]),
                                 plan["method"])
else:
    raw = func
raw = inspect.unwrap(getattr(raw, "__func__", raw))
CODE = getattr(raw, "__code__", None)
HIT = {"entered": False}

def _prof(frame, event, arg):
    if event == "call" and frame.f_code is CODE:
        HIT["entered"] = True

def stage(data):
    fdp = atheris.FuzzedDataProvider(data)
    total = fdp.remaining_bytes()
    rec = {"receiver_built": None, "arguments_built": False, "consumed_bytes": 0,
           "target_invoked": False, "outcome": None}
    try:
        target = func
        if plan.get("receiver"):
            r = plan["receiver"]
            rargs, rkwargs = build(fdp, r["plan"])
            target = getattr(resolve(r["module"], r["qualname"])(*rargs, **rkwargs), plan["method"])
            rec["receiver_built"] = True
        args, kwargs = build(fdp, plan["params"])
        rec["arguments_built"] = True
    except SystemExit as exc:
        if plan.get("receiver") and not rec["receiver_built"]:
            rec["receiver_built"] = False
        rec["outcome"] = ["system_exit_before_target", str(exc.code)]
        rec["consumed_bytes"] = total - fdp.remaining_bytes()
        return rec
    except BaseException as exc:
        if plan.get("receiver") and not rec["receiver_built"]:
            rec["receiver_built"] = False
        rec["outcome"] = ["build_error", type(exc).__name__]
        rec["consumed_bytes"] = total - fdp.remaining_bytes()
        return rec
    rec["consumed_bytes"] = total - fdp.remaining_bytes()
    HIT["entered"] = False
    sys.setprofile(_prof)
    try:
        value = target(*args, **kwargs)
        rec["outcome"] = ["ok", None]
        try:
            rec["outcome"][1] = canonical(value)
        except Opaque as exc:
            rec["outcome"] = ["opaque", str(exc)]
    except SystemExit as exc:
        rec["outcome"] = ["system_exit", str(exc.code)]
    except Exception as exc:
        rec["outcome"] = ["raise", type(exc).__name__]
    except BaseException as exc:
        rec["outcome"] = ["base_exception", type(exc).__name__]
    finally:
        sys.setprofile(None)
    rec["target_invoked"] = HIT["entered"]
    return rec

out = [stage(open(p, "rb").read()) for p in spec["inputs"]]
print(json.dumps({"code_object_found": CODE is not None, "seeds": out}))
'''

SCRUB = [(re.compile(r"/root/oneiros_v27_prep/[^/\s'\"]+"), "<prep>"),
         (re.compile(r"/root/oneiros_[A-Za-z0-9_]+"), "<root-tmp>"),
         (re.compile(r"/tmp/[A-Za-z0-9_.-]+"), "<tmp>"),
         (re.compile(r"/mnt/c/Users/[^/\s'\"]+"), "<user>")]


def scrub(text: str) -> str:
    for pattern, repl in SCRUB:
        text = pattern.sub(repl, text)
    return text


def run_child(script: str, spec: dict, view: Path, *, files: dict | None = None,
              cpu: int = 60, wall: int = 120) -> dict:
    """One sandboxed child with full exit evidence; result is its last stdout JSON line."""
    out = Path(tempfile.mkdtemp(prefix="oneiros_v27e2_out_"))
    proc, work = ath._launch(script, spec, view, out, cpu=cpu, wall=wall, files=files,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    timed_out = False
    try:
        stdout, stderr = proc.communicate(timeout=wall + 30)
    except subprocess.TimeoutExpired:
        timed_out = True
        proc.kill()
        stdout, stderr = proc.communicate()
    finally:
        shutil.rmtree(work, ignore_errors=True)
        shutil.rmtree(out, ignore_errors=True)
    stdout, stderr = stdout or "", stderr or ""
    code = proc.returncode
    evidence = {"returncode": code, "signal": -code if code is not None and code < 0 else None,
                "timed_out": timed_out,
                "stdout_sha256": hashlib.sha256(stdout.encode()).hexdigest(),
                "stderr_sha256": hashlib.sha256(stderr.encode()).hexdigest(),
                "stdout_tail": scrub(stdout[-TAIL:]), "stderr_tail": scrub(stderr[-TAIL:])}
    try:
        result = json.loads(stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        result = None
    return {"result": result, "process": evidence,
            "normal_exit": code == 0 and not timed_out and result is not None}


def evaluate(rows_by_label: dict) -> dict:
    """Pure decision from the retained invocation evidence (unit-tested)."""
    reps = rows_by_label
    if any(not r["normal_exit"] for runs in reps.values() for r in runs):
        return {"eligible_for_fuzzing": False, "reason": "invocation_process_failure",
                "failure_class": "harness"}
    seeds = {label: [r["result"]["seeds"] for r in runs] for label, runs in reps.items()}
    deterministic = all(runs[0] == runs[1] for runs in seeds.values())
    flags = {}
    for label, runs in seeds.items():
        first = runs[0]
        flags[label] = {
            "receiver_built": any(s["receiver_built"] in (True, None) and s["arguments_built"]
                                  for s in first) if first else False,
            "arguments_built": any(s["arguments_built"] for s in first),
            "target_invoked": any(s["target_invoked"] for s in first),
            "consumes_fuzz_input": any(s["arguments_built"] and s["target_invoked"]
                                       and s["consumed_bytes"] > 0 for s in first),
            "v5_terminating": any((s["outcome"] or [""])[0] in
                                  ("system_exit", "base_exception", "system_exit_before_target")
                                  for s in first),
            "outcomes": sorted({(s["outcome"] or ["none"])[0] for s in first}),
        }
    summary = {k: all(flags[l][k] for l in flags) for k in
               ("receiver_built", "arguments_built", "target_invoked", "consumes_fuzz_input")}
    summary["v5_terminating"] = any(flags[l]["v5_terminating"] for l in flags)
    summary["deterministic"] = deterministic
    summary["per_revision"] = flags
    if not deterministic:
        return {**summary, "eligible_for_fuzzing": False,
                "reason": "nondeterministic_invocation", "failure_class": "target"}
    if summary["v5_terminating"]:
        return {**summary, "eligible_for_fuzzing": False,
                "reason": "target_terminates_interpreter:" + ",".join(
                    sorted({o for l in flags for o in flags[l]["outcomes"]})),
                "failure_class": "target"}
    if not summary["arguments_built"]:
        return {**summary, "eligible_for_fuzzing": False,
                "reason": "receiver_or_arguments_never_built", "failure_class": "adapter"}
    if not summary["target_invoked"]:
        return {**summary, "eligible_for_fuzzing": False,
                "reason": "target_never_entered", "failure_class": "adapter"}
    if not summary["consumes_fuzz_input"]:
        return {**summary, "eligible_for_fuzzing": False,
                "reason": "zero_input_invocable", "failure_class": "zero_input"}
    return {**summary, "eligible_for_fuzzing": True, "reason": None, "failure_class": None}


def plan_classification(reason: str | None) -> str | None:
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


def probe_target(row: dict, scratch: Path) -> dict:
    runtime = ath.runtime_for(row["python_path"], row["env_dir"])
    out = {"runtime": ath.runtime_identity(runtime)}
    if not runtime.get("available"):
        reason = runtime.get("reason")
        return {**out, "adapter_covered": False, "eligible_for_fuzzing": False,
                "reason": reason, "failure_class": plan_classification(reason)}
    ath.set_runtime(runtime)
    spec = {"module": row["module"], "qualname": row["qualname"]}
    probes = {label: run_child(ath.PROBE, spec, ath.fresh_view(Path(row["views"][label]),
                                                                scratch / f"probe_{label}"))
              for label in ("buggy", "fixed")}
    out["probe"] = probes
    for label in ("buggy", "fixed"):
        if not probes[label]["normal_exit"]:
            return {**out, "adapter_covered": False, "eligible_for_fuzzing": False,
                    "reason": f"probe_process_failure:{label}", "failure_class": "harness"}
    b, f = probes["buggy"]["result"], probes["fixed"]["result"]
    if not b.get("eligible"):
        reason = b.get("reason") or "probe_failure"
        return {**out, "adapter_covered": False, "eligible_for_fuzzing": False,
                "reason": reason, "failure_class": plan_classification(reason)}
    if not f.get("eligible"):
        reason = "revision_load_mismatch:" + str(f.get("reason"))
        return {**out, "adapter_covered": False, "eligible_for_fuzzing": False,
                "reason": reason, "failure_class": "target"}
    if b.get("plan") != f.get("plan"):
        return {**out, "adapter_covered": False, "eligible_for_fuzzing": False,
                "reason": "revision_plan_mismatch", "failure_class": "target"}
    plan = b["plan"]
    out["plan"] = plan
    out["structural_fuzz_inputs"] = len(plan["params"]) + len(
        (plan.get("receiver") or {}).get("plan") or [])
    inst = run_child(INSTRUMENT, spec, ath.fresh_view(Path(row["views"]["buggy"]),
                                                      scratch / "instrument"))
    out["instrumentation"] = inst
    if not inst["normal_exit"] or not inst["result"].get("instrumented"):
        return {**out, "adapter_covered": True, "eligible_for_fuzzing": False,
                "reason": "instrumentation_failure", "failure_class": "harness"
                if not inst["normal_exit"] else "environment"}
    files = {f"{i:05d}": s for i, s in enumerate(SEEDS)}
    ispec = {**spec, "plan": plan, "inputs": [f"/tmp/work/inputs/{n}" for n in files]}
    reps = {label: [run_child(INVOKE, ispec, ath.fresh_view(Path(row["views"][label]),
                                                            scratch / f"inv_{label}_{r}"),
                              files=files) for r in range(2)]
            for label in ("buggy", "fixed")}
    out["invocations"] = reps
    decision = evaluate(reps)
    return {**out, "adapter_covered": True, **decision}


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


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
    base = Path(tempfile.mkdtemp(prefix="oneiros_v27_elig2_", dir="/root"))
    try:
        for i, key in enumerate(keys):
            row = prepared["rows"][key]
            rec = {"target_id": key, "sandbox_compatible": bool(control[key]["control_passed"]),
                   "live_views_sha256": live[key]}
            if not control[key]["control_passed"]:
                rec.update(probed=False, adapter_covered=False, eligible_for_fuzzing=False,
                           reason="sandbox_control_failed:sandbox_policy_blocks_ctypes_dlopen",
                           failure_class="policy")
            else:
                try:
                    result = probe_target(row, base / f"t{i:02d}")
                except Exception as exc:                     # noqa: BLE001 - recorded
                    result = {"adapter_covered": False, "eligible_for_fuzzing": False,
                              "reason": f"harness_exception:{type(exc).__name__}",
                              "failure_class": "harness", "detail": scrub(str(exc))[:300]}
                finally:
                    ath.RUNTIME.clear()
                evidence = out_dir / "evidence" / f"{row['tag']}.json"
                evidence.write_text(json.dumps(result, indent=1, sort_keys=True, default=str)
                                    + "\n", encoding="utf-8")
                rec.update(probed=True, evidence={"path": evidence.relative_to(REPO).as_posix(),
                                                  "sha256": sha(evidence)},
                           **{k: result.get(k) for k in (
                               "adapter_covered", "eligible_for_fuzzing", "reason",
                               "failure_class", "plan", "structural_fuzz_inputs",
                               "receiver_built", "arguments_built", "target_invoked",
                               "consumes_fuzz_input", "deterministic", "v5_terminating",
                               "per_revision", "runtime")})
            rows.append(rec)
            print(json.dumps({"target": key[-50:], "covered": rec["adapter_covered"],
                              "fuzzable": rec["eligible_for_fuzzing"], "reason": rec["reason"]}),
                  flush=True)
    finally:
        shutil.rmtree(base, ignore_errors=True)
        ath.RUNTIME.clear()
    results = out_dir / "eligibility.jsonl"
    results.write_text("".join(json.dumps(r, sort_keys=True, default=str) + "\n" for r in rows),
                       encoding="utf-8")
    contract = {"schema_version": PROBE_VERSION, "design_version": ath.DESIGN_VERSION,
                "harness_script_sha256": sha(HERE / "native_generated_tests_atheris_wsl.py"),
                "probe_script_sha256": sha(Path(__file__)), "inner_sha256": sha(ath.INNER),
                "cohort": {"path": args.cohort, "sha256": sha(cohort_path)},
                "control": {"path": args.control, "sha256": sha(REPO / args.control)},
                "prep": {"path": prepared["path"], "sha256": prepared["sha256"]},
                "seeds_sha256": [hashlib.sha256(s).hexdigest() for s in SEEDS],
                "fuzzing": False, "model": False,
                "results": {"path": results.relative_to(REPO).as_posix(), "sha256": sha(results)},
                "counts": {"cohort": len(rows),
                           "sandbox_compatible": sum(r["sandbox_compatible"] for r in rows),
                           "adapter_covered": sum(bool(r["adapter_covered"]) for r in rows),
                           "eligible_for_fuzzing": sum(bool(r["eligible_for_fuzzing"])
                                                       for r in rows)}}
    (out_dir / "contract.json").write_text(json.dumps(contract, indent=1, sort_keys=True) + "\n",
                                           encoding="utf-8")
    print(json.dumps(contract["counts"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
