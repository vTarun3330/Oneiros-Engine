"""Actual atheris 2.3.0 comparison modes for native generated tests (protocol v2, section 7).

Runs under the WSL atheris interpreter (/opt/atheris311/bin/python, Python 3.11; stdlib +
atheris). Three separately labelled modes:

  ordinary      fuzz the BUGGY target; a finding is an uncaught exception or a violated
                declared return-type contract; it is a bug-specific kill only if the same
                input, replayed on the fixed revision in a separate process, does not fail
  posthoc       buggy-only coverage search (exceptions caught; fixed behaviour never guides
                the search); the saved corpus is replayed afterwards on both revisions and
                any difference in value or exception type is a kill
  differential  online buggy-vs-fixed comparison inside the fuzz loop: an ORACLE-ASSISTED
                UPPER BOUND, reported separately (single-file modules only)

Eligibility comes from annotations: every parameter needs a supported type or a default;
instance methods (receiver needed) and variadic signatures are ineligible, with reasons.
The preregistered full budget is 600 CPU-seconds per target, mode and seed (enforced as
-max_total_time; actual CPU time is recorded). Canaries use a short budget.

    python native_generated_tests_atheris_wsl.py canaries <out_dir>
"""
from __future__ import annotations

import hashlib
import inspect
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import tempfile
import time
import typing

DESIGN_VERSION = "oneiros_native_generated_tests_atheris_v1"
ATHERIS_PYTHON = "/opt/atheris311/bin/python"
FULL_BUDGET_CPU_SECONDS = 600
CANARY_BUDGET_SECONDS = 30
SEEDS = (42, 43, 44)
SUPPORTED = {"int", "float", "str", "bytes", "bool", "list[int]", "list[str]", "list[float]"}


def type_name(annotation) -> str | None:
    if annotation is inspect.Parameter.empty:
        return None
    if isinstance(annotation, str):
        return annotation.replace("List[", "list[").replace("typing.", "")
    origin = typing.get_origin(annotation)
    if origin is list:
        (arg,) = typing.get_args(annotation) or (None,)
        return f"list[{getattr(arg, '__name__', arg)}]"
    return getattr(annotation, "__name__", None)


def eligibility(func, qualname: str) -> dict:
    """Typed-argument applicability; never a silent zero."""
    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):
        return {"eligible": False, "reason": "no_signature"}
    params = list(signature.parameters.values())
    if "." in qualname and params and params[0].name == "self":
        return {"eligible": False, "reason": "instance_method_receiver"}
    types = []
    for p in params:
        if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            return {"eligible": False, "reason": "variadic_signature"}
        name = type_name(p.annotation)
        if name in SUPPORTED:
            types.append(name)
        elif p.default is not inspect.Parameter.empty:
            continue
        else:
            return {"eligible": False, "reason": f"unsupported_parameter:{p.name}:{name}"}
    return {"eligible": True, "types": types,
            "returns": type_name(signature.return_annotation)}


FUZZ_CHILD = r'''
import atheris, importlib.util, json, os, sys
spec = json.load(open(sys.argv[1]))

def load(path):
    # Import BY NAME inside instrument_imports(): loading from a file location bypasses the
    # import hook, and instrument_all() fails on this runtime, leaving coverage at 1.
    import importlib, os
    directory, stem = os.path.split(os.path.splitext(path)[0])
    sys.path.insert(0, directory)
    return importlib.import_module(stem)

with atheris.instrument_imports():
    buggy = load(spec["buggy_file"])
    fixed = load(spec["fixed_file"]) if spec["mode"] == "differential" else None
fb = getattr(buggy, spec["function"])
ff = getattr(fixed, spec["function"]) if fixed else None
TYPES = spec["types"]
RETURNS = spec.get("returns")
MARK = spec["reach_marker"]

def build(fdp):
    out = []
    for t in TYPES:
        if t == "int": out.append(fdp.ConsumeIntInRange(-10**6, 10**6))
        elif t == "float": out.append(fdp.ConsumeRegularFloat())
        elif t == "str": out.append(fdp.ConsumeUnicodeNoSurrogates(32))
        elif t == "bytes": out.append(fdp.ConsumeBytes(32))
        elif t == "bool": out.append(fdp.ConsumeBool())
        elif t == "list[int]": out.append([fdp.ConsumeIntInRange(-1000, 1000) for _ in range(fdp.ConsumeIntInRange(0, 8))])
        elif t == "list[str]": out.append([fdp.ConsumeUnicodeNoSurrogates(8) for _ in range(fdp.ConsumeIntInRange(0, 8))])
        elif t == "list[float]": out.append([fdp.ConsumeRegularFloat() for _ in range(fdp.ConsumeIntInRange(0, 8))])
    return out

def reached():
    if not os.path.exists(MARK):
        open(MARK, "w").close()

class ContractViolation(Exception):
    pass

def check_contract(value):
    names = {"int": int, "float": float, "str": str, "bytes": bytes, "bool": bool}
    if RETURNS in names and not isinstance(value, names[RETURNS]):
        raise ContractViolation(f"returned {type(value).__name__}, declared {RETURNS}")

def one(data):
    args = build(atheris.FuzzedDataProvider(data))
    reached()
    if spec["mode"] == "ordinary":
        check_contract(fb(*args))
    elif spec["mode"] == "posthoc":
        try:
            fb(*args)
        except Exception:
            pass
    else:
        def run(f):
            try:
                return ("ok", repr(f(*args)))
            except Exception as exc:
                return ("raise", type(exc).__name__)
        if run(fb) != run(ff):
            raise AssertionError("differential mismatch")

argv = [sys.argv[0], f"-seed={spec['seed']}", f"-max_total_time={spec['budget']}",
        f"-artifact_prefix={spec['artifacts']}/", "-print_final_stats=1", "-use_value_profile=1"]
if spec["mode"] == "posthoc":
    argv.append(spec["corpus"])
atheris.Setup(argv, one)
atheris.Fuzz()
'''

REPLAY_CHILD = r'''
import atheris, importlib.util, json, sys
spec = json.load(open(sys.argv[1]))
s = importlib.util.spec_from_file_location("oneiros_replay_target", spec["file"])
m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
f = getattr(m, spec["function"])
TYPES = spec["types"]
exec(spec["build_source"])
RETURNS = spec.get("returns")
out = []
for path in spec["inputs"]:
    args = build(atheris.FuzzedDataProvider(open(path, "rb").read()))
    try:
        value = f(*args)
        names = {"int": int, "float": float, "str": str, "bytes": bytes, "bool": bool}
        if RETURNS in names and not isinstance(value, names[RETURNS]):
            out.append(["raise", "ContractViolation"])
        else:
            out.append(["ok", repr(value)])
    except Exception as exc:
        out.append(["raise", type(exc).__name__])
print(json.dumps(out))
'''
BUILD_SOURCE = FUZZ_CHILD[FUZZ_CHILD.index("def build(fdp):"):FUZZ_CHILD.index("def reached():")]


def replay(file: str, function: str, types, returns, inputs, work: Path) -> list:
    spec = work / f"replay_{hashlib.sha256(file.encode()).hexdigest()[:8]}.json"
    spec.write_text(json.dumps({"file": file, "function": function, "types": types,
                                "returns": returns, "inputs": [str(i) for i in inputs],
                                "build_source": BUILD_SOURCE}))
    script = work / "replay_child.py"
    script.write_text(REPLAY_CHILD)
    done = subprocess.run([ATHERIS_PYTHON, str(script), str(spec)], capture_output=True,
                          text=True, timeout=600)
    try:
        return json.loads(done.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return [["replay_error", done.stderr[-200:]]] * len(inputs)


def fuzz(mode: str, buggy_file: str, fixed_file: str, function: str, info: dict, seed: int,
         budget: int, work: Path) -> dict:
    """One mode, one seed. Returns the finding and the bug-specific kill decision."""
    run_dir = work / f"{function}_{mode}_{seed}"
    (run_dir / "artifacts").mkdir(parents=True)
    (run_dir / "corpus").mkdir()
    spec = {"mode": mode, "buggy_file": buggy_file, "fixed_file": fixed_file,
            "function": function, "types": info["types"], "returns": info.get("returns"),
            "seed": seed, "budget": budget, "artifacts": str(run_dir / "artifacts"),
            "corpus": str(run_dir / "corpus"), "reach_marker": str(run_dir / "reached")}
    (run_dir / "spec.json").write_text(json.dumps(spec))
    (run_dir / "fuzz_child.py").write_text(FUZZ_CHILD)
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    started = time.time()
    done = subprocess.run([ATHERIS_PYTHON, str(run_dir / "fuzz_child.py"),
                           str(run_dir / "spec.json")], capture_output=True, text=True,
                          timeout=budget + 120, cwd=run_dir)
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    cpu = round((after.ru_utime + after.ru_stime) - (before.ru_utime + before.ru_stime), 2)
    crashes = sorted((run_dir / "artifacts").glob("crash-*"))
    result = {"mode": mode, "seed": seed, "budget_seconds": budget, "cpu_seconds": cpu,
              "wall_seconds": round(time.time() - started, 2), "exit": done.returncode,
              "reached": (run_dir / "reached").exists(), "findings": len(crashes)}
    if mode in ("ordinary", "differential"):
        if not crashes:
            return {**result, "kill": False}
        if mode == "differential":
            return {**result, "kill": True, "label": "oracle-assisted upper bound"}
        b = replay(buggy_file, function, info["types"], info.get("returns"), crashes[:1], run_dir)
        f = replay(fixed_file, function, info["types"], info.get("returns"), crashes[:1], run_dir)
        return {**result, "buggy_replay": b[0], "fixed_replay": f[0],
                "kill": b[0][0] == "raise" and f[0][0] == "ok"}
    corpus = sorted((run_dir / "corpus").iterdir())[:2000]
    if not corpus:
        return {**result, "kill": False, "corpus": 0}
    b = replay(buggy_file, function, info["types"], info.get("returns"), corpus, run_dir)
    f = replay(fixed_file, function, info["types"], info.get("returns"), corpus, run_dir)
    diffs = sum(x != y for x, y in zip(b, f))
    return {**result, "corpus": len(corpus), "differences": diffs, "kill": diffs > 0}


CANARY_BUGGY = '''
def crash(x: int) -> int:
    if x == 91357:
        raise ValueError("buggy crash")
    return x

def same_crash(x: int) -> int:
    if x == 91357:
        raise ValueError("shared crash")
    return x

def wrong(x: int) -> int:
    if x == 424242:
        return x + 1
    return x

class Box:
    def total(self, n: int) -> int:
        return n

def untyped(x):
    return x
'''
CANARY_FIXED = CANARY_BUGGY.replace('raise ValueError("buggy crash")', "return 0") \
                           .replace("return x + 1", "return x")
EXPECT = {("crash", "ordinary"): True, ("same_crash", "ordinary"): False,
          ("wrong", "ordinary"): False, ("wrong", "posthoc"): True,
          ("wrong", "differential"): True, ("same_crash", "posthoc"): False}


def canaries(out_dir: Path) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="oneiros_atheris_canary_"))
    try:
        (work / "buggy.py").write_text(CANARY_BUGGY)
        (work / "fixed.py").write_text(CANARY_FIXED)
        version = subprocess.run([ATHERIS_PYTHON, "-c", "import importlib.metadata as m, sys; "
                                  "print(m.version('atheris'), sys.version.split()[0])"],
                                 capture_output=True, text=True).stdout.split()
        sys.path.insert(0, str(work))
        import importlib
        buggy = importlib.import_module("buggy")
        elig = {name: eligibility(getattr(buggy, name), name) for name in
                ("crash", "same_crash", "wrong", "untyped")}
        elig["Box.total"] = eligibility(buggy.Box.total, "Box.total")
        results = {}
        for (function, mode), expected in EXPECT.items():
            r = fuzz(mode, str(work / "buggy.py"), str(work / "fixed.py"), function,
                     elig[function], 42, CANARY_BUDGET_SECONDS, work)
            results[f"{function}:{mode}"] = {**r, "expected_kill": expected}
            print(f"{function:11s} {mode:12s} kill={r['kill']} expected={expected} "
                  f"reached={r['reached']} findings={r['findings']}", flush=True)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    checks = {"atheris_2_3_0_py311": version[:1] == ["2.3.0"] and version[1].startswith("3.11"),
              "eligibility_typed": elig["crash"]["eligible"] and not elig["untyped"]["eligible"]
              and elig["Box.total"]["reason"] == "instance_method_receiver",
              **{f"canary_{k}": v["kill"] == v["expected_kill"] and v["reached"]
                 for k, v in results.items()}}
    receipt = {"schema_version": "oneiros_native_atheris_canaries_v1",
               "design_version": DESIGN_VERSION, "atheris": version,
               "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "canary_budget_seconds": CANARY_BUDGET_SECONDS,
               "full_budget_cpu_seconds": FULL_BUDGET_CPU_SECONDS, "seeds": list(SEEDS),
               "eligibility": elig, "results": results, "checks": checks,
               "passed": all(checks.values()),
               "note": ("ordinary Atheris cannot kill the wrong-answer canary (no semantic "
                        "oracle): that is the expected, documented limitation")}
    (out_dir / "atheris_canary_receipt.json").write_text(json.dumps(receipt, indent=1,
                                                                    sort_keys=True) + "\n")
    print(json.dumps({"passed": receipt["passed"],
                      "failed": [k for k, v in checks.items() if not v]}, indent=1))
    return 0 if receipt["passed"] else 1


def main(argv=None) -> int:
    args = argv or sys.argv[1:]
    if args[:1] == ["canaries"]:
        return canaries(Path(args[1]))
    raise SystemExit("usage: canaries <out_dir>")


if __name__ == "__main__":
    raise SystemExit(main())
