"""Actual atheris 2.3.0 comparison for native generated tests (protocol v2 s7 + amendment v2.1 D).

Runs in WSL as root. Runtime parity: every child uses CPython 3.11 (/usr/bin/python3.11, the
base of the atheris venv) with atheris taken from its site-packages. Each revision runs in
its OWN process and imports the target package BY NAME from its canonical view, so files
with the same basename never share a ``sys.modules`` identity and package-relative imports
work. Arguments are rebuilt from the input bytes independently for every call.

Modes (separately labelled):
  ordinary      buggy-only search; every distinct exception signature is kept as a witness
                (the loop continues rather than stopping at the first); EVERY witness is
                replayed on both revisions; a kill needs buggy failing and fixed returning a
                canonical value, confirmed in fresh single-input processes twice per revision
  posthoc       buggy-only coverage search; the corpus (sorted by file name, capped at 2,000,
                truncation recorded) is replayed on both revisions; differences in canonical
                value or exception type are re-confirmed singly before counting
  differential  online comparison with a separate fixed-revision worker process: an
                ORACLE-ASSISTED UPPER BOUND; witnesses are re-confirmed like the others
Results are canonicalised (None, bool, int, float, str, bytes, list, tuple, dict, set);
anything else is ``opaque`` and never compared. Replay errors are reported separately and are
never kills. Budget: RLIMIT_CPU plus a wall-clock backstop; network disabled (unshare -n).

    python native_generated_tests_atheris_wsl.py canaries <out_dir>
    python native_generated_tests_atheris_wsl.py run --prep <records.jsonl> --manifest <m.json>
        --out <dir> [--budget 600] [--seeds 42,43,44]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

DESIGN_VERSION = "oneiros_native_generated_tests_atheris_v2"
PYTHON = "/usr/bin/python3.11"
ATHERIS_SITE = "/opt/atheris311/lib/python3.11/site-packages"
FULL_BUDGET_CPU_SECONDS = 600
CANARY_BUDGET_SECONDS = 20
WALL_BACKSTOP = 300
CORPUS_CAP = 2000
MAX_WITNESSES = 50
CONFIRMATIONS = 2
SEEDS = (42, 43, 44)
MODES = ("ordinary", "posthoc", "differential")

COMMON = r'''
import base64, importlib, inspect, json, math, sys, typing

SUPPORTED = {"int", "float", "str", "bytes", "bool", "list[int]", "list[str]", "list[float]"}

class Opaque(Exception):
    pass

def type_name(annotation):
    if annotation is inspect.Parameter.empty:
        return None
    if isinstance(annotation, str):
        return annotation.replace("List[", "list[").replace("typing.", "")
    if typing.get_origin(annotation) is list:
        args = typing.get_args(annotation)
        return "list[%s]" % getattr(args[0], "__name__", args[0]) if args else None
    return getattr(annotation, "__name__", None)

def resolve(module, qualname):
    obj = importlib.import_module(module)
    for part in qualname.split("."):
        obj = getattr(obj, part)
    return obj

def eligibility(func, qualname):
    try:
        sig = inspect.signature(func)
    except (TypeError, ValueError):
        return {"eligible": False, "reason": "no_signature"}
    params = list(sig.parameters.values())
    if "." in qualname and params and params[0].name == "self":
        return {"eligible": False, "reason": "instance_method_receiver"}
    plan = []
    for p in params:
        if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            return {"eligible": False, "reason": "variadic_signature"}
        name = type_name(p.annotation)
        if name in SUPPORTED:
            plan.append({"name": p.name, "type": name,
                         "keyword": p.kind == p.KEYWORD_ONLY})
        elif p.default is inspect.Parameter.empty:
            return {"eligible": False, "reason": "unsupported_parameter:%s:%s" % (p.name, name)}
    return {"eligible": True, "plan": plan, "returns": type_name(sig.return_annotation)}

def build(fdp, plan):
    args, kwargs = [], {}
    for p in plan:
        t = p["type"]
        if t == "int": v = fdp.ConsumeIntInRange(-10**6, 10**6)
        elif t == "float": v = fdp.ConsumeRegularFloat()
        elif t == "str": v = fdp.ConsumeUnicodeNoSurrogates(32)
        elif t == "bytes": v = fdp.ConsumeBytes(32)
        elif t == "bool": v = fdp.ConsumeBool()
        elif t == "list[int]": v = [fdp.ConsumeIntInRange(-1000, 1000) for _ in range(fdp.ConsumeIntInRange(0, 8))]
        elif t == "list[str]": v = [fdp.ConsumeUnicodeNoSurrogates(8) for _ in range(fdp.ConsumeIntInRange(0, 8))]
        else: v = [fdp.ConsumeRegularFloat() for _ in range(fdp.ConsumeIntInRange(0, 8))]
        if p["keyword"]:
            kwargs[p["name"]] = v
        else:
            args.append(v)
    return args, kwargs

def canonical(value, depth=0):
    if depth > 6:
        raise Opaque("too deep")
    if value is None or type(value) in (bool, int, str):
        return [type(value).__name__, value]
    if type(value) is float:
        return ["float", "nan" if math.isnan(value) else repr(value)]
    if type(value) is bytes:
        return ["bytes", base64.b64encode(value).decode()]
    if type(value) in (list, tuple):
        return [type(value).__name__, [canonical(v, depth + 1) for v in value]]
    if type(value) is dict:
        items = [[canonical(k, depth + 1), canonical(v, depth + 1)] for k, v in value.items()]
        return ["dict", sorted(items, key=json.dumps)]
    if type(value) in (set, frozenset):
        return [type(value).__name__, sorted((canonical(v, depth + 1) for v in value), key=json.dumps)]
    raise Opaque(type(value).__name__)

def outcome(func, plan, data, atheris, returns):
    args, kwargs = build(atheris.FuzzedDataProvider(data), plan)
    try:
        value = func(*args, **kwargs)
    except Exception as exc:
        return ["raise", type(exc).__name__]
    names = {"int": int, "float": float, "str": str, "bytes": bytes, "bool": bool}
    if returns in names and not isinstance(value, names[returns]):
        return ["raise", "ContractViolation"]
    try:
        return ["ok", canonical(value)]
    except Opaque as exc:
        return ["opaque", str(exc)]
'''

PROBE = COMMON + r'''
spec = json.load(open(sys.argv[1]))
try:
    func = resolve(spec["module"], spec["qualname"])
except SyntaxError as exc:
    print(json.dumps({"eligible": False, "reason": "runtime_mismatch:" + type(exc).__name__}))
    raise SystemExit(0)
except Exception as exc:
    print(json.dumps({"eligible": False, "reason": "import_failure:" + type(exc).__name__}))
    raise SystemExit(0)
print(json.dumps({**eligibility(func, spec["qualname"]), "python": sys.version.split()[0]}))
'''

FUZZ = COMMON + r'''
import atheris, hashlib, os, subprocess
spec = json.load(open(sys.argv[1]))
with atheris.instrument_imports():
    func = resolve(spec["module"], spec["qualname"])
PLAN, RETURNS, MODE = spec["plan"], spec["returns"], spec["mode"]
SEEN = set()
worker = None
if MODE == "differential":
    worker = subprocess.Popen([sys.executable, spec["worker_script"], spec["worker_spec"]],
                              stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)

def witness(data, signature):
    if signature in SEEN or len(SEEN) >= spec["max_witnesses"]:
        return
    SEEN.add(signature)
    name = hashlib.sha256(data).hexdigest()
    with open(os.path.join(spec["witnesses"], name), "wb") as handle:
        handle.write(data)

def one(data):
    if not os.path.exists(spec["reached"]):
        open(spec["reached"], "w").close()
    got = outcome(func, PLAN, data, atheris, RETURNS)
    if MODE == "ordinary" and got[0] == "raise":
        witness(data, got[1])
    elif MODE == "differential" and got[0] in ("ok", "raise"):
        worker.stdin.write(base64.b64encode(data).decode() + "\n")
        worker.stdin.flush()
        other = json.loads(worker.stdout.readline())
        if other[0] in ("ok", "raise") and other != got:
            witness(data, json.dumps([got, other])[:200])

argv = [sys.argv[0], "-seed=%d" % spec["seed"], "-max_total_time=%d" % spec["budget"],
        "-print_final_stats=1", "-use_value_profile=1"]
if MODE == "posthoc":
    argv.append(spec["corpus"])
atheris.Setup(argv, one)
atheris.Fuzz()
'''

WORKER = COMMON + r'''
import atheris
spec = json.load(open(sys.argv[1]))
func = resolve(spec["module"], spec["qualname"])
for line in sys.stdin:
    data = base64.b64decode(line.strip())
    print(json.dumps(outcome(func, spec["plan"], data, atheris, spec["returns"])), flush=True)
'''

REPLAY = COMMON + r'''
import atheris
spec = json.load(open(sys.argv[1]))
try:
    func = resolve(spec["module"], spec["qualname"])
except Exception as exc:
    print(json.dumps({"error": "import:" + type(exc).__name__}))
    raise SystemExit(0)
out = [outcome(func, spec["plan"], open(p, "rb").read(), atheris, spec["returns"])
       for p in spec["inputs"]]
print(json.dumps({"results": out}))
'''


def _child(script: str, spec: dict, work: Path, view: Path, name: str, cpu: int | None = None,
           wall: int = 600) -> subprocess.CompletedProcess:
    path = work / f"{name}.py"
    path.write_text(script, encoding="utf-8")
    spec_path = work / f"{name}.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    env = {"PATH": "/usr/bin:/bin", "PYTHONPATH": f"{view}:{ATHERIS_SITE}",
           "PYTHONDONTWRITEBYTECODE": "1", "PYTHONNOUSERSITE": "1", "PYTHONHASHSEED": "0",
           "HOME": str(work)}
    command = [PYTHON, "-B", str(path), str(spec_path)]
    if cpu is not None:
        command = ["prlimit", f"--cpu={cpu}", "--", *command]
    command = ["unshare", "-n", *command]
    return subprocess.run(command, cwd=work, env=env, capture_output=True, text=True, timeout=wall)


def probe(view: Path, module: str, qualname: str, work: Path) -> dict:
    done = _child(PROBE, {"module": module, "qualname": qualname}, work, view, "probe", wall=120)
    try:
        return json.loads(done.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return {"eligible": False, "reason": "probe_failure", "tail": done.stderr[-200:]}


def replay(view: Path, target: dict, info: dict, inputs: list, work: Path, name: str) -> dict:
    spec = {"module": target["module"], "qualname": target["qualname"], "plan": info["plan"],
            "returns": info["returns"], "inputs": [str(p) for p in inputs]}
    try:
        done = _child(REPLAY, spec, work, view, name, wall=900)
        payload = json.loads(done.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError, subprocess.TimeoutExpired):
        return {"error": "replay_process_failure"}
    return payload


def confirmed(views: dict, target: dict, info: dict, witness: Path, work: Path) -> dict:
    """Fresh single-input processes, CONFIRMATIONS times per revision; stable and different."""
    seen = {"buggy": [], "fixed": []}
    for label in ("buggy", "fixed"):
        for i in range(CONFIRMATIONS):
            r = replay(views[label], target, info, [witness], work, f"confirm_{label}_{i}")
            if "error" in r:
                return {"kill": False, "replay_error": r["error"]}
            seen[label].append(r["results"][0])
    stable = all(x == seen["buggy"][0] for x in seen["buggy"]) and \
        all(x == seen["fixed"][0] for x in seen["fixed"])
    b, f = seen["buggy"][0], seen["fixed"][0]
    comparable = b[0] in ("ok", "raise") and f[0] in ("ok", "raise")
    return {"kill": stable and comparable and b != f, "stable": stable, "buggy": b, "fixed": f}


def fuzz(mode: str, views: dict, target: dict, info: dict, seed: int, budget: int,
         work: Path) -> dict:
    run_dir = work / f"{mode}_{seed}"
    for sub in ("witnesses", "corpus"):
        (run_dir / sub).mkdir(parents=True, exist_ok=True)
    spec = {"module": target["module"], "qualname": target["qualname"], "plan": info["plan"],
            "returns": info["returns"], "mode": mode, "seed": seed, "budget": budget,
            "witnesses": str(run_dir / "witnesses"), "corpus": str(run_dir / "corpus"),
            "reached": str(run_dir / "reached"), "max_witnesses": MAX_WITNESSES}
    if mode == "differential":
        worker_spec = run_dir / "worker.json"
        worker_spec.write_text(json.dumps({"module": target["module"], "qualname": target["qualname"],
                                           "plan": info["plan"], "returns": info["returns"]}))
        (run_dir / "worker.py").write_text(WORKER, encoding="utf-8")
        # the worker imports the FIXED view; it is started by the fuzz child with the fixed path
        spec["worker_script"] = str(run_dir / "worker_launcher.py")
        spec["worker_spec"] = str(worker_spec)
        (run_dir / "worker_launcher.py").write_text(
            "import os, sys\nos.environ['PYTHONPATH'] = %r\nos.execv(%r, [%r, '-B', %r, sys.argv[1]])\n"
            % (f"{views['fixed']}:{ATHERIS_SITE}", PYTHON, PYTHON, str(run_dir / "worker.py")),
            encoding="utf-8")
    import resource
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    started = time.time()
    try:
        done = _child(FUZZ, spec, run_dir, views["buggy"], "fuzz", cpu=budget + 5,
                      wall=budget + WALL_BACKSTOP)
        exit_code = done.returncode
    except subprocess.TimeoutExpired:
        exit_code = "wall_backstop"
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    result = {"mode": mode, "seed": seed, "budget_cpu_seconds": budget,
              "cpu_seconds": round((after.ru_utime + after.ru_stime) -
                                   (before.ru_utime + before.ru_stime), 2),
              "wall_seconds": round(time.time() - started, 2), "exit": exit_code,
              "reached": (run_dir / "reached").exists()}
    if mode == "posthoc":
        corpus = sorted((run_dir / "corpus").iterdir())
        result.update(corpus=len(corpus), corpus_truncated=len(corpus) > CORPUS_CAP)
        corpus = corpus[:CORPUS_CAP]
        batch = {label: replay(views[label], target, info, corpus, run_dir, f"batch_{label}")
                 for label in ("buggy", "fixed")}
        if any("error" in b for b in batch.values()):
            return {**result, "kill": False, "replay_errors": [b.get("error") for b in batch.values()]}
        candidates = [p for p, x, y in zip(corpus, batch["buggy"]["results"],
                                           batch["fixed"]["results"])
                      if x != y and x[0] in ("ok", "raise") and y[0] in ("ok", "raise")]
        witnesses = candidates[:MAX_WITNESSES]
        result["opaque_results"] = sum(x[0] == "opaque" for x in batch["buggy"]["results"])
    else:
        witnesses = sorted((run_dir / "witnesses").iterdir())
    checks = []
    for w in witnesses:
        verdict = confirmed(views, target, info, w, run_dir)
        if mode == "ordinary" and verdict.get("buggy", [None])[0] != "raise":
            verdict["kill"] = False
        checks.append({"witness": w.name[:16], **verdict})
    replay_errors = [c for c in checks if "replay_error" in c]
    return {**result, "witnesses": len(witnesses), "confirmed": sum(c["kill"] for c in checks),
            "replay_errors": len(replay_errors), "kill": any(c["kill"] for c in checks),
            **({"label": "oracle-assisted upper bound"} if mode == "differential" else {})}


# --- canaries ---------------------------------------------------------------------------------

CANARY_HELPERS = "MAGIC = 424242\n"
CANARY_BUGGY = '''
from .helpers import MAGIC

COUNTER = [0]


def crash(x: int) -> int:
    if x == 91357:
        raise ValueError("buggy crash")
    return x


def same_crash(x: int) -> int:
    if x == 91357:
        raise ValueError("shared crash")
    return x


def later(x: int) -> int:
    if x == 111:
        raise KeyError("shared")
    if x == 777:
        raise IndexError("buggy only")
    return x


def wrong(x: int) -> int:
    if x == MAGIC:
        return x + 1
    return x


def mutable(xs: list[int]) -> int:
    xs.append(1)
    return len(xs)


def stateful(x: int) -> int:
    COUNTER[0] += 1
    return x


def opaque(x: int) -> object:
    return object()


def keyword(a: int, /, b: int = 2, *, c: int, d: str = "z") -> int:
    if c == 6262:
        raise ValueError("kw bug")
    return a + c


def variadic(*xs: int) -> int:
    return len(xs)


class Box:
    def total(self, n: int) -> int:
        return n


def untyped(x):
    return x
'''
CANARY_FIXED = CANARY_BUGGY.replace('raise ValueError("buggy crash")', "return 0") \
    .replace('raise IndexError("buggy only")', "return 0") \
    .replace("return x + 1", "return x").replace('raise ValueError("kw bug")', "return 0")
EXPECT = {("crash", "ordinary"): True, ("same_crash", "ordinary"): False,
          ("later", "ordinary"): True, ("wrong", "ordinary"): False,
          ("wrong", "posthoc"): True, ("wrong", "differential"): True,
          ("same_crash", "posthoc"): False, ("mutable", "differential"): False,
          ("stateful", "posthoc"): False, ("opaque", "posthoc"): False,
          ("keyword", "ordinary"): True}


def canaries(out_dir: Path) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="oneiros_atheris_canary_"))
    try:
        views = {}
        for label, body in (("buggy", CANARY_BUGGY), ("fixed", CANARY_FIXED)):
            pkg = work / "views" / label / "canarypkg"
            pkg.mkdir(parents=True)
            (pkg / "__init__.py").write_text("")
            (pkg / "helpers.py").write_text(CANARY_HELPERS)
            (pkg / "core.py").write_text(body)          # the SAME basename in both views
            views[label] = work / "views" / label
        mismatch = work / "views" / "py312" / "canarypkg"
        mismatch.mkdir(parents=True)
        (mismatch / "__init__.py").write_text("")
        (mismatch / "core.py").write_text("type Alias = int\n\ndef f(x: int) -> int:\n    return x\n")
        version = subprocess.run([PYTHON, "-c", "import sys; sys.path.insert(0, %r); import atheris, "
                                  "importlib.metadata as m; print(m.version('atheris'), "
                                  "sys.version.split()[0])" % ATHERIS_SITE],
                                 capture_output=True, text=True).stdout.split()
        names = ("crash", "same_crash", "later", "wrong", "mutable", "stateful", "opaque",
                 "keyword", "variadic", "Box.total", "untyped")
        elig = {n: probe(views["buggy"], "canarypkg.core", n, work) for n in names}
        elig["runtime_mismatch"] = probe(work / "views" / "py312", "canarypkg.core", "f", work)
        results = {}
        for (function, mode), expected in EXPECT.items():
            target = {"module": "canarypkg.core", "qualname": function}
            r = fuzz(mode, views, target, elig[function], 42, CANARY_BUDGET_SECONDS,
                     work / "runs" / function)
            results[f"{function}:{mode}"] = {**r, "expected_kill": expected}
            print(f"{function:11s} {mode:12s} kill={r['kill']} expected={expected} "
                  f"witnesses={r['witnesses']} confirmed={r['confirmed']} reached={r['reached']}",
                  flush=True)
        bad = replay(views["buggy"], {"module": "canarypkg.core", "qualname": "missing"},
                     elig["crash"], [], work, "bad_replay")
    finally:
        shutil.rmtree(work, ignore_errors=True)
    checks = {
        "atheris_2_3_0_py311": version[:1] == ["2.3.0"] and version[1].startswith("3.11"),
        "eligibility": elig["crash"]["eligible"] and elig["keyword"]["eligible"]
        and not elig["untyped"]["eligible"]
        and elig["Box.total"]["reason"] == "instance_method_receiver"
        and elig["variadic"]["reason"] == "variadic_signature"
        and elig["runtime_mismatch"]["reason"].startswith("runtime_mismatch"),
        "keyword_plan": [p["keyword"] for p in elig["keyword"]["plan"]] == [False, False, True, True],
        "replay_error_is_not_a_kill": "error" in bad,
        "cpu_budget_enforced": all(r["cpu_seconds"] <= CANARY_BUDGET_SECONDS + 120
                                   for r in results.values()),
        **{f"canary_{k}": v["kill"] == v["expected_kill"] and v["reached"]
           for k, v in results.items()}}
    receipt = {"schema_version": "oneiros_native_atheris_canaries_v2",
               "design_version": DESIGN_VERSION, "atheris": version,
               "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "canary_budget_cpu_seconds": CANARY_BUDGET_SECONDS,
               "full_budget_cpu_seconds": FULL_BUDGET_CPU_SECONDS, "corpus_cap": CORPUS_CAP,
               "eligibility": elig, "results": results, "checks": checks,
               "passed": all(checks.values()),
               "note": "ordinary Atheris cannot kill the wrong-answer canary (no semantic oracle)"}
    (out_dir / "atheris_canary_receipt_v2.json").write_text(json.dumps(receipt, indent=1,
                                                                       sort_keys=True) + "\n")
    print(json.dumps({"passed": receipt["passed"],
                      "failed": [k for k, v in checks.items() if not v]}, indent=1))
    return 0 if receipt["passed"] else 1


# --- durable real-panel run (NOT executed in this work block) -------------------------------

def run(prep_path: Path, manifest_path: Path, out: Path, budget: int, seeds) -> int:
    prep = {}
    for line in prep_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            prep[row["key"]] = row
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    contract = {"design_version": DESIGN_VERSION,
                "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "prep_sha256": hashlib.sha256(prep_path.read_bytes()).hexdigest(),
                "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                "budget_cpu_seconds": budget, "seeds": list(seeds), "modes": list(MODES),
                "corpus_cap": CORPUS_CAP, "python": PYTHON}
    chash = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    out.mkdir(parents=True, exist_ok=True)
    cfile = out / "atheris_contract.json"
    if cfile.exists() and json.loads(cfile.read_text()) != contract:
        raise SystemExit("REFUSED: Atheris contract differs; prior results preserved")
    cfile.write_text(json.dumps(contract, indent=1, sort_keys=True) + "\n")
    results = out / "atheris_results.jsonl"
    done = set()
    if results.exists():
        for line in results.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row.get("contract_sha256") != chash:
                raise SystemExit("REFUSED: stale Atheris rows; use a new output directory")
            done.add(row["key"])
    with results.open("a", encoding="utf-8") as handle:
        for key in manifest["kept_targets"]:
            row = prep[key]
            target = {"module": row["module"], "qualname": row["qualname"]}
            views = {k: Path(v) for k, v in row["views"].items()}
            work = Path(tempfile.mkdtemp(prefix="oneiros_atheris_"))
            try:
                info = probe(views["buggy"], row["module"], row["qualname"], work)
                for mode in MODES:
                    for seed in seeds:
                        rkey = f"{key}::{mode}::{seed}"
                        if rkey in done:
                            continue
                        r = (fuzz(mode, views, target, info, seed, budget, work)
                             if info.get("eligible") else {"kill": False})
                        handle.write(json.dumps({"key": rkey, "contract_sha256": chash,
                                                 "target_key": key, "mode": mode, "seed": seed,
                                                 "eligible": bool(info.get("eligible")),
                                                 "reason": info.get("reason"), **r},
                                                sort_keys=True, default=str) + "\n")
                        handle.flush()
                        os.fsync(handle.fileno())
            finally:
                shutil.rmtree(work, ignore_errors=True)
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    c = sub.add_parser("canaries")
    c.add_argument("out_dir")
    r = sub.add_parser("run")
    r.add_argument("--prep", required=True)
    r.add_argument("--manifest", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--budget", type=int, default=FULL_BUDGET_CPU_SECONDS)
    r.add_argument("--seeds", default="42,43,44")
    args = parser.parse_args(argv)
    if args.command == "canaries":
        return canaries(Path(args.out_dir))
    return run(Path(args.prep), Path(args.manifest), Path(args.out), args.budget,
               tuple(int(s) for s in args.seeds.split(",")))


if __name__ == "__main__":
    raise SystemExit(main())
