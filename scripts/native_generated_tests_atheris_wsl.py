"""Actual atheris 2.3.0 comparison for native generated tests (protocol v2 s7, amendment v2.1 D,
amendment v2.2 E, amendment v2.4 E).

v2.4: the kill rule is scripts/native_atheris_results.py:judge (ordinary = buggy raise AND
fixed ok, stable; two different exceptions are not a kill); every row keeps its per-witness
confirmation replays so the authoritative loader can recompute each kill; the contract binds
the exact manifest-declared preparation file.

Runs in WSL as root. Runtime parity: every child uses CPython 3.11 (/usr/bin/python3.11, the
base of the atheris venv) with atheris taken from its read-only site-packages.

Isolation (v2.2 E.2): every child runs in the shared generated-code sandbox
(scripts/native_sandbox_inner.sh): private mount, PID and network namespaces, nobody with
no_new_privs and no capabilities, prlimit limits, tmpfs over host homes/drives/caches. Each
target, mode, seed and revision gets a FRESH copy of its sanitised view, mounted read-only at
the canonical path /target; PYTHONPATH is /target plus the read-only atheris site; no host
checkout path is visible. The target package is imported BY NAME, so same-basename files never
share a sys.modules identity and package-relative imports work. The differential fixed-revision
worker is a separate sandbox connected only by pipes.

Budget (v2.2 E.1): 600 CPU-seconds per target/mode/seed over the WHOLE search process tree.
The fuzz sandbox and the worker sandbox are placed in sub-groups of one cgroup-v2 group; the
harness polls cpu.stat and kills the whole group (cgroup.kill) at the budget. Per-process
RLIMIT_CPU and a wall backstop (budget + 300 s) are secondary guards. Main, worker, replay and
aggregate CPU, wall time and the end reason (completed, cpu_budget_exhausted, wall_timeout,
crashed, infrastructure_failure) are recorded separately. Declared tolerance: 1.0 CPU-second
+ 2% of the budget.

Modes (separately labelled):
  ordinary      buggy-only search; every distinct exception signature is kept as a witness;
                EVERY witness is replayed on both revisions; a kill needs buggy failing and fixed
                returning a canonical value, confirmed in fresh single-input processes twice per
                revision
  posthoc       buggy-only coverage search; the corpus (sorted by file name, capped at 2,000,
                truncation recorded) is replayed on both revisions; differences in canonical
                value or exception type are re-confirmed singly before counting
  differential  online comparison with the fixed-revision worker: an ORACLE-ASSISTED UPPER
                BOUND; witnesses are re-confirmed like the others
Results are canonicalised (None, bool, int, float, str, bytes, list, tuple, dict, set);
anything else is ``opaque`` and never compared. Replay errors are reported separately and are
never kills. Witness files whose content does not match their name (a write cut off by a kill)
are dropped and counted.

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

DESIGN_VERSION = "oneiros_native_generated_tests_atheris_v5"
HERE = Path(__file__).resolve().parent
INNER = HERE / "native_sandbox_inner.sh"
REPO_ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import native_generation_io as gio  # noqa: E402  (stdlib only)
import native_atheris_results as verdicts  # noqa: E402  (stdlib only; the one kill rule)
def _cgroup2_mount() -> Path:
    """The cgroup-v2 hierarchy: the hybrid layout's /sys/fs/cgroup/unified (older WSL kernels)
    or, on a pure cgroup-v2 kernel (WSL 6.x), /sys/fs/cgroup itself. Same controller files and
    cgroup.kill / cpu.stat accounting either way."""
    for candidate in (Path("/sys/fs/cgroup/unified"), Path("/sys/fs/cgroup")):
        if (candidate / "cgroup.controllers").is_file():
            return candidate
    return Path("/sys/fs/cgroup/unified")


CGROUP_ROOT = _cgroup2_mount() / "oneiros_atheris"
FULL_BUDGET_CPU_SECONDS = 600
CANARY_BUDGET_SECONDS = 20
WALL_BACKSTOP = 300
CORPUS_CAP = 2000
MAX_WITNESSES = 50
CONFIRMATIONS = 2
SEEDS = (42, 43, 44)
MODES = ("ordinary", "posthoc", "differential")
POLL_SECONDS = 0.05
SANDBOX_LIMITS = {"as": 8 * 2 ** 30, "nproc": 256, "nofile": 1024, "fsize": 64 * 2 ** 20}
REPLAY_CPU, REPLAY_WALL = 600, 900
END_REASONS = ("completed", "cpu_budget_exhausted", "wall_timeout", "crashed",
               "infrastructure_failure")


def tolerance(budget: float) -> float:
    return 1.0 + 0.02 * budget


# --- v5 runtime: the target's own interpreter and locked environment + a clean overlay -------

ATHERIS_V5_ROOT = Path("/opt/oneiros_atheris_v5")
RUNTIME: dict = {}


def tree_manifest_sha256(root: Path) -> str:
    """Hash of every file (relative path -> sha256) under ``root``; bytecode caches ignored."""
    files = {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted(Path(root).rglob("*"))
             if p.is_file() and "__pycache__" not in p.parts}
    return hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()


def environment_sha256(env_dir: str | None) -> str | None:
    """Identity of a locked environment's site-packages (v2.5 C.2: hashed before and after)."""
    if not env_dir:
        return None
    sites = sorted(Path(env_dir).glob("lib/python*/site-packages"))
    return hashlib.sha256("".join(tree_manifest_sha256(s) for s in sites).encode()).hexdigest()


def runtime_for(python: str, env_dir: str | None) -> dict:
    """The v5 runtime of one target: its prepared interpreter and environment, plus the clean
    read-only Atheris overlay for that CPython minor version. No overlay -> the infrastructure
    category ``atheris_abi_unavailable``."""
    version = subprocess.run([python, "-c", "import sys; print('%d.%d' % sys.version_info[:2])"],
                             capture_output=True, text=True).stdout.strip()
    overlay = ATHERIS_V5_ROOT / f"cp{version.replace('.', '')}"
    if not (overlay / "atheris").is_dir():
        return {"available": False, "reason": "atheris_abi_unavailable", "version": version}
    base = Path(os.path.realpath(python)).parent.parent        # the interpreter's installation
    ro = [str(overlay)] + ([env_dir] if env_dir else []) + \
        ([str(base)] if not str(base).startswith("/usr") else [])
    return {"available": True, "python": python, "env_dir": env_dir, "version": version,
            "overlay": str(overlay), "overlay_manifest_sha256": tree_manifest_sha256(overlay),
            "ro": ro}


def set_runtime(runtime: dict) -> None:
    if not runtime.get("available"):
        raise RuntimeError(f"no usable Atheris runtime: {runtime.get('reason')}")
    RUNTIME.clear()
    RUNTIME.update(runtime)


COMMON = r'''
import base64, enum, importlib, inspect, json, math, sys, types, typing

# Atheris adapter v5 (protocol v2.5 C.3-C.4). Supported: the primitives and lists below,
# Optional/Union of supported types, Any (a FROZEN primitive menu), Enum members and Literal
# values that are JSON primitives; instance-method receivers are constructed ONLY through the
# class's own public __init__ with supported parameters. Anything else is explicitly
# "adapter_unsupported:<reason>" - never an import/infrastructure failure, never a non-kill.
PRIMS = ("int", "float", "str", "bytes", "bool")
LISTS = ("list[int]", "list[str]", "list[float]")
ANY_MENU = ("int", "float", "str", "bytes", "bool", "none")

class Opaque(Exception):
    pass

class Unsupported(Exception):
    pass

def resolve(module, qualname):
    obj = importlib.import_module(module)
    for part in qualname.split("."):
        obj = getattr(obj, part)
    return obj

def type_name(annotation):
    """Return-type name used only for the contract-violation check."""
    if isinstance(annotation, str):
        return annotation.replace("typing.", "")
    return getattr(annotation, "__name__", None)

def type_spec(a, depth=0):
    if depth > 3:
        raise Unsupported("nested_too_deep")
    if a is inspect.Parameter.empty:
        raise Unsupported("None")
    if a is typing.Any:
        return {"kind": "any"}
    if a is None or a is type(None):
        return {"kind": "none"}
    if isinstance(a, str):
        name = a.replace("List[", "list[").replace("typing.", "")
        if name in PRIMS or name in LISTS:
            return {"kind": "prim", "type": name}
        raise Unsupported(name)
    origin, args = typing.get_origin(a), typing.get_args(a)
    if origin is list and len(args) == 1 and getattr(args[0], "__name__", None) in ("int", "str", "float"):
        return {"kind": "prim", "type": "list[%s]" % args[0].__name__}
    if origin is typing.Union or (getattr(types, "UnionType", None) is not None and origin is types.UnionType):
        return {"kind": "union", "options": [type_spec(x, depth + 1) for x in args]}
    if origin is typing.Literal:
        if args and all(v is None or type(v) in (bool, int, float, str) for v in args):
            return {"kind": "literal", "values": list(args)}
        raise Unsupported("Literal")
    if inspect.isclass(a) and issubclass(a, enum.Enum):
        members = list(a.__members__)
        if not members:
            raise Unsupported(a.__name__)
        return {"kind": "enum", "module": a.__module__, "qualname": a.__qualname__, "members": members}
    name = getattr(a, "__name__", None)
    if name in PRIMS:
        return {"kind": "prim", "type": name}
    raise Unsupported(str(name or a)[:40])

class UnsupportedParam(Exception):
    pass

def param_plan(func, skip_first=False):
    sig = inspect.signature(func)
    try:
        hints = typing.get_type_hints(func)
    except Exception:
        hints = {}
    params = list(sig.parameters.values())[1 if skip_first else 0:]
    plan = []
    for p in params:
        if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            raise UnsupportedParam("variadic_signature")
        try:
            spec = type_spec(hints.get(p.name, p.annotation))
        except Unsupported as exc:
            if p.default is inspect.Parameter.empty:
                raise UnsupportedParam("unsupported_parameter:%s:%s" % (p.name, exc))
            continue
        plan.append({"name": p.name, "spec": spec, "keyword": p.kind == p.KEYWORD_ONLY})
    return plan

def eligibility(func, qualname, module):
    def no(reason):
        return {"eligible": False, "reason": "adapter_unsupported:" + reason}
    receiver, method = None, None
    try:
        owner = resolve(module, qualname.rsplit(".", 1)[0]) if "." in qualname else None
        raw = inspect.getattr_static(owner, qualname.rsplit(".", 1)[1]) \
            if inspect.isclass(owner) else None
        if inspect.isfunction(raw):                 # a plain instance method: needs a receiver
            if owner.__init__ is object.__init__:
                cplan = []
            else:
                try:
                    cplan = param_plan(owner.__init__, skip_first=True)
                except UnsupportedParam as exc:
                    return no("receiver_constructor:" + str(exc))
            receiver = {"module": module, "qualname": qualname.rsplit(".", 1)[0], "plan": cplan}
            method = qualname.rsplit(".", 1)[1]
            params = param_plan(raw, skip_first=True)
        else:
            params = param_plan(func)
    except UnsupportedParam as exc:
        return no(str(exc))
    except (TypeError, ValueError):
        return no("no_signature")
    try:
        returns = typing.get_type_hints(func).get("return", inspect.signature(func).return_annotation)
    except Exception:
        returns = inspect.Parameter.empty
    return {"eligible": True, "plan": {"params": params, "receiver": receiver, "method": method},
            "returns": type_name(returns) if returns is not inspect.Parameter.empty else None}

def value_for(fdp, s):
    k = s["kind"]
    if k == "none":
        return None
    if k == "any":
        pick = ANY_MENU[fdp.ConsumeIntInRange(0, len(ANY_MENU) - 1)]
        return None if pick == "none" else value_for(fdp, {"kind": "prim", "type": pick})
    if k == "union":
        return value_for(fdp, s["options"][fdp.ConsumeIntInRange(0, len(s["options"]) - 1)])
    if k == "literal":
        return s["values"][fdp.ConsumeIntInRange(0, len(s["values"]) - 1)]
    if k == "enum":
        return resolve(s["module"], s["qualname"])[s["members"][fdp.ConsumeIntInRange(0, len(s["members"]) - 1)]]
    t = s["type"]
    if t == "int": return fdp.ConsumeIntInRange(-10**6, 10**6)
    if t == "float": return fdp.ConsumeRegularFloat()
    if t == "str": return fdp.ConsumeUnicodeNoSurrogates(32)
    if t == "bytes": return fdp.ConsumeBytes(32)
    if t == "bool": return fdp.ConsumeBool()
    if t == "list[int]": return [fdp.ConsumeIntInRange(-1000, 1000) for _ in range(fdp.ConsumeIntInRange(0, 8))]
    if t == "list[str]": return [fdp.ConsumeUnicodeNoSurrogates(8) for _ in range(fdp.ConsumeIntInRange(0, 8))]
    return [fdp.ConsumeRegularFloat() for _ in range(fdp.ConsumeIntInRange(0, 8))]

def build(fdp, params):
    args, kwargs = [], {}
    for p in params:
        v = value_for(fdp, p["spec"])
        if p["keyword"]:
            kwargs[p["name"]] = v
        else:
            args.append(v)
    return args, kwargs

def invoke(func, plan, fdp):
    """Call the target once; a receiver is built (constructor arguments consumed first)."""
    target = func
    if plan.get("receiver"):
        r = plan["receiver"]
        rargs, rkwargs = build(fdp, r["plan"])
        target = getattr(resolve(r["module"], r["qualname"])(*rargs, **rkwargs), plan["method"])
    args, kwargs = build(fdp, plan["params"])
    return target(*args, **kwargs)

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
    try:
        value = invoke(func, plan, atheris.FuzzedDataProvider(data))
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
print(json.dumps({**eligibility(func, spec["qualname"], spec["module"]),
                  "python": sys.version.split()[0]}))
'''

FUZZ = COMMON + r'''
import atheris, hashlib, os
spec = json.load(open(sys.argv[1]))
OUT = "/sandbox_out"
open(os.path.join(OUT, "started"), "w").close()
with atheris.instrument_imports():
    func = resolve(spec["module"], spec["qualname"])
PLAN, RETURNS, MODE = spec["plan"], spec["returns"], spec["mode"]
SEEN = set()
os.makedirs(os.path.join(OUT, "witnesses"), exist_ok=True)
os.makedirs(os.path.join(OUT, "corpus"), exist_ok=True)
request = response = None
if MODE == "differential":
    request = os.fdopen(spec["request_fd"], "w")
    response = os.fdopen(spec["response_fd"], "r")
    while True:
        line = response.readline()
        if not line:
            open(os.path.join(OUT, "worker_lost"), "w").close()
            os._exit(3)
        if line.strip() == "READY":
            break

def witness(data, signature):
    if signature in SEEN or len(SEEN) >= spec["max_witnesses"]:
        return
    SEEN.add(signature)
    name = hashlib.sha256(data).hexdigest()
    with open(os.path.join(OUT, "witnesses", name), "wb") as handle:
        handle.write(data)

REACHED = os.path.join(OUT, "reached")

def one(data):
    if not os.path.exists(REACHED):
        open(REACHED, "w").close()
    got = outcome(func, PLAN, data, atheris, RETURNS)
    if MODE == "ordinary" and got[0] == "raise":
        witness(data, got[1])
    elif MODE == "differential" and got[0] in ("ok", "raise"):
        request.write(base64.b64encode(data).decode() + "\n")
        request.flush()
        line = response.readline()
        if not line:
            open(os.path.join(OUT, "worker_lost"), "w").close()
            os._exit(3)
        other = json.loads(line)
        if other[0] in ("ok", "raise") and other != got:
            witness(data, json.dumps([got, other])[:200])

argv = [sys.argv[0], "-seed=%d" % spec["seed"], "-max_total_time=%d" % spec["budget"],
        "-print_final_stats=1", "-use_value_profile=1"]
if MODE == "posthoc":
    argv.append(os.path.join(OUT, "corpus"))
atheris.Setup(argv, one)
atheris.Fuzz()
'''

WORKER = COMMON + r'''
import atheris
spec = json.load(open(sys.argv[1]))
func = resolve(spec["module"], spec["qualname"])
print("READY", flush=True)
for line in sys.stdin:
    data = base64.b64decode(line.strip())
    print(json.dumps(outcome(func, spec["plan"], data, atheris, spec["returns"])), flush=True)
'''

REPLAY = COMMON + r'''
import atheris, os
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

BUSY = r'''
import json, os, sys
open("/sandbox_out/started", "w").close()
for _ in range(json.load(open(sys.argv[1]))["children"]):
    if os.fork() == 0:
        while True:
            pass
while True:
    pass
'''


# --- sandbox and cgroup plumbing ----------------------------------------------------------------

def tree_sha256(root: Path) -> str:
    """Content hash of a view (relative paths and bytes), to prove nothing mutated it."""
    digest = hashlib.sha256()
    for path in sorted(p for p in Path(root).rglob("*") if p.is_file()):
        digest.update(path.relative_to(root).as_posix().encode() + b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def fresh_view(source: Path, dest: Path) -> Path:
    """A fresh private copy of a sanitised revision view (one per target/mode/seed/revision)."""
    if dest.exists():
        raise RuntimeError(f"view copy {dest} already exists")
    shutil.copytree(source, dest, symlinks=False)
    return dest


class Group:
    """One cgroup-v2 group for one search: sub-groups main and worker, plus replay."""

    def __init__(self, label: str):
        CGROUP_ROOT.mkdir(exist_ok=True)
        self.path = CGROUP_ROOT / f"{label}_{os.getpid()}_{time.time_ns()}"
        self.path.mkdir()
        for sub in ("main", "worker"):
            (self.path / sub).mkdir()

    def usage(self, sub: str = "") -> float:
        text = ((self.path / sub) if sub else self.path).joinpath("cpu.stat").read_text()
        for line in text.splitlines():
            key, value = line.split()
            if key == "usage_usec":
                return int(value) / 1e6
        return 0.0

    def procs(self) -> list:
        out = []
        for p in (self.path, self.path / "main", self.path / "worker"):
            if p.exists():
                out += [x for x in (p / "cgroup.procs").read_text().split() if x]
        return out

    def kill(self) -> None:
        (self.path / "cgroup.kill").write_text("1")

    def remove(self) -> bool:
        deadline = time.time() + 10
        while self.procs() and time.time() < deadline:
            self.kill()
            time.sleep(0.05)
        try:
            for sub in ("main", "worker"):
                (self.path / sub).rmdir()
            self.path.rmdir()
        except OSError:
            return False
        return True


def _launch(script: str, spec: dict, view: Path, out_dir: Path, *, cpu: int, wall: int,
            cgroup: Path | None = None, files: dict | None = None, **popen) -> tuple:
    """Start one sandboxed child; returns (Popen, work_dir). /target = ``view`` (read-only)."""
    work = Path(tempfile.mkdtemp(prefix="oneiros_ath_work_"))
    (work / "child.py").write_text(script, encoding="utf-8")
    (work / "spec.json").write_text(json.dumps(spec), encoding="utf-8")
    for name, data in (files or {}).items():
        (work / "inputs").mkdir(exist_ok=True)
        (work / "inputs" / name).write_bytes(data)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not RUNTIME:
        raise RuntimeError("v5: set_runtime() must select the target's runtime first")
    # v5: the target's own interpreter and locked environment (read-only), Atheris from the
    # clean overlay appended after /target; nothing from the system site-packages
    env = {"PATH": "/usr/bin:/bin", "SB_RO": "\n".join(RUNTIME["ro"]), "SB_TARGET": str(view),
           "SB_OUT": str(out_dir), "SB_WORK_SRC": str(work),
           "SB_PYTHONPATH": f"/target:{RUNTIME['overlay']}", "SB_CPU": str(cpu),
           "SB_AS": str(SANDBOX_LIMITS["as"]), "SB_NPROC": str(SANDBOX_LIMITS["nproc"]),
           "SB_NOFILE": str(SANDBOX_LIMITS["nofile"]), "SB_FSIZE": str(SANDBOX_LIMITS["fsize"]),
           "SB_WALL": str(wall)}
    argv = ["unshare", "--mount", "--pid", "--net", "--fork", "--mount-proc", "--",
            "/bin/bash", str(INNER), RUNTIME["python"], "-B", "-s", "/tmp/work/child.py",
            "/tmp/work/spec.json"]
    if cgroup is not None:
        argv = ["/bin/sh", "-c", 'echo $$ > "$0/cgroup.procs" && exec "$@"', str(cgroup), *argv]
    return subprocess.Popen(argv, env=env, **popen), work


def run_once(script: str, spec: dict, view: Path, *, cpu: int = REPLAY_CPU,
             wall: int = REPLAY_WALL, files: dict | None = None,
             cgroup: Path | None = None) -> dict:
    """A sandboxed child run to completion; the last stdout line is its JSON payload."""
    out = Path(tempfile.mkdtemp(prefix="oneiros_ath_out_"))
    proc, work = _launch(script, spec, view, out, cpu=cpu, wall=wall, files=files, cgroup=cgroup,
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        stdout, stderr = proc.communicate(timeout=wall + 30)
    except subprocess.TimeoutExpired:
        proc.kill()
        stdout, stderr = proc.communicate()
    finally:
        shutil.rmtree(work, ignore_errors=True)
        shutil.rmtree(out, ignore_errors=True)
    try:
        return json.loads(stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return {"error": "process_failure", "tail": (stderr or "")[-300:]}


def supervise(procs: list, group: Group, budget: float, wall_limit: float) -> dict:
    """Poll the group's aggregate CPU; kill the whole group at the budget or the wall limit."""
    started = time.monotonic()
    reason = None
    while any(p.poll() is None for p in procs):
        if reason is None and group.usage() >= budget:
            reason = "cpu_budget_exhausted"
            group.kill()
        elif reason is None and time.monotonic() - started > wall_limit:
            reason = "wall_timeout"
            group.kill()
        time.sleep(POLL_SECONDS)
    for p in procs:
        p.wait()
    return {"supervisor_reason": reason, "wall_seconds": round(time.monotonic() - started, 2),
            "aggregate_cpu_seconds": round(group.usage(), 3),
            "main_cpu_seconds": round(group.usage("main"), 3),
            "worker_cpu_seconds": round(group.usage("worker"), 3)}


# --- probe, replay, confirmation ----------------------------------------------------------------

def probe(view: Path, module: str, qualname: str) -> dict:
    result = run_once(PROBE, {"module": module, "qualname": qualname}, view, cpu=60, wall=120)
    if "error" in result:
        return {"eligible": False, "reason": "probe_failure", "tail": result.get("tail")}
    return result


def replay(view: Path, target: dict, info: dict, inputs: list, cgroup: Path | None = None
           ) -> dict:
    files = {f"{i:05d}": Path(p).read_bytes() for i, p in enumerate(inputs)}
    spec = {"module": target["module"], "qualname": target["qualname"], "plan": info["plan"],
            "returns": info["returns"], "inputs": [f"/tmp/work/inputs/{n}" for n in files]}
    result = run_once(REPLAY, spec, view, files=files, cgroup=cgroup)
    if "error" in result:
        return {"error": "replay_process_failure" if result["error"] == "process_failure"
                else result["error"]}
    return result


def confirmed(mode: str, views: dict, target: dict, info: dict, witness: Path,
              cgroup: Path) -> dict:
    """Fresh single-input processes, CONFIRMATIONS times per revision, judged by the one
    kill rule (native_atheris_results.judge)."""
    seen = {"buggy": [], "fixed": []}
    for label in ("buggy", "fixed"):
        for _ in range(CONFIRMATIONS):
            r = replay(views[label], target, info, [witness], cgroup)
            if "error" in r:
                return verdicts.judge(mode, [], [], r["error"])
            seen[label].append(r["results"][0])
    return verdicts.judge(mode, seen["buggy"], seen["fixed"])


def _end_reason(sup: dict, code, out: Path, budget: float) -> str:
    if sup["supervisor_reason"]:
        return sup["supervisor_reason"]
    if not (out / "started").exists() or (out / "worker_lost").exists():
        return "infrastructure_failure"
    if code == 0:
        return "completed"
    if code in (124, 137) and sup["wall_seconds"] >= budget + WALL_BACKSTOP:
        return "wall_timeout"
    if code in (152, -24):
        return "cpu_budget_exhausted"        # the per-process RLIMIT_CPU secondary guard
    return "crashed"


def _intact(directory: Path, algorithm: str = "sha256") -> tuple:
    """Files whose name is the hash of their content (witnesses: SHA-256, written here;
    libFuzzer corpus units: SHA-1). Anything else was cut off by a kill and is dropped."""
    kept, dropped = [], 0
    for path in sorted(directory.iterdir()) if directory.exists() else []:
        if hashlib.new(algorithm, path.read_bytes()).hexdigest() == path.name:
            kept.append(path)
        else:
            dropped += 1
    return kept, dropped


def fuzz(mode: str, sources: dict, target: dict, info: dict, seed: int, budget: int,
         work: Path) -> dict:
    """One search with fresh read-only views, aggregate CPU accounting and full cleanup."""
    run_dir = work / f"{mode}_{seed}"
    run_dir.mkdir(parents=True)
    views = {label: fresh_view(sources[label], run_dir / "views" / label)
             for label in ("buggy", "fixed")}
    source_hashes = {label: tree_sha256(sources[label]) for label in views}
    env_before = environment_sha256(RUNTIME.get("env_dir"))      # v2.5 C.2
    out = run_dir / "out"
    spec = {"module": target["module"], "qualname": target["qualname"], "plan": info["plan"],
            "returns": info["returns"], "mode": mode, "seed": seed, "budget": budget,
            "max_witnesses": MAX_WITNESSES}
    group = Group(f"{mode}_{seed}")
    replay_group = None
    procs, works, pipes = [], [], []
    logs = [open(run_dir / "fuzz.log", "w"), open(run_dir / "worker.log", "w")]
    try:
        if mode == "differential":
            request_r, request_w = os.pipe()
            response_r, response_w = os.pipe()
            pipes = [request_r, request_w, response_r, response_w]
            worker, w_work = _launch(WORKER, spec, views["fixed"], run_dir / "worker_out",
                                     cpu=budget + 5, wall=budget + WALL_BACKSTOP,
                                     cgroup=group.path / "worker", stdin=request_r,
                                     stdout=response_w, stderr=logs[1])
            procs.append(worker)
            works.append(w_work)
            spec = {**spec, "request_fd": request_w, "response_fd": response_r}
            main, m_work = _launch(FUZZ, spec, views["buggy"], out, cpu=budget + 5,
                                   wall=budget + WALL_BACKSTOP, cgroup=group.path / "main",
                                   pass_fds=(request_w, response_r), stdout=logs[0],
                                   stderr=subprocess.STDOUT)
            for fd in pipes:
                os.close(fd)
            pipes = []
        else:
            main, m_work = _launch(FUZZ, spec, views["buggy"], out, cpu=budget + 5,
                                   wall=budget + WALL_BACKSTOP, cgroup=group.path / "main",
                                   stdout=logs[0], stderr=subprocess.STDOUT)
        procs.insert(0, main)
        works.append(m_work)
        sup = supervise(procs, group, budget, budget + WALL_BACKSTOP)
        reason = _end_reason(sup, main.returncode, out, budget)
    finally:
        for fd in pipes:
            os.close(fd)
        for p in procs:
            if p.poll() is None:
                p.kill()
                p.wait()
        for handle in logs:
            handle.close()
        for w in works:
            shutil.rmtree(w, ignore_errors=True)
        cleanup_ok = group.remove()
    result = {"mode": mode, "seed": seed, "budget_cpu_seconds": budget,
              "tolerance_cpu_seconds": tolerance(budget), **sup, "end_reason": reason,
              "exit": main.returncode, "reached": (out / "reached").exists(),
              "within_budget": sup["aggregate_cpu_seconds"] <= budget + tolerance(budget),
              "cleanup_ok": cleanup_ok,
              "views_unchanged": all(tree_sha256(views[k]) == h for k, h in source_hashes.items()),
              "environment_sha256": env_before,
              "environment_unchanged": environment_sha256(RUNTIME.get("env_dir")) == env_before}
    replay_group = Group(f"replay_{mode}_{seed}")
    try:
        verdict = _verify(mode, views, target, info, out, result, reason, replay_group)
    finally:
        replay_cpu = round(replay_group.usage(), 3)
        replay_cleanup = replay_group.remove()
    return {**verdict, "replay_cpu_seconds": replay_cpu, "replay_cleanup_ok": replay_cleanup}


def _verify(mode: str, views: dict, target: dict, info: dict, out: Path, result: dict,
            reason: str, replay_group: Group) -> dict:
    """Replay and confirmation after the search; never part of the search budget."""
    if reason == "infrastructure_failure":
        return {**result, "kill": False, "witnesses": 0, "confirmed": 0, "replay_errors": 0,
                "confirmations": []}
    if mode == "posthoc":
        corpus, dropped = _intact(out / "corpus", "sha1")
        result.update(corpus=len(corpus), corpus_truncated=len(corpus) > CORPUS_CAP,
                      dropped_partial_inputs=dropped)
        corpus = corpus[:CORPUS_CAP]
        batch = {label: replay(views[label], target, info, corpus, replay_group.path)
                 for label in ("buggy", "fixed")}
        if any("error" in b for b in batch.values()):
            return {**result, "kill": False, "witnesses": 0, "confirmed": 0,
                    "replay_errors": 1, "confirmations": [],
                    "replay_error_detail": [b.get("error") for b in batch.values()]}
        candidates = [p for p, x, y in zip(corpus, batch["buggy"]["results"],
                                           batch["fixed"]["results"])
                      if x != y and x[0] in ("ok", "raise") and y[0] in ("ok", "raise")]
        witnesses = candidates[:MAX_WITNESSES]
        result["opaque_results"] = sum(x[0] == "opaque" for x in batch["buggy"]["results"])
    else:
        witnesses, dropped = _intact(out / "witnesses")
        result["dropped_partial_witnesses"] = dropped
    checks = []
    for w in witnesses:
        verdict = confirmed(mode, views, target, info, w, replay_group.path)
        checks.append({"witness": w.name[:16], **verdict})
    return {**result, "witnesses": len(witnesses), "confirmations": checks,
            "confirmed": sum(c["kill"] for c in checks),
            "replay_errors": sum("replay_error" in c for c in checks),
            "kill": any(c["kill"] for c in checks),
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


def swap_exception(x: int) -> int:
    if x == 91357:
        raise ValueError("buggy exception")
    return x


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


def write_probe(x: int) -> int:
    import os
    try:
        with open(os.path.join(os.path.dirname(__file__), "written.txt"), "w") as handle:
            handle.write("x")
        return 1
    except OSError:
        return 0


def where(x: int) -> str:
    return __file__


def cross_run(x: int) -> int:
    import os
    path = "/tmp/oneiros_cross_run_state"
    count = int(open(path).read()) if os.path.exists(path) else 0
    with open(path, "w") as handle:
        handle.write(str(count + 1))
    return count + 1


def network(x: int) -> int:
    import socket
    try:
        socket.create_connection(("1.1.1.1", 53), timeout=2).close()
        return 1
    except OSError:
        return 0


def identity(x: int) -> int:
    import os
    return os.getuid()


# --- v5 adapter canaries (protocol v2.5 C.3-C.5) ---
import enum
import typing


class Color(enum.Enum):
    RED = 1
    BLUE = 2


def shade(c: Color) -> int:
    return c.value


def union_val(x: typing.Optional[int]) -> int:
    if x == 4242:
        raise ValueError("union bug")
    return 0 if x is None else x


def any_val(x: typing.Any) -> str:
    return type(x).__name__


def lit(mode: typing.Literal["a", "b"]) -> str:
    return mode


class Account:
    def __init__(self, balance: int):
        self.balance = balance

    def withdraw(self, amount: int) -> int:
        if amount == 31337:
            raise ValueError("receiver bug")
        return self.balance - amount


class Widget:
    pass


def custom(o: Widget) -> int:
    return 1
'''
# a module needing packages that exist ONLY in the target's locked environment (cattrs and the
# newer attrs whose NothingType the system python3.11 dist-packages attrs 23.2.0 lacks)
CANARY_ENVMOD = "from attrs import NothingType\nimport cattrs\n\n\ndef f(x: int) -> int:\n    return x\n"
CANARY_ENV_PACKAGES = ("attrs==25.3.0", "cattrs==25.1.1")
CANARY_FIXED = CANARY_BUGGY.replace('raise ValueError("buggy crash")', "return 0") \
    .replace('raise IndexError("buggy only")', "return 0") \
    .replace("return x + 1", "return x").replace('raise ValueError("kw bug")', "return 0") \
    .replace('raise ValueError("buggy exception")', 'raise TypeError("fixed exception")') \
    .replace('raise ValueError("union bug")', "return 0") \
    .replace('raise ValueError("receiver bug")', "return 0")
EXPECT = {("crash", "ordinary"): True, ("same_crash", "ordinary"): False,
          ("later", "ordinary"): True, ("wrong", "ordinary"): False,
          ("wrong", "posthoc"): True, ("wrong", "differential"): True,
          ("same_crash", "posthoc"): False, ("mutable", "differential"): False,
          ("stateful", "posthoc"): False, ("opaque", "posthoc"): False,
          ("keyword", "ordinary"): True,
          ("swap_exception", "ordinary"): False,      # v2.4 E.1: different exceptions
          ("union_val", "ordinary"): True,            # v2.5 C.4: Optional/Union adapter
          ("Account.withdraw", "ordinary"): True}     # v2.5 C.3: constructed receiver


def canary_runtime(work: Path, python: str, packages=()) -> dict:
    """A locked canary environment built like a prepared target's (uv venv + pinned packages),
    and its v5 runtime."""
    env = work / f"env_{Path(python).name}"
    subprocess.run(["uv", "venv", "-q", "--python", python, str(env)], check=True)
    if packages:
        subprocess.run(["uv", "pip", "install", "-q", "--python", str(env / "bin" / "python"),
                        *packages], check=True)
    return runtime_for(str(env / "bin" / "python"), str(env))


def aggregate_budget_canary(work: Path, view: Path, budget: float = 4.0) -> dict:
    """Two busy child processes (plus their parent) must not exceed the aggregate allowance,
    although each process alone stays far below its own RLIMIT_CPU."""
    group = Group("busy")
    out = work / "busy_out"
    proc, w = _launch(BUSY, {"children": 2}, view, out, cpu=int(budget) + 60, wall=120,
                      cgroup=group.path / "main", stdout=subprocess.DEVNULL,
                      stderr=subprocess.DEVNULL)
    try:
        sup = supervise([proc], group, budget, 60)
    finally:
        shutil.rmtree(w, ignore_errors=True)
        leftover = group.procs()
        removed = group.remove()
    return {**sup, "budget": budget, "tolerance": tolerance(budget),
            "ok": sup["supervisor_reason"] == "cpu_budget_exhausted"
            and sup["aggregate_cpu_seconds"] <= budget + tolerance(budget)
            and sup["aggregate_cpu_seconds"] >= budget and not leftover and removed,
            "leftover_processes": leftover, "cgroup_removed": removed}


def isolation_canaries(views: dict, work: Path) -> dict:
    """Write attempts, observed path, cross-run state, network and privileges."""
    target = {"module": "canarypkg.core"}
    params = [{"name": "x", "spec": {"kind": "prim", "type": "int"}, "keyword": False}]
    int_plan = {"plan": {"params": params, "receiver": None, "method": None}, "returns": "int"}
    str_plan = {"plan": {"params": params, "receiver": None, "method": None}, "returns": "str"}
    data = work / "one_input"
    data.write_bytes(b"\x00" * 8)
    out = {}

    def call(label: str, name: str, plan: dict) -> list:
        copy = fresh_view(views[label], work / "iso" / f"{name}_{label}_{time.time_ns()}")
        r = replay(copy, {**target, "qualname": name}, plan, [data])
        return r.get("results", [r])[0]
    write = [call(label, "write_probe", int_plan) for label in ("buggy", "fixed")]
    out["write_to_target_refused"] = {"results": write,
                                      "ok": all(r == ["ok", ["int", 0]] for r in write)}
    paths = [call(label, "where", str_plan) for label in ("buggy", "fixed")]
    out["canonical_path_identical"] = {
        "results": paths, "ok": paths[0] == paths[1] == ["ok", ["str", "/target/canarypkg/core.py"]]}
    runs = [call("buggy", "cross_run", int_plan) for _ in range(2)]
    out["no_cross_run_state"] = {"results": runs, "ok": runs == [["ok", ["int", 1]]] * 2}
    net = call("buggy", "network", int_plan)
    out["network_disabled"] = {"results": net, "ok": net == ["ok", ["int", 0]]}
    uid = call("buggy", "identity", int_plan)
    out["unprivileged_nobody"] = {"results": uid, "ok": uid == ["ok", ["int", 65534]]}
    return out


def live_view_canaries(sources: dict, work: Path) -> dict:
    """v2.4 C: the authoritative view-manifest hash refuses drift, symlinks and missing views
    (checked here inside WSL, where symlinks are available)."""
    copies = {label: fresh_view(sources[label], work / "live" / label) for label in sources}
    row = {"views": {k: str(v) for k, v in copies.items()},
           "view_manifest_sha256": {k: gio.view_manifest_sha256(v) for k, v in copies.items()}}
    out = {}
    try:
        gio.verify_live_views({"t": row}, ["t"])
        out["unchanged_views_verify"] = True
    except SystemExit:
        out["unchanged_views_verify"] = False

    def refused(fn) -> bool:
        try:
            fn()
        except SystemExit:
            return True
        return False
    for label in ("buggy", "fixed"):
        drifted = fresh_view(sources[label], work / "live" / f"drift_{label}")
        (drifted / "canarypkg" / "core.py").write_text("# drifted after preparation\n")
        out[f"{label}_drift_refused"] = refused(lambda d=drifted, l=label: gio.verify_live_views(
            {"t": {**row, "views": {**row["views"], l: str(d)}}}, ["t"]))
    linked = fresh_view(sources["buggy"], work / "live" / "symlinked")
    os.symlink(linked / "canarypkg" / "core.py", linked / "canarypkg" / "alias.py")
    out["symlink_refused"] = refused(lambda: gio.view_manifest_sha256(linked))
    out["missing_view_refused"] = refused(lambda: gio.view_manifest_sha256(work / "live" / "gone"))
    return out


def canaries(out_dir: Path) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="oneiros_atheris_canary_"))
    try:
        sources = {}
        for label, body in (("buggy", CANARY_BUGGY), ("fixed", CANARY_FIXED)):
            pkg = work / "views" / label / "canarypkg"
            pkg.mkdir(parents=True)
            (pkg / "__init__.py").write_text("")
            (pkg / "helpers.py").write_text(CANARY_HELPERS)
            (pkg / "core.py").write_text(body)          # the SAME basename in both views
            (pkg / "envmod.py").write_text(CANARY_ENVMOD)
            sources[label] = work / "views" / label
        mismatch = work / "views" / "py312" / "canarypkg"
        mismatch.mkdir(parents=True)
        (mismatch / "__init__.py").write_text("")
        (mismatch / "core.py").write_text("type Alias = int\n\ndef f(x: int) -> int:\n    return x\n")
        # v5: every child runs in a locked canary environment built like a prepared target's
        runtime = canary_runtime(work, "/usr/bin/python3.11", CANARY_ENV_PACKAGES)
        set_runtime(runtime)
        env_before = environment_sha256(runtime["env_dir"])
        version = subprocess.run([runtime["python"], "-s", "-c",
                                  "import sys; sys.path.append(%r); import atheris, "
                                  "importlib.metadata as m; print(m.version('atheris'), "
                                  "sys.version.split()[0]); print(atheris.__file__)"
                                  % runtime["overlay"]],
                                 capture_output=True, text=True).stdout.split()
        abi = canary_runtime(work, "/usr/bin/python3.12")      # no Atheris 2.3.0 wheel
        names = ("crash", "same_crash", "later", "wrong", "mutable", "stateful", "opaque",
                 "keyword", "variadic", "Box.total", "untyped", "swap_exception", "shade",
                 "union_val", "any_val", "lit", "Account.withdraw", "custom")
        elig = {n: probe(sources["buggy"], "canarypkg.core", n) for n in names}
        elig["envmod.f"] = probe(sources["buggy"], "canarypkg.envmod", "f")
        elig["runtime_mismatch"] = probe(work / "views" / "py312", "canarypkg.core", "f")
        elig["import_failure"] = probe(sources["buggy"], "canarypkg.no_such_module", "f")
        elig["abi_unavailable"] = {"eligible": False, "reason": abi.get("reason")}
        classified = {n: verdicts.classify_probe(info) for n, info in elig.items()}
        live = live_view_canaries(sources, work)
        print(f"live views: {json.dumps(live)}", flush=True)
        print(f"probe classes: {json.dumps(classified)}", flush=True)
        busy = aggregate_budget_canary(work, sources["buggy"])
        print(f"aggregate budget: {json.dumps(busy)}", flush=True)
        isolation = isolation_canaries(sources, work)
        print(f"isolation: {json.dumps({k: v['ok'] for k, v in isolation.items()})}", flush=True)
        results = {}
        for (function, mode), expected in EXPECT.items():
            target = {"module": "canarypkg.core", "qualname": function}
            r = fuzz(mode, sources, target, elig[function], 42, CANARY_BUDGET_SECONDS,
                     work / "runs" / function)
            results[f"{function}:{mode}"] = {**r, "expected_kill": expected}
            print(f"{function:11s} {mode:12s} kill={r['kill']} expected={expected} "
                  f"end={r['end_reason']} agg={r['aggregate_cpu_seconds']} "
                  f"main={r['main_cpu_seconds']} worker={r['worker_cpu_seconds']} "
                  f"replay={r.get('replay_cpu_seconds')} cleanup={r['cleanup_ok']}", flush=True)
        # the same seed and mode twice with fresh views: identical search outcome, no carry-over
        repeat = [fuzz("posthoc", sources, {"module": "canarypkg.core", "qualname": "stateful"},
                       elig["stateful"], 43, 5, work / "repeat" / str(i)) for i in range(2)]
        bad = replay(sources["buggy"], {"module": "canarypkg.core", "qualname": "missing"},
                     elig["crash"], [])
        leftovers = ([p.name for p in CGROUP_ROOT.iterdir() if p.is_dir()]
                     if CGROUP_ROOT.exists() else [])
        nobody = subprocess.run(["pgrep", "-u", "65534"], capture_output=True, text=True).stdout.split()
        env_after = environment_sha256(runtime["env_dir"])
    finally:
        RUNTIME.clear()
        shutil.rmtree(work, ignore_errors=True)
    unsupported = "adapter_unsupported:"
    checks = {
        "atheris_2_3_0_from_overlay_in_target_runtime": version[:1] == ["2.3.0"]
        and version[1].startswith("3.11") and version[2].startswith(runtime["overlay"]),
        "eligibility": elig["crash"]["eligible"] and elig["keyword"]["eligible"]
        and not elig["untyped"]["eligible"]
        and elig["variadic"]["reason"] == unsupported + "variadic_signature"
        and elig["runtime_mismatch"]["reason"].startswith("runtime_mismatch"),
        "keyword_plan": [p["keyword"] for p in elig["keyword"]["plan"]["params"]]
        == [False, False, True, True],
        "env_only_dependency_imports_in_target_runtime": elig["envmod.f"].get("eligible") is True,
        "environment_unchanged_by_every_search": env_after == env_before and all(
            r.get("environment_unchanged") is True for r in [*results.values(), *repeat]),
        "receiver_constructed_through_public_constructor":
            elig["Account.withdraw"].get("eligible") is True
            and elig["Account.withdraw"]["plan"]["receiver"]["plan"][0]["name"] == "balance"
            and elig["Box.total"].get("eligible") is True
            and results["Account.withdraw:ordinary"]["kill"] is True,
        "union_any_enum_literal_supported": all(
            elig[n].get("eligible") is True for n in ("shade", "union_val", "any_val", "lit"))
            and results["union_val:ordinary"]["kill"] is True,
        "unsupported_stays_explicit": classified["variadic"] == "adapter_unsupported"
            and elig["untyped"]["reason"].startswith(unsupported + "unsupported_parameter:x:")
            and elig["custom"]["reason"].startswith(unsupported + "unsupported_parameter:o:"),
        "abi_unavailable_is_infrastructure": elig["abi_unavailable"]["reason"]
            == "atheris_abi_unavailable"
            and classified["abi_unavailable"] == "infrastructure_failure",
        "replay_error_is_not_a_kill": "error" in bad,
        "aggregate_budget_two_busy_children": busy["ok"],
        "every_search_within_aggregate_budget": all(r["within_budget"] for r in results.values()),
        "differential_worker_cpu_counted": all(
            r["worker_cpu_seconds"] > 0 for k, r in results.items() if k.endswith(":differential")),
        "end_reasons_declared": all(r["end_reason"] in END_REASONS for r in results.values())
        and all(r["end_reason"] in ("completed", "cpu_budget_exhausted") for r in results.values()),
        "worker_and_group_cleanup": all(r["cleanup_ok"] and r["replay_cleanup_ok"]
                                        for r in results.values())
        and not leftovers and not nobody,
        "ordinary_different_exceptions_not_a_kill":
            results["swap_exception:ordinary"]["kill"] is False
            and results["swap_exception:ordinary"]["witnesses"] >= 1
            and all(c.get("buggy", [None])[0] == "raise" and c.get("fixed", [None])[0] == "raise"
                    for c in results["swap_exception:ordinary"]["confirmations"]),
        "ordinary_raise_versus_ok_is_a_kill": results["crash:ordinary"]["kill"] is True,
        "ordinary_same_exception_not_a_kill": results["same_crash:ordinary"]["kill"] is False,
        "every_kill_recomputes_from_confirmations": all(
            r["kill"] == any(verdicts.judge(r["mode"], c.get("buggy_all") or [],
                                            c.get("fixed_all") or [],
                                            c.get("replay_error"))["kill"]
                             for c in r["confirmations"]) for r in results.values()),
        "live_view_drift_refused": all(live.values()),
        # the shared resume/final validator accepts the rows REAL searches record, all modes
        "real_rows_structurally_valid": not any(
            verdicts.evidence_problems({k: v for k, v in r.items() if k != "expected_kill"},
                                       "eligible", budget)
            for rows, budget in ((results.values(), CANARY_BUDGET_SECONDS), (repeat, 5))
            for r in rows),
        "applicability_versus_infrastructure": (
            classified["crash"] == "eligible" and classified["keyword"] == "eligible"
            and classified["variadic"] == "adapter_unsupported"
            and classified["Box.total"] == "eligible"
            and classified["untyped"] == "adapter_unsupported"
            and classified["runtime_mismatch"] == "infrastructure_failure"
            and classified["import_failure"] == "infrastructure_failure"
            and elig["import_failure"]["reason"].startswith("import_failure")),
        "fresh_views_unmutated_by_any_search": all(
            r["views_unchanged"] for r in [*results.values(), *repeat]),
        "repeat_same_seed_mode_independent": all(
            r["reached"] and r["views_unchanged"] and r["kill"] is False
            and r["end_reason"] in ("completed", "cpu_budget_exhausted") for r in repeat),
        **{f"isolation_{k}": v["ok"] for k, v in isolation.items()},
        **{f"canary_{k}": v["kill"] == v["expected_kill"] and v["reached"]
           for k, v in results.items()}}
    receipt = {"schema_version": "oneiros_native_atheris_canaries_v4",
               "design_version": DESIGN_VERSION, "atheris": version,
               "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "verdicts_sha256": hashlib.sha256((HERE / "native_atheris_results.py")
                                                 .read_bytes()).hexdigest(),
               "inner_sha256": hashlib.sha256(INNER.read_bytes()).hexdigest(),
               "canary_budget_cpu_seconds": CANARY_BUDGET_SECONDS,
               "full_budget_cpu_seconds": FULL_BUDGET_CPU_SECONDS, "corpus_cap": CORPUS_CAP,
               "tolerance_rule": "1.0 CPU-second + 2% of the budget",
               "runtime": {"version": runtime["version"],
                           "overlay_manifest_sha256": runtime["overlay_manifest_sha256"],
                           "environment_packages": list(CANARY_ENV_PACKAGES),
                           "environment_sha256_before": env_before,
                           "environment_sha256_after": env_after},
               "abi_probe": {"version": abi.get("version"), "reason": abi.get("reason")},
               "eligibility": elig, "probe_classes": classified, "live_views": live,
               "aggregate_budget": busy, "isolation": isolation,
               "results": results, "repeat": repeat, "cgroup_leftovers": leftovers,
               "nobody_processes_after": nobody, "checks": checks,
               "passed": all(checks.values()),
               "note": "ordinary Atheris cannot kill the wrong-answer canary (no semantic oracle)"}
    import receipt_sanitize                             # tracked receipt: no user paths
    receipt = receipt_sanitize.scrub_json(receipt)
    (out_dir / "atheris_canary_receipt_v4.json").write_text(json.dumps(receipt, indent=1,
                                                                       sort_keys=True) + "\n")
    print(json.dumps({"passed": receipt["passed"],
                      "failed": [k for k, v in checks.items() if not v]}, indent=1))
    return 0 if receipt["passed"] else 1


# --- durable real-panel run (NOT executed: no real-target Atheris is authorised) -------------

def runtime_identity(runtime: dict) -> dict:
    """What the contract binds about a target's v5 runtime (no host paths)."""
    if not runtime.get("available"):
        return {"available": False, "reason": runtime.get("reason"),
                "version": runtime.get("version")}
    return {"available": True, "version": runtime["version"],
            "overlay_manifest_sha256": runtime["overlay_manifest_sha256"],
            "environment_sha256": environment_sha256(runtime.get("env_dir"))}


def eligibility_entry(key: str, row: dict, info: dict, views: dict,
                      runtime: dict | None = None) -> dict:
    status = verdicts.classify_probe(info)
    return {"target_key": key, "module": row["module"], "qualname": row["qualname"],
            "status": status,
            "reason": None if status == "eligible" else (info.get("reason") or "probe_failure"),
            "plan": info.get("plan") if status == "eligible" else None,
            "returns": info.get("returns") if status == "eligible" else None,
            "runtime": runtime, "views": views, "prep_record_sha256": gio.contract_sha(row)}


def prepare_contract(prep_path: Path, manifest_path: Path, budget: int, seeds,
                     root: Path | None = None, probe_fn=None, runtime_fn=None) -> tuple:
    """Everything the contract binds, established BEFORE any write (amendment v2.4 C/E,
    protocol v2.5 C): the exact preparation file, BOTH live views of all qualified targets
    re-hashed against their preparation manifests, each target's v5 runtime identity (its own
    interpreter and locked environment + the clean overlay), and a frozen per-target
    eligibility map from probes on fresh copies of the buggy views run IN that runtime."""
    probe_fn = probe_fn or probe
    runtime_fn = runtime_fn or (lambda row: runtime_for(row["python_path"], row["env_dir"]))
    prepared = gio.resolve_prep(prep_path, manifest_path, root or REPO_ROOT)
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    qualified = list(manifest["qualified_targets"])
    live = gio.verify_live_views(prepared["rows"], qualified)
    eligibility = {}
    scratch = Path(tempfile.mkdtemp(prefix="oneiros_atheris_probe_"))
    try:
        for i, key in enumerate(qualified):
            row = prepared["rows"][key]
            runtime = runtime_fn(row)
            if not runtime.get("available"):                     # infrastructure, never ABI
                info = {"eligible": False, "reason": runtime.get("reason")}   # -> unsupported
            else:
                set_runtime(runtime)
                view = fresh_view(Path(row["views"]["buggy"]), scratch / f"t{i:02d}")
                try:
                    info = probe_fn(view, row["module"], row["qualname"])
                except Exception as exc:                      # noqa: BLE001 - infrastructure
                    info = {"eligible": False, "reason": f"probe_failure:{type(exc).__name__}"}
            eligibility[key] = eligibility_entry(key, row, info, live[key],
                                                 runtime_identity(runtime))
    finally:
        RUNTIME.clear()
        shutil.rmtree(scratch, ignore_errors=True)
    contract = {"design_version": DESIGN_VERSION,
                "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "inner_sha256": hashlib.sha256(INNER.read_bytes()).hexdigest(),
                "verdicts_sha256": hashlib.sha256((HERE / "native_atheris_results.py")
                                                  .read_bytes()).hexdigest(),
                "prep": {"path": prepared["path"], "sha256": prepared["sha256"]},
                "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                "budget_cpu_seconds": budget, "tolerance": tolerance(budget),
                "seeds": list(seeds), "modes": list(MODES), "corpus_cap": CORPUS_CAP,
                "runtime_policy": "each target's prepared interpreter and locked environment "
                                  "(read-only), Atheris 2.3.0 from a clean read-only overlay "
                                  "per CPython minor version; environment hashed before and "
                                  "after every search", "live_views": live,
                "live_views_sha256": gio.contract_sha(live), "eligibility": eligibility,
                "eligibility_sha256": gio.contract_sha(eligibility)}
    return contract, prepared, eligibility


def run(prep_path: Path, manifest_path: Path, out: Path, budget: int, seeds,
        root: Path | None = None, runtime_fn=None) -> int:
    runtime_fn = runtime_fn or (lambda row: runtime_for(row["python_path"], row["env_dir"]))
    contract, prepared, eligibility = prepare_contract(prep_path, manifest_path, budget, seeds,
                                                       root, runtime_fn=runtime_fn)
    prep = prepared["rows"]
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    # v2.3 D.3: Atheris may cover every QUALIFIED target (explicit in the successor manifest);
    # joining with Oneiros happens in the analysis on the valid intersection.
    if "qualified_targets" not in manifest:
        raise SystemExit("REFUSED: manifest lacks an explicit qualified_targets cohort")
    qualified = list(manifest["qualified_targets"])
    chash = verdicts.contract_hash(contract)
    cfile = out / "atheris_contract.json"
    results = out / "atheris_results.jsonl"
    rejected = out / "atheris_rejected_rows.jsonl"
    # Resume (v2.4 E): EVERY retained row is validated by the authoritative shared validator
    # before any is skipped; any corrupt, stale, unexpected, duplicate or partial row refuses
    # the run and leaves the contract and results byte-for-byte unchanged.
    if cfile.exists() and json.loads(cfile.read_text(encoding="utf-8")) != contract:
        raise SystemExit("REFUSED: Atheris contract differs; prior results preserved")
    if results.exists() and not cfile.exists():
        raise SystemExit("REFUSED: Atheris results exist without their contract; preserved "
                         "unchanged; use a new output directory")
    if rejected.exists():
        raise SystemExit("REFUSED: a previously rejected Atheris row is preserved in "
                         f"{rejected.name}; resolve it before resuming")
    done = set()
    if results.exists():
        try:
            done = set(verdicts.validate_rows(results.read_bytes(), contract,
                                              qualified=qualified, seeds=seeds,
                                              budget=budget, complete=False))
        except SystemExit as exc:
            raise SystemExit(f"{exc} -- resume refused; {results} preserved unchanged; "
                             "quarantine it and use a new output directory") from None
    out.mkdir(parents=True, exist_ok=True)
    if not cfile.exists():
        cfile.write_bytes((json.dumps(contract, indent=1, sort_keys=True) + "\n").encode())
    with results.open("a", encoding="utf-8", newline="\n") as handle:
        for key in qualified:
            row = prep[key]
            entry = eligibility[key]                   # frozen in the contract (v2.4 E)
            target = {"module": row["module"], "qualname": row["qualname"]}
            info = {"eligible": True, "plan": entry["plan"], "returns": entry["returns"]}
            sources = {k: Path(v) for k, v in row["views"].items()}
            if entry["status"] == "eligible":
                runtime = runtime_fn(row)                  # v5: the target's own runtime,
                if runtime_identity(runtime) != entry["runtime"]:   # unchanged since probing
                    raise SystemExit(f"REFUSED: the runtime of {key} changed after the "
                                     "contract froze it; results preserved")
                set_runtime(runtime)
            work = Path(tempfile.mkdtemp(prefix="oneiros_atheris_"))
            try:
                for mode in MODES:
                    for seed in seeds:
                        rkey = f"{key}::{mode}::{seed}"
                        if rkey in done:
                            continue
                        r = (fuzz(mode, sources, target, info, seed, budget, work)
                             if entry["status"] == "eligible" else {"kill": False})
                        line = json.dumps({"key": rkey, "contract_sha256": chash,
                                           "target_key": key, "mode": mode, "seed": seed,
                                           "eligible": entry["status"] == "eligible",
                                           "status": entry["status"],
                                           "reason": entry["reason"], **r},
                                          sort_keys=True, default=str) + "\n"
                        try:                           # the same validator, before any append
                            verdicts.validate_rows(line.encode("utf-8"), contract,
                                                   qualified=qualified, seeds=seeds,
                                                   budget=budget, complete=False)
                        except SystemExit as exc:
                            rejected.write_bytes((json.dumps(
                                {"reason": str(exc), "row": json.loads(line)},
                                sort_keys=True, default=str) + "\n").encode("utf-8"))
                            raise SystemExit(f"{exc} -- new row NOT appended; preserved in "
                                             f"{rejected}") from None
                        handle.write(line)
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
