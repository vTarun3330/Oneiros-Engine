"""Execute model-generated pytest modules on buggy and fixed revisions inside a sandbox.

Protocol: docs/SFT_ROOT_CAUSE_NATIVE_GENERATED_TEST_PROTOCOL_V2.md (sections 4 and 5).
Runs INSIDE WSL as root (stdlib only); every generated module executes as ``nobody`` in
fresh mount/PID/network namespaces (scripts/native_sandbox_inner.sh) with prlimit limits,
an in-process audit hook, bytecode writes disabled and an imported-path attestation.

Classification is a pure function (``classify``) so it is unit-tested on any machine.

    python native_generated_tests_execute_wsl.py canaries <out_dir>
        builds a synthetic buggy/fixed package and proves the sandbox and the taxonomy
    python native_generated_tests_execute_wsl.py run <job.json> <out_dir>
        executes candidates for revalidated targets (not used in this work block)
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

DESIGN_VERSION = "oneiros_native_generated_tests_execute_v2"
HERE = Path(__file__).resolve().parent
INNER = HERE / "native_sandbox_inner.sh"
LIMITS = {"cpu_seconds": 60, "address_space_bytes": 4 * 2 ** 30, "nproc": 64, "nofile": 256,
          "fsize_bytes": 50 * 2 ** 20, "wall_seconds": 90, "per_test_seconds": 10}
MODULE_LIMITS = {"tests": 25, "asserts": 100, "target_calls": 200}
UV_PYTHON = "/root/.local/share/uv/python"
MASK_NAMES = ("tests", "test", "testing")

SITECUSTOMIZE = r'''
import sys
_BLOCKED = ("subprocess.Popen", "os.system", "os.exec", "os.posix_spawn", "os.fork",
            "os.forkpty", "os.spawn", "pty.spawn", "socket.__new__", "socket.connect",
            "socket.bind", "ctypes.dlopen", "ctypes.dlsym")
def _oneiros_sandbox_hook(event, args):
    if event.startswith(_BLOCKED):
        raise PermissionError("blocked by the Oneiros sandbox: " + event)
sys.addaudithook(_oneiros_sandbox_hook)
'''

PLUGIN = r'''
import importlib, inspect, json, os, signal, sys, threading, traceback
import pytest

SPEC = json.load(open("/tmp/work/sandbox_spec.json", encoding="utf-8"))
OUT = "/sandbox_out/report.json"
STATE = {"attestation": None, "target_error": None, "collected": [], "collection_errors": [],
         "nodes": {}, "current": None}
TARGET = {"code": None, "package_root": None}

class SandboxTestTimeout(BaseException):
    pass

# An oversized write must fail the test (OSError), not kill pytest and look like a harness
# failure that would exclude the candidate.
signal.signal(signal.SIGXFSZ, signal.SIG_IGN)

def _alarm(signum, frame):
    raise SandboxTestTimeout("per-test time limit")

def pytest_configure(config):
    try:
        module = importlib.import_module(SPEC["module"])
        path = os.path.realpath(module.__file__)
        root = os.path.realpath(SPEC["expected_root"])
        STATE["attestation"] = {"module_file": path, "expected_root": root,
                                "ok": path.startswith(root + os.sep)}
        TARGET["package_root"] = os.path.join(root, *SPEC["package_path"])
        obj = module
        for part in SPEC["qualname"].split("."):
            obj = inspect.getattr_static(obj, part) if inspect.isclass(obj) else getattr(obj, part)
        if isinstance(obj, (staticmethod, classmethod)):
            obj = obj.__func__
        TARGET["code"] = getattr(inspect.unwrap(obj), "__code__", None)
        if TARGET["code"] is None:
            STATE["target_error"] = "no code object"
    except BaseException as exc:
        STATE["target_error"] = type(exc).__name__ + ": " + str(exc)[:200]

def _node(nodeid):
    return STATE["nodes"].setdefault(nodeid, {"reached": False, "phases": {}})

def _profile(frame, event, arg):
    if event == "call" and frame.f_code is TARGET["code"] and STATE["current"]:
        _node(STATE["current"])["reached"] = True

def pytest_collectreport(report):
    if report.failed:
        text = str(report.longrepr)[-600:]
        STATE["collection_errors"].append({"nodeid": report.nodeid, "text": text})

def pytest_collection_finish(session):
    STATE["collected"] = [item.nodeid for item in session.items]

@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):
    STATE["current"] = item.nodeid
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(SPEC["per_test_seconds"])
    sys.setprofile(_profile)
    threading.setprofile(_profile)
    try:
        yield
    finally:
        sys.setprofile(None)
        threading.setprofile(None)
        signal.alarm(0)
        STATE["current"] = None

def _excinfo(call):
    if call.excinfo is None:
        return None
    tb = call.excinfo.tb
    target_in_tb, project_in_tb, frames = False, False, []
    while tb is not None:
        code = tb.tb_frame.f_code
        path = os.path.realpath(code.co_filename)
        frames.append(path)
        if code is TARGET["code"]:
            target_in_tb = True
        if TARGET["package_root"] and path.startswith(TARGET["package_root"] + os.sep):
            project_in_tb = True
        tb = tb.tb_next
    etype = call.excinfo.type
    return {"type": etype.__name__, "assertion": issubclass(etype, AssertionError),
            "timeout": etype.__name__ == "SandboxTestTimeout",
            "sandbox_block": issubclass(etype, PermissionError)
            and "Oneiros sandbox" in str(call.excinfo.value),
            "target_in_traceback": target_in_tb, "project_in_traceback": project_in_tb}

@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    entry = _node(item.nodeid)
    entry["phases"][call.when] = {"outcome": report.outcome,
                                  "xfail": bool(getattr(report, "wasxfail", None) is not None),
                                  "exception": _excinfo(call)}

def pytest_sessionfinish(session, exitstatus):
    out = {"attestation": STATE["attestation"], "target_error": STATE["target_error"],
           "collected": STATE["collected"], "collection_errors": STATE["collection_errors"],
           "nodes": STATE["nodes"], "exitstatus": int(exitstatus), "uid": os.getuid()}
    with open(OUT, "w", encoding="utf-8") as handle:
        json.dump(out, handle)
'''


# --- static admission -------------------------------------------------------------------

def static_check(source: str, target_name: str) -> dict:
    """Syntax and module-size limits before any execution."""
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError) as exc:
        return {"status": "syntax_failure", "detail": type(exc).__name__}
    tests = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
             and n.name.startswith("test")]
    asserts = sum(isinstance(n, ast.Assert) for n in ast.walk(tree))
    calls = sum(isinstance(n, ast.Call) and (
        (isinstance(n.func, ast.Name) and n.func.id == target_name)
        or (isinstance(n.func, ast.Attribute) and n.func.attr == target_name))
        for n in ast.walk(tree))
    counts = {"tests": len(tests), "asserts": asserts, "target_calls": calls}
    over = [k for k, v in counts.items() if v > MODULE_LIMITS[k]]
    return {"status": "over_limits" if over else "ok", "counts": counts, "over": over}


# --- classification (pure) ----------------------------------------------------------------

INFRA = ("harness_failure", "dependency_failure", "environment_failure")
KILLS = ("semantic_kill", "crash_kill")


def _executed(node) -> bool:
    call = node["phases"].get("call")
    return bool(call) and call["outcome"] in ("passed", "failed") and not call["xfail"]


def _failed(node) -> bool:
    return any(p["outcome"] == "failed" for p in node["phases"].values())


def _fabricated(errors) -> bool:
    return any(("ModuleNotFoundError" in e["text"] or "ImportError" in e["text"]
                or "cannot import name" in e["text"] or "AttributeError" in e["text"])
               for e in errors)


def revision_outcome(report) -> dict:
    """Summarise one revision's sandbox report."""
    if report is None or "nodes" not in report:
        return {"status": "harness_failure"}
    if not (report.get("attestation") or {}).get("ok"):
        return {"status": "harness_failure", "detail": "revision attestation failed"}
    if report.get("wall_timeout"):
        return {"status": "timeout"}
    if report["collection_errors"]:
        return {"status": "fabricated_import" if _fabricated(report["collection_errors"])
                else "collection_failure"}
    if not report["collected"]:
        return {"status": "no_tests_collected"}
    return {"status": "ran"}


def classify(static: dict, buggy: dict | None, fixed: dict | None,
             rerun: tuple | None = None) -> dict:
    """One candidate's outcome class under protocol v2 section 4."""
    if static["status"] != "ok":
        return {"class": static["status"]}
    for report in (buggy, fixed):
        summary = revision_outcome(report)
        if summary["status"] != "ran":
            return {"class": summary["status"], "detail": summary.get("detail")}
    if set(buggy["collected"]) != set(fixed["collected"]):
        return {"class": "collection_failure", "detail": "node sets differ"}
    nodes = sorted(buggy["collected"])
    b, f = buggy["nodes"], fixed["nodes"]
    executed = [n for n in nodes if n in b and n in f and _executed(b[n]) and _executed(f[n])]
    if not executed:
        return {"class": "skipped_or_xfail"}
    for rep in (b, f):
        for n in executed:
            exc = (rep[n]["phases"].get("call") or {}).get("exception") or {}
            if exc.get("timeout"):
                return {"class": "timeout"}
    reached = [n for n in executed if b[n]["reached"] and f[n]["reached"]]
    if not reached:
        return {"class": "target_not_reached"}
    if any(_failed(f[n]) for n in nodes if n in f):
        subtype = "inverted" if not any(_failed(b[n]) for n in nodes if n in b) else "fail_both"
        return {"class": "fixed_side_failure", "subtype": subtype}
    failing = [n for n in reached if _failed(b[n])]
    if not failing:
        return {"class": "pass_both"}
    kinds = []
    for n in failing:
        exc = (b[n]["phases"].get("call") or {}).get("exception") or {}
        if exc.get("assertion"):
            kinds.append("semantic_kill")
        elif exc.get("target_in_traceback") or exc.get("project_in_traceback"):
            kinds.append("crash_kill")
        else:
            kinds.append("buggy_test_error")
    result = ("semantic_kill" if "semantic_kill" in kinds else
              "crash_kill" if "crash_kill" in kinds else "buggy_test_error")
    if result in KILLS and rerun is not None:
        again = classify(static, rerun[0], rerun[1], None)
        if again["class"] != result:
            return {"class": "nondeterminism", "first": result, "rerun": again["class"]}
    return {"class": result, "kill_nodes": [n for n, k in zip(failing, kinds) if k in KILLS]}


# --- sandbox execution (WSL, root) ----------------------------------------------------------

def mask_dirs(revision: Path) -> list[str]:
    out = []
    for name in MASK_NAMES:
        for path in [revision / name, *revision.glob(f"*/{name}")]:
            if path.is_dir():
                out.append(str(path))
    return sorted(set(out))


def run_sandboxed(python: str, env_dir: Path, revision: Path, module: str, package_path: list,
                  qualname: str, candidate: str, out_dir: Path) -> dict:
    """Run one candidate module on one revision in a fresh sandbox; return its report."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob("*"):
        stale.unlink()
    work = Path(tempfile.mkdtemp(prefix="oneiros_sbwork_"))
    try:
        (work / "plugin").mkdir()
        (work / "plugin" / "sitecustomize.py").write_text(SITECUSTOMIZE, encoding="utf-8")
        (work / "plugin" / "oneiros_native_plugin.py").write_text(PLUGIN, encoding="utf-8")
        (work / "test_candidate.py").write_text(candidate, encoding="utf-8")
        (work / "pytest.ini").write_text("[pytest]\n", encoding="utf-8")
        (work / "sandbox_spec.json").write_text(json.dumps({
            "module": module, "qualname": qualname, "package_path": package_path,
            "expected_root": str(revision), "per_test_seconds": LIMITS["per_test_seconds"]}),
            encoding="utf-8")
        env = {"PATH": "/usr/bin:/bin", "SB_RO": "\n".join([UV_PYTHON, str(env_dir), str(revision)]),
               "SB_MASK": "\n".join(mask_dirs(revision)), "SB_OUT": str(out_dir),
               "SB_WORK_SRC": str(work), "SB_CPU": str(LIMITS["cpu_seconds"]),
               "SB_AS": str(LIMITS["address_space_bytes"]), "SB_NPROC": str(LIMITS["nproc"]),
               "SB_NOFILE": str(LIMITS["nofile"]), "SB_FSIZE": str(LIMITS["fsize_bytes"]),
               "SB_WALL": str(LIMITS["wall_seconds"])}
        argv = ["unshare", "--mount", "--pid", "--net", "--fork", "--mount-proc", "--",
                "/bin/bash", str(INNER), python, "-B", "-m", "pytest", "-q", "-p",
                "no:cacheprovider", "-c", "/tmp/work/pytest.ini", "--rootdir=/tmp/work",
                "-p", "oneiros_native_plugin", "test_candidate.py"]
        started = time.time()
        try:
            done = subprocess.run(argv, env=env, capture_output=True, text=True,
                                  timeout=LIMITS["wall_seconds"] + 30)
            code, tail = done.returncode, (done.stdout + done.stderr)[-800:]
        except subprocess.TimeoutExpired:
            code, tail = "backstop_timeout", ""
        seconds = round(time.time() - started, 2)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    report_path = out_dir / "report.json"
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
    else:
        report = {"harness_error": True}
    report.update(process_exit=code, seconds=seconds, tail=tail,
                  wall_timeout=code in (124, 137, "backstop_timeout"))
    return report


def switch(python: str, revision: Path, module: str = "") -> None:
    """Point the environment's editable install at ``revision`` and PROVE it did.

    The venv root is ``python``'s grandparent WITHOUT resolving the symlink (resolving it
    lands in the uv-managed runtime, not the venv). Stale editable-finder bytecode is
    removed, and when ``module`` is given it must import from under ``revision``.
    """
    done = subprocess.run(["uv", "pip", "install", "--python", python, "--no-deps", "-e",
                           str(revision)], capture_output=True, text=True, timeout=900)
    if done.returncode != 0:
        raise RuntimeError("switch failed: " + done.stderr[-300:])
    venv = Path(python).parent.parent
    for cache in venv.glob("lib/python*/site-packages/__pycache__"):
        for pyc in cache.glob("__editable__*"):
            pyc.unlink()
    if module:
        probe = subprocess.run([python, "-B", "-c", f"import os, {module} as m; "
                                "print(os.path.realpath(m.__file__))"], cwd="/tmp",
                               capture_output=True, text=True, timeout=120,
                               env={"PATH": "/usr/bin:/bin", "PYTHONDONTWRITEBYTECODE": "1"})
        path = probe.stdout.strip()
        if not path.startswith(os.path.realpath(revision) + os.sep):
            raise RuntimeError(f"switch did not take effect: {module} imports from {path!r}")


# --- canaries (synthetic package; proves the sandbox and the taxonomy) ----------------------

TOY_BUGGY = '''
import time
def add(a, b):
    return a - b
def crash(x):
    return 10 // (x - x)
def slow(x):
    return x
def flaky(x):
    return x
'''
TOY_FIXED = '''
import time
def add(a, b):
    return a + b
def crash(x):
    return 10
def slow(x):
    return x
def flaky(x):
    return x
'''

CANDIDATES = {
    # taxonomy
    "semantic_kill": ("add", "from toy.core import add\ndef test_a():\n    assert add(2, 3) == 5\n"),
    "crash_kill": ("crash", "from toy.core import crash\ndef test_c():\n    assert crash(1) == 10\n"),
    "pass_both": ("add", "from toy.core import add\ndef test_p():\n    assert add(2, 0) == 2\n"),
    "fail_both": ("add", "from toy.core import add\ndef test_f():\n    add(1, 1)\n    assert False\n"),
    "inverted": ("add", "from toy.core import add\ndef test_i():\n    assert add(2, 3) == -1\n"),
    "skip": ("add", "import pytest\nfrom toy.core import add\n@pytest.mark.skip\ndef test_s():\n    assert add(1, 1) == 2\n"),
    "xfail": ("add", "import pytest\nfrom toy.core import add\n@pytest.mark.xfail\ndef test_x():\n    assert add(2, 3) == 5\n"),
    "timeout": ("add", "from toy.core import add\ndef test_t():\n    add(1, 1)\n    while True:\n        pass\n"),
    "fabricated_import": ("add", "from toy.core import add_numbers\ndef test_m():\n    assert add_numbers(1, 2) == 3\n"),
    "zero_tests": ("add", "from toy.core import add\nVALUE = add(1, 2)\n"),
    "not_reached": ("add", "def test_n():\n    assert 1 + 1 == 2\n"),
    "syntax": ("add", "def test_(:\n    pass\n"),
    # malicious (must be contained)
    "read_windows_drive": ("add", "import os\nfrom toy.core import add\ndef test_r():\n    add(1, 1)\n    assert os.path.exists('/mnt/c/Users')\n"),
    "read_root_home": ("add", "import os\nfrom toy.core import add\ndef test_h():\n    add(1, 1)\n    assert any(os.path.exists(p) for p in ('/root/.oneiros_decoy/secret.txt', '/root/.ssh', '/root/.cache', '/root/oneiros_rehearsal'))\n"),
    "read_other_worktree": ("add", "import os\nfrom toy.core import add\ndef test_o():\n    add(1, 1)\n    assert os.path.exists(os.environ.get('ONEIROS_PROBE', '/root/oneiros_rehearsal'))\n"),
    "network": ("add", "import socket\nfrom toy.core import add\ndef test_net():\n    add(1, 1)\n    socket.create_connection(('1.1.1.1', 80), timeout=2)\n"),
    "subprocess": ("add", "import subprocess\nfrom toy.core import add\ndef test_sp():\n    add(1, 1)\n    subprocess.run(['id'])\n"),
    "write_source": ("add", "import toy.core\nfrom toy.core import add\ndef test_w():\n    add(1, 1)\n    open(toy.core.__file__, 'a').write('#x')\n"),
    "uid": ("add", "import os\nfrom toy.core import add\ndef test_u():\n    add(1, 1)\n    assert os.getuid() == 0\n"),
    "shadow": ("add", "from toy.core import add\ndef test_sh():\n    add(1, 1)\n    open('/etc/shadow').read()\n"),
    "big_file": ("add", "from toy.core import add\ndef test_b():\n    add(1, 1)\n    open('/tmp/work/big', 'wb').write(b'0' * (200 * 2 ** 20))\n"),
    "memory": ("add", "from toy.core import add\ndef test_mem():\n    add(1, 1)\n    x = bytearray(8 * 2 ** 30)\n"),
    "tests_dir_masked": ("add", "import os, toy\nfrom toy.core import add\ndef test_td():\n    add(1, 1)\n    root = os.path.dirname(os.path.dirname(toy.__file__))\n    assert os.listdir(os.path.join(root, 'tests'))\n"),
}
EXPECTED = {
    "semantic_kill": "semantic_kill", "crash_kill": "crash_kill", "pass_both": "pass_both",
    "fail_both": "fixed_side_failure", "inverted": "fixed_side_failure",
    "skip": "skipped_or_xfail", "xfail": "skipped_or_xfail", "timeout": "timeout",
    "fabricated_import": "fabricated_import", "zero_tests": "no_tests_collected",
    "not_reached": "target_not_reached", "syntax": "syntax_failure",
    # malicious probes must FAIL on both revisions, i.e. fixed_side_failure (never a kill)
    "read_windows_drive": "fixed_side_failure", "read_root_home": "fixed_side_failure",
    "read_other_worktree": "fixed_side_failure", "network": "fixed_side_failure",
    "subprocess": "fixed_side_failure", "write_source": "fixed_side_failure",
    "uid": "fixed_side_failure", "shadow": "fixed_side_failure",
    "big_file": "fixed_side_failure", "memory": "fixed_side_failure",
    "tests_dir_masked": "fixed_side_failure",
}


def build_toy(root: Path) -> dict:
    """A buggy and a fixed revision of a tiny package plus one shared uv environment."""
    sys.path.insert(0, str(HERE))
    from native_rehearsal_wsl import build_env
    for label, body in (("buggy", TOY_BUGGY), ("fixed", TOY_FIXED)):
        rev = root / label
        (rev / "toy").mkdir(parents=True)
        (rev / "tests").mkdir()
        (rev / "tests" / "test_official.py").write_text("SECRET_OFFICIAL = 1\n", encoding="utf-8")
        (rev / "toy" / "__init__.py").write_text("", encoding="utf-8")
        (rev / "toy" / "core.py").write_text(body, encoding="utf-8")
        (rev / "pyproject.toml").write_text(
            '[project]\nname = "toy"\nversion = "0.1"\n[build-system]\nrequires = '
            '["setuptools"]\nbuild-backend = "setuptools.build_meta"\n[tool.setuptools]\n'
            'packages = ["toy"]\n', encoding="utf-8")
    env = build_env(root / "env", root / "fixed")
    if not env["ok"]:
        raise RuntimeError("toy environment failed")
    return {"python": env["python_path"], "env_dir": root / "env"}


def run_candidate(toy: dict, root: Path, name: str, qualname: str, source: str,
                  out: Path, rerun: bool = True) -> dict:
    static = static_check(source, qualname.split(".")[-1])
    if static["status"] != "ok":
        return {"classification": classify(static, None, None), "static": static}
    reports = {}
    for label in ("buggy", "fixed"):
        switch(toy["python"], root / label, "toy.core")
        reports[label] = run_sandboxed(toy["python"], toy["env_dir"], root / label, "toy.core",
                                       ["toy"], qualname, source, out / name / label)
    again = None
    first = classify(static, reports["buggy"], reports["fixed"])
    if rerun and first["class"] in KILLS:
        again_reports = {}
        for label in ("buggy", "fixed"):
            switch(toy["python"], root / label, "toy.core")
            again_reports[label] = run_sandboxed(toy["python"], toy["env_dir"], root / label,
                                                 "toy.core", ["toy"], qualname, source,
                                                 out / name / f"{label}_rerun")
        again = (again_reports["buggy"], again_reports["fixed"])
    return {"classification": classify(static, reports["buggy"], reports["fixed"], again),
            "static": static,
            "uids": [reports[l].get("uid") for l in reports],
            "attestation": [reports[l].get("attestation") for l in reports],
            "tails": {l: reports[l].get("tail", "")[-200:] for l in reports}}


def canaries(out_dir: Path) -> int:
    """Run outputs stay on the WSL filesystem (drvfs cannot be chowned to nobody)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix="oneiros_canary_", dir="/root"))
    runs = root / "runs"
    decoy = Path("/root/.oneiros_decoy")
    decoy.mkdir(exist_ok=True)
    (decoy / "secret.txt").write_text("decoy credential\n", encoding="utf-8")
    try:
        toy = build_toy(root)
        results = {}
        for name, (qualname, source) in CANDIDATES.items():
            results[name] = run_candidate(toy, root, name, qualname, source, runs)
            observed = results[name]["classification"]["class"]
            print(f"{name:22s} -> {observed}", flush=True)
            if observed != EXPECTED[name]:
                print(json.dumps(results[name].get("tails"), indent=1)[:1500], flush=True)
        # stale-bytecode / wrong-revision canary: the attestation must follow each switch
        attest = []
        for label in ("buggy", "fixed", "buggy"):
            switch(toy["python"], root / label, "toy.core")
            rep = run_sandboxed(toy["python"], toy["env_dir"], root / label, "toy.core", ["toy"],
                                "add", CANDIDATES["pass_both"][1], runs / "attest")
            attest.append({"label": label, **(rep.get("attestation") or {})})
        wrong = run_sandboxed(toy["python"], toy["env_dir"], root / "fixed", "toy.core", ["toy"],
                              "add", CANDIDATES["pass_both"][1], runs / "wrong")
        # environment currently points at buggy while fixed is mounted: must refuse
        wrong_refused = not (wrong.get("attestation") or {}).get("ok", False)
    finally:
        shutil.rmtree(root, ignore_errors=True)
        shutil.rmtree(decoy, ignore_errors=True)
    checks = {name: results[name]["classification"]["class"] == EXPECTED[name]
              for name in CANDIDATES}
    checks["ran_as_nobody"] = all(u == 65534 for r in results.values() for u in r.get("uids", [])
                                  if u is not None)
    checks["attestation_follows_switch"] = [a["label"] in a.get("module_file", "")
                                            for a in attest] == [True, True, True]
    checks["wrong_revision_refused"] = wrong_refused
    receipt = {"schema_version": "oneiros_native_sandbox_canaries_v1",
               "design_version": DESIGN_VERSION,
               "executor_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "inner_sha256": hashlib.sha256(INNER.read_bytes()).hexdigest(),
               "limits": LIMITS, "module_limits": MODULE_LIMITS,
               "results": {n: {"expected": EXPECTED[n],
                               "observed": r["classification"]["class"]}
                           for n, r in results.items()},
               "checks": checks, "passed": all(checks.values()),
               "nondeterminism": ("covered by classifier unit tests: the sandbox gives every "
                                  "run a fresh filesystem, so no live canary can carry state "
                                  "between runs")}
    (out_dir / "canary_receipt.json").write_text(json.dumps(receipt, indent=1, sort_keys=True)
                                                 + "\n", encoding="utf-8")
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
