"""Execute model-generated pytest modules on buggy and fixed revisions inside a sandbox.

Protocol v2 (sections 4-5) with amendment v2.1 (section B). Runs INSIDE WSL as root, stdlib
only. Every candidate executes as ``nobody`` in fresh mount/PID/network namespaces
(scripts/native_sandbox_inner.sh) with prlimit limits and an in-process audit hook.

Revision identity: the trusted harness builds a sanitised VIEW of each revision's import
root (no test directories or test files at any depth, no .git, no metadata) and mounts it
at the canonical path /target; the project is imported through PYTHONPATH=/target from a
dependency-only environment (the project's editable finder is stripped). Everything the
candidate can see - paths, sandbox specification, module file names - is identical for
both revisions; the revision attestation (hash of the imported target module) is checked
by the harness OUTSIDE the sandbox.

    python native_generated_tests_execute_wsl.py canaries <out_dir>
    python native_generated_tests_execute_wsl.py run --prep <records.jsonl> --manifest <m.json>
        --job <job.json> (--generations <root with base/ and sft/> |
        --base-generations <dir> --sft-generations <dir>)
        --condition primary_whole_module --out <dir>

Amendment v2.3: only the generation cohort resolved from the exact job artifact is executed
(never all kept/qualified targets); both arm directories and their generation contracts are
hashed and validated (scripts/native_generation_io.py); exactly 23 x 3 x 8 x 2 = 1,104 rows
are required for the rehearsal job; every execution row carries its candidate's generation
telemetry. Pre-generation exclusions are reported, never counted as model failures.
"""
from __future__ import annotations

import argparse
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

DESIGN_VERSION = "oneiros_native_generated_tests_execute_v5"
REPO_ROOT = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent
INNER = HERE / "native_sandbox_inner.sh"
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import native_generation_io as gio  # noqa: E402  (stdlib only)
APPROVED_PYTHON = "/usr/bin/python3.11"
LIMITS = {"cpu_seconds": 60, "address_space_bytes": 4 * 2 ** 30, "nproc": 64, "nofile": 256,
          "fsize_bytes": 50 * 2 ** 20, "wall_seconds": 90, "per_test_seconds": 10}
MODULE_LIMITS = {"tests": 25, "asserts": 100, "target_calls": 200}
ARMS = ("base", "sft")
SEEDS = (42, 43, 44)
SLOTS = 8
TEST_DIRS = {"test", "tests", "testing"}
SKIP_DIRS = TEST_DIRS | {".git", "__pycache__", ".tox", ".nox", ".venv", "venv", "build",
                         "dist", ".eggs", ".mypy_cache", ".pytest_cache", "docs", "doc"}

POLICY_NAMES = {"open", "eval", "exec", "compile", "__import__", "globals", "vars", "breakpoint"}
POLICY_IMPORTS = {"os", "sys", "pathlib", "io", "inspect", "dis", "importlib", "pkgutil",
                  "subprocess", "socket", "ctypes", "shutil", "glob", "tempfile", "builtins",
                  "gc", "marshal", "pickle", "linecache", "traceback", "code", "runpy", "signal"}
POLICY_ATTRS = {"__file__", "__code__", "__dict__", "__globals__", "__builtins__", "__loader__",
                "__spec__", "__closure__", "__subclasses__", "f_code", "co_code", "gi_code"}

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
import hashlib, importlib, inspect, json, os, signal, sys, threading
import pytest

SPEC = json.load(open("/tmp/work/sandbox_spec.json", encoding="utf-8"))
OUT = "/sandbox_out/report.json"
STATE = {"attestation": None, "target_error": None, "collected": [], "collection_errors": [],
         "nodes": {}, "current": None}
TARGET = {"code": None, "package_root": os.path.join("/target", SPEC["top_package"])}

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
        STATE["attestation"] = {"module_file": path,
                                "module_sha256": hashlib.sha256(open(path, "rb").read()).hexdigest()}
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
        STATE["collection_errors"].append({"nodeid": report.nodeid,
                                           "text": str(report.longrepr)[-600:]})

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
    target_in_tb, project_in_tb = False, False
    while tb is not None:
        code = tb.tb_frame.f_code
        path = os.path.realpath(code.co_filename)
        if code is TARGET["code"]:
            target_in_tb = True
        if path.startswith(TARGET["package_root"] + os.sep):
            project_in_tb = True
        tb = tb.tb_next
    etype = call.excinfo.type
    return {"type": etype.__name__, "assertion": issubclass(etype, AssertionError),
            "timeout": etype.__name__ == "SandboxTestTimeout",
            "target_in_traceback": target_in_tb, "project_in_traceback": project_in_tb}

@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    _node(item.nodeid)["phases"][call.when] = {
        "outcome": report.outcome, "xfail": bool(getattr(report, "wasxfail", None) is not None),
        "exception": _excinfo(call)}

def pytest_sessionfinish(session, exitstatus):
    with open(OUT, "w", encoding="utf-8") as handle:
        json.dump({"attestation": STATE["attestation"], "target_error": STATE["target_error"],
                   "collected": STATE["collected"], "collection_errors": STATE["collection_errors"],
                   "nodes": STATE["nodes"], "exitstatus": int(exitstatus), "uid": os.getuid()},
                  handle)
'''


# --- static admission and policy ------------------------------------------------------------

def static_check(source: str, target_name: str, enforce_policy: bool = True) -> dict:
    """Syntax, the v2.1 candidate policy, and module-size limits - before any execution."""
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError) as exc:
        return {"status": "syntax_failure", "detail": type(exc).__name__}
    violations = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in POLICY_NAMES:
            violations.append(f"name:{node.id}")
        elif isinstance(node, ast.Attribute) and node.attr in POLICY_ATTRS:
            violations.append(f"attribute:{node.attr}")
        elif isinstance(node, ast.Import):
            violations += [f"import:{a.name}" for a in node.names
                           if a.name.split(".")[0] in POLICY_IMPORTS]
        elif isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] in POLICY_IMPORTS:
            violations.append(f"import:{node.module}")
    if violations and enforce_policy:
        return {"status": "policy_refused", "violations": sorted(set(violations))[:10]}
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


# --- classification (pure) ------------------------------------------------------------------

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


def revision_outcome(report, expected_sha: str | None = None) -> dict:
    if report is None or "nodes" not in report:
        return {"status": "harness_failure"}
    attest = report.get("attestation") or {}
    if not attest or (expected_sha and attest.get("module_sha256") != expected_sha):
        return {"status": "harness_failure", "detail": "revision attestation failed"}
    if report.get("wall_timeout"):
        return {"status": "timeout"}
    if report["collection_errors"]:
        return {"status": "fabricated_import" if _fabricated(report["collection_errors"])
                else "collection_failure"}
    if not report["collected"]:
        return {"status": "no_tests_collected"}
    return {"status": "ran"}


def classify(static: dict, buggy: dict | None, fixed: dict | None, rerun: tuple | None = None,
             expected: dict | None = None) -> dict:
    """One candidate's class under v2 section 4 with the v2.1 whole-module rule."""
    expected = expected or {}
    if static["status"] != "ok":
        return {"class": static["status"]}
    for label, report in (("buggy", buggy), ("fixed", fixed)):
        summary = revision_outcome(report, expected.get(label))
        if summary["status"] != "ran":
            return {"class": summary["status"], "detail": summary.get("detail")}
    if set(buggy["collected"]) != set(fixed["collected"]):
        return {"class": "collection_failure", "detail": "node sets differ"}
    nodes = sorted(buggy["collected"])
    b, f = buggy["nodes"], fixed["nodes"]
    if not all(n in b and n in f and _executed(b[n]) and _executed(f[n]) for n in nodes):
        return {"class": "skipped_or_xfail"}
    for rep in (b, f):
        for n in nodes:
            if ((rep[n]["phases"].get("call") or {}).get("exception") or {}).get("timeout"):
                return {"class": "timeout"}
    if not all(b[n]["reached"] and f[n]["reached"] for n in nodes):
        return {"class": "target_not_reached"}
    if any(_failed(f[n]) for n in nodes):
        subtype = "inverted" if not any(_failed(b[n]) for n in nodes) else "fail_both"
        return {"class": "fixed_side_failure", "subtype": subtype}
    failing = [n for n in nodes if _failed(b[n])]
    evidence = {"fixed_all_executed_passed_reached": True}
    if not failing:
        return {"class": "pass_both", **evidence}
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
    if result in KILLS:
        if rerun is None:
            return {"class": result, **evidence, "rerun_agrees": None}
        again = classify(static, rerun[0], rerun[1], None, expected)
        if again["class"] != result:
            return {"class": "nondeterminism", "first": result, "rerun": again["class"],
                    "fixed_all_executed_passed_reached": False}
        return {"class": result, **evidence, "rerun_agrees": True,
                "kill_nodes": [n for n, k in zip(failing, kinds) if k in KILLS]}
    return {"class": result, **evidence}


# --- environments and views (trusted harness) ----------------------------------------------

def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def import_root(target_file: str) -> str:
    first = Path(target_file).parts[0]
    return first if first in ("src", "lib") else ""


def _is_test_file(name: str) -> bool:
    return name == "conftest.py" or (name.startswith("test_") and name.endswith(".py")) \
        or name.endswith("_test.py")


def build_view(checkout: Path, root_rel: str, dest: Path, extra: dict | None = None) -> dict:
    """Sanitised copy of the import root: Python packages only; tests removed at any depth."""
    source_root = checkout / root_rel if root_rel else checkout
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    tops = []
    for entry in sorted(source_root.iterdir()):
        if entry.is_dir() and entry.name not in SKIP_DIRS and (entry / "__init__.py").exists():
            tops.append(entry)
        elif entry.is_file() and entry.suffix == ".py" and not _is_test_file(entry.name) and \
                entry.name not in ("setup.py", "noxfile.py", "tasks.py", "conftest.py"):
            tops.append(entry)
    for top in tops:
        if top.is_file():
            shutil.copy2(top, dest / top.name)
            continue
        for path in sorted(top.rglob("*")):
            rel = path.relative_to(source_root)
            if any(part in SKIP_DIRS for part in rel.parts) or path.is_dir():
                continue
            if _is_test_file(path.name) or path.name.startswith(".git"):
                continue
            (dest / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dest / rel)
    for rel, data in (extra or {}).items():
        target = dest / rel
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
    files = {p.relative_to(dest).as_posix(): sha256_file(p) for p in sorted(dest.rglob("*"))
             if p.is_file()}
    return {"files": len(files), "manifest_sha256": hashlib.sha256(
        json.dumps(files, sort_keys=True).encode()).hexdigest(), "tops": [t.name for t in tops]}


def generated_files(checkout: Path, root_rel: str) -> dict:
    """Build-generated (untracked or ignored) Python files inside the import root."""
    done = subprocess.run(["git", "-C", str(checkout), "ls-files", "--others", "--exclude-standard",
                           "--ignored", "--directory", "--no-empty-directory"],
                          capture_output=True, text=True)
    out = {}
    prefix = (root_rel + "/") if root_rel else ""
    for rel in done.stdout.splitlines():
        path = checkout / rel
        if rel.startswith(prefix) and path.is_file() and path.suffix == ".py" and \
                "__pycache__" not in rel and not _is_test_file(path.name):
            out[rel[len(prefix):]] = path.read_bytes()
    return out


def strip_project_install(python: str, checkout: Path) -> list:
    """Remove the project's editable finder/.pth so it is importable ONLY via PYTHONPATH."""
    removed = []
    venv = Path(python).parent.parent
    for site in venv.glob("lib/python*/site-packages"):
        for path in list(site.glob("*.pth")) + list(site.glob("__editable__*.py")):
            text = path.read_text(encoding="utf-8", errors="replace")
            if str(checkout) in text or path.name.startswith("__editable__"):
                path.unlink()
                removed.append(path.name)
        cache = site / "__pycache__"
        if cache.exists():
            for pyc in cache.glob("__editable__*"):
                pyc.unlink()
    return removed


def env_lock(python: str) -> dict:
    """Environment identity: sanitised freeze plus a site-packages manifest hash."""
    freeze = subprocess.run(["uv", "pip", "freeze", "--python", python], capture_output=True,
                            text=True).stdout
    venv = Path(python).parent.parent
    files = {}
    for site in venv.glob("lib/python*/site-packages"):
        for p in sorted(site.rglob("*")):
            if p.is_file() and "__pycache__" not in p.parts:
                files[p.relative_to(site).as_posix()] = sha256_file(p)
    version = subprocess.run([python, "-c", "import sys; print(sys.version.split()[0])"],
                             capture_output=True, text=True).stdout.strip()
    return {"python": version, "freeze_sha256": hashlib.sha256(
        "\n".join(l for l in freeze.splitlines() if not l.startswith("-e ")).encode()).hexdigest(),
        "site_packages_manifest_sha256": hashlib.sha256(
            json.dumps(files, sort_keys=True).encode()).hexdigest(),
        "site_packages_files": len(files)}


def verify_import(python: str, view: Path, module: str) -> dict:
    """Trusted attestation outside the sandbox: the module imports from the view only."""
    done = subprocess.run([python, "-B", "-c", f"import os, hashlib, {module} as m; "
                           "p = os.path.realpath(m.__file__); "
                           "print(p); print(hashlib.sha256(open(p, 'rb').read()).hexdigest())"],
                          capture_output=True, text=True, cwd="/",
                          env={"PATH": "/usr/bin:/bin", "PYTHONPATH": str(view),
                               "PYTHONDONTWRITEBYTECODE": "1", "PYTHONNOUSERSITE": "1"})
    lines = done.stdout.split()
    ok = done.returncode == 0 and len(lines) == 2 and lines[0].startswith(str(view) + os.sep)
    bare = subprocess.run([python, "-B", "-c", f"import {module}"], capture_output=True, cwd="/",
                          env={"PATH": "/usr/bin:/bin", "PYTHONNOUSERSITE": "1"})
    return {"ok": ok and bare.returncode != 0, "module_sha256": lines[1] if len(lines) == 2 else None,
            "relative": lines[0][len(str(view)) + 1:] if ok else None,
            "not_importable_without_view": bare.returncode != 0,
            "error": done.stderr[-300:] if not ok else None}


# --- sandbox execution ---------------------------------------------------------------------

def spec_for(module: str, qualname: str) -> dict:
    """The only harness data the candidate could read: identical for both revisions."""
    return {"module": module, "qualname": qualname, "top_package": module.split(".")[0],
            "per_test_seconds": LIMITS["per_test_seconds"]}


def run_sandboxed(python: str, env_dir: Path, view: Path, spec: dict, candidate: str,
                  out_dir: Path) -> dict:
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
        (work / "sandbox_spec.json").write_text(json.dumps(spec, sort_keys=True), encoding="utf-8")
        env = {"PATH": "/usr/bin:/bin", "SB_RO": str(env_dir), "SB_TARGET": str(view),
               "SB_OUT": str(out_dir), "SB_WORK_SRC": str(work),
               "SB_PYTHONPATH": "/tmp/work/plugin:/target",
               "SB_CPU": str(LIMITS["cpu_seconds"]), "SB_AS": str(LIMITS["address_space_bytes"]),
               "SB_NPROC": str(LIMITS["nproc"]), "SB_NOFILE": str(LIMITS["nofile"]),
               "SB_FSIZE": str(LIMITS["fsize_bytes"]), "SB_WALL": str(LIMITS["wall_seconds"])}
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
    path = out_dir / "report.json"
    raw = path.read_bytes() if path.exists() else b""
    report = json.loads(raw.decode("utf-8")) if raw else {"harness_error": True}
    report.update(process_exit=code, seconds=seconds, tail=tail,
                  wall_timeout=code in (124, 137, "backstop_timeout"),
                  raw_report_sha256=hashlib.sha256(raw).hexdigest() if raw else None)
    return report


# Evidence kept per report (v2.4 D): everything classify() reads, nothing host-derived.
EVIDENCE_FIELDS = ("attestation", "target_error", "collected", "collection_errors", "nodes",
                   "exitstatus", "uid", "process_exit", "seconds", "wall_timeout",
                   "harness_error", "raw_report_sha256")


def sanitise(report: dict | None) -> dict | None:
    """Sandbox-relative evidence only; the host-side process tail is dropped."""
    if report is None:
        return None
    return {k: report[k] for k in EVIDENCE_FIELDS if k in report}


def execute_candidate(target: dict, source: str, scratch: Path, enforce_policy: bool = True
                      ) -> dict:
    """Static policy, buggy and fixed runs, rerun of potential kills, classification."""
    name = target["qualname"].split(".")[-1]
    static = static_check(source, name, enforce_policy)
    expected = {label: target["module_sha256"][label] for label in ("buggy", "fixed")}
    if static["status"] != "ok":
        return {"classification": classify(static, None, None), "static": static,
                "evidence": {"static": static, "expected": expected, "reports": {}}}
    spec = spec_for(target["module"], target["qualname"])
    runs = {label: run_sandboxed(target["python"], Path(target["env_dir"]),
                                 Path(target["views"][label]), spec, source, scratch / label)
            for label in ("buggy", "fixed")}
    first = classify(static, runs["buggy"], runs["fixed"], None, expected)
    rerun = None
    if first["class"] in KILLS:
        rerun = tuple(run_sandboxed(target["python"], Path(target["env_dir"]),
                                    Path(target["views"][label]), spec, source,
                                    scratch / f"{label}_rerun") for label in ("buggy", "fixed"))
    final = classify(static, runs["buggy"], runs["fixed"], rerun, expected)
    summary = {label: {"uid": r.get("uid"), "collected": len(r.get("collected") or []),
                       "seconds": r.get("seconds"),
                       "attestation_ok": (r.get("attestation") or {}).get("module_sha256")
                       == expected[label]} for label, r in runs.items()}
    reports = {"buggy": sanitise(runs["buggy"]), "fixed": sanitise(runs["fixed"])}
    if rerun is not None:
        reports.update(rerun_buggy=sanitise(rerun[0]), rerun_fixed=sanitise(rerun[1]))
    return {"classification": final, "static": static, "runs": summary,
            "rerun": rerun is not None,
            "evidence": {"static": static, "expected": expected, "reports": reports}}


def fixed_valid_of(cls: dict) -> bool:
    return (cls.get("fixed_all_executed_passed_reached") is True
            and cls["class"] not in ("nondeterminism",)
            and (cls["class"] not in KILLS or cls.get("rerun_agrees") is True))


def canary_status(reports: dict, expected: dict) -> bool:
    """The known-good module ran, executed, passed and attested on both revisions."""
    for label in ("buggy", "fixed"):
        report = reports.get(label)
        status = revision_outcome(report, expected.get(label))
        node = ((report or {}).get("nodes") or {}).get(((report or {}).get("collected") or [""])[0], {})
        if status["status"] != "ran" or not node or not _executed(node) or _failed(node):
            return False
    return True


def verify_row(row: dict, module_source: str | None = None, target_name: str | None = None
               ) -> list:
    """Recompute a stored execution row from its retained evidence (v2.4 D). Returns the
    disagreements; an empty list means the row is internally consistent."""
    problems = []
    evidence = row.get("evidence")
    if not isinstance(evidence, dict):
        return ["no retained evidence"]
    canary = evidence.get("canary")
    if not isinstance(canary, dict) or "reports" not in canary:
        return ["no retained canary evidence"]
    canary_passed = canary_status(canary["reports"], canary.get("expected") or {})
    if row.get("canary_failed") is not (not canary_passed):
        problems.append("canary_failed disagrees with the canary evidence")
    if not canary_passed:
        if row.get("class") != "environment_failure":
            problems.append("failed canary but class is not environment_failure")
        return problems
    static, reports = evidence.get("static"), evidence.get("reports") or {}
    if not isinstance(static, dict) or "status" not in static:
        return problems + ["no static evidence"]
    if module_source is not None and target_name is not None and \
            static != static_check(module_source, target_name, True):
        problems.append("static evidence disagrees with the module")
    rerun = None
    if "rerun_buggy" in reports or "rerun_fixed" in reports:
        rerun = (reports.get("rerun_buggy"), reports.get("rerun_fixed"))
    recomputed = classify(static, reports.get("buggy"), reports.get("fixed"), rerun,
                          evidence.get("expected") or {})
    if recomputed != row.get("classification") or recomputed["class"] != row.get("class"):
        problems.append(f"class {row.get('class')!r} disagrees with evidence "
                        f"({recomputed['class']!r})")
    if recomputed["class"] in KILLS and rerun is None:
        problems.append("kill without retained rerun evidence")
    if fixed_valid_of(recomputed) is not row.get("fixed_valid"):
        problems.append("fixed_valid disagrees with evidence")
    return problems


# --- durable run over real generations ------------------------------------------------------

def canary_check(target: dict, scratch: Path) -> dict:
    """Known-good module in the same environment: proves the environment, not the model.
    Returns the verdict with its retained evidence (v2.4 D)."""
    top = target["qualname"].split(".")[0]
    good = f"from {target['module']} import {top}\n\n\ndef test_environment():\n    assert {top} is not None\n"
    spec = spec_for(target["module"], target["qualname"])
    expected = {label: target["module_sha256"][label] for label in ("buggy", "fixed")}
    reports = {label: sanitise(run_sandboxed(target["python"], Path(target["env_dir"]),
                                             Path(target["views"][label]), spec, good,
                                             scratch / f"canary_{label}"))
               for label in ("buggy", "fixed")}
    return {"ok": canary_status(reports, expected), "reports": reports, "expected": expected,
            "module_sha256": hashlib.sha256(good.encode()).hexdigest()}


def canary_ok(target: dict, scratch: Path) -> bool:
    return canary_check(target, scratch)["ok"]


GENERATION_FIELDS = ("raw_sha256", "generated_tokens", "eos_reached", "finish_reason",
                     "hit_completion_limit", "fence_stripped")


def run(prep_path: Path, manifest_path: Path, job_path: Path, arms: dict, condition: str,
        out: Path, root: Path | None = None) -> int:
    cohort = gio.resolve_cohort(job_path, manifest_path, condition)
    prepared = gio.resolve_prep(prep_path, manifest_path, root or REPO_ROOT)  # v2.4 C
    prep = prepared["rows"]
    targets = []
    for key in cohort["generation"]:                  # the generation cohort only
        row = prep[key]
        for label in ("buggy", "fixed"):
            view = build_view_hash(Path(row["views"][label]))
            if view != row["view_manifest_sha256"][label]:
                raise SystemExit(f"REFUSED: view for {key}/{label} changed since preparation")
        targets.append({"target_key": key, "module": row["module"], "qualname": row["qualname"],
                        "python": row["python_path"], "env_dir": row["env_dir"],
                        "views": row["views"], "module_sha256": row["module_sha256"]})
    gens = gio.load_arm_generations(arms, cohort)
    contract = {"design_version": DESIGN_VERSION,
                "executor_sha256": sha256_file(Path(__file__)), "inner_sha256": sha256_file(INNER),
                "io_sha256": sha256_file(HERE / "native_generation_io.py"),
                "prep": {"path": prepared["path"], "sha256": prepared["sha256"]},
                "manifest_sha256": cohort["manifest_sha256"],
                "job_file_sha256": cohort["job_file_sha256"], "job_sha256": cohort["job_sha256"],
                "cohort": {"qualified": len(cohort["qualified"]),
                           "generation_targets": cohort["generation"],
                           "pre_generation_exclusions": cohort["pre_generation_exclusions"],
                           "expected": cohort["expected"]},
                "generations_sha256": gens["files_sha256"],
                "generation_contracts_sha256": gens["contracts_sha256"],
                "generation_identity_sha256": gens["identity_sha256"],
                "telemetry_schema": gio.TELEMETRY_SCHEMA,
                "condition": condition, "limits": LIMITS, "module_limits": MODULE_LIMITS}
    chash = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    out.mkdir(parents=True, exist_ok=True)
    contract_file = out / f"execute_contract_{condition}.json"
    if contract_file.exists():
        if json.loads(contract_file.read_text(encoding="utf-8")) != contract:
            raise SystemExit("REFUSED: execution contract differs; prior results are preserved; "
                             "use a new output directory")
    else:
        contract_file.write_text(json.dumps(contract, indent=1, sort_keys=True) + "\n",
                                 encoding="utf-8")
    results = out / f"results_{condition}.jsonl"
    expected = {f"{arm}::{t['target_key']}::{s}::{slot}" for arm in ARMS for t in targets
                for s in SEEDS for slot in range(SLOTS)}
    if len(expected) != cohort["expected"]["candidates_total"]:
        raise SystemExit("REFUSED: execution grid differs from the frozen cohort size")
    done, problems = {}, []
    if results.exists():
        data = results.read_bytes()
        if data and not data.endswith(b"\n"):
            problems.append("partial final line")
        for number, line in enumerate(data.decode("utf-8").splitlines(), 1):
            try:
                row = json.loads(line)
            except ValueError:
                problems.append(f"malformed {number}")
                continue
            if row.get("contract_sha256") != chash or row.get("key") not in expected or \
                    row["key"] in done:
                problems.append(f"line {number} stale/unexpected/duplicate")
            else:
                done[row["key"]] = row
        if problems:
            qdir = out / "quarantine"
            qdir.mkdir(exist_ok=True)
            shutil.move(str(results), str(qdir / f"{int(time.time() * 1000)}_{results.name}"))
            done = {}
    scratch_root = Path(tempfile.mkdtemp(prefix="oneiros_exec_"))
    try:
        with results.open("a", encoding="utf-8") as handle:
            for t in targets:
                canary = canary_check(t, scratch_root / t["target_key"].replace("/", "_"))
                env_ok = canary["ok"]
                for arm in ARMS:
                    for seed in SEEDS:
                        grow = gens["rows"][(arm, t["target_key"], seed)]
                        for slot, cand in enumerate(grow["candidates"]):
                            key = f"{arm}::{t['target_key']}::{seed}::{slot}"
                            if key in done:
                                continue
                            module = cand["module"]
                            if env_ok:
                                outcome = execute_candidate(t, module, scratch_root / "c")
                            else:
                                outcome = {"classification": {"class": "environment_failure"},
                                           "canary_failed": True, "evidence": {}}
                            cls = outcome["classification"]
                            handle.write(json.dumps({
                                "key": key, "contract_sha256": chash, "arm": arm, "seed": seed,
                                "target_key": t["target_key"], "slot": slot,
                                "module_sha256": hashlib.sha256(module.encode()).hexdigest(),
                                "generation": {**{f: cand[f] for f in GENERATION_FIELDS},
                                               "prompt_tokens": grow["prompt_tokens"],
                                               "target_seed": grow["target_seed"],
                                               "row_wall_seconds": grow["wall_seconds"]},
                                "class": cls["class"], "classification": cls,
                                "static": outcome.get("static"), "runs": outcome.get("runs"),
                                "canary_failed": not env_ok,
                                "evidence": {**outcome["evidence"], "canary": canary},
                                "fixed_valid": fixed_valid_of(cls)},
                                sort_keys=True) + "\n")
                            handle.flush()
                            os.fsync(handle.fileno())
    finally:
        shutil.rmtree(scratch_root, ignore_errors=True)
    rows = [json.loads(l) for l in results.read_text(encoding="utf-8").splitlines() if l.strip()]
    keys = {r["key"] for r in rows}
    complete = keys == expected and len(rows) == len(expected)
    print(json.dumps({"rows": len(rows), "expected": len(expected), "complete": complete,
                      "qualified": len(cohort["qualified"]),
                      "generation_targets": len(cohort["generation"]),
                      "pre_generation_excluded": [e["target_key"] for e in
                                                  cohort["pre_generation_exclusions"]],
                      "results_sha256": sha256_file(results)}))
    return 0 if complete else 1


def build_view_hash(view: Path) -> str:
    files = {p.relative_to(view).as_posix(): sha256_file(p) for p in sorted(view.rglob("*"))
             if p.is_file()}
    return hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()


# --- synthetic toy target (canaries and the pipeline test) ----------------------------------

TOY_BUGGY = '''
def add(a, b):
    return a - b
def crash(x):
    return 10 // (x - 1)
'''
TOY_FIXED = TOY_BUGGY.replace("return a - b", "return a + b").replace("return 10 // (x - 1)",
                                                                       "return 10")


def build_toy_repo(root: Path) -> dict:
    """A git repository with a buggy and a fixed commit, test clutter and .git metadata."""
    repo = root / "toy_repo"
    (repo / "src" / "toy" / "sub" / "tests").mkdir(parents=True)
    files = {
        "pyproject.toml": '[project]\nname = "toy"\nversion = "0.1"\nrequires-python = ">=3.8"\n'
                          '[build-system]\nrequires = ["setuptools"]\nbuild-backend = '
                          '"setuptools.build_meta"\n[tool.setuptools.packages.find]\n'
                          'where = ["src"]\n',
        "src/toy/__init__.py": "", "src/toy/core.py": TOY_BUGGY,
        "src/toy/conftest.py": "OFFICIAL_CONFTEST = 1\n",
        "src/toy/test_scattered.py": "def test_x():\n    pass\n",
        "src/toy/core_test.py": "def test_y():\n    pass\n",
        "src/toy/sub/__init__.py": "", "src/toy/sub/tests/test_nested.py": "SECRET = 1\n",
        "tests/test_official.py": "from toy.core import add\n\n\ndef test_add():\n"
                                  "    assert add(2, 3) == 5\n",
    }
    for rel, text in files.items():
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        (repo / rel).write_text(text, encoding="utf-8")
    def g(*a):
        subprocess.run(["git", "-C", str(repo), *a], check=True, capture_output=True)
    g("init", "-q")
    g("-c", "user.email=t@t", "-c", "user.name=t", "add", "-A")
    g("-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "buggy")
    buggy = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True,
                           text=True).stdout.strip()
    (repo / "src/toy/core.py").write_text(TOY_FIXED, encoding="utf-8")
    g("-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qam", "fixed")
    fixed = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True,
                           text=True).stdout.strip()
    return {"repo": repo, "buggy_commit": buggy, "fixed_commit": fixed}


def prepare_toy(root: Path, qualname: str = "add") -> dict:
    """Exactly the formal preparation code path, on the toy repository."""
    sys.path.insert(0, str(HERE))
    from native_rehearsal_prepare_wsl import prepare_target
    toy = build_toy_repo(root)
    target = {"key": f"cand:synthetic/toy@{toy['fixed_commit']}", "repository": "synthetic/toy",
              "repository_url": f"file://{toy['repo']}", "buggy_commit": toy["buggy_commit"],
              "fixed_commit": toy["fixed_commit"], "target": qualname,
              "target_file": "src/toy/core.py", "regression_test_files": ["tests/test_official.py"],
              "difference_exposing_tests": ["tests.test_official::test_add"]}
    row = prepare_target(target, toy["repo"], root / "prep", root / "exports", keep=True)
    return {"target": target, "prep": row}


def synthetic_prepare(out_dir: Path) -> int:
    """Toy cohort for the end-to-end pipeline test, prepared by the FORMAL preparation path.

    Writes records.jsonl (prep rows), manifest.json and exports/{buggy_view,verifier} to
    ``out_dir``; environments and views stay under /root.
    """
    sys.path.insert(0, str(HERE))
    from native_rehearsal_prepare_wsl import prepare_target
    root = Path("/root/oneiros_native_v21_synthetic")
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in ("records.jsonl", "manifest.json"):
        if (out_dir / stale).exists():
            (out_dir / stale).unlink()
    if (out_dir / "exports").exists():
        shutil.rmtree(out_dir / "exports")
    toy = build_toy_repo(root)
    targets, rows = [], []
    for qualname in ("add", "crash"):
        target = {"key": f"cand:synthetic/toy@{toy['fixed_commit']}-{qualname}",
                  "repository": "synthetic/toy", "repository_url": f"file://{toy['repo']}",
                  "buggy_commit": toy["buggy_commit"], "fixed_commit": toy["fixed_commit"],
                  "target": qualname, "target_file": "src/toy/core.py",
                  "regression_test_files": ["tests/test_official.py"],
                  "difference_exposing_tests": ["tests.test_official::test_add"]}
        rows.append(prepare_target(target, toy["repo"], root / "prep", out_dir / "exports"))
        targets.append(target)
    with (out_dir / "records.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps({**row, "contract_sha256": "synthetic"}, sort_keys=True) + "\n")
    kept = [r["key"] for r in rows if r["category"] == "requalified"]
    (out_dir / "manifest.json").write_text(json.dumps(
        {"kept_targets": kept, "targets": targets, "synthetic": True}, indent=1), encoding="utf-8")
    print(json.dumps({"kept": kept, "categories": [r["category"] for r in rows]}))
    return 0 if len(kept) == 2 else 1


CANARIES = {
    # taxonomy (policy on)
    "semantic_kill": ("add", "from toy.core import add\ndef test_a():\n    assert add(2, 3) == 5\n", True),
    "crash_kill": ("crash", "from toy.core import crash\ndef test_c():\n    assert crash(1) == 10\n", True),
    "pass_both": ("add", "from toy.core import add\ndef test_p():\n    assert add(2, 0) == 2\n", True),
    "fail_both": ("add", "from toy.core import add\ndef test_f():\n    add(1, 1)\n    assert False\n", True),
    "inverted": ("add", "from toy.core import add\ndef test_i():\n    assert add(2, 3) == -1\n", True),
    "skip": ("add", "import pytest\nfrom toy.core import add\n@pytest.mark.skip\ndef test_s():\n    assert add(1, 1) == 2\n", True),
    "mixed_skip": ("add", "import pytest\nfrom toy.core import add\ndef test_k():\n    assert add(2, 3) == 5\n@pytest.mark.skip\ndef test_s():\n    assert add(1, 1) == 2\n", True),
    "xfail": ("add", "import pytest\nfrom toy.core import add\n@pytest.mark.xfail\ndef test_x():\n    assert add(2, 3) == 5\n", True),
    "timeout": ("add", "from toy.core import add\ndef test_t():\n    add(1, 1)\n    while True:\n        pass\n", True),
    "fabricated_import": ("add", "from toy.core import add_numbers\ndef test_m():\n    assert add_numbers(1, 2) == 3\n", True),
    "zero_tests": ("add", "from toy.core import add\nVALUE = add(1, 2)\n", True),
    "not_reached": ("add", "def test_n():\n    assert 1 + 1 == 2\n", True),
    "partly_not_reached": ("add", "from toy.core import add\ndef test_a():\n    assert add(2, 3) == 5\ndef test_b():\n    assert 1 == 1\n", True),
    "syntax": ("add", "def test_(:\n    pass\n", True),
    # policy refusals (static; never executed)
    "policy_open": ("add", "from toy.core import add\ndef test_o():\n    open('/tmp/work/sandbox_spec.json').read()\n", True),
    "policy_file": ("add", "import toy.core\ndef test_f():\n    assert 'buggy' not in toy.core.__file__\n", True),
    "policy_code": ("add", "from toy.core import add\ndef test_c():\n    assert add.__code__.co_consts\n", True),
    "policy_inspect": ("add", "import inspect\nfrom toy.core import add\ndef test_i():\n    assert '-' not in inspect.getsource(add)\n", True),
    "policy_dis": ("add", "import dis\nfrom toy.core import add\ndef test_d():\n    dis.dis(add)\n", True),
    "policy_sys_modules": ("add", "import sys\ndef test_s():\n    assert sys.modules\n", True),
    "policy_subprocess": ("add", "import subprocess\ndef test_p():\n    subprocess.run(['id'])\n", True),
    # sandbox indistinguishability (policy OFF): must never be a kill
    "probe_spec": ("add", "from toy.core import add\ndef test_s():\n    add(1, 1)\n    assert 'buggy' not in open('/tmp/work/sandbox_spec.json').read()\n", False),
    "probe_file": ("add", "import toy.core\nfrom toy.core import add\ndef test_f():\n    add(1, 1)\n    assert toy.core.__file__ == '/target/toy/core.py'\n", False),
    "probe_sys_modules": ("add", "import sys\nfrom toy.core import add\ndef test_m():\n    add(1, 1)\n    assert not any('buggy' in str(getattr(m, '__file__', '')) or 'fixed' in str(getattr(m, '__file__', '')) for m in list(sys.modules.values()))\n", False),
    "probe_git": ("add", "import os\nfrom toy.core import add\ndef test_g():\n    add(1, 1)\n    assert not any(os.path.exists(p) for p in ('/target/.git', '/target/toy/.git', '/target/../.git'))\n", False),
    "probe_tests": ("add", "import os\nfrom toy.core import add\ndef test_t():\n    add(1, 1)\n    names = [n for _, _, fs in os.walk('/target') for n in fs]\n    assert not any(n == 'conftest.py' or n.startswith('test_') or n.endswith('_test.py') for n in names)\n    assert not os.path.exists('/target/toy/sub/tests')\n", False),
    "probe_host_paths": ("add", "import os\nfrom toy.core import add\ndef test_h():\n    add(1, 1)\n    assert not any(os.path.exists(p) for p in ('/mnt/c/Users', '/root/.oneiros_decoy/secret.txt', '/root/.ssh'))\n", False),
    # containment (policy OFF): must fail on both revisions (never a kill)
    "network": ("add", "import socket\nfrom toy.core import add\ndef test_net():\n    add(1, 1)\n    socket.create_connection(('1.1.1.1', 80), timeout=2)\n", False),
    "subprocess": ("add", "import subprocess\nfrom toy.core import add\ndef test_sp():\n    add(1, 1)\n    subprocess.run(['id'])\n", False),
    "write_target": ("add", "import toy.core\nfrom toy.core import add\ndef test_w():\n    add(1, 1)\n    open(toy.core.__file__, 'a').write('#x')\n", False),
    "uid": ("add", "import os\nfrom toy.core import add\ndef test_u():\n    add(1, 1)\n    assert os.getuid() == 0\n", False),
    "shadow": ("add", "from toy.core import add\ndef test_sh():\n    add(1, 1)\n    open('/etc/shadow').read()\n", False),
    "big_file": ("add", "from toy.core import add\ndef test_b():\n    add(1, 1)\n    open('/tmp/work/big', 'wb').write(b'0' * (200 * 2 ** 20))\n", False),
    "memory": ("add", "from toy.core import add\ndef test_mem():\n    add(1, 1)\n    x = bytearray(8 * 2 ** 30)\n", False),
}
EXPECTED = {
    "semantic_kill": "semantic_kill", "crash_kill": "crash_kill", "pass_both": "pass_both",
    "fail_both": "fixed_side_failure", "inverted": "fixed_side_failure",
    "skip": "skipped_or_xfail", "mixed_skip": "skipped_or_xfail", "xfail": "skipped_or_xfail",
    "timeout": "timeout", "fabricated_import": "fabricated_import",
    "zero_tests": "no_tests_collected", "not_reached": "target_not_reached",
    "partly_not_reached": "target_not_reached", "syntax": "syntax_failure",
    **{k: "policy_refused" for k in CANARIES if k.startswith("policy_")},
    **{k: "pass_both" for k in CANARIES if k.startswith("probe_")},
    **{k: "fixed_side_failure" for k in ("network", "subprocess", "write_target", "uid",
                                         "shadow", "big_file", "memory")},
}


def canaries(out_dir: Path) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix="oneiros_canary_", dir="/root"))
    decoy = Path("/root/.oneiros_decoy")
    decoy.mkdir(exist_ok=True)
    (decoy / "secret.txt").write_text("decoy credential\n", encoding="utf-8")
    results, views = {}, {}
    try:
        toys = {q: prepare_toy(root / q, q) for q in ("add", "crash")}
        for name, (qualname, source, policy) in CANARIES.items():
            row = toys[qualname]["prep"]
            if row["category"] != "requalified" and qualname == "add":
                raise SystemExit(f"toy preparation failed: {row}")
            target = {"target_key": name, "module": row["module"], "qualname": row["qualname"],
                      "python": row["python_path"], "env_dir": row["env_dir"],
                      "views": row["views"], "module_sha256": row["module_sha256"]}
            outcome = execute_candidate(target, source, root / "runs" / name, policy)
            results[name] = outcome
            observed = outcome["classification"]["class"]
            print(f"{name:20s} -> {observed}", flush=True)
            if observed != EXPECTED[name]:
                print(json.dumps(outcome, default=str)[:1200], flush=True)
        views = {q: toys[q]["prep"]["view_manifest_sha256"] for q in toys}
        wrong = dict(toys["add"]["prep"])
        wrong_target = {"target_key": "wrong", "module": wrong["module"], "qualname": "add",
                        "python": wrong["python_path"], "env_dir": wrong["env_dir"],
                        "views": {"buggy": wrong["views"]["fixed"], "fixed": wrong["views"]["fixed"]},
                        "module_sha256": wrong["module_sha256"]}
        wrong_outcome = execute_candidate(wrong_target, CANARIES["pass_both"][1], root / "runs" / "wrong")
    finally:
        shutil.rmtree(root, ignore_errors=True)
        shutil.rmtree(decoy, ignore_errors=True)
    checks = {name: results[name]["classification"]["class"] == EXPECTED[name] for name in CANARIES}
    checks["ran_as_nobody"] = all(run["uid"] == 65534 for r in results.values()
                                  for run in (r.get("runs") or {}).values() if run.get("uid") is not None)
    checks["wrong_revision_refused"] = wrong_outcome["classification"]["class"] == "harness_failure"
    checks["no_kill_from_any_probe"] = not any(
        results[n]["classification"]["class"] in KILLS for n in CANARIES
        if n.startswith(("probe_", "policy_")) or EXPECTED[n] == "fixed_side_failure")
    receipt = {"schema_version": "oneiros_native_sandbox_canaries_v2",
               "design_version": DESIGN_VERSION,
               "executor_sha256": sha256_file(Path(__file__)), "inner_sha256": sha256_file(INNER),
               "prepare_sha256": sha256_file(HERE / "native_rehearsal_prepare_wsl.py"),
               "limits": LIMITS, "module_limits": MODULE_LIMITS, "view_manifests": views,
               "results": {n: {"expected": EXPECTED[n],
                               "observed": r["classification"]["class"]} for n, r in results.items()},
               "checks": checks, "passed": all(checks.values()),
               "nondeterminism": "covered by classifier unit tests (fresh filesystem per run)"}
    import receipt_sanitize                             # tracked receipt: no user paths
    receipt = receipt_sanitize.scrub_json(receipt)
    (out_dir / "canary_receipt_v2.json").write_text(json.dumps(receipt, indent=1, sort_keys=True)
                                                    + "\n", encoding="utf-8")
    print(json.dumps({"passed": receipt["passed"],
                      "failed": [k for k, v in checks.items() if not v]}, indent=1))
    return 0 if receipt["passed"] else 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    c = sub.add_parser("canaries")
    c.add_argument("out_dir")
    s = sub.add_parser("synthetic-prepare")
    s.add_argument("out_dir")
    r = sub.add_parser("run")
    r.add_argument("--prep", required=True)
    r.add_argument("--manifest", required=True)
    r.add_argument("--job", required=True)
    r.add_argument("--generations", default=None, help="root containing base/ and sft/")
    r.add_argument("--base-generations", default=None)
    r.add_argument("--sft-generations", default=None)
    r.add_argument("--condition", required=True, choices=("primary_whole_module",))
    r.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    if args.command == "canaries":
        return canaries(Path(args.out_dir))
    if args.command == "synthetic-prepare":
        return synthetic_prepare(Path(args.out_dir))
    opt = lambda value: Path(value) if value else None  # noqa: E731
    arms = gio.arm_paths(opt(args.generations), opt(args.base_generations),
                         opt(args.sft_generations), args.condition)
    return run(Path(args.prep), Path(args.manifest), Path(args.job), arms, args.condition,
               Path(args.out))


if __name__ == "__main__":
    raise SystemExit(main())
