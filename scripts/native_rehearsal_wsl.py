"""Bounded native rehearsal (runs INSIDE WSL with uv-managed CPython; stdlib only).

For each target of the frozen manifest:

1. clone the repository once (blob-less), check out the buggy (parent) and fixed
   commits as separate worktrees;
2. build a uv virtual environment (Python 3.13, falling back to 3.10: decision D6),
   install the project from the FIXED checkout with its declared test extras /
   dependency groups / test requirement files, plus pytest;
3. run the FIXED revision's regression test file(s) against the buggy source (the
   test files are copied into the buggy checkout, as in the BugsInPy protocol) and
   against the fixed source, each with the package reinstalled editable from that
   checkout (``--no-deps``); results come from JUnit XML;
4. fixed-call probe: calls to the target function with literal-only arguments found in
   the regression test are executed on both revisions; a "safe fixed call" is one whose
   fixed result is a short round-trippable literal and differs from the buggy result;
5. record timings and exactly one category.  ENVIRONMENT failures (clone, install,
   pytest could not run) are never counted as semantic negatives.

The verifier alone sees fixed code, patches and gold tests; nothing here involves a
model, and nothing is written into any prompt or training input.
"""
from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
import tomllib
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor

WORK = Path("/root/oneiros_rehearsal")
PYTHONS = ("3.13", "3.10")
INSTALL_TIMEOUT = 900
TEST_TIMEOUT = 900
PROBE_TIMEOUT = 60
EXTRA_NAMES = ("test", "tests", "testing", "dev")
REQUIREMENT_FILES = ("requirements-test.txt", "requirements-tests.txt", "test-requirements.txt",
                     "requirements/test.txt", "requirements/tests.txt",
                     "requirements/testing.txt", "requirements-dev.txt", "dev-requirements.txt",
                     "requirements/dev.txt")
LOCK = threading.Lock()


def run(cmd, cwd=None, timeout=600, env=None):
    started = time.time()
    try:
        done = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout,
                              env=env)
        code, out = done.returncode, (done.stdout + done.stderr)
    except subprocess.TimeoutExpired as exc:
        code, out = "timeout", str(exc)
    return {"code": code, "seconds": round(time.time() - started, 1), "tail": out[-1500:]}


def group_requirements(pyproject: dict, name: str, seen=None) -> list[str]:
    seen = seen or set()
    if name in seen:
        return []
    seen.add(name)
    out = []
    for item in (pyproject.get("dependency-groups") or {}).get(name, []):
        if isinstance(item, str):
            out.append(item)
        elif isinstance(item, dict) and "include-group" in item:
            out += group_requirements(pyproject, item["include-group"], seen)
    return out


def install_plan(checkout: Path) -> dict:
    plan = {"extras": [], "groups": [], "requirement_files": []}
    pyproject_path = checkout / "pyproject.toml"
    pyproject = {}
    if pyproject_path.exists():
        pyproject = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
        optional = (pyproject.get("project") or {}).get("optional-dependencies") or {}
        plan["extras"] = [e for e in EXTRA_NAMES if e in optional]
        groups = pyproject.get("dependency-groups") or {}
        plan["groups"] = [g for g in EXTRA_NAMES if g in groups]
    plan["requirement_files"] = [f for f in REQUIREMENT_FILES if (checkout / f).exists()]
    plan["group_requirements"] = sorted({r for g in plan["groups"]
                                         for r in group_requirements(pyproject, g)})
    return plan


def build_env(env_dir: Path, fixed: Path) -> dict:
    plan = install_plan(fixed)
    attempts = []
    for py in PYTHONS:
        if env_dir.exists():
            shutil.rmtree(env_dir)
        venv = run(["uv", "venv", "--python", py, str(env_dir)], timeout=300)
        if venv["code"] != 0:
            attempts.append({"python": py, "step": "venv", **venv})
            continue
        python = str(env_dir / "bin" / "python")
        spec = str(fixed) + (f"[{','.join(plan['extras'])}]" if plan["extras"] else "")
        cmd = ["uv", "pip", "install", "--python", python, "-e", spec, "pytest"]
        cmd += plan["group_requirements"]
        for req in plan["requirement_files"]:
            cmd += ["-r", str(fixed / req)]
        result = run(cmd, timeout=INSTALL_TIMEOUT)
        attempts.append({"python": py, "step": "install", "command_kind": plan, **result})
        if result["code"] == 0:
            return {"ok": True, "python": py, "python_path": python, "attempts": attempts}
    return {"ok": False, "attempts": attempts}


def junit(path: Path) -> dict:
    if not path.exists():
        return {}
    out = {}
    for case in ET.parse(path).getroot().iter("testcase"):
        node = f"{case.get('classname')}::{case.get('name')}"
        tags = {child.tag for child in case}
        out[node] = ("failed" if tags & {"failure", "error"} else
                     "skipped" if "skipped" in tags else "passed")
    return out


def run_tests(python: str, checkout: Path, tests: list[str], label: str, outdir: Path) -> dict:
    xml = outdir / f"{label}.xml"
    cmd = [python, "-m", "pytest", "-p", "no:cacheprovider", "-q", "-o", "addopts=",
           f"--junitxml={xml}", *tests]
    result = run(cmd, cwd=checkout, timeout=TEST_TIMEOUT)
    cases = junit(xml)
    return {"exit": result["code"], "seconds": result["seconds"], "cases": cases,
            "counts": {s: sum(1 for v in cases.values() if v == s)
                       for s in ("passed", "failed", "skipped")},
            "tail": result["tail"][-600:]}


def module_name(target_file: str) -> str:
    parts = Path(target_file).with_suffix("").parts
    if parts and parts[0] in ("src", "lib"):
        parts = parts[1:]
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def literal_calls(test_sources: list[str], target: str, limit: int = 8) -> list[dict]:
    calls, seen = [], set()
    for source in test_sources:
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = node.func.id if isinstance(node.func, ast.Name) else (
                node.func.attr if isinstance(node.func, ast.Attribute) else None)
            if name != target:
                continue
            try:
                args = [ast.literal_eval(a) for a in node.args]
                kwargs = {k.arg: ast.literal_eval(k.value) for k in node.keywords if k.arg}
            except (ValueError, SyntaxError, TypeError):
                continue
            if any(k.arg is None for k in node.keywords):
                continue
            key = repr((args, kwargs))
            if key not in seen:
                seen.add(key)
                calls.append({"source": ast.unparse(node)[:200], "args": args, "kwargs": kwargs})
            if len(calls) >= limit:
                return calls
    return calls


PROBE = r"""
import importlib, json, sys
spec = json.loads(sys.argv[1])
out = []
try:
    fn = getattr(importlib.import_module(spec["module"]), spec["target"])
except Exception as exc:
    print(json.dumps({"import_error": type(exc).__name__ + ": " + str(exc)[:200]})); sys.exit(0)
for call in spec["calls"]:
    try:
        value = fn(*call["args"], **call["kwargs"])
        out.append(["ok", repr(value)[:500]])
    except Exception as exc:
        out.append(["error", type(exc).__name__])
print(json.dumps({"results": out}))
"""


def probe(python: str, cwd: Path, module: str, target: str, calls: list[dict]) -> dict:
    spec = json.dumps({"module": module, "target": target,
                       "calls": [{"args": c["args"], "kwargs": c["kwargs"]} for c in calls]},
                      default=repr)
    result = run([python, "-c", PROBE, spec], cwd=cwd, timeout=PROBE_TIMEOUT)
    try:
        return json.loads(result["tail"].strip().splitlines()[-1])
    except (ValueError, IndexError):
        return {"probe_error": result["tail"][-300:]}


def round_trips(text: str) -> bool:
    try:
        return repr(ast.literal_eval(text)) == text and len(text) <= 120
    except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
        return False


def rehearse(target: dict, repo_dir: Path, out_path: Path) -> None:
    started = time.time()
    tag = target["key"].replace("/", "__").replace("@", "_").replace("cand:", "")[:80]
    wt = WORK / "wt" / tag
    record = {"key": target["key"], "repository": target["repository"],
              "buggy_commit": target["buggy_commit"], "fixed_commit": target["fixed_commit"],
              "target": target["target"], "target_file": target["target_file"],
              "tier": target["tier"], "bug_family": target["bug_family_heuristic"],
              "regression_test_files": target["regression_test_files"], "steps": {}}
    try:
        for label, commit in (("fixed", target["fixed_commit"]), ("buggy", target["buggy_commit"])):
            step = run(["git", "-C", str(repo_dir), "worktree", "add", "--force", "--detach",
                        str(wt / label), commit], timeout=900)
            record["steps"][f"worktree_{label}"] = {k: step[k] for k in ("code", "seconds")}
            if step["code"] != 0:
                record.update(category="checkout_failed", environment_failure=True,
                              detail=step["tail"][-400:])
                return
        fixed, buggy = wt / "fixed", wt / "buggy"
        env = build_env(wt / "env", fixed)
        record["steps"]["environment"] = {"ok": env["ok"], "python": env.get("python"),
                                          "attempts": [{k: a.get(k) for k in
                                                        ("python", "step", "code", "seconds",
                                                         "command_kind")}
                                                       for a in env["attempts"]],
                                          "seconds": round(sum(a.get("seconds", 0)
                                                               for a in env["attempts"]), 1)}
        if not env["ok"]:
            record.update(category="environment_install_failed", environment_failure=True,
                          detail=env["attempts"][-1].get("tail", "")[-400:])
            return
        python = env["python_path"]
        tests = [t for t in target["regression_test_files"] if (fixed / t).exists()]
        if not tests:
            record.update(category="no_regression_test_in_fixed_revision",
                          environment_failure=False)
            return
        for t in tests:
            (buggy / t).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(fixed / t, buggy / t)
        sources = [(fixed / t).read_text(encoding="utf-8", errors="replace") for t in tests]
        calls = literal_calls(sources, target["target"])
        module = module_name(target["target_file"])
        outdir = wt / "junit"
        outdir.mkdir(parents=True, exist_ok=True)
        results = {}
        probes = {}
        for label, checkout in (("buggy", buggy), ("fixed", fixed)):
            switch = run(["uv", "pip", "install", "--python", python, "--no-deps", "-e",
                          str(checkout)], timeout=INSTALL_TIMEOUT)
            record["steps"][f"switch_{label}"] = {k: switch[k] for k in ("code", "seconds")}
            if switch["code"] != 0:
                record.update(category="environment_install_failed", environment_failure=True,
                              detail=switch["tail"][-400:])
                return
            results[label] = run_tests(python, checkout, tests, label, outdir)
            probes[label] = probe(python, checkout, module, target["target"], calls) \
                if calls else {"results": []}
        record["tests"] = {label: {k: r[k] for k in ("exit", "seconds", "counts")}
                           for label, r in results.items()}
        runnable = all(results[l]["exit"] in (0, 1) and results[l]["cases"]
                       for l in ("buggy", "fixed"))
        if not runnable:
            record.update(category="test_infrastructure_error", environment_failure=True,
                          detail={l: results[l]["tail"][-300:] for l in results})
            return
        exposing = sorted(n for n, s in results["buggy"]["cases"].items()
                          if s == "failed" and results["fixed"]["cases"].get(n) == "passed")
        record["difference_exposing_tests"] = exposing[:20]
        record["difference_exposing_count"] = len(exposing)
        call_rows = []
        b, f = probes["buggy"].get("results"), probes["fixed"].get("results")
        if b is not None and f is not None:
            for c, rb, rf in zip(calls, b, f):
                safe = rf[0] == "ok" and round_trips(rf[1]) and rb != rf
                call_rows.append({"call": c["source"], "buggy": rb, "fixed": rf,
                                  "safe_fixed_call": safe})
        record["fixed_call_probe"] = {
            "module": module, "literal_calls_found": len(calls),
            "import_error": probes["fixed"].get("import_error") or probes["buggy"].get(
                "import_error"),
            "calls": call_rows,
            "safe_fixed_call_constructed": any(r["safe_fixed_call"] for r in call_rows)}
        if exposing:
            record.update(category="natively_qualified", environment_failure=False)
        else:
            record.update(category="official_tests_do_not_distinguish",
                          environment_failure=False)
    except Exception as exc:          # recorded, never silently dropped
        record.update(category="runner_error", environment_failure=True,
                      detail=f"{type(exc).__name__}: {exc}"[:400])
    finally:
        record["wall_seconds"] = round(time.time() - started, 1)
        with LOCK:
            with out_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
            print(f"[{record['wall_seconds']:7.1f}s] {record.get('category'):40s} "
                  f"{target['key']}", flush=True)
        shutil.rmtree(wt, ignore_errors=True)
        run(["git", "-C", str(repo_dir), "worktree", "prune"], timeout=120)


def rehearse_repository(repository: str, url: str, targets: list[dict], out_path: Path,
                        clone_log: dict) -> None:
    repo_dir = WORK / "repos" / repository.replace("/", "__")
    if not repo_dir.exists():
        result = run(["git", "clone", "--filter=blob:none", "--no-checkout", url, str(repo_dir)],
                     timeout=1800)
        clone_log[repository] = {k: result[k] for k in ("code", "seconds")}
        if result["code"] != 0:
            for target in targets:
                with LOCK:
                    with out_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps({"key": target["key"],
                                                 "repository": repository,
                                                 "category": "clone_failed",
                                                 "environment_failure": True,
                                                 "wall_seconds": result["seconds"],
                                                 "detail": result["tail"][-300:]}) + "\n")
            return
    for target in targets:
        rehearse(target, repo_dir, out_path)


def main() -> int:
    manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    out_dir = Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "records.jsonl"
    done = set()
    if out_path.exists():
        done = {json.loads(line)["key"] for line in out_path.read_text().splitlines() if line}
    (WORK / "repos").mkdir(parents=True, exist_ok=True)
    by_repo: dict[str, list[dict]] = {}
    for target in manifest["targets"]:
        if target["key"] not in done:
            by_repo.setdefault(target["repository"], []).append(target)
    urls = {t["repository"]: t["repository_url"] for t in manifest["targets"]}
    clone_log: dict = {}
    started = time.time()
    print(f"rehearsing {sum(len(v) for v in by_repo.values())} targets in {len(by_repo)} "
          f"repositories ({len(done)} already done)", flush=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        for repository, targets in by_repo.items():
            pool.submit(rehearse_repository, repository, urls[repository], targets, out_path,
                        clone_log)
    (out_dir / "run_summary.json").write_text(json.dumps({
        "wall_seconds": round(time.time() - started, 1), "clone_log": clone_log,
        "uv": run(["uv", "--version"])["tail"].strip(),
        "pythons": {p: run(["uv", "python", "find", p])["tail"].strip() for p in PYTHONS},
        "host": run(["uname", "-a"])["tail"].strip(), "cpus": os.cpu_count()}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
