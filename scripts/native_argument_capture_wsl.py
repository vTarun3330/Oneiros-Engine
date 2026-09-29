"""Bounded argument-capture pilot (runs INSIDE WSL with uv-managed CPython; stdlib only).

Choice A's blocker: literal calls in regression-test SOURCE gave 0/24 safe fixed calls,
because tests reach targets through parametrisation, objects and wrappers. This pilot
captures the target's ACTUAL arguments while the difference-exposing official tests run.

Per natively qualified target:

1. check out fixed and buggy worktrees, build the environment from the fixed checkout
   (native_rehearsal_wsl.build_env, which also records the dependency resolution);
2. CAPTURE on the fixed revision: a pytest plugin installs a profile hook only while a
   difference-exposing test runs, and matches frames by the target's code object (so the
   import binding does not matter). Arguments are serialised at call entry, before the
   body can mutate them, and only if every value is a supported literal (None, bool, int,
   finite float, str, bytes, list, tuple, dict, set) of exact type, within depth, item and
   size bounds, and ``ast.literal_eval(repr(v)) == v``. Anything else - objects, handles,
   generators, modules, subclasses, opaque state - is rejected with a reason. No pickle.
3. REPLAY each unique captured call in a fresh subprocess per revision, reconstructing the
   arguments independently from their literal text for each revision; functions,
   staticmethods and classmethods are replayable, instance methods are not (the receiver
   is object state) and are recorded as such;
4. keep a call as a short verified oracle only if the fixed revision returns a short
   round-trippable literal and the buggy revision differs (other value or an exception).

The fixed implementation and gold tests are verifier-only; nothing here involves a model,
and a kept oracle is only ``call`` plus ``value``.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from native_rehearsal_wsl import WORK, build_env, module_name, run  # noqa: E402

CAPTURE_TIMEOUT = 900
REPLAY_TIMEOUT = 300
LIMITS = {"max_depth": 4, "max_items": 64, "max_chars": 500, "max_calls_per_target": 200,
          "max_unique_replayed": 60, "oracle_max_chars": 120, "per_call_seconds": 5}
LOCK = threading.Lock()

COMMON = r'''
import ast, importlib, inspect, math

class Unsupported(Exception):
    pass

SCALARS = (type(None), bool, int, str, bytes)

def literal(value, limits, depth=0, count=None):
    count = count if count is not None else [0]
    count[0] += 1
    if depth > limits["max_depth"]:
        raise Unsupported("too_deep")
    if count[0] > limits["max_items"]:
        raise Unsupported("too_many_items")
    kind = type(value)
    if kind in SCALARS:
        pass
    elif kind is float:
        if not math.isfinite(value):
            raise Unsupported("non_finite_float")
    elif kind in (list, tuple, set):
        for item in value:
            literal(item, limits, depth + 1, count)
    elif kind is dict:
        for key, item in value.items():
            literal(key, limits, depth + 1, count)
            literal(item, limits, depth + 1, count)
    else:
        raise Unsupported("type:" + kind.__module__ + "." + kind.__qualname__)
    if depth == 0:
        text = repr(value)
        if len(text) > limits["max_chars"]:
            raise Unsupported("too_large")
        try:
            back = ast.literal_eval(text)
        except Exception:
            raise Unsupported("not_round_trippable")
        if back != value or repr(back) != text:
            raise Unsupported("not_round_trippable")
        return text
    return None

def resolve(module, qualname):
    obj = importlib.import_module(module)
    parts = qualname.split(".")
    parent = obj
    for part in parts[:-1]:
        parent = getattr(parent, part)
    static = inspect.getattr_static(parent, parts[-1])
    if inspect.ismodule(parent):
        kind, func = "function", static
    elif isinstance(static, staticmethod):
        kind, func = "staticmethod", static.__func__
    elif isinstance(static, classmethod):
        kind, func = "classmethod", static.__func__
    elif inspect.isfunction(static):
        kind, func = "instance_method", static
    else:
        kind, func = "unsupported:" + type(static).__name__, None
    code = getattr(inspect.unwrap(func), "__code__", None) if func is not None else None
    return kind, parent, parts[-1], code
'''

PLUGIN = COMMON + r'''
import json, os, sys, threading
from collections import Counter

SPEC = json.loads(os.environ["ONEIROS_CAPTURE_SPEC"])
WANTED = set(SPEC["wanted"])
STATE = {"test": None, "observed": 0, "calls": [], "keys": set(), "rejected": Counter(),
         "per_test": Counter(), "kind": None, "resolve_error": None}
TARGET = {"code": None}

def junit_id(nodeid):
    parts = nodeid.split("::")
    module = parts[0][:-3].replace("/", ".") if parts[0].endswith(".py") else parts[0]
    return ".".join([module] + parts[1:-1]) + "::" + parts[-1]

def arguments(frame, kind):
    code = frame.f_code
    names = code.co_varnames
    loc = frame.f_locals
    positional = [loc[n] for n in names[:code.co_argcount]]
    keywords = {n: loc[n] for n in names[code.co_argcount:code.co_argcount + code.co_kwonlyargcount]}
    index = code.co_argcount + code.co_kwonlyargcount
    if code.co_flags & 0x04:
        positional += list(loc[names[index]])
        index += 1
    if code.co_flags & 0x08:
        keywords.update(loc[names[index]])
    receiver = None
    if kind in ("instance_method", "classmethod") and positional:
        receiver, positional = positional[0], positional[1:]
    return receiver, positional, keywords

def profile(frame, event, arg):
    if event != "call" or frame.f_code is not TARGET["code"]:
        return
    test = STATE["test"]
    if test not in WANTED:
        return
    STATE["observed"] += 1
    STATE["per_test"][test] += 1
    if len(STATE["calls"]) >= SPEC["limits"]["max_calls_per_target"]:
        STATE["rejected"]["call_limit"] += 1
        return
    try:
        receiver, positional, keywords = arguments(frame, STATE["kind"])
        if STATE["kind"] == "instance_method":
            raise Unsupported("instance_method_receiver:" + type(receiver).__qualname__)
        args_text = literal(tuple(positional), SPEC["limits"])
        kwargs_text = literal(dict(keywords), SPEC["limits"])
    except Unsupported as exc:
        STATE["rejected"][str(exc)] += 1
        return
    except Exception as exc:
        STATE["rejected"]["capture_error:" + type(exc).__name__] += 1
        return
    key = (args_text, kwargs_text)
    if key in STATE["keys"]:
        STATE["rejected"]["duplicate"] += 1
        return
    STATE["keys"].add(key)
    STATE["calls"].append({"test": test, "args": args_text, "kwargs": kwargs_text})

def pytest_configure(config):
    try:
        kind, parent, name, code = resolve(SPEC["module"], SPEC["qualname"])
        STATE["kind"], TARGET["code"] = kind, code
        if code is None:
            STATE["resolve_error"] = "no code object for " + kind
    except Exception as exc:
        STATE["resolve_error"] = type(exc).__name__ + ": " + str(exc)[:200]

import pytest

@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):
    STATE["test"] = junit_id(item.nodeid)
    if TARGET["code"] is not None and STATE["test"] in WANTED:
        sys.setprofile(profile)
        threading.setprofile(profile)
    try:
        yield
    finally:
        sys.setprofile(None)
        threading.setprofile(None)
        STATE["test"] = None

def pytest_unconfigure(config):
    out = {"kind": STATE["kind"], "resolve_error": STATE["resolve_error"],
           "observed": STATE["observed"], "calls": STATE["calls"],
           "rejected": dict(STATE["rejected"]),
           "wanted_tests_reaching_target": len(STATE["per_test"])}
    with open(SPEC["out"], "w", encoding="utf-8") as handle:
        json.dump(out, handle)
'''

REPLAY = COMMON + r'''
import json, signal, sys
spec = json.loads(open(sys.argv[1], encoding="utf-8").read())

class Timeout(Exception):
    pass

def alarm(signum, frame):
    raise Timeout()

signal.signal(signal.SIGALRM, alarm)
results = []
try:
    kind, parent, name, code = resolve(spec["module"], spec["qualname"])
    target = getattr(parent, name)
except Exception as exc:
    print(json.dumps({"import_error": type(exc).__name__ + ": " + str(exc)[:200]}))
    sys.exit(0)
for call in spec["calls"]:
    args = ast.literal_eval(call["args"])          # independent reconstruction
    kwargs = ast.literal_eval(call["kwargs"])
    signal.alarm(spec["limits"]["per_call_seconds"])
    try:
        value = target(*args, **kwargs)
        signal.alarm(0)
        try:
            results.append(["ok", literal(value, spec["limits"])])
        except Unsupported as exc:
            results.append(["ok_unserializable", str(exc)])
    except Timeout:
        results.append(["timeout", None])
    except BaseException as exc:
        signal.alarm(0)
        results.append(["raise", type(exc).__name__])
print(json.dumps({"kind": kind, "results": results}))
'''


def capture(python: str, checkout: Path, target: dict, tests: list[str], tmp: Path) -> dict:
    tmp.mkdir(parents=True, exist_ok=True)
    (tmp / "oneiros_capture.py").write_text(PLUGIN, encoding="utf-8")
    out = tmp / "captured.json"
    spec = {"module": module_name(target["target_file"]), "qualname": target["target"],
            "wanted": target["difference_exposing_tests"], "out": str(out), "limits": LIMITS}
    env = dict(os.environ, ONEIROS_CAPTURE_SPEC=json.dumps(spec),
               PYTHONPATH=str(tmp) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    result = run([python, "-m", "pytest", "-p", "no:cacheprovider", "-q", "-o", "addopts=",
                  "-p", "oneiros_capture", *tests], cwd=checkout, timeout=CAPTURE_TIMEOUT,
                 env=env)
    if not out.exists():
        return {"error": "capture plugin wrote nothing", "tail": result["tail"][-400:],
                "seconds": result["seconds"]}
    data = json.loads(out.read_text(encoding="utf-8"))
    data["seconds"] = result["seconds"]
    data["pytest_exit"] = result["code"]
    return data


def replay(python: str, checkout: Path, target: dict, calls: list[dict], tmp: Path,
           label: str) -> dict:
    script = tmp / "oneiros_replay.py"
    script.write_text(REPLAY, encoding="utf-8")
    spec_path = tmp / f"replay_{label}.json"
    spec_path.write_text(json.dumps({"module": module_name(target["target_file"]),
                                     "qualname": target["target"], "calls": calls,
                                     "limits": LIMITS}), encoding="utf-8")
    result = run([python, str(script), str(spec_path)], cwd=checkout, timeout=REPLAY_TIMEOUT)
    try:
        return json.loads(result["tail"].strip().splitlines()[-1])
    except (ValueError, IndexError):
        return {"replay_error": result["tail"][-300:], "code": result["code"]}


def classify(calls: list[dict], fixed: dict, buggy: dict) -> list[dict]:
    rows = []
    for call, rf, rb in zip(calls, fixed.get("results") or [], buggy.get("results") or []):
        both = rf[0] in ("ok", "ok_unserializable", "raise") and \
            rb[0] in ("ok", "ok_unserializable", "raise")
        fixed_literal = rf[0] == "ok" and len(rf[1]) <= LIMITS["oracle_max_chars"]
        differs = rb[0] != "ok" or rb[1] != rf[1]
        rows.append({**call, "fixed": rf, "buggy": rb, "replayable_both": both,
                     "fixed_short_literal": fixed_literal,
                     "short_verified_oracle": both and fixed_literal and differs})
    return rows


def capture_target(target: dict, repo_dir: Path, out_path: Path) -> None:
    started = time.time()
    tag = target["key"].replace("/", "__").replace("@", "_").replace("cand:", "")[:80]
    wt = WORK / "capture_wt" / tag
    record = {"key": target["key"], "repository": target["repository"],
              "target": target["target"], "target_file": target["target_file"]}
    try:
        for label, commit in (("fixed", target["fixed_commit"]),
                              ("buggy", target["buggy_commit"])):
            step = run(["git", "-C", str(repo_dir), "worktree", "add", "--force", "--detach",
                        str(wt / label), commit], timeout=900)
            if step["code"] != 0:
                record.update(category="checkout_failed", detail=step["tail"][-300:])
                return
        fixed, buggy = wt / "fixed", wt / "buggy"
        env = build_env(wt / "env", fixed)
        record["environment"] = {"ok": env["ok"], "python": env.get("python"),
                                 "dependencies": env.get("dependencies")}
        if not env["ok"]:
            record.update(category="environment_install_failed")
            return
        python = env["python_path"]
        tests = [t for t in target["regression_test_files"] if (fixed / t).exists()]
        for t in tests:
            (buggy / t).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(fixed / t, buggy / t)
        switch = run(["uv", "pip", "install", "--python", python, "--no-deps", "-e", str(fixed)],
                     timeout=900)
        if switch["code"] != 0:
            record.update(category="environment_install_failed")
            return
        captured = capture(python, fixed, target, tests, wt / "capture")
        record["capture"] = {k: v for k, v in captured.items() if k != "calls"}
        calls = (captured.get("calls") or [])[:LIMITS["max_unique_replayed"]]
        record["calls_serializable"] = len(captured.get("calls") or [])
        if not calls:
            record.update(category="no_serializable_call", calls=[])
            return
        fixed_result = replay(python, fixed, target, calls, wt / "capture", "fixed")
        switch = run(["uv", "pip", "install", "--python", python, "--no-deps", "-e", str(buggy)],
                     timeout=900)
        if switch["code"] != 0:
            record.update(category="environment_install_failed")
            return
        buggy_result = replay(python, buggy, target, calls, wt / "capture", "buggy")
        rows = classify(calls, fixed_result, buggy_result)
        record["replay_errors"] = {k: v for k, v in (("fixed", fixed_result), ("buggy", buggy_result))
                                   if "results" not in v}
        record["calls"] = rows
        record["category"] = ("usable_fixed_call" if any(r["short_verified_oracle"] for r in rows)
                              else "no_short_verified_oracle")
    except Exception as exc:          # recorded, never silently dropped
        record.update(category="runner_error", detail=f"{type(exc).__name__}: {exc}"[:400])
    finally:
        record["wall_seconds"] = round(time.time() - started, 1)
        with LOCK:
            with out_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
            print(f"[{record['wall_seconds']:7.1f}s] {record.get('category'):28s} "
                  f"obs={record.get('capture', {}).get('observed')} "
                  f"ser={record.get('calls_serializable')} {target['key']}", flush=True)
        shutil.rmtree(wt, ignore_errors=True)
        run(["git", "-C", str(repo_dir), "worktree", "prune"], timeout=120)


def capture_repository(repository: str, url: str, targets: list[dict], out_path: Path) -> None:
    repo_dir = WORK / "repos" / repository.replace("/", "__")
    if not repo_dir.exists():
        result = run(["git", "clone", "--filter=blob:none", "--no-checkout", url, str(repo_dir)],
                     timeout=1800)
        if result["code"] != 0:
            for target in targets:
                with LOCK:
                    with out_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps({"key": target["key"], "category": "clone_failed",
                                                 "repository": repository}) + "\n")
            return
    for target in targets:
        capture_target(target, repo_dir, out_path)


def main() -> int:
    manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    out_dir = Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "records.jsonl"
    done = set()
    if out_path.exists():
        done = {json.loads(line)["key"] for line in out_path.read_text().splitlines() if line}
    by_repo: dict[str, list[dict]] = {}
    for target in manifest["targets"]:
        if target["key"] not in done:
            by_repo.setdefault(target["repository"], []).append(target)
    urls = {t["repository"]: t["repository_url"] for t in manifest["targets"]}
    started = time.time()
    print(f"capturing {sum(map(len, by_repo.values()))} targets in {len(by_repo)} repositories",
          flush=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        for repository, targets in by_repo.items():
            pool.submit(capture_repository, repository, urls[repository], targets, out_path)
    (out_dir / "run_summary.json").write_text(json.dumps({
        "wall_seconds": round(time.time() - started, 1), "limits": LIMITS,
        "uv": run(["uv", "--version"])["tail"].strip(), "cpus": os.cpu_count()}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
