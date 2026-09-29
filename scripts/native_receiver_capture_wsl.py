"""Receiver-aware argument capture and replay, v2 (WSL, uv-managed CPython; stdlib only).

Implements docs/SFT_ROOT_CAUSE_CHOICE_A_RECEIVER_REPLAY_PROTOCOL_V2.md on the 24 natively
qualified development targets. v1 (native_argument_capture_wsl.py) is unchanged.

Capture (fixed revision, only while a difference-exposing official test runs):
  * free functions, staticmethods and classmethods: literal-only arguments, as in v1;
  * instance methods: the receiver must have been built in the same test by an
    ALLOWLISTED construction - ``cls(...)`` of a class in the target package (no custom
    metaclass, no ``__new__`` override, the outermost ``__init__`` is the class's own) or
    a package classmethod factory that returns a new instance of exactly its class - with
    literal-only arguments; receivers are held by STRONG reference in a per-test registry
    (so ``id`` cannot be reused), snapshotted (``__dict__`` / ``__slots__`` fields, each
    literal-only) after construction and again before the target call, and rejected if
    the snapshots differ.
Replay: fresh process per revision and repetition (2 x fixed, 2 x buggy), temporary cwd,
network disabled (``unshare -n`` when available plus an in-process socket guard), 5 s
per-call watchdog, reconstruction from literal text only, reconstructed snapshot must
equal the captured one. No pickle, no object deserialisation, no test assertion values.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from native_argument_capture_wsl import COMMON, LIMITS  # noqa: E402
from native_rehearsal_wsl import WORK, build_env, module_name, run  # noqa: E402

DESIGN_VERSION = "oneiros_receiver_capture_v2"
CAPTURE_TIMEOUT = 900
REPLAY_TIMEOUT = 600
REPEATS = 2
LIMITS_V2 = {**LIMITS, "max_registry": 5000, "snapshot_max_chars": 4000}
LOCK = threading.Lock()

SNAPSHOT = r'''
def slot_names(cls):
    names = []
    for klass in cls.__mro__:
        slots = klass.__dict__.get("__slots__", ())
        if isinstance(slots, str):
            slots = (slots,)
        for name in slots:
            if name in ("__dict__", "__weakref__"):
                continue
            if name.startswith("__") and not name.endswith("__"):
                name = "_" + klass.__name__.lstrip("_") + name
            names.append(name)
    return sorted(set(names))

def snapshot(obj, limits):
    """Literal text of every slot and __dict__ field; Unsupported('opaque_state:...')."""
    fields = {}
    unset = []
    for name in slot_names(type(obj)):
        try:
            value = object.__getattribute__(obj, name)
        except AttributeError:
            unset.append(name)
            continue
        try:
            fields["slot:" + name] = literal(value, limits)
        except Unsupported as exc:
            raise Unsupported("opaque_state:" + str(exc))
    try:
        mapping = object.__getattribute__(obj, "__dict__")
    except AttributeError:
        mapping = None
    if mapping is not None:
        if type(mapping) is not dict:
            raise Unsupported("opaque_state:non_dict_namespace")
        for key in sorted(mapping):
            try:
                fields["dict:" + str(key)] = literal(mapping[key], limits)
            except Unsupported as exc:
                raise Unsupported("opaque_state:" + str(exc))
    text = repr((sorted(fields.items()), unset))
    if len(text) > limits["snapshot_max_chars"]:
        raise Unsupported("opaque_state:snapshot_too_large")
    return text

def in_package(cls, package):
    return (getattr(cls, "__module__", "") or "").split(".")[0] == package

def allowlisted_class(cls, package):
    if not isinstance(cls, type) or not in_package(cls, package):
        return "foreign_class"
    if type(cls) is not type:
        return "metaclass"
    if cls.__new__ is not object.__new__:
        return "custom_new"
    return None
'''

PLUGIN = COMMON + SNAPSHOT + r'''
import json, os, sys, threading, inspect
from collections import Counter

SPEC = json.loads(os.environ["ONEIROS_CAPTURE_SPEC"])
LIMITS = SPEC["limits"]
PACKAGE = SPEC["package"]
WANTED = set(SPEC["wanted"])
STATE = {"test": None, "observed": 0, "calls": [], "keys": set(), "rejected": Counter(),
         "recipes": 0, "per_test": Counter(), "kind": None, "resolve_error": None}
REG = {}          # id -> {"obj": strong ref, "recipe": ..., "snapshot": ..., "reject": ...}
PENDING = {}      # id(frame) -> pending construction
TARGET = {"code": None, "name": None}

def junit_id(nodeid):
    parts = nodeid.split("::")
    module = parts[0][:-3].replace("/", ".") if parts[0].endswith(".py") else parts[0]
    return ".".join([module] + parts[1:-1]) + "::" + parts[-1]

def split_args(frame, drop_first):
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
    first = positional[0] if (drop_first and positional) else None
    return first, (positional[1:] if drop_first else positional), keywords

def literal_args(positional, keywords):
    return literal(tuple(positional), LIMITS), literal(dict(keywords), LIMITS)

def register(obj, recipe, reject=None):
    if len(REG) >= LIMITS["max_registry"]:
        STATE["rejected"]["registry_full"] += 1
        return
    entry = {"obj": obj, "recipe": recipe, "reject": reject, "snapshot": None}
    if reject is None:
        try:
            entry["snapshot"] = snapshot(obj, LIMITS)
        except Unsupported as exc:
            entry["reject"] = str(exc)
    REG[id(obj)] = entry
    if entry["reject"] is None:
        STATE["recipes"] += 1

def on_call(frame):
    code = frame.f_code
    if code is TARGET["code"]:
        return on_target(frame)
    if code.co_name == "__init__" and code.co_argcount >= 1:
        obj = frame.f_locals.get(code.co_varnames[0])
        cls = type(obj)
        if id(obj) in REG or not in_package(cls, PACKAGE):
            return
        if any(p.get("obj") is obj for p in PENDING.values()):
            return                               # a super().__init__ inside the outer one
        reason = allowlisted_class(cls, PACKAGE)
        own = getattr(inspect.unwrap(cls.__init__), "__code__", None)
        if reason is None and own is not code:
            reason = "init_not_class_constructor"
        recipe, error = None, reason
        if reason is None:
            try:
                _, positional, keywords = split_args(frame, True)
                args_text, kwargs_text = literal_args(positional, keywords)
                recipe = {"kind": "constructor", "module": cls.__module__,
                          "qualname": cls.__qualname__, "args": args_text,
                          "kwargs": kwargs_text}
            except Unsupported as exc:
                error = "constructor_" + str(exc)
        PENDING[id(frame)] = {"obj": obj, "recipe": recipe, "reject": error, "factory": False}
        return
    if code.co_varnames[:1] == ("cls",) and code.co_argcount >= 1:
        cls = frame.f_locals.get("cls")
        if not isinstance(cls, type) or not in_package(cls, PACKAGE):
            return
        static = inspect.getattr_static(cls, code.co_name, None)
        if not isinstance(static, classmethod) or \
                getattr(inspect.unwrap(static.__func__), "__code__", None) is not code:
            return
        reason = allowlisted_class(cls, PACKAGE)
        recipe = None
        if reason is None:
            try:
                _, positional, keywords = split_args(frame, True)
                args_text, kwargs_text = literal_args(positional, keywords)
                recipe = {"kind": "factory", "module": cls.__module__,
                          "qualname": cls.__qualname__, "factory": code.co_name,
                          "args": args_text, "kwargs": kwargs_text}
            except Unsupported as exc:
                reason = "factory_" + str(exc)
        PENDING[id(frame)] = {"cls": cls, "recipe": recipe, "reject": reason, "factory": True,
                              "before": set(REG)}

def on_return(frame, value):
    pending = PENDING.pop(id(frame), None)
    if pending is None:
        return
    if not pending["factory"]:
        register(pending["obj"], pending["recipe"], pending["reject"])
        return
    if type(value) is not pending["cls"]:
        return                                   # not a constructor-like factory
    if id(value) in pending["before"]:
        entry = REG.get(id(value))
        if entry is not None and entry["obj"] is value:
            entry["reject"] = entry["reject"] or "cached_factory"
        return
    entry = REG.get(id(value))
    if entry is not None and entry["obj"] is value:
        return                                   # built inside by an allowlisted constructor
    register(value, pending["recipe"], pending["reject"])

def on_target(frame):
    test = STATE["test"]
    STATE["observed"] += 1
    STATE["per_test"][test] += 1
    if len(STATE["calls"]) >= LIMITS["max_calls_per_target"]:
        STATE["rejected"]["call_limit"] += 1
        return
    kind = STATE["kind"]
    try:
        receiver, positional, keywords = split_args(frame, kind in ("instance_method",
                                                                   "classmethod"))
        recipe, snap = None, None
        if kind == "instance_method":
            entry = REG.get(id(receiver))
            if entry is None or entry["obj"] is not receiver:
                raise Unsupported("receiver_without_allowlisted_construction:"
                                  + type(receiver).__qualname__)
            if entry["reject"]:
                raise Unsupported("receiver_" + entry["reject"])
            try:
                now = snapshot(receiver, LIMITS)
            except Unsupported as exc:
                raise Unsupported("receiver_" + str(exc))
            if now != entry["snapshot"]:
                raise Unsupported("receiver_mutated")
            recipe, snap = entry["recipe"], entry["snapshot"]
        args_text, kwargs_text = literal_args(positional, keywords)
    except Unsupported as exc:
        STATE["rejected"][str(exc)] += 1
        return
    except Exception as exc:
        STATE["rejected"]["capture_error:" + type(exc).__name__] += 1
        return
    key = (json.dumps(recipe, sort_keys=True), args_text, kwargs_text)
    if key in STATE["keys"]:
        STATE["rejected"]["duplicate"] += 1
        return
    STATE["keys"].add(key)
    STATE["calls"].append({"test": test, "recipe": recipe, "snapshot": snap,
                           "args": args_text, "kwargs": kwargs_text})

def profile(frame, event, arg):
    if STATE["test"] not in WANTED:
        return
    if event == "call":
        on_call(frame)
    elif event == "return" and PENDING:
        on_return(frame, arg)

def pytest_configure(config):
    try:
        kind, parent, name, code = resolve(SPEC["module"], SPEC["qualname"])
        STATE["kind"], TARGET["code"], TARGET["name"] = kind, code, name
        if code is None:
            STATE["resolve_error"] = "no code object for " + kind
    except Exception as exc:
        STATE["resolve_error"] = type(exc).__name__ + ": " + str(exc)[:200]

import pytest

@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):
    STATE["test"] = junit_id(item.nodeid)
    active = TARGET["code"] is not None and STATE["test"] in WANTED
    if active:
        sys.setprofile(profile)
        threading.setprofile(profile)
    try:
        yield
    finally:
        sys.setprofile(None)
        threading.setprofile(None)
        STATE["test"] = None
        REG.clear()
        PENDING.clear()

def pytest_unconfigure(config):
    out = {"kind": STATE["kind"], "resolve_error": STATE["resolve_error"],
           "observed": STATE["observed"], "recipes_registered": STATE["recipes"],
           "calls": STATE["calls"], "rejected": dict(STATE["rejected"]),
           "wanted_tests_reaching_target": len(STATE["per_test"])}
    with open(SPEC["out"], "w", encoding="utf-8") as handle:
        json.dump(out, handle)
'''

REPLAY = COMMON + SNAPSHOT + r'''
import json, socket, sys, threading, _thread

class NetworkDisabled(OSError):
    pass

def _refuse(*args, **kwargs):
    raise NetworkDisabled("network disabled during replay")

socket.socket = _refuse
socket.create_connection = _refuse
socket.getaddrinfo = _refuse

class Timeout(Exception):
    pass

spec = json.loads(open(sys.argv[1], encoding="utf-8").read())
LIMITS = spec["limits"]
out = {"results": []}
try:
    kind, parent, name, code = resolve(spec["module"], spec["qualname"])
except Exception as exc:
    print(json.dumps({"import_error": type(exc).__name__ + ": " + str(exc)[:200]}))
    sys.exit(0)
import os as _os
_module_file = _os.path.realpath(importlib.import_module(spec["module"]).__file__)
out["module_file"] = _module_file
_root = spec.get("expected_root")
if _root and not _module_file.startswith(_os.path.realpath(_root) + _os.sep):
    print(json.dumps({"revision_mismatch": _module_file}))
    sys.exit(0)

def build(recipe):
    import importlib
    module = importlib.import_module(recipe["module"])
    cls = module
    for part in recipe["qualname"].split("."):
        cls = getattr(cls, part)
    reason = allowlisted_class(cls, spec["package"])
    if reason:
        raise Unsupported("recipe_class_" + reason)
    args = ast.literal_eval(recipe["args"])
    kwargs = ast.literal_eval(recipe["kwargs"])
    if recipe["kind"] == "constructor":
        return cls(*args, **kwargs)
    obj = getattr(cls, recipe["factory"])(*args, **kwargs)
    if type(obj) is not cls:
        raise Unsupported("factory_returned_other_type")
    return obj

for call in spec["calls"]:
    fired = []
    timer = threading.Timer(LIMITS["per_call_seconds"],
                            lambda: (fired.append(1), _thread.interrupt_main()))
    timer.start()
    try:
        try:
            if call["recipe"] is not None:
                try:
                    obj = build(call["recipe"])
                except Unsupported as exc:
                    out["results"].append(["reconstruction_refused", str(exc)])
                    continue
                except BaseException as exc:
                    if fired:
                        raise
                    out["results"].append(["constructor_raise", type(exc).__name__])
                    continue
                if snapshot(obj, LIMITS) != call["snapshot"]:
                    out["results"].append(["reconstruction_mismatch", None])
                    continue
                target = getattr(obj, name)
            else:
                target = getattr(parent, name)
            args = ast.literal_eval(call["args"])          # fresh objects in every process
            kwargs = ast.literal_eval(call["kwargs"])
            value = target(*args, **kwargs)
            try:
                out["results"].append(["ok", literal(value, LIMITS)])
            except Unsupported as exc:
                out["results"].append(["ok_unserializable", str(exc)])
        except KeyboardInterrupt:
            out["results"].append(["timeout", None] if fired else ["raise", "KeyboardInterrupt"])
        except BaseException as exc:
            out["results"].append(["timeout", None] if fired else ["raise", type(exc).__name__])
    finally:
        timer.cancel()
print(json.dumps(out))
'''

SEMANTIC = ("ok", "ok_unserializable", "raise")


def pick(replay_run, index):
    results = replay_run.get("results")
    return results[index] if isinstance(results, list) and index < len(results) else None


def classify(calls, fixed_runs, buggy_runs, oracle_max_chars=LIMITS["oracle_max_chars"]):
    """Per call: determinism on both revisions, a deterministic difference, a short oracle."""
    rows = []
    for index, call in enumerate(calls):
        f = [pick(run_, index) for run_ in fixed_runs]
        b = [pick(run_, index) for run_ in buggy_runs]
        if any(x is None for x in f + b):
            status = "replay_process_failure"
        elif any(x[0] == "timeout" for x in f + b):
            status = "timeout"
        elif any(x[0] not in SEMANTIC for x in f + b):
            status = next(x[0] for x in f + b if x[0] not in SEMANTIC)
        elif any(x != f[0] for x in f):
            status = "nondeterministic_fixed"
        elif any(x != b[0] for x in b):
            status = "nondeterministic_buggy"
        elif f[0] == b[0]:
            status = "no_difference"
        elif f[0][0] != "ok":
            status = "fixed_not_literal_value"
        elif len(f[0][1]) > oracle_max_chars:
            status = "fixed_value_too_long"
        else:
            status = "usable"
        rows.append({"recipe_kind": (call.get("recipe") or {}).get("kind"), "test": call["test"],
                     "args": call["args"], "kwargs": call["kwargs"],
                     "recipe": call.get("recipe"), "fixed": f[0] if f else None,
                     "buggy": b[0] if b else None, "status": status,
                     "usable": status == "usable"})
    return rows


def isolated(command):
    """Prefix with a fresh network namespace when available (root in WSL)."""
    return (["unshare", "-n", *command]
            if shutil.which("unshare") and hasattr(os, "geteuid") and os.geteuid() == 0
            else command)


def capture(python, checkout, target, tests, tmp):
    tmp.mkdir(parents=True, exist_ok=True)
    (tmp / "oneiros_receiver_capture.py").write_text(PLUGIN, encoding="utf-8")
    out = tmp / "captured.json"
    module = module_name(target["target_file"])
    spec = {"module": module, "qualname": target["target"], "package": module.split(".")[0],
            "wanted": target["difference_exposing_tests"], "out": str(out), "limits": LIMITS_V2}
    env = dict(os.environ, ONEIROS_CAPTURE_SPEC=json.dumps(spec), **NO_BYTECODE,
               PYTHONPATH=str(tmp) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    result = run([python, "-m", "pytest", "-p", "no:cacheprovider", "-q", "-o", "addopts=",
                  "-p", "oneiros_receiver_capture", *tests], cwd=checkout,
                 timeout=CAPTURE_TIMEOUT, env=env)
    if not out.exists():
        return {"error": "capture plugin wrote nothing", "tail": result["tail"][-300:],
                "seconds": result["seconds"]}
    data = json.loads(out.read_text(encoding="utf-8"))
    data.update(seconds=result["seconds"], pytest_exit=result["code"])
    return data


NO_BYTECODE = {"PYTHONDONTWRITEBYTECODE": "1"}


def clear_stale_finder_bytecode(python):
    """Editable finders are rewritten in place with the same size; a stale .pyc from the
    same second would keep importing the previous revision. Remove it after each switch."""
    for cache in Path(python).resolve().parent.parent.glob("lib/python*/site-packages/__pycache__"):
        for pyc in cache.glob("__editable__*"):
            pyc.unlink()


def replay(python, target, calls, tmp, label, expected_root=None):
    script = tmp / "oneiros_receiver_replay.py"
    script.write_text(REPLAY, encoding="utf-8")
    module = module_name(target["target_file"])
    spec_path = tmp / f"replay_{label}.json"
    spec_path.write_text(json.dumps({"module": module, "qualname": target["target"],
                                     "package": module.split(".")[0], "calls": calls,
                                     "limits": LIMITS_V2,
                                     "expected_root": str(expected_root) if expected_root
                                     else None}), encoding="utf-8")
    cwd = Path(tempfile.mkdtemp(prefix="oneiros_replay_"))
    try:
        result = run(isolated([python, str(script), str(spec_path)]), cwd=cwd,
                     timeout=REPLAY_TIMEOUT, env=dict(os.environ, **NO_BYTECODE))
    finally:
        shutil.rmtree(cwd, ignore_errors=True)
    try:
        return json.loads(result["tail"].strip().splitlines()[-1])
    except (ValueError, IndexError):
        return {"replay_error": result["tail"][-300:], "code": result["code"]}


def capture_target(target, repo_dir, out_path):
    started = time.time()
    tag = target["key"].replace("/", "__").replace("@", "_").replace("cand:", "")[:80]
    wt = WORK / "receiver_wt" / tag
    record = {"key": target["key"], "repository": target["repository"],
              "target": target["target"], "target_file": target["target_file"],
              "design_version": DESIGN_VERSION}
    try:
        for label, commit in (("fixed", target["fixed_commit"]), ("buggy", target["buggy_commit"])):
            step = run(["git", "-C", str(repo_dir), "worktree", "add", "--force", "--detach",
                        str(wt / label), commit], timeout=900)
            if step["code"] != 0:
                record.update(category="environment_failure", failure="checkout_failed")
                return
        fixed, buggy = wt / "fixed", wt / "buggy"
        env = build_env(wt / "env", fixed)
        record["environment"] = {"ok": env["ok"], "python": env.get("python"),
                                 "dependencies": env.get("dependencies")}
        if not env["ok"]:
            record.update(category="environment_failure", failure="install_failed")
            return
        python = env["python_path"]
        tests = [t for t in target["regression_test_files"] if (fixed / t).exists()]
        for t in tests:
            (buggy / t).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(fixed / t, buggy / t)
        runs = {}
        for label, checkout in (("fixed", fixed), ("buggy", buggy)):
            switch = run(["uv", "pip", "install", "--python", python, "--no-deps", "-e",
                          str(checkout)], timeout=900)
            if switch["code"] != 0:
                record.update(category="environment_failure", failure=f"switch_{label}_failed")
                return
            clear_stale_finder_bytecode(python)
            if label == "fixed":
                captured = capture(python, fixed, target, tests, wt / "capture")
                record["capture"] = {k: v for k, v in captured.items() if k != "calls"}
                calls = (captured.get("calls") or [])[:LIMITS["max_unique_replayed"]]
                record["calls_captured"] = len(captured.get("calls") or [])
                if not calls:
                    record.update(category="no_replayable_call", calls=[])
                    return
            runs[label] = [replay(python, target, calls, wt / "capture", f"{label}{i}",
                                  expected_root=checkout) for i in range(REPEATS)]
        mismatched = [r for label in runs for r in runs[label] if "revision_mismatch" in r]
        if mismatched:
            record.update(category="environment_failure", failure="revision_mismatch",
                          detail=mismatched[:2])
            return
        rows = classify(calls, runs["fixed"], runs["buggy"])
        record["replay_errors"] = [r for label in runs for r in runs[label] if "results" not in r]
        record["calls"] = rows
        record["category"] = ("usable_fixed_call" if any(r["usable"] for r in rows)
                              else "no_usable_call")
    except Exception as exc:          # recorded, never silently dropped
        record.update(category="runner_error", failure=f"{type(exc).__name__}: {exc}"[:300])
    finally:
        record["wall_seconds"] = round(time.time() - started, 1)
        with LOCK:
            with out_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            print(f"[{record['wall_seconds']:7.1f}s] {record.get('category'):20s} "
                  f"obs={record.get('capture', {}).get('observed')} "
                  f"cap={record.get('calls_captured')} "
                  f"use={sum(r['usable'] for r in record.get('calls') or [])} {target['key']}",
                  flush=True)
        shutil.rmtree(wt, ignore_errors=True)
        run(["git", "-C", str(repo_dir), "worktree", "prune"], timeout=120)


def capture_repository(repository, url, targets, out_path):
    repo_dir = WORK / "repos" / repository.replace("/", "__")
    if not repo_dir.exists():
        result = run(["git", "clone", "--filter=blob:none", "--no-checkout", url,
                      str(repo_dir)], timeout=1800)
        if result["code"] != 0:
            for target in targets:
                with LOCK, out_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps({"key": target["key"], "repository": repository,
                                             "category": "environment_failure",
                                             "failure": "clone_failed"}) + "\n")
            return
    for target in targets:             # one repository = one sequential worker
        capture_target(target, repo_dir, out_path)


def main() -> int:
    manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    out_dir = Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "records.jsonl"
    done = set()
    if out_path.exists():
        done = {json.loads(line)["key"] for line in out_path.read_text().splitlines() if line}
    by_repo = {}
    for target in manifest["targets"]:
        if target["key"] not in done:
            by_repo.setdefault(target["repository"], []).append(target)
    urls = {t["repository"]: t["repository_url"] for t in manifest["targets"]}
    started = time.time()
    print(f"receiver capture: {sum(map(len, by_repo.values()))} targets, {len(by_repo)} "
          f"repositories ({len(done)} already done)", flush=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        for repository, targets in by_repo.items():
            pool.submit(capture_repository, repository, urls[repository], targets, out_path)
    (out_dir / "run_summary.json").write_text(json.dumps({
        "design_version": DESIGN_VERSION, "wall_seconds": round(time.time() - started, 1),
        "limits": LIMITS_V2, "repeats": REPEATS,
        "network_isolation": "unshare -n" if isolated(["x"])[0] == "unshare" else
        "in-process socket guard only",
        "uv": run(["uv", "--version"])["tail"].strip(), "cpus": os.cpu_count()}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
