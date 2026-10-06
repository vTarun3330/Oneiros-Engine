"""v2.7 panel admission: NATIVE target reach of the official difference-exposing tests (WSL,
stdlib only; run inside scripts/wsl_isolated.sh).

The official tests need their project's own test support (conftest fixtures, helper modules),
so reach is measured where they were qualified: a fresh worktree of each revision (the buggy
qualification copy receives the fixed revision's test files, exactly as in qualification), the
target's prepared environment and the same pytest invocation, plus a tiny tracing plugin
(sys.setprofile) that records whether the target function's code object (name + target file)
is entered. ``reach_verified``: entered on BOTH revisions while the selected tests run.

v2 (measurement defects found on r4; generic, applied to every target, outcome-independent):
- test ids ``pkg.mod.Cls::test[p]`` resolve to the longest dotted prefix that is an existing
  ``.py`` file (v1 turned class names into file paths -> pytest usage error, exit 4);
- a run whose pytest exit code is not 0/1 (usage error, no tests collected, crash) is
  ``unmeasured`` (infrastructure), never ``not_reached``;
- the tracing plugin survives interpreter shutdown (v1 raised when module globals were None)
  and writes its result the moment the target is entered (v3: a project plugin crashing in
  its own session teardown, e.g. pytest_pyvista with the cache provider disabled, lost it);
- build-generated files missing from a fresh worktree (e.g. a hatch-vcs ``_version.py``) are
  copied from the target's prepared views, as qualification did.

    bash scripts/wsl_isolated.sh bash scripts/wsl_native_python.sh scripts/v27_native_reach_wsl.py \
        --manifest M --prep DIR --out FILE
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
from native_generated_tests_execute_wsl import import_root  # noqa: E402

REPOS = Path("/root/oneiros_rehearsal/repos")
PLUGIN = '''
import json, os, sys
_NAME = os.environ["ONEIROS_REACH_NAME"]
_FILE = os.environ["ONEIROS_REACH_FILE"]
_OUT = os.environ["ONEIROS_REACH_OUT"]
_HIT = {"entered": False}
def _save(_hit=_HIT, _out=_OUT, _dump=json.dump):
    with open(_out, "w") as fh:
        _dump(_hit, fh)
def _prof(frame, event, arg, _hit=_HIT, _name=_NAME, _file=_FILE, _save=_save):
    try:
        if event == "call" and not _hit["entered"]:
            co = frame.f_code
            if co.co_name == _name and co.co_filename.replace("\\\\", "/").endswith(_file):
                _hit["entered"] = True
                _save()          # recorded at once: a later plugin crash cannot lose it
    except Exception:
        pass
    return None
def pytest_configure(config):
    sys.setprofile(_prof)
    import threading
    threading.setprofile(_prof)
def pytest_sessionstart(session):
    _save()
def pytest_unconfigure(config):
    sys.setprofile(None)
    _save()
'''


def git(repo: Path, *args) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True,
                          timeout=900)


def pytest_ids(test_ids: list, checkout: Path) -> list:
    """``pkg.mod.Cls::test[p]`` -> ``pkg/mod.py::Cls::test[p]`` (longest existing module)."""
    out = []
    for tid in test_ids:
        dotted, _, rest = tid.partition("::")
        parts = dotted.split(".")
        for cut in range(len(parts), 0, -1):
            if (checkout / ("/".join(parts[:cut]) + ".py")).is_file():
                break
        else:
            cut = len(parts)
        node = "::".join(["/".join(parts[:cut]) + ".py", *parts[cut:], *([rest] if rest else [])])
        out.append(node)
    return out


def copy_generated(view: Path, checkout: Path) -> list:
    """Files the prepared view has but the fresh worktree lacks (build-generated only)."""
    copied = []
    if view.is_dir():
        for src in view.rglob("*.py"):
            rel = src.relative_to(view)
            dst = checkout / rel
            if not dst.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src, dst)
                copied.append(rel.as_posix())
    return sorted(copied)


def run_reach(python: str, checkout: Path, root_rel: str, tests: list, name: str,
              target_file: str, work: Path, label: str) -> dict:
    plugin_dir = work / "plugin"
    plugin_dir.mkdir(parents=True, exist_ok=True)
    (plugin_dir / "oneiros_reach_plugin.py").write_text(PLUGIN, encoding="utf-8")
    out = work / f"{label}.json"
    path = str(checkout / root_rel) if root_rel else str(checkout)
    env = {"PATH": "/usr/bin:/bin", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONNOUSERSITE": "1",
           "PYTHONPATH": f"{plugin_dir}:{path}", "HOME": str(work),
           "ONEIROS_REACH_NAME": name, "ONEIROS_REACH_FILE": target_file,
           "ONEIROS_REACH_OUT": str(out)}
    done = subprocess.run([python, "-B", "-m", "pytest", "-p", "no:cacheprovider", "-p",
                           "oneiros_reach_plugin", "-q", "-o", "addopts=", *tests],
                          cwd=checkout, env=env, capture_output=True, text=True, timeout=900)
    hit = json.loads(out.read_text()) if out.exists() else {"entered": None}
    entered = hit.get("entered") if done.returncode in (0, 1) else None   # tests did not run
    return {"exit": done.returncode, "entered": entered,
            "tail": (done.stdout + done.stderr)[-300:] if entered is None else None}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--prep", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    out = REPO / args.out
    if out.exists():
        raise SystemExit(f"REFUSED: {out} exists")
    targets = {t["key"]: t for t in json.loads((REPO / args.manifest).read_text(
        encoding="utf-8"))["targets"]}
    records = {}
    for line in (REPO / args.prep / "records.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            records[r["key"]] = r
    rows = []
    base = Path(tempfile.mkdtemp(prefix="oneiros_v27_nreach_"))
    try:
        for key, rec in sorted(records.items()):
            t = targets[key]
            row = {"key": key, "repository": t["repository"]}
            if rec.get("category") != "requalified":
                rows.append({**row, "reach": "not_requalified"})
                continue
            repo = REPOS / t["repository"].replace("/", "__")
            work = base / rec["tag"]
            res = {}
            try:
                for label, commit in (("fixed", t["fixed_commit"]), ("buggy", t["buggy_commit"])):
                    wt = work / label
                    if git(repo, "worktree", "add", "--force", "--detach", str(wt),
                           commit).returncode != 0:
                        raise RuntimeError(f"worktree {label}")
                tests = [p for p in t["regression_test_files"] if (work / "fixed" / p).exists()]
                for p in tests:
                    (work / "buggy" / p).parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(work / "fixed" / p, work / "buggy" / p)
                root_rel = import_root(t["target_file"])
                rel_file = t["target_file"][len(root_rel) + 1:] if root_rel else t["target_file"]
                name = rec["qualname"].split(".")[-1]
                views = Path(rec["env_dir"]).parent / "views"
                generated = {}
                for label in ("fixed", "buggy"):
                    dest = work / label / root_rel if root_rel else work / label
                    generated[label] = copy_generated(views / label, dest)
                for label in ("fixed", "buggy"):
                    res[label] = run_reach(rec["python_path"], work / label, root_rel,
                                           pytest_ids(t["difference_exposing_tests"],
                                                      work / label), name,
                                           rel_file, work / f"r_{label}", label)
            except Exception as exc:                      # noqa: BLE001 - recorded
                rows.append({**row, "reach": f"runner_error:{type(exc).__name__}"})
                continue
            finally:
                for label in ("fixed", "buggy"):
                    git(repo, "worktree", "remove", "--force", str(work / label))
                git(repo, "worktree", "prune")
            ok = res["fixed"]["entered"] is True and res["buggy"]["entered"] is True
            measured = all(v["entered"] is not None for v in res.values())
            rows.append({**row, "reach": "reach_verified" if ok else
                         "not_reached" if measured else "unmeasured",
                         "generated_files_copied": generated,
                         "entered": {k: v["entered"] for k, v in res.items()},
                         "exit": {k: v["exit"] for k, v in res.items()},
                         "detail": {k: v["tail"] for k, v in res.items() if v["tail"]} or None})
            print(json.dumps({"key": key[-40:], "reach": rows[-1]["reach"],
                              "entered": rows[-1]["entered"]}), flush=True)
    finally:
        shutil.rmtree(base, ignore_errors=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(("\n".join(json.dumps(r, sort_keys=True) for r in rows) + "\n")
                    .encode("utf-8"))
    print(json.dumps({"rows": len(rows), "sha256": hashlib.sha256(out.read_bytes())
                      .hexdigest()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
