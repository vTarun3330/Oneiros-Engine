"""v2.7 panel admission: NATIVE target reach of the official difference-exposing tests (WSL,
stdlib only; run inside scripts/wsl_isolated.sh).

The official tests need their project's own test support (conftest fixtures, helper modules),
so reach is measured where they were qualified: a fresh worktree of each revision (the buggy
qualification copy receives the fixed revision's test files, exactly as in qualification), the
target's prepared environment and the same pytest invocation, plus a tiny tracing plugin
(sys.setprofile) that records whether the target function's code object (name + target file)
is entered. ``reach_verified``: entered on BOTH revisions while the selected tests run.

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
_HIT = {"entered": False}
def _prof(frame, event, arg):
    if event == "call" and not _HIT["entered"]:
        co = frame.f_code
        if co.co_name == _NAME and co.co_filename.replace("\\\\", "/").endswith(_FILE):
            _HIT["entered"] = True
    return None
def pytest_configure(config):
    sys.setprofile(_prof)
    import threading
    threading.setprofile(_prof)
def pytest_unconfigure(config):
    sys.setprofile(None)
    with open(os.environ["ONEIROS_REACH_OUT"], "w") as fh:
        json.dump(_HIT, fh)
'''


def git(repo: Path, *args) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True,
                          timeout=900)


def pytest_ids(test_ids: list) -> list:
    """``pkg.mod::Cls::test`` -> ``pkg/mod.py::Cls::test``."""
    out = []
    for tid in test_ids:
        mod, _, rest = tid.partition("::")
        out.append(mod.replace(".", "/") + ".py" + ("::" + rest if rest else ""))
    return out


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
    return {"exit": done.returncode, "entered": hit.get("entered"),
            "tail": (done.stdout + done.stderr)[-300:] if hit.get("entered") is None else None}


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
                for label in ("fixed", "buggy"):
                    res[label] = run_reach(rec["python_path"], work / label, root_rel,
                                           pytest_ids(t["difference_exposing_tests"]), name,
                                           rel_file, work / f"r_{label}", label)
            except Exception as exc:                      # noqa: BLE001 - recorded
                rows.append({**row, "reach": f"runner_error:{type(exc).__name__}"})
                continue
            finally:
                for label in ("fixed", "buggy"):
                    git(repo, "worktree", "remove", "--force", str(work / label))
                git(repo, "worktree", "prune")
            ok = res["fixed"]["entered"] is True and res["buggy"]["entered"] is True
            rows.append({**row, "reach": "reach_verified" if ok else "not_reached",
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
