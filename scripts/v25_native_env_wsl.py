"""v2.5 native buggy/fixed environments for TRAIN repositories (WSL, root; stdlib only).

Generalises the v2.1 preparation (``native_rehearsal_prepare_wsl``) without modifying it: the same
sanitised views (tests removed at any depth), import attestation, project-install stripping and
environment lock, plus
- a pinned interpreter per repository/version (canary spec), not only 3.11;
- dependency resolution frozen in time: ``uv pip install --exclude-newer <buggy commit date>``,
  BugsInPy's own pinned requirements.txt where present;
- SWE-bench fixed revision = base commit + gold patch (hash-verified); the official test patch is
  applied ONLY to the two qualification copies, never to the views given to generated tests;
- the official targeted tests qualified 3/3 (fail on buggy, pass on fixed) with pytest, or
  Django's own ``tests/runtests.py`` (settings test_sqlite) for Django;
- the environment is built TWICE and must give an identical lock (reproducibility);
- every failure gets an explicit category; nothing is upgraded silently.

    wsl -u root -- bash scripts/wsl_native_python.sh scripts/v25_native_env_wsl.py \
        --targets results/sft_root_cause/v25_native/targets.jsonl --canary-only
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from native_rehearsal_wsl import junit, module_name, run  # noqa: E402
from native_generated_tests_execute_wsl import (  # noqa: E402
    build_view, env_lock, generated_files, import_root)

REPOS = Path("/root/oneiros_v25_repos")
WORK = Path("/root/oneiros_v25_native")
INTERPRETERS = {"3.7": "/usr/bin/python3.7", "3.8": "/usr/bin/python3.8",
                "3.9": "/usr/bin/python3.9", "3.10": "python3.10", "3.11": "/usr/bin/python3.11",
                "3.12": "/usr/bin/python3.12"}
REPEATS = 3
INSTALL_TIMEOUT = 1800
TEST_TIMEOUT = 1200
DJANGO_LINE = re.compile(r"^(test\w*) \(([\w.]+?)(?:\.(test\w*))?\)\s*\.\.\.\s*(ok|FAIL|ERROR|"
                         r"skipped|expected failure|unexpected success)", re.M)


def git(repo: Path, *args, timeout=1800) -> dict:
    return run(["git", "-C", str(repo), *args], timeout=timeout)


def clone(url: str) -> Path:
    dest = REPOS / url.rstrip("/").split("github.com/")[1].replace("/", "__")
    if not dest.exists():
        REPOS.mkdir(parents=True, exist_ok=True)
        done = run(["git", "clone", "--quiet", url, str(dest)], timeout=3600)
        if done["code"] != 0:
            raise RuntimeError(f"clone failed: {done['tail'][-200:]}")
    return dest


def apply_patch(checkout: Path, patch: str) -> bool:
    done = subprocess.run(["git", "-C", str(checkout), "apply", "--whitespace=nowarn", "-"],
                          input=patch, capture_output=True, text=True)
    return done.returncode == 0


def interpreter(version: str) -> str | None:
    path = INTERPRETERS.get(version)
    if path and not path.startswith("/"):
        found = subprocess.run(["uv", "python", "find", version], capture_output=True, text=True)
        path = found.stdout.strip() or None
    return path


# Explicit runtime dependencies for SWE-bench repositories (resolved at the buggy commit's date);
# the project itself is NEVER installed - it is importable only from a checkout or view on
# PYTHONPATH, exactly as in the generated-test sandbox.
SWEBENCH_DEPENDENCIES = {
    # SWE-bench's documented sympy pin; old mpmath sdists cannot build under a 2016 cutoff
    "sympy": ["mpmath==1.3.0", "pytest==7.4.4"],
    "django": ["asgiref", "pytz", "sqlparse", "pytest"]}


def build_env(env_dir: Path, python_path: str, project: str, date: str,
              requirements: Path | None) -> dict:
    if env_dir.exists():
        shutil.rmtree(env_dir)
    venv = run(["uv", "venv", "-q", "--python", python_path, str(env_dir)], timeout=300)
    if venv["code"] != 0:
        return {"ok": False, "failure": "environment_install_failed", "tail": venv["tail"][-300:]}
    python = str(env_dir / "bin" / "python")
    dropped = []
    if requirements is not None:
        # BugsInPy: its exact pins (they fix every version, so no date cutoff), minus the line
        # naming the project itself as an editable git requirement
        kept = []
        for line in requirements.read_text(encoding="utf-8").splitlines():
            (dropped if "git+" in line else kept).append(line)
        pinned = env_dir.parent / f"{env_dir.name}_requirements.txt"
        pinned.write_text("\n".join(kept) + "\n", encoding="utf-8")
        cmd = ["uv", "pip", "install", "-q", "--python", python, "-r", str(pinned)]
        if not any(l.lower().startswith("pytest==") for l in kept):
            cmd.append("pytest")
    else:
        deps = SWEBENCH_DEPENDENCIES.get(project, ["pytest"])
        # exact pins are reproducible on their own; anything unpinned resolves at the buggy
        # commit's date
        cutoff = [] if all("==" in d for d in deps) else ["--exclude-newer", date]
        cmd = ["uv", "pip", "install", "-q", "--python", python, *cutoff, *deps]
    done = run(cmd, timeout=INSTALL_TIMEOUT)
    if done["code"] != 0:
        text = done["tail"].lower()
        failure = ("no_approved_interpreter" if "requires-python" in text
                   or "requires python" in text else "environment_install_failed")
        return {"ok": False, "failure": failure, "tail": done["tail"][-400:]}
    return {"ok": True, "python": python, "lock": env_lock(python),
            "dropped_self_requirements": dropped,
            "install_command": [p.replace(str(env_dir.parent), "<prep>") for p in cmd]}


def attest(python: str, view: Path, module: str) -> dict:
    """Same checks as ``verify_import`` (imported from the view only; not importable without
    it), but importing exactly as the sandbox plugin does (``importlib.import_module``): the
    ``import a.b.c as m`` form breaks on packages that re-bind a submodule name (old sympy)."""
    code = ("import hashlib, importlib, os; m = importlib.import_module(%r); "
            "p = os.path.realpath(m.__file__); print(p); "
            "print(hashlib.sha256(open(p, 'rb').read()).hexdigest())" % module)
    env = {"PATH": "/usr/bin:/bin", "PYTHONPATH": str(view), "PYTHONDONTWRITEBYTECODE": "1",
           "PYTHONNOUSERSITE": "1"}
    done = subprocess.run([python, "-B", "-c", code], capture_output=True, text=True, cwd="/",
                          env=env)
    lines = done.stdout.split()
    ok = done.returncode == 0 and len(lines) == 2 and lines[0].startswith(str(view) + "/")
    bare = subprocess.run([python, "-B", "-c", "import importlib; importlib.import_module(%r)"
                           % module], capture_output=True, cwd="/",
                          env={"PATH": "/usr/bin:/bin", "PYTHONNOUSERSITE": "1"})
    return {"ok": ok and bare.returncode != 0,
            "module_sha256": lines[1] if len(lines) == 2 else None,
            "not_importable_without_view": bare.returncode != 0,
            "error": done.stderr[-300:] if not ok else None}


def pytest_cases(python: str, checkout: Path, selectors: list, outdir: Path, label: str) -> dict:
    xml = outdir / f"{label}.xml"
    env = {"PATH": "/usr/bin:/bin", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONNOUSERSITE": "1",
           "PYTHONPATH": str(checkout), "HOME": str(outdir)}
    run([python, "-B", "-m", "pytest", "-p", "no:cacheprovider", "-q", "-o", "addopts=",
         f"--junitxml={xml}", *selectors], cwd=checkout, timeout=TEST_TIMEOUT, env=env)
    cases = junit(xml)
    out = {}
    for sel in selectors:
        path, *middle, name = sel.split("::")     # path[::Class...]::test_name
        cls = path[:-3].replace("/", ".") if path.endswith(".py") else path
        if middle:
            cls = ".".join([cls, *middle])
        hits = {k: v for k, v in cases.items()
                if k.split("::")[0].endswith(cls) and (
                    k.split("::")[1] == name or k.split("::")[1].startswith(name + "["))}
        out[sel] = ("missing" if not hits else "failed" if "failed" in hits.values() else
                    "skipped" if all(v == "skipped" for v in hits.values()) else "passed")
    return out


def run_full(cmd, cwd, env, timeout):
    """run() keeps only a tail; Django's per-test lines need the full output."""
    started = time.time()
    try:
        p = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True, timeout=timeout)
        return {"code": p.returncode, "full": p.stdout + p.stderr,
                "tail": (p.stdout + p.stderr)[-1500:], "seconds": round(time.time() - started, 1)}
    except subprocess.TimeoutExpired:
        return {"code": "timeout", "full": "", "tail": "", "seconds": timeout}


def prepare(t: dict, patches: Path) -> dict:
    started = time.time()
    tag = re.sub(r"[^\w.-]", "_", t["task"])
    base = WORK / tag
    row = {"task": t["task"], "project": t["project"], "dataset": t["dataset"],
           "interpreter_version": t["interpreter"], "target_file": t["target_file"]}
    co = base / "checkouts"
    try:
        python_path = interpreter(t["interpreter"])
        if not python_path:
            return {**row, "category": "no_approved_interpreter"}
        repo = clone(t["repository_url"])
        if git(repo, "cat-file", "-e", f"{t['buggy_commit']}^{{commit}}")["code"] != 0:
            git(repo, "fetch", "--quiet", "origin", t["buggy_commit"])
        if base.exists():
            shutil.rmtree(base)
        swe = t["dataset"] == "SWE-bench Verified"
        patch = None
        if swe:
            pfile = patches / f"{t['task']}.json"
            if not pfile.is_file():
                return {**row, "category": "patch_unavailable"}
            patch = json.loads(pfile.read_text(encoding="utf-8"))
            for key, want in (("patch", "gold_patch_sha256"), ("test_patch", "test_patch_sha256")):
                if hashlib.sha256(patch[key].encode()).hexdigest() != t[want]:
                    return {**row, "category": "patch_hash_mismatch"}
        commits = {"buggy": t["buggy_commit"], "qual_buggy": t["buggy_commit"],
                   "fixed": t["buggy_commit"] if swe else t["fixed_commit"],
                   "qual_fixed": t["buggy_commit"] if swe else t["fixed_commit"]}
        for label, commit in commits.items():
            got = git(repo, "worktree", "add", "--force", "--detach", str(co / label), commit)
            if got["code"] != 0:
                return {**row, "category": "commit_not_exact", "failure": label}
        if swe:
            ok = (apply_patch(co / "fixed", patch["patch"])
                  and apply_patch(co / "qual_fixed", patch["patch"])
                  and apply_patch(co / "qual_fixed", patch["test_patch"])
                  and apply_patch(co / "qual_buggy", patch["test_patch"]))
            if not ok:
                return {**row, "category": "patch_apply_failed"}
        else:
            for sel in t["selectors"]:
                rel = sel.split("::")[0]
                src = co / "fixed" / rel
                if src.is_file():
                    (co / "qual_buggy" / rel).parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(src, co / "qual_buggy" / rel)
        date = git(repo, "show", "-s", "--format=%cI", t["buggy_commit"])["tail"].strip()
        requirements = None
        if t.get("bugsinpy_bug"):
            requirements = (REPO_ROOT / "data" / "BugsInPy_repo" / "projects" / t["project"] /
                            "bugs" / t["bugsinpy_bug"] / "requirements.txt")
            requirements = requirements if requirements.is_file() else None
        # built twice AT THE SAME PATH (console-script RECORD hashes embed the venv path), the
        # second build replacing the first; byte-identical locks prove reproducibility
        envs = []
        for _ in range(2):
            built = build_env(base / "env", python_path, t["project"], date, requirements)
            if not built["ok"]:
                return {**row, "category": built["failure"], "failure": built.get("tail")}
            envs.append(built)
        row["environment_lock"] = envs[1]["lock"]
        row["environment_lock_first_build"] = envs[0]["lock"]
        row["environment_reproducible"] = envs[0]["lock"] == envs[1]["lock"]
        row["install_command"] = envs[1]["install_command"]
        row["dropped_self_requirements"] = envs[1]["dropped_self_requirements"]
        row["resolution_cutoff"] = date
        python = envs[1]["python"]
        root_rel = import_root(t["target_file"])
        extra = generated_files(co / "fixed", root_rel)
        views = {label: base / "views" / label for label in ("buggy", "fixed")}
        manifests = {label: build_view(co / label, root_rel, views[label], extra)
                     for label in views}
        row["views"] = {k: str(v) for k, v in views.items()}
        row["view_manifest_sha256"] = {k: v["manifest_sha256"] for k, v in manifests.items()}
        module = module_name(t["target_file"])
        attest_rows = {label: attest(python, views[label], module) for label in views}
        row["module"] = module
        row["module_sha256"] = {k: v.get("module_sha256") for k, v in attest_rows.items()}
        if not all(a["ok"] for a in attest_rows.values()):
            return {**row, "category": "view_attestation_failed",
                    "failure": {k: (v.get("error") or "")[-200:] for k, v in attest_rows.items()}}
        if any(s.startswith("unresolved:") for s in t["selectors"]) or not t["selectors"]:
            return {**row, "category": "runner_incompatible", "failure": "unresolved selector"}
        outdir = base / "official"
        outdir.mkdir(parents=True, exist_ok=True)
        runs = {"buggy": [], "fixed": []}
        django = all(s.startswith("django:") for s in t["selectors"])
        for label in ("buggy", "fixed"):
            for i in range(REPEATS):
                checkout = co / f"qual_{label}"
                if django:
                    env = {"PATH": "/usr/bin:/bin", "PYTHONDONTWRITEBYTECODE": "1",
                           "PYTHONNOUSERSITE": "1", "PYTHONPATH": str(checkout),
                           "HOME": str(outdir), "LANG": "C.UTF-8"}
                    done = run_full([python, "-B", "tests/runtests.py", "--settings=test_sqlite",
                                     "--parallel", "1", "-v", "2",
                                     *[s.split(":", 1)[1] for s in t["selectors"]]],
                                    checkout, env, TEST_TIMEOUT)
                    status = {}
                    for m in DJANGO_LINE.finditer(done["full"]):
                        method, owner, inner, verdict = m.groups()
                        full = f"{owner}.{inner}" if inner else f"{owner}.{method}"
                        status[full] = {"ok": "passed", "FAIL": "failed",
                                        "ERROR": "failed"}.get(verdict, "skipped")
                    runs[label].append({s: status.get(s.split(":", 1)[1], "missing")
                                        for s in t["selectors"]})
                else:
                    runs[label].append(pytest_cases(python, checkout, t["selectors"], outdir,
                                                    f"{label}{i}"))
        stable = [s for s in t["selectors"]
                  if all(r[s] == "failed" for r in runs["buggy"])
                  and all(r[s] == "passed" for r in runs["fixed"])]
        row["qualification"] = {"repeats": REPEATS, "selectors": len(t["selectors"]),
                                "stable_3_of_3": len(stable),
                                "observed": {"buggy": runs["buggy"][0], "fixed": runs["fixed"][0]}}
        row["python_path"] = python
        row["env_dir"] = str(base / "env")
        if not row["environment_reproducible"]:
            return {**row, "category": "environment_not_reproducible"}
        if not stable or len(stable) != len(t["selectors"]):
            missing = any(v == "missing" for r in runs["buggy"] + runs["fixed"] for v in r.values())
            return {**row, "category": "runner_incompatible" if missing else "not_stable_3_of_3"}
        return {**row, "category": "qualified"}
    except Exception as exc:                                  # noqa: BLE001 - recorded
        return {**row, "category": "runner_error", "failure": f"{type(exc).__name__}: {exc}"[:300]}
    finally:
        for label in ("qual_buggy", "qual_fixed"):
            shutil.rmtree(co / label, ignore_errors=True)
        try:
            for r in REPOS.iterdir():
                run(["git", "-C", str(r), "worktree", "prune"])
        except OSError:
            pass


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--targets", required=True)
    parser.add_argument("--canary-only", action="store_true")
    parser.add_argument("--projects", default="")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    tpath = REPO_ROOT / args.targets
    targets = [json.loads(l) for l in tpath.read_text(encoding="utf-8").splitlines()]
    wanted = set(filter(None, args.projects.split(",")))
    targets = [t for t in targets if (t["canary"] or not args.canary_only)
               and (not wanted or t["project"] in wanted)]
    out = REPO_ROOT / args.out
    if out.exists():
        raise SystemExit(f"REFUSED: {out} exists")
    rows = []
    for t in targets:
        started = time.time()
        row = {**prepare(t, tpath.parent / "patches"),
               "wall_seconds": round(time.time() - started, 1)}
        rows.append(row)
        print(json.dumps({k: row.get(k) for k in ("task", "category", "wall_seconds",
                                                  "environment_reproducible")}), flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(("\n".join(json.dumps(r, sort_keys=True, default=str) for r in rows) + "\n")
                    .encode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
