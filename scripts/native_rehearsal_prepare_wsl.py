"""Formal re-qualification and preparation of rehearsal targets (WSL, root; stdlib only).

Amendment v2.1 sections A and B. For each target of the frozen manifest:
  1. verify the exact buggy and fixed commit SHAs;
  2. build ONE environment with the approved interpreter (CPython 3.11, /usr/bin/python3.11)
     from the fixed revision's declared test dependencies; strip the project's editable
     finder so the project is importable only through a view; record the environment lock
     (sanitised freeze + site-packages manifest hash); a project that cannot install under
     3.11 because of its Python requirement is ``no_approved_interpreter``;
  3. build sanitised canonical VIEWS of both revisions (tests and metadata removed at any
     depth; build-generated files present in the fixed tree are copied into both) and
     attest the imported target module from each view, outside any sandbox;
  4. QUALIFY with the official regression tests under the same interpreter and environment:
     EVERY recorded difference-exposing test must fail 3/3 on buggy (fixed tests copied into a
     separate qualification copy, BugsInPy style) and pass 3/3 on fixed;
  5. export buggy_view/<tag>.json (sanitised DTO + buggy file; the only input to the prompt
     builder) and verifier/<tag>.json (fixed file, patch, official tests, commit message;
     verifier-only), in separate directories.
Rows are appended durably and bound to a contract (sources, manifest, protocol, interpreter,
runner, limits); stale, partial, duplicate or wrong-contract rows are quarantined; failures
that may be transient are retried on resume, semantic failures are final.

    python native_rehearsal_prepare_wsl.py <manifest.json> <prep_dir> <protocol files...>
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from native_rehearsal_wsl import WORK, install_plan, junit, module_name, run  # noqa: E402
from native_generated_tests_execute_wsl import (  # noqa: E402
    APPROVED_PYTHON, build_view, env_lock, generated_files, import_root, sha256_file,
    strip_project_install, verify_import)

REPEATS = 3
LOCK = threading.Lock()
DTO_KEYS = ("target_key", "repository", "buggy_commit", "target_file", "qualname")
FINAL = {"requalified", "not_stable_3_of_3", "no_approved_interpreter", "commit_not_exact",
         "target_file_missing", "view_attestation_failed"}
INSTALL_TIMEOUT = 900
TEST_TIMEOUT = 900


def tag_of(key: str) -> str:
    """Filesystem- and URL-safe directory name for a target key (no '#', '%', spaces...)."""
    import re
    base = key.replace("cand:", "").replace("/", "__").replace("@", "_")
    return re.sub(r"[^A-Za-z0-9._-]", "_", base)[:96]


def git(repo: Path, *args: str) -> dict:
    return run(["git", "-C", str(repo), *args], timeout=600)


def show(repo: Path, commit: str, path: str):
    done = subprocess.run(["git", "-C", str(repo), "show", f"{commit}:{path}"],
                          capture_output=True, text=True, timeout=300)
    return done.stdout if done.returncode == 0 else None


def build_env_311(env_dir: Path, fixed: Path) -> dict:
    plan = install_plan(fixed)
    if env_dir.exists():
        shutil.rmtree(env_dir)
    venv = run(["uv", "venv", "--python", APPROVED_PYTHON, str(env_dir)], timeout=300)
    if venv["code"] != 0:
        return {"ok": False, "failure": "environment_install_failed", "tail": venv["tail"][-300:]}
    python = str(env_dir / "bin" / "python")
    spec = str(fixed) + (f"[{','.join(plan['extras'])}]" if plan["extras"] else "")
    cmd = ["uv", "pip", "install", "--python", python, "-e", spec, "pytest",
           *plan["group_requirements"]]
    for req in plan["requirement_files"]:
        cmd += ["-r", str(fixed / req)]
    result = run(cmd, timeout=INSTALL_TIMEOUT)
    if result["code"] != 0:
        text = result["tail"].lower()
        failure = ("no_approved_interpreter" if "requires-python" in text
                   or "requires python" in text else "environment_install_failed")
        return {"ok": False, "failure": failure, "tail": result["tail"][-300:]}
    return {"ok": True, "python": python, "plan": plan,
            "install_command": [p.replace(str(fixed), "<checkout>").replace(str(WORK), "<work>")
                                for p in cmd]}


def run_official(python: str, checkout: Path, root_rel: str, tests: list, label: str,
                 outdir: Path) -> dict:
    xml = outdir / f"{label}.xml"
    env = {"PATH": "/usr/bin:/bin", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONNOUSERSITE": "1",
           "PYTHONPATH": str(checkout / root_rel) if root_rel else str(checkout),
           "HOME": str(outdir)}
    result = run([python, "-B", "-m", "pytest", "-p", "no:cacheprovider", "-q", "-o", "addopts=",
                  f"--junitxml={xml}", *tests], cwd=checkout, timeout=TEST_TIMEOUT, env=env)
    return {"exit": result["code"], "cases": junit(xml)}


def prepare_target(target: dict, repo: Path, prep_root: Path, exports: Path,
                   keep: bool = True) -> dict:
    started = time.time()
    tag = tag_of(target["key"])
    base = prep_root / tag
    co = base / "checkouts"
    row = {"key": target["key"], "repository": target["repository"], "tag": tag,
           "module": module_name(target["target_file"]), "qualname": target["target"],
           "interpreter": APPROVED_PYTHON}
    try:
        for label in ("buggy", "fixed"):
            got = git(repo, "rev-parse", "--verify", f"{target[f'{label}_commit']}^{{commit}}")
            if got["code"] != 0 or got["tail"].strip() != target[f"{label}_commit"]:
                return {**row, "category": "commit_not_exact", "failure": label}
        if base.exists():
            shutil.rmtree(base)
        for label, commit in (("fixed", target["fixed_commit"]), ("buggy", target["buggy_commit"]),
                              ("qualify_buggy", target["buggy_commit"])):
            step = git(repo, "worktree", "add", "--force", "--detach", str(co / label), commit)
            if step["code"] != 0:
                return {**row, "category": "environment_install_failed",
                        "failure": f"checkout_{label}"}
        fixed, buggy, qual = co / "fixed", co / "buggy", co / "qualify_buggy"
        env = build_env_311(base / "env", fixed)
        if not env["ok"]:
            return {**row, "category": env["failure"], "failure": env.get("tail")}
        python = env["python"]
        row["stripped_install_files"] = strip_project_install(python, fixed)
        row["environment_lock"] = env_lock(python)
        row["install_command"] = env["install_command"]
        root_rel = import_root(target["target_file"])
        extra = generated_files(fixed, root_rel)
        row["generated_files_copied"] = sorted(extra)
        views = {label: base / "views" / label for label in ("buggy", "fixed")}
        manifests = {label: build_view(checkout, root_rel, views[label], extra)
                     for label, checkout in (("buggy", buggy), ("fixed", fixed))}
        row["views"] = {k: str(v) for k, v in views.items()}
        row["view_manifest_sha256"] = {k: v["manifest_sha256"] for k, v in manifests.items()}
        attest = {label: verify_import(python, views[label], row["module"]) for label in views}
        row["attestation"] = {k: {x: v[x] for x in ("ok", "relative", "not_importable_without_view")}
                              for k, v in attest.items()}
        row["module_sha256"] = {k: v["module_sha256"] for k, v in attest.items()}
        if not all(a["ok"] for a in attest.values()) or \
                attest["buggy"]["relative"] != attest["fixed"]["relative"]:
            return {**row, "category": "view_attestation_failed",
                    "failure": {k: v.get("error") for k, v in attest.items()}}
        tests = [t for t in target["regression_test_files"] if (fixed / t).exists()]
        for t in tests:
            (qual / t).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(fixed / t, qual / t)
        outdir = base / "junit"
        outdir.mkdir(parents=True, exist_ok=True)
        runs = {"buggy": [], "fixed": []}
        for label, checkout in (("buggy", qual), ("fixed", fixed)):
            for i in range(REPEATS):
                runs[label].append(run_official(python, checkout, root_rel, tests,
                                                f"{label}{i}", outdir)["cases"])
        wanted = list(target["difference_exposing_tests"])
        stable = [n for n in wanted if all(r.get(n) == "failed" for r in runs["buggy"])
                  and all(r.get(n) == "passed" for r in runs["fixed"])]
        row["qualification"] = {"repeats": REPEATS, "recorded": len(wanted),
                                "stable_3_of_3": len(stable),
                                "unstable": sorted(set(wanted) - set(stable))[:10],
                                "rule": "every recorded difference-exposing test"}
        buggy_source = show(repo, target["buggy_commit"], target["target_file"])
        fixed_source = show(repo, target["fixed_commit"], target["target_file"])
        if buggy_source is None or fixed_source is None:
            return {**row, "category": "target_file_missing"}
        (exports / "buggy_view").mkdir(parents=True, exist_ok=True)
        (exports / "verifier").mkdir(parents=True, exist_ok=True)
        dto = {"target_key": target["key"], "repository": target["repository"],
               "buggy_commit": target["buggy_commit"], "target_file": target["target_file"],
               "qualname": target["target"]}
        (exports / "buggy_view" / f"{tag}.json").write_text(json.dumps(
            {"dto": {k: dto[k] for k in DTO_KEYS}, "buggy_source": buggy_source}), encoding="utf-8")
        patch = git(repo, "diff", target["buggy_commit"], target["fixed_commit"], "--",
                    target["target_file"])
        (exports / "verifier" / f"{tag}.json").write_text(json.dumps({
            "buggy_source": buggy_source, "fixed_source": fixed_source,
            "patch": patch["tail"] if patch["code"] == 0 else "",
            "official_tests": [show(repo, target["fixed_commit"], t) or "" for t in tests],
            "commit_message": git(repo, "log", "-1", "--format=%B", target["fixed_commit"])["tail"]}),
            encoding="utf-8")
        row["python_path"] = python
        row["env_dir"] = str(base / "env")
        ok = bool(wanted) and len(stable) == len(wanted)
        return {**row, "category": "requalified" if ok else "not_stable_3_of_3"}
    except Exception as exc:          # recorded, never silently dropped
        return {**row, "category": "runner_error", "failure": f"{type(exc).__name__}: {exc}"[:300]}
    finally:
        row["wall_seconds"] = round(time.time() - started, 1)
        shutil.rmtree(co, ignore_errors=True)
        git(repo, "worktree", "prune")
        if not keep:
            shutil.rmtree(base, ignore_errors=True)


def contract_for(manifest_path: Path, protocol_files: list) -> dict:
    return {"prepare_sha256": sha256_file(Path(__file__)),
            "executor_sha256": sha256_file(HERE / "native_generated_tests_execute_wsl.py"),
            "rehearsal_wsl_sha256": sha256_file(HERE / "native_rehearsal_wsl.py"),
            "manifest_sha256": sha256_file(manifest_path),
            "protocol_sha256": {Path(p).name: sha256_file(Path(p)) for p in protocol_files},
            "interpreter": APPROVED_PYTHON, "repeats": REPEATS,
            "install_timeout": INSTALL_TIMEOUT, "test_timeout": TEST_TIMEOUT}


def load_rows(path: Path, chash: str) -> dict:
    rows, problems = {}, []
    if not path.exists():
        return rows
    data = path.read_bytes()
    if data and not data.endswith(b"\n"):
        problems.append("partial final line")
    for number, line in enumerate(data.decode("utf-8").splitlines(), 1):
        try:
            row = json.loads(line)
        except ValueError:
            problems.append(f"malformed {number}")
            continue
        if row.get("contract_sha256") != chash:
            problems.append(f"line {number}: different contract")
        else:
            rows.setdefault(row["key"], []).append(row)
    if problems:
        qdir = path.parent / "quarantine"
        qdir.mkdir(exist_ok=True)
        shutil.move(str(path), str(qdir / f"{int(time.time() * 1000)}_{path.name}"))
        with (path.parent / "resume_log.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"quarantined": path.name, "reasons": problems[:20]}) + "\n")
        return {}
    return rows


def main() -> int:
    manifest_path, prep_root = Path(sys.argv[1]), Path(sys.argv[2])
    protocol_files = sys.argv[3:]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    prep_root.mkdir(parents=True, exist_ok=True)
    contract = contract_for(manifest_path, protocol_files)
    chash = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    (prep_root / "contract.json").write_text(json.dumps(contract, indent=1, sort_keys=True) + "\n")
    records = prep_root / "records.jsonl"
    previous = load_rows(records, chash)
    todo = []
    for t in manifest["targets"]:
        history = previous.get(t["key"], [])
        if history and history[-1]["category"] in FINAL:
            continue
        todo.append((t, len(history) + 1))
    by_repo: dict = {}
    for t, attempt in todo:
        by_repo.setdefault(t["repository"], []).append((t, attempt))
    urls = {t["repository"]: t["repository_url"] for t in manifest["targets"]}
    exports = prep_root / "exports"

    def work(repository, items):
        repo = WORK / "repos" / repository.replace("/", "__")
        if not repo.exists():
            clone = run(["git", "clone", "--filter=blob:none", "--no-checkout", urls[repository],
                         str(repo)], timeout=1800)
            if clone["code"] != 0:
                for t, attempt in items:
                    write({"key": t["key"], "category": "environment_install_failed",
                           "failure": "clone", "attempt": attempt})
                return
        for t, attempt in items:
            write({**prepare_target(t, repo, prep_root, exports), "attempt": attempt})

    def write(row):
        row["contract_sha256"] = chash
        with LOCK, records.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        print(f"[{row.get('wall_seconds', 0):6.1f}s] {row['category']:26s} {row['key']}", flush=True)

    started = time.time()
    with ThreadPoolExecutor(max_workers=4) as pool:
        for repository, items in by_repo.items():
            pool.submit(work, repository, items)
    (prep_root / "run_summary.json").write_text(json.dumps(
        {"wall_seconds": round(time.time() - started, 1), "contract_sha256": chash,
         "uv": run(["uv", "--version"])["tail"].strip()}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
