"""v2.5 Phase 7: verify converted repository ``pytest_module_v1`` candidates in their qualified
native environments with the FROZEN v2.4 executor (static policy, sandboxed buggy/fixed runs,
reach tracking, kill reruns, classification).

Target (verifier-side, never shown to any test): the function or method of the buggy target
file that encloses the first line changed by the fix (the records' benchmark-declared affected
region), computed from the two qualified views. A change outside any function is
``target_unlocalizable``. Django fragments need Django's own runner settings, which the frozen
pytest sandbox does not provide: they are ``runner_incompatible`` and NOT executed (no
misleading collection failures). Only rows of ``qualified`` environments are executed.
Verdict files hold only deterministic fields (byte-comparable across repeatability runs).

    wsl -u root -- bash scripts/wsl_native_python.sh scripts/v25_verify_repository_wsl.py \
        --envs <env rows .jsonl> --run 1 --out <dir>
"""
from __future__ import annotations

import argparse
import ast
import difflib
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
import native_generated_tests_execute_wsl as ex  # noqa: E402
import v25_django_layer_wsl as dj  # noqa: E402

CANDIDATES = "results/sft_root_cause/v25_corpus_stage1_r2/converted_candidates.jsonl"
TARGETS = "results/sft_root_cause/v25_native/targets.jsonl"
SANDBOX_RUNNERS = ("thefuck", "sympy")          # pytest-native; django needs its own runner


def localize(buggy: str, fixed: str) -> str | None:
    """Qualname of the innermost function/method enclosing the first changed buggy line."""
    a, b = buggy.splitlines(), fixed.splitlines()
    changed = None
    for tag, i1, i2, _, _ in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if tag != "equal":
            changed = i1 + 1 if i2 > i1 else max(i1, 1)
            break
    if changed is None:
        return None
    best = None

    def walk(node, prefix):
        nonlocal best
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                name = f"{prefix}{child.name}"
                start = min([d.lineno for d in getattr(child, "decorator_list", [])]
                            + [child.lineno])
                if start <= changed <= child.end_lineno:
                    if not isinstance(child, ast.ClassDef):
                        best = name
                    walk(child, name + ".")
    walk(ast.parse(buggy), "")
    return best


def target_for(env: dict) -> tuple:
    rel = env["target_file"]
    root = ex.import_root(rel)
    path_in_view = rel[len(root) + 1:] if root else rel
    sources = {label: (Path(env["views"][label]) / path_in_view).read_text(encoding="utf-8")
               for label in ("buggy", "fixed")}
    qualname = localize(sources["buggy"], sources["fixed"])
    if qualname is None:
        return None, "target_unlocalizable"
    return {"module": env["module"], "qualname": qualname, "python": env["python_path"],
            "env_dir": env["env_dir"], "views": env["views"],
            "module_sha256": env["module_sha256"]}, None


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--envs", required=True)
    parser.add_argument("--run", type=int, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--candidates", default=CANDIDATES)
    parser.add_argument("--targets", default=TARGETS)
    parser.add_argument("--runners", default=",".join(SANDBOX_RUNNERS),
                        help="projects run in the pytest sandbox, or 'all'")
    parser.add_argument("--django-layer", action="store_true",
                        help="run django through the v2.5 Django settings layer")
    args = parser.parse_args(argv)
    runners = None if args.runners == "all" else set(args.runners.split(","))
    out_dir = REPO / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"repository_verification_run{args.run}.jsonl"
    if out.exists():
        raise SystemExit(f"REFUSED: {out.name} exists")
    envs = {r["task"]: r for r in map(json.loads, (REPO / args.envs).read_text(encoding="utf-8")
                                      .splitlines())}
    targets = [json.loads(l) for l in (REPO / args.targets).read_text(encoding="utf-8")
               .splitlines()]
    cands = {c["index"]: c for c in map(json.loads, (REPO / args.candidates).read_text(
        encoding="utf-8").splitlines())}
    layers = {}
    verdicts = []
    scratch = Path(tempfile.mkdtemp(prefix="oneiros_v25_repo_"))
    try:
        for t in targets:
            env = envs.get(t["task"])
            django = t["project"] == "django"
            runnable = (django and args.django_layer) or (not django and (
                runners is None or t["project"] in runners))
            target, why = (None, "no_environment_row") if env is None else (
                (None, f"environment:{env['category']}") if env["category"] != "qualified"
                else (None, "runner_incompatible") if not runnable
                else target_for(env))
            if target is not None and django:
                layered = dj.layered_env(env["env_dir"])
                python = str(layered / "bin" / Path(env["python_path"]).name)
                check = dj.canary(python, env["views"]["buggy"])
                layers[t["task"]] = {"layer": dj.VERSION, "files": dj.layer_files_sha256(),
                                     "canary_ok": check["ok"], "canary_error": check["error"],
                                     "lock": ex.env_lock(python)}
                if not check["ok"]:
                    target, why = None, "django_layer_canary_failed"
                else:
                    target = {**target, "python": python, "env_dir": str(layered)}
            for frag in t["fragments"]:
                cand = cands[frag["index"]]
                row = {"index": frag["index"], "task": t["task"], "project": t["project"],
                       "module_sha256": frag["module_sha256"],
                       "qualname": target["qualname"] if target else None}
                if target is None:
                    verdicts.append({**row, "status": why, "accepted": False})
                    continue
                outcome = ex.execute_candidate(target, cand["conversion"]["module"],
                                               scratch / "c")
                cls = outcome["classification"]
                fixed_valid = ex.fixed_valid_of(cls)
                verdicts.append({**row, "status": "executed", "class": cls["class"],
                                 "fixed_valid": fixed_valid,
                                 "rerun_agrees": cls.get("rerun_agrees"),
                                 "accepted": cls["class"] in ex.KILLS and fixed_valid})
                print(json.dumps({k: verdicts[-1][k] for k in ("task", "status", "accepted")}
                                 | {"class": verdicts[-1].get("class")}), flush=True)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    out.write_bytes(("\n".join(json.dumps(v, sort_keys=True) for v in verdicts) + "\n")
                    .encode("utf-8"))
    if layers:
        (out_dir / f"django_layers_run{args.run}.json").write_bytes(
            (json.dumps(layers, indent=1, sort_keys=True) + "\n").encode("utf-8"))
    print(json.dumps({"rows": len(verdicts), "accepted": sum(v["accepted"] for v in verdicts),
                      "sha256": hashlib.sha256(out.read_bytes()).hexdigest()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
