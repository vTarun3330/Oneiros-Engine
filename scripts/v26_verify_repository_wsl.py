"""v2.6 repository-candidate verification (WSL, stdlib only) with target rule F2.

Identical to the v2.5 verifier (frozen executor, static policy, sandboxed buggy/fixed runs, kill
reruns; Django through the v2.5 settings layer) except the verifier-side TARGET:

F2 (``oneiros_v26_target_any_changed_function_v1``): the target set is every buggy-side
function or method containing a line changed by the fix, ordered by definition line with an
enclosing function before its nested functions. The candidate is executed against the targets
in that order and the first target it REACHES defines the verdict; if it reaches none, the
verdict is ``target_not_reached``. A fix that changes no function is ``target_unlocalizable``.
(v2.5 used only the innermost function enclosing the FIRST changed line, which failed when that
line was an import or decorator, or when the test exercised the enclosing function.)

    wsl -u root -- bash scripts/wsl_native_python.sh scripts/v26_verify_repository_wsl.py \
        --envs FILE... --targets FILE --candidates FILE --run N --out DIR
"""
from __future__ import annotations

import argparse
import ast
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
from v26_funnel_diag_wsl import changed_functions  # noqa: E402

TARGET_RULE = "oneiros_v26_target_any_changed_function_v1"


def ordered_targets(buggy: str, fixed: str) -> list:
    """Changed functions by definition line; an enclosing function precedes nested ones."""
    names = changed_functions(buggy, fixed)
    lines = {}

    def walk(node, prefix):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                name = f"{prefix}{child.name}"
                lines.setdefault(name, child.lineno)
                walk(child, name + ".")
    walk(ast.parse(buggy), "")
    return sorted(names, key=lambda n: (lines.get(n, 0), n.count(".")))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--envs", nargs="+", required=True)
    parser.add_argument("--targets", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--run", type=int, required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    out_dir = REPO / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"repository_verification_run{args.run}.jsonl"
    if out.exists():
        raise SystemExit(f"REFUSED: {out.name} exists")
    envs = {}
    for rel in args.envs:
        for row in map(json.loads, (REPO / rel).read_text(encoding="utf-8").splitlines()):
            envs[row["task"]] = row
    targets = [json.loads(l) for l in (REPO / args.targets).read_text(encoding="utf-8")
               .splitlines()]
    cands = {c["index"]: c for c in map(json.loads, (REPO / args.candidates).read_text(
        encoding="utf-8").splitlines())}
    verdicts, layers = [], {}
    scratch = Path(tempfile.mkdtemp(prefix="oneiros_v26_repo_"))
    try:
        for t in targets:
            env = envs.get(t["task"])
            for frag in t["fragments"]:
                row = {"index": frag["index"], "task": t["task"], "project": t["project"],
                       "module_sha256": frag["module_sha256"], "target_rule": TARGET_RULE}
                if env is None or env["category"] != "qualified":
                    why = "no_environment_row" if env is None else f"environment:{env['category']}"
                    verdicts.append({**row, "status": why, "qualname": None, "accepted": False})
                    continue
                rel = t["target_file"]
                root = ex.import_root(rel)
                inview = rel[len(root) + 1:] if root else rel
                src = {lab: (Path(env["views"][lab]) / inview).read_text(encoding="utf-8")
                       for lab in ("buggy", "fixed")}
                order = ordered_targets(src["buggy"], src["fixed"])
                if not order:
                    verdicts.append({**row, "status": "target_unlocalizable", "qualname": None,
                                     "accepted": False})
                    continue
                base = {"module": env["module"], "python": env["python_path"],
                        "env_dir": env["env_dir"], "views": env["views"],
                        "module_sha256": env["module_sha256"]}
                if t["project"] == "django":
                    layered = dj.layered_env(env["env_dir"])
                    python = str(layered / "bin" / Path(env["python_path"]).name)
                    check = dj.canary(python, env["views"]["buggy"])
                    layers[t["task"]] = {"layer": dj.VERSION, "canary_ok": check["ok"],
                                         "lock": ex.env_lock(python)}
                    if not check["ok"]:
                        verdicts.append({**row, "status": "django_layer_canary_failed",
                                         "qualname": None, "accepted": False})
                        continue
                    base.update(python=python, env_dir=str(layered))
                module = cands[frag["index"]]["conversion"]["module"]
                tried = []
                for qual in order:
                    outcome = ex.execute_candidate({**base, "qualname": qual}, module,
                                                   scratch / "c")
                    cls = outcome["classification"]
                    tried.append(qual)
                    if cls["class"] != "target_not_reached":
                        break
                fixed_valid = ex.fixed_valid_of(cls)
                verdicts.append({**row, "status": "executed", "qualname": qual,
                                 "targets_tried": tried, "class": cls["class"],
                                 "fixed_valid": fixed_valid,
                                 "rerun_agrees": cls.get("rerun_agrees"),
                                 "accepted": cls["class"] in ex.KILLS and fixed_valid})
                print(json.dumps({"task": t["task"], "class": cls["class"]}), flush=True)
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
