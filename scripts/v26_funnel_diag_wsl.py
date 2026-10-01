"""v2.6 Phase 2 (WSL, stdlib only): diagnostic evidence for the train-only failure funnel.

For every eligible repository fragment (v2.5 r2 targets + the Phase 6 thefuck environments)
records the environment / localisation stage and, for executed fragments, re-executes the
candidate with the FROZEN executor and keeps the evidence the verdict files drop: collection
errors, import errors (missing module / name), per-node exception type and message head, reach
flags, every function the fix changed (not only the first), and the names the fragment calls.
Diagnostic only: verdicts are NOT replaced; train-side artifacts only.

    wsl -u root -- bash scripts/wsl_native_python.sh scripts/v26_funnel_diag_wsl.py \
        --envs FILE... --targets FILE --candidates FILE --out FILE
"""
from __future__ import annotations

import argparse
import ast
import difflib
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys
import tempfile

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
import native_generated_tests_execute_wsl as ex  # noqa: E402
import v25_django_layer_wsl as dj  # noqa: E402
from v25_verify_repository_wsl import localize  # noqa: E402

MISSING = re.compile(r"No module named '([\w.]+)'")
CANNOT = re.compile(r"cannot import name '(\w+)' from '([\w.]+)'")


def changed_functions(buggy: str, fixed: str) -> list:
    """Qualnames of every buggy-side function/method containing a changed line."""
    a, b = buggy.splitlines(), fixed.splitlines()
    lines = set()
    for tag, i1, i2, _, _ in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if tag != "equal":
            lines.update(range(i1 + 1, i2 + 1) if i2 > i1 else [max(i1, 1)])
    out = []

    def walk(node, prefix):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                name = f"{prefix}{child.name}"
                start = min([d.lineno for d in child.decorator_list] + [child.lineno])
                if not isinstance(child, ast.ClassDef) and \
                        any(start <= ln <= child.end_lineno for ln in lines):
                    out.append(name)
                walk(child, name + ".")
    try:
        walk(ast.parse(buggy), "")
    except SyntaxError:
        return []
    return sorted(set(out))


def called_names(source: str) -> list:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    names = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Call):
            f = n.func
            names.add(f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", None))
    return sorted(x for x in names if x)


def text_of(report: dict) -> str:
    parts = [json.dumps(report.get("collection_errors") or ""),
             json.dumps(report.get("target_error") or "")]
    for node in (report.get("nodes") or {}).values():
        for phase in (node.get("phases") or {}).values():
            exc = (phase or {}).get("exception") or {}
            parts.append(json.dumps(exc))
    return "\n".join(parts)


def summarise(outcome: dict) -> dict:
    reports = outcome["evidence"]["reports"]
    out = {}
    for label in ("buggy", "fixed"):
        r = reports.get(label) or {}
        text = text_of(r)
        nodes = r.get("nodes") or {}
        excs = []
        for n, node in nodes.items():
            call = (node.get("phases") or {}).get("call") or {}
            exc = call.get("exception") or {}
            if exc:
                excs.append({k: (str(v)[:160] if isinstance(v, str) else v)
                             for k, v in exc.items() if k in ("type", "message", "assertion",
                                                              "target_in_traceback",
                                                              "project_in_traceback",
                                                              "timeout")})
        out[label] = {"collected": len(r.get("collected") or []),
                      "collection_errors": bool(r.get("collection_errors")),
                      "target_error": (str(r.get("target_error"))[:200]
                                       if r.get("target_error") else None),
                      "missing_modules": sorted(set(MISSING.findall(text))),
                      "cannot_import": sorted(set(f"{m}.{n}" for n, m in CANNOT.findall(text))),
                      "reached": [bool(node.get("reached")) for node in nodes.values()],
                      "exceptions": excs[:3],
                      "error_head": text[:300] if (r.get("collection_errors")
                                                   or r.get("target_error")) else None}
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--envs", nargs="+", required=True)
    parser.add_argument("--targets", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    out = REPO / args.out
    if out.exists():
        raise SystemExit(f"REFUSED: {out} exists")
    envs = {}
    for rel in args.envs:
        for row in map(json.loads, (REPO / rel).read_text(encoding="utf-8").splitlines()):
            envs[row["task"]] = row
    targets = [json.loads(l) for l in (REPO / args.targets).read_text(encoding="utf-8")
               .splitlines()]
    cands = {c["index"]: c for c in map(json.loads, (REPO / args.candidates).read_text(
        encoding="utf-8").splitlines())}
    rows = []
    scratch = Path(tempfile.mkdtemp(prefix="oneiros_v26_diag_"))
    try:
        for t in targets:
            env = envs.get(t["task"])
            for frag in t["fragments"]:
                cand = cands[frag["index"]]
                module = cand["conversion"]["module"]
                row = {"index": frag["index"], "task": t["task"], "project": t["project"],
                       "dataset": t["dataset"], "recovered": frag.get("recovered", False),
                       "fragment_calls": called_names(module),
                       "async_test": "async def test" in module,
                       "target_symbols": t.get("target_symbols")}
                if env is None:
                    rows.append({**row, "stage": "environment", "detail": "no_environment_row"})
                    continue
                row["environment_category"] = env["category"]
                if env["category"] != "qualified":
                    rows.append({**row, "stage": "environment", "detail": env["category"],
                                 "failure": str(env.get("failure"))[-240:]})
                    continue
                rel = t["target_file"]
                root = ex.import_root(rel)
                inview = rel[len(root) + 1:] if root else rel
                src = {lab: (Path(env["views"][lab]) / inview).read_text(encoding="utf-8")
                       for lab in ("buggy", "fixed")}
                qual = localize(src["buggy"], src["fixed"])
                row["changed_functions"] = changed_functions(src["buggy"], src["fixed"])
                row["localized"] = qual
                if qual is None:
                    rows.append({**row, "stage": "localization"})
                    continue
                target = {"module": env["module"], "qualname": qual,
                          "python": env["python_path"], "env_dir": env["env_dir"],
                          "views": env["views"], "module_sha256": env["module_sha256"]}
                if t["project"] == "django":
                    layered = dj.layered_env(env["env_dir"])
                    target.update(python=str(layered / "bin" / Path(env["python_path"]).name),
                                  env_dir=str(layered))
                outcome = ex.execute_candidate(target, module, scratch / "c")
                cls = outcome["classification"]
                rows.append({**row, "stage": "executed", "class": cls["class"],
                             "fixed_valid": ex.fixed_valid_of(cls),
                             "static": outcome["static"].get("status"),
                             "evidence": summarise(outcome)})
                print(json.dumps({"task": t["task"], "class": cls["class"]}), flush=True)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(("\n".join(json.dumps(r, sort_keys=True) for r in rows) + "\n")
                    .encode("utf-8"))
    print(json.dumps({"rows": len(rows), "sha256": hashlib.sha256(out.read_bytes())
                      .hexdigest()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
