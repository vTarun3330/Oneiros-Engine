"""v2.7 panel admission: target reach and direction of the OFFICIAL difference-exposing tests in
the FROZEN evaluation sandbox (WSL, stdlib only; run inside scripts/wsl_isolated.sh).

For each requalified target, the official test file (verifier-only material, never shown to a
model) is reduced by AST to its recorded difference-exposing tests (imports, module-level
helpers and fixtures kept; every other test dropped) and executed with the frozen executor
(``execute_candidate``: sandboxed buggy and fixed runs against the prepared views, target-reach
tracking, kill rerun). Static candidate policy is not enforced (this is the official test, not
a model candidate); the sandbox is. ``reach_verified`` requires a semantic or crash kill:
every reduced test reached the target, passed on fixed and failed on buggy, twice.

    bash scripts/wsl_isolated.sh bash scripts/wsl_native_python.sh scripts/v27_reach_wsl.py \
        --manifest M --prep DIR --out FILE
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


def exposing_names(test_ids: list) -> dict:
    """{module dotted path: {(class or None, function)}} from ``pkg.mod::[Cls::]test``."""
    out = {}
    for tid in test_ids:
        mod, _, rest = tid.partition("::")
        parts = rest.split("::")
        cls, fn = (parts[0], parts[-1]) if len(parts) > 1 else (None, parts[0])
        out.setdefault(mod, set()).add((cls, fn.split("[")[0]))
    return out


def reduce_module(source: str, wanted: set) -> str | None:
    tree = ast.parse(source)
    keep_fns = {fn for cls, fn in wanted if cls is None}
    keep_cls = {}
    for cls, fn in wanted:
        if cls is not None:
            keep_cls.setdefault(cls, set()).add(fn)
    body, kept = [], 0
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and \
                node.name.startswith("test"):
            if node.name in keep_fns:
                body.append(node)
                kept += 1
        elif isinstance(node, ast.ClassDef) and (node.name.startswith("Test")
                                                 or node.name in keep_cls):
            if node.name in keep_cls:
                node.body = [n for n in node.body
                             if not (isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                                     and n.name.startswith("test")
                                     and n.name not in keep_cls[node.name])] or [ast.Pass()]
                body.append(node)
                kept += sum(1 for n in node.body
                            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                            and n.name in keep_cls[node.name])
        else:
            body.append(node)
    if kept == 0:
        return None
    tree.body = body
    return ast.unparse(ast.fix_missing_locations(tree)) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--prep", required=True, help="results/.../<name>_prep directory")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    out = REPO / args.out
    if out.exists():
        raise SystemExit(f"REFUSED: {out} exists")
    manifest = json.loads((REPO / args.manifest).read_text(encoding="utf-8"))
    targets = {t["key"]: t for t in manifest["targets"]}
    prep = REPO / args.prep
    records = {}
    for line in (prep / "records.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            records[r["key"]] = r
    rows = []
    scratch = Path(tempfile.mkdtemp(prefix="oneiros_v27_reach_"))
    try:
        for key, rec in sorted(records.items()):
            row = {"key": key, "repository": rec.get("repository")}
            if rec.get("category") != "requalified":
                rows.append({**row, "reach": "not_requalified", "prep": rec.get("category")})
                continue
            verifier = json.loads((prep / "verifier" / f"{rec['tag']}.json").read_text(
                encoding="utf-8"))
            wanted = exposing_names(targets[key]["difference_exposing_tests"])
            reduced = None
            for src in verifier["official_tests"]:
                for names in wanted.values():
                    try:
                        reduced = reduce_module(src, names) if src else None
                    except SyntaxError:
                        reduced = None
                    if reduced:
                        break
                if reduced:
                    break
            if not reduced:
                rows.append({**row, "reach": "official_test_not_reducible"})
                continue
            target = {"module": rec["module"], "qualname": rec["qualname"],
                      "python": rec["python_path"], "env_dir": rec["env_dir"],
                      "views": rec["views"], "module_sha256": rec["module_sha256"]}
            outcome = ex.execute_candidate(target, reduced, scratch / "c", enforce_policy=False)
            cls = outcome["classification"]
            rows.append({**row, "reach": "reach_verified" if cls["class"] in ex.KILLS
                         and ex.fixed_valid_of(cls) else f"not_verified:{cls['class']}",
                         "class": cls["class"], "rerun_agrees": cls.get("rerun_agrees"),
                         "reduced_test_sha256": hashlib.sha256(reduced.encode()).hexdigest()})
            print(json.dumps({"key": key[-40:], "reach": rows[-1]["reach"]}), flush=True)
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
