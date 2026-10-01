"""v2.5 CPU program 2, Phase 6A: audit (and, narrowly, recover) repository fragments rejected
by the frozen candidate policy. Train records only (strict loader, fail-closed access audit).

For every rejected repository row the converted module is rebuilt exactly as stage1_r2 built
it, and every static-policy violation is classified:
- where it comes from: the public import HEADER or the verified test FRAGMENT;
- for a forbidden import, whether any bound name is referenced anywhere in the module (AST);
- whether project test-support imports (``tests.*``, ``conftest``, relative imports) are
  required.

The ONLY transformation (recovery rule ``oneiros_v25_unused_forbidden_import_removal_v1``):
remove a forbidden import alias from the header when AST analysis proves the alias unused,
and nothing else. A row is recoverable only if, afterwards, the module parses, the fragment
(test body) AST is byte-identical, the frozen static policy reports ``ok`` (no name,
attribute or import violation remains: the sandbox and the policy are never weakened), and
no test-support import is needed. Assertions and expected values are never touched. Before
and after sources and hashes are written; recovered rows still need native verification.

    python scripts/v25_policy_recovery.py --examples PATH --out DIR
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RULE = "oneiros_v25_unused_forbidden_import_removal_v1"
RECEIPT = "results/sft_root_cause_v25_policy_recovery_audit.json"
R2 = "results/sft_root_cause/v25_corpus_stage1_r2/converted_candidates.jsonl"
TEST_SUPPORT_ROOTS = {"tests", "test", "testing", "conftest"}


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def referenced_names(tree: ast.AST) -> set:
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
    return names


def forbidden_aliases(stmt: ast.stmt, policy_imports: set) -> list:
    """(bound name, dotted module) for every forbidden alias of an import statement."""
    out = []
    if isinstance(stmt, ast.Import):
        for a in stmt.names:
            if a.name.split(".")[0] in policy_imports:
                out.append((a.asname or a.name.split(".")[0], a.name))
    elif isinstance(stmt, ast.ImportFrom) and (stmt.module or "").split(".")[0] in policy_imports:
        for a in stmt.names:
            out.append((a.asname or a.name, f"{stmt.module}.{a.name}"))
    return out


def test_support(tree: ast.AST) -> list:
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.level or (node.module or "")
                                                  .split(".")[0] in TEST_SUPPORT_ROOTS):
            out.append(f"from {'.' * node.level}{node.module or ''}")
        elif isinstance(node, ast.Import):
            out += [a.name for a in node.names if a.name.split(".")[0] in TEST_SUPPORT_ROOTS]
    return out


def remove_aliases(header: str, unused: set, policy_imports: set) -> str:
    """Drop exactly the unused forbidden aliases; keep every other alias and line."""
    kept = []
    for line in header.splitlines():
        try:
            stmts = ast.parse(line).body
        except SyntaxError:
            kept.append(line)
            continue
        if len(stmts) != 1 or not isinstance(stmts[0], (ast.Import, ast.ImportFrom)):
            kept.append(line)
            continue
        stmt = stmts[0]
        drop = {bound for bound, _ in forbidden_aliases(stmt, policy_imports)} & unused
        if not drop:
            kept.append(line)
            continue
        names = [a for a in stmt.names if (a.asname or (a.name.split(".")[0] if isinstance(
            stmt, ast.Import) else a.name)) not in drop]
        if names:
            stmt.names = names
            kept.append(ast.unparse(stmt))
    return "\n".join(kept)


def non_import_statements(source: str) -> list:
    """AST dumps of every top-level statement that is not an import (the test body)."""
    return [ast.dump(n) for n in ast.parse(source).body
            if not isinstance(n, (ast.Import, ast.ImportFrom))]


def remove_aliases_from_module(source: str, unused: set, policy_imports: set) -> str:
    """Drop unused forbidden aliases from TOP-LEVEL import statements only."""
    if not unused:
        return source
    tree = ast.parse(source)
    lines = source.splitlines()
    for stmt in reversed(tree.body):
        if not isinstance(stmt, (ast.Import, ast.ImportFrom)):
            continue
        drop = {b for b, _ in forbidden_aliases(stmt, policy_imports)} & unused
        if not drop:
            continue
        keep = [a for a in stmt.names if (a.asname or (a.name.split(".")[0] if isinstance(
            stmt, ast.Import) else a.name)) not in drop]
        replacement = []
        if keep:
            stmt.names = keep
            replacement = [ast.unparse(stmt)]
        lines[stmt.lineno - 1:stmt.end_lineno] = replacement
    return "\n".join(lines)


def audit_row(fragment: str, support_context: str, target: str) -> dict:
    from scripts import v25_module_conversion as mc
    from scripts.native_generated_tests_execute_wsl import POLICY_IMPORTS, static_check
    conv = mc.convert_repository(fragment, support_context)
    if not conv["accepted"]:
        return {"category": f"conversion:{conv['reason']}"}
    module = conv["module"]
    before = static_check(module, target)
    header = mc.repository_header(support_context) or ""
    frag_tree = ast.parse(fragment.strip())
    tree = ast.parse(module)
    used = referenced_names(tree)
    for node in ast.walk(tree):                       # attribute roots (os.path -> os)
        if isinstance(node, ast.Attribute):
            root = node
            while isinstance(root, ast.Attribute):
                root = root.value
            if isinstance(root, ast.Name):
                used.add(root.id)
    header_aliases, top_aliases, nested_aliases, inserted = [], [], [], []
    for stmt in ast.parse(header).body if header.strip() else []:
        header_aliases += forbidden_aliases(stmt, POLICY_IMPORTS)
    for stmt in frag_tree.body:                       # module level of the copied fragment
        top_aliases += forbidden_aliases(stmt, POLICY_IMPORTS)
        if not isinstance(stmt, (ast.Import, ast.ImportFrom)):
            for node in ast.walk(stmt):               # inside test functions / classes
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    nested_aliases += forbidden_aliases(node, POLICY_IMPORTS)
    for line in conv.get("inserted_imports") or []:   # converter-inserted: used by definition
        inserted += forbidden_aliases(ast.parse(line).body[0], POLICY_IMPORTS)
    non_import = [v for v in before.get("violations", []) if not v.startswith("import:")]
    unused_header = {b for b, _ in header_aliases if b not in used}
    unused_top = {b for b, _ in top_aliases if b not in used}
    used_forbidden = sorted({m for b, m in header_aliases + top_aliases if b in used}
                            | {m for _, m in inserted})
    support = test_support(tree)
    row = {"violations": before.get("violations", []),
           "violation_origin": {
               "header_imports": sorted({m for _, m in header_aliases}),
               "fragment_module_level_imports": sorted({m for _, m in top_aliases}),
               "fragment_nested_imports": sorted({m for _, m in nested_aliases}),
               "converter_inserted_imports": sorted({m for _, m in inserted}),
               "non_import": non_import},
           "forbidden_imports_referenced": used_forbidden,
           "unused_forbidden_aliases": sorted(unused_header | unused_top),
           "test_support_imports": support,
           "package_testing_imports": sorted({
               (n.module if isinstance(n, ast.ImportFrom) else a.name)
               for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))
               for a in n.names
               if ".testing" in "." + ((n.module or "") if isinstance(n, ast.ImportFrom)
                                       else a.name)}),
           "before_sha256": sha(module)}
    if before["status"] != "policy_refused":
        row["category"] = f"not_policy:{before['status']}"
        return row
    if non_import:
        row["category"] = "unrecoverable:forbidden_name_or_attribute_used"
    elif nested_aliases:
        row["category"] = "unrecoverable:forbidden_import_inside_test"
    elif used_forbidden:
        row["category"] = "unrecoverable:forbidden_module_used_by_test"
    elif support:
        row["category"] = "unrecoverable:test_support_import_required"
    else:
        new_header = remove_aliases(header, unused_header, POLICY_IMPORTS)
        # a minimal context carrying exactly the reduced header (the converter re-extracts it)
        new_ctx = f"{mc.HEADER_MARK}\n{new_header}\n" if header else support_context
        new_fragment = remove_aliases_from_module(fragment.strip(), unused_top, POLICY_IMPORTS)
        recovered = mc.convert_repository(new_fragment, new_ctx)
        after = static_check(recovered["module"], target) if recovered["accepted"] else {}
        body_same = recovered["accepted"] and \
            non_import_statements(new_fragment) == non_import_statements(fragment.strip())
        if after.get("status") == "ok" and body_same:
            row.update(category="recoverable:unused_forbidden_import_removed",
                       after_sha256=sha(recovered["module"]),
                       removed_aliases=sorted(unused_header | unused_top),
                       before_module=module, after_module=recovered["module"])
        else:
            row["category"] = f"unrecoverable:after_removal_{after.get('status')}"
    return row


def main(argv=None) -> int:
    from scripts.native_rehearsal_rebuild_v22 import publish_once
    from scripts.v25_converted_corpus import (EXAMPLES_SHA, access_receipt, install_audit,
                                              load_train_records)
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--examples", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    install_audit()
    examples = Path(args.examples)
    if hashlib.sha256(examples.read_bytes()).hexdigest() != EXAMPLES_SHA:
        raise SystemExit("REFUSED: examples differ from the audited exact rebuild")
    records = {r["id"]: r for r in load_train_records()}
    rows = [json.loads(l) for l in examples.read_text(encoding="utf-8").splitlines()]
    cands = [json.loads(l) for l in (ROOT / R2).read_text(encoding="utf-8").splitlines()]
    audited = []
    for c in cands:
        if c["execution_mode"] == "function_assertion" or c["conversion"]["accepted"]:
            continue
        ex, rec = rows[c["index"]], records[c["id"]]
        target = rec["entry_point"] or ((rec.get("target_symbols") or ["target"])[0])
        result = audit_row(ex["completion"], rec.get("support_context") or "",
                           target.split(".")[-1] or "target")
        audited.append({"index": c["index"], "id": c["id"], "project": c["project"],
                        "r2_reason": c["conversion"]["reason"], **result})
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    detail = out / "policy_recovery_rows.jsonl"
    if detail.exists():
        raise SystemExit(f"REFUSED: {detail} exists")
    detail.write_bytes(("\n".join(json.dumps(r, sort_keys=True) for r in audited) + "\n")
                       .encode("utf-8"))
    access = access_receipt()
    cats = Counter(r["category"] for r in audited)
    receipt = {
        "schema_version": "oneiros_v25_policy_recovery_audit_v1", "rule": RULE,
        "rejected_repository_rows": len(audited),
        "categories": dict(sorted(cats.items())),
        "by_project": {p: dict(Counter(r["category"] for r in audited if r["project"] == p))
                       for p in sorted({r["project"] for r in audited})},
        "violations": dict(Counter(v for r in audited for v in r.get("violations", []))),
        "recoverable_before_execution": sum(k.startswith("recoverable") for k in
                                            (r["category"] for r in audited)),
        "recoverable_indices": [r["index"] for r in audited
                                if r["category"].startswith("recoverable")],
        "rows_file": {"path": f"{args.out}/policy_recovery_rows.jsonl",
                      "sha256": hashlib.sha256(detail.read_bytes()).hexdigest()},
        "never": ["global os/sys allowance", "test-support imports", "assertion or expected "
                  "value edits", "sandbox weakening", "filesystem/network/process access"],
        "status": "AUDIT: recovered rows are candidates only; they still need native "
                  "buggy/fixed verification twice",
        "access_audit": {k: access[k] for k in ("passed", "opened")},
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    status = publish_once(RECEIPT, receipt)
    print(json.dumps({"status": status, **{k: receipt[k] for k in (
        "rejected_repository_rows", "categories", "recoverable_before_execution")},
        "access": receipt["access_audit"]}, indent=1))
    return 0 if access["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
