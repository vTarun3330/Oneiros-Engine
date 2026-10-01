"""Shape audit of the exact SFT training completions versus the v2.4 whole-module contract."""
import ast
import builtins
import collections
import json
from pathlib import Path
import re
import sys
import warnings

warnings.filterwarnings("ignore")
rows = [json.loads(l) for l in Path(sys.argv[1]).read_text(encoding="utf-8").splitlines()]
from transformers import AutoTokenizer  # noqa: E402
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-1.5B-Instruct",
                                    revision="2e1fd397ee46e1388853d2af2c993145b0f1098a",
                                    local_files_only=True)
BUILTINS = set(dir(builtins))


def undefined_names(tree):
    defined, used = set(), set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for a in node.names:
                defined.add((a.asname or a.name).split(".")[0])
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            defined.add(node.name)
            if not isinstance(node, ast.ClassDef):
                for a in [*node.args.args, *node.args.kwonlyargs, *node.args.posonlyargs]:
                    defined.add(a.arg)
                for a in (node.args.vararg, node.args.kwarg):
                    if a:
                        defined.add(a.arg)
        elif isinstance(node, ast.Name):
            (used if isinstance(node.ctx, ast.Load) else defined).add(node.id)
        elif isinstance(node, ast.arg):
            defined.add(node.arg)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            defined.add(node.name)
        elif isinstance(node, ast.alias):
            pass
    return used - defined - BUILTINS


def shape(r):
    c = r["completion"]
    f = {"fence": "```" in c}
    try:
        tree = ast.parse(c)
        f["parses"] = True
    except SyntaxError:
        tree, f["parses"] = None, False
    f["imports"] = bool(re.search(r"^\s*(import|from)\s+\w", c, re.M))
    f["uses_pytest"] = "pytest." in c or "@pytest" in c
    f["imports_pytest"] = bool(re.search(r"^\s*(import pytest|from pytest\b)", c, re.M))
    f["test_function"] = bool(re.search(r"^\s*(async\s+)?def test\w*\s*\(", c, re.M))
    f["test_class"] = bool(re.search(r"^\s*class Test\w*", c, re.M))
    ep = (r.get("entry_point") or "").split(".")[0]
    f["imports_target"] = bool(ep) and bool(re.search(
        rf"^\s*(from\s+[\w.]+\s+import\s+[^\n]*\b{re.escape(ep)}\b|import\s+[\w.]*\b{re.escape(ep)}\b)",
        c, re.M))
    if tree is not None:
        top = [type(n).__name__ for n in tree.body]
        f["assertion_only"] = bool(top) and all(t == "Assert" for t in top)
        und = undefined_names(tree)
        f["undefined_names"] = sorted(und)
        f["target_undefined"] = bool(ep) and ep in und
        f["self_contained_module"] = (f["test_function"] or f["test_class"]) and not und
    else:
        f.update(assertion_only=False, undefined_names=["<unparsed>"], target_undefined=False,
                 self_contained_module=False)
    lines = [l for l in c.splitlines() if l.strip()]
    f["prose_lines"] = sum(bool(re.match(r"^[A-Z][a-z]+( [a-z]+){3,}", l.strip())) for l in lines)
    f["tokens"] = len(tok(c, add_special_tokens=False)["input_ids"])
    return f


FLAGS = ("parses", "fence", "imports", "uses_pytest", "imports_pytest", "test_function",
         "test_class", "assertion_only", "imports_target", "target_undefined",
         "self_contained_module")
shapes = [shape(r) for r in rows]
groups = collections.defaultdict(list)
for r, s in zip(rows, shapes):
    groups["ALL"].append((r, s))
    groups[f"mode:{r['execution_mode']}"].append((r, s))
    groups[f"dataset:{r['dataset']}"].append((r, s))
    groups[f"complexity:{r.get('complexity')}"].append((r, s))


def summarise(items):
    n = len(items)
    w = sum(r["effective_repeats"] for r, _ in items)
    out = {"unique": n, "effective": w}
    for flag in FLAGS:
        u = sum(bool(s[flag]) for _, s in items)
        e = sum(r["effective_repeats"] for r, s in items if s[flag])
        out[flag] = f"{u}/{n} unique; {e}/{w} effective ({100 * e / w:.1f}%)"
    out["pytest_used_but_not_imported"] = sum(
        1 for _, s in items if s["uses_pytest"] and not s["imports_pytest"])
    out["prose_examples"] = sum(1 for _, s in items if s["prose_lines"])
    toks = sorted(s["tokens"] for _, s in items)
    out["tokens"] = {"p50": toks[n // 2], "p90": toks[int(0.9 * (n - 1))], "max": toks[-1]}
    comps = [r["completion"] for r, _ in items]
    out["exact_duplicate_completions"] = n - len(set(comps))
    out["near_duplicate_completions"] = n - len({re.sub(r"\s+", "", c) for c in comps})
    und = collections.Counter(u for _, s in items for u in s["undefined_names"])
    out["top_undefined_names"] = und.most_common(6)
    return out


report = {k: summarise(v) for k, v in sorted(groups.items())}
Path(sys.argv[2]).write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
for k in ("ALL", "mode:function_assertion", *[k for k in report if k.startswith("mode:") and k != "mode:function_assertion"]):
    if k in report:
        print("==", k, json.dumps(report[k], indent=1))
syn = next(r for r in rows if r["execution_mode"] == "function_assertion")
rep = next(r for r in rows if r["execution_mode"] != "function_assertion")
print("\n--- synthetic completion example:\n" + syn["completion"][:300])
print("--- synthetic prompt tail:\n" + syn["prompt_tail"][-450:])
print("\n--- repository completion example:\n" + rep["completion"][:500])
print("--- repository prompt tail:\n" + rep["prompt_tail"][-450:])
