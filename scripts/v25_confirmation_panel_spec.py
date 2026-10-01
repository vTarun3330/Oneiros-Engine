"""v2.5 Phase 9: specification and mechanism of the untouched confirmation panel (protocol v2.5 D,
addendum 1 section 3). Builds NO panel, runs no model and no fuzzing.

Freezes, before any training:
- the REPOSITORY EXCLUSION LEDGER, built only from sources this stage may read:
  training repositories (strict train-only loader), the v2.4 diagnostic panel (manifest v7),
  every repository of the three earlier acquisition pilots (directory names only), and - so the
  panel is disjoint from every validation / reserved / sealed record WITHOUT opening them -
  every BugsInPy project and every SWE-bench repository (the corpus' repository sources);
- TRAINING FUNCTION FINGERPRINTS (canonical-AST hashes of every function in every train
  record's code under test) for the near-duplicate gate;
- the structural selection rules, minimum sizes (addendum 1) and the lineage fingerprint;
- the access/usage ledger format every later panel access must append to.

    python scripts/v25_confirmation_panel_spec.py
"""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RECEIPT = "results/sft_root_cause_v25_confirmation_panel_spec.json"
FINGERPRINTS = "results/sft_root_cause/v25_panel/training_function_fingerprints.json"
MANIFEST_V7 = "results/sft_root_cause_native_v24_rehearsal_manifest_v7.json"
PILOT_DIRS = ("acquisition_pilot", "aprime_confirmation_pilot", "aprime_fresh_confirmation_pilot")
SWEBENCH_REPOSITORIES = ("astropy/astropy", "django/django", "matplotlib/matplotlib",
                         "mwaskom/seaborn", "pallets/flask", "psf/requests", "pydata/xarray",
                         "pylint-dev/pylint", "pytest-dev/pytest", "scikit-learn/scikit-learn",
                         "sphinx-doc/sphinx", "sympy/sympy")


def repo_key(name: str) -> str:
    """Owner-independent repository key (forks and renames of one project collide)."""
    name = name.lower().rstrip("/").removesuffix(".git")
    name = name.split("github.com/")[-1]
    return name.replace("__", "/").split("/")[-1]


class _Canon(ast.NodeTransformer):
    """Rename local identifiers and drop docstrings: textual edits do not hide a duplicate."""

    def visit_FunctionDef(self, node):
        node.name = "f"
        node.returns = None
        if node.body and isinstance(node.body[0], ast.Expr) and \
                isinstance(getattr(node.body[0], "value", None), ast.Constant) and \
                isinstance(node.body[0].value.value, str):
            node.body = node.body[1:] or [ast.Pass()]
        self.generic_visit(node)
        return node

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_arg(self, node):
        node.annotation = None
        return node


def function_fingerprint(source: str) -> str:
    """Canonical-AST hash of one function definition (identifier-renamed, docstring-free)."""
    tree = ast.parse(source)
    node = next(n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef,
                                                             ast.AsyncFunctionDef)))
    names = {}

    class Rename(ast.NodeTransformer):
        def visit_Name(self, n):
            n.id = names.setdefault(n.id, f"v{len(names)}")
            return n

        def visit_arg(self, n):
            n.arg = names.setdefault(n.arg, f"v{len(names)}")
            n.annotation = None
            return n
    canon = Rename().visit(_Canon().visit(ast.parse(ast.unparse(node))))
    return hashlib.sha256(ast.dump(canon, annotate_fields=False).encode()).hexdigest()


def functions_in(code: str) -> list:
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        return []
    out = []
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            try:
                out.append(function_fingerprint(ast.unparse(n)))
            except (StopIteration, SyntaxError, ValueError):
                pass
    return out


def lineage_fingerprint(repository_url: str, fix_commit: str, target_function_source: str) -> dict:
    return {"repository": repo_key(repository_url), "fix_commit": fix_commit.lower(),
            "function": function_fingerprint(target_function_source)}


def main() -> int:
    from scripts.native_rehearsal_rebuild_v22 import publish_once
    from scripts.v25_converted_corpus import install_audit, load_train_records
    install_audit()
    train = load_train_records()
    train_repos = sorted({repo_key((r.get("provenance") or {}).get("repository")
                                   or (r.get("provenance") or {}).get("repository_url") or "")
                          for r in train if (r.get("provenance") or {}).get("repository")
                          or (r.get("provenance") or {}).get("repository_url")})
    v24 = json.loads((ROOT / MANIFEST_V7).read_text(encoding="utf-8"))
    v24_repos = sorted({repo_key(t["repository"]) for t in v24["targets"]})
    pilot = {d: sorted(repo_key(p.name) for p in (ROOT / "data" / "repository_native" / d /
                                                  "repos").iterdir())
             for d in PILOT_DIRS}
    bugsinpy = sorted(p.name.lower() for p in (ROOT / "data" / "BugsInPy_repo" / "projects")
                      .iterdir() if p.is_dir())
    swebench = sorted(repo_key(r) for r in SWEBENCH_REPOSITORIES)
    excluded = sorted(set(train_repos) | set(v24_repos) | set().union(*pilot.values())
                      | set(bugsinpy) | set(swebench))
    fps = sorted({fp for r in train for fp in functions_in(r.get("code_under_test") or "")})
    fp_path = ROOT / FINGERPRINTS
    fp_path.parent.mkdir(parents=True, exist_ok=True)
    fp_path.write_bytes((json.dumps(fps) + "\n").encode("utf-8"))
    spec = {
        "schema_version": "oneiros_v25_confirmation_panel_spec_v1",
        "status": "SPECIFICATION ONLY: no panel acquired, no model generation, no fuzzing",
        "exclusion_ledger": {
            "rule": "a candidate whose repository key is in this set is excluded",
            "repository_key": "lower-case repository name without owner (forks/renames collide)",
            "sources": {"training_records": train_repos, "v24_diagnostic_panel": v24_repos,
                        "earlier_acquisition_pilots": pilot, "bugsinpy_projects": bugsinpy,
                        "swebench_repositories": swebench},
            "excluded_repository_keys": excluded,
            "protected_material": "validation shards, reserved confirmation ids and sealed "
                                  "results are NEVER opened; disjointness from them is "
                                  "guaranteed at repository level by excluding every "
                                  "repository source of the corpus (BugsInPy, SWE-bench) and "
                                  "every earlier pilot"},
        "near_duplicate_gate": {
            "training_function_fingerprints_sha256": hashlib.sha256(
                fp_path.read_bytes()).hexdigest(),
            "training_function_fingerprints": len(fps),
            "fingerprint": "sha256 of the canonical AST of the function: identifiers renamed "
                           "in order of appearance, annotations and docstring removed",
            "rule": "a candidate whose buggy target-function fingerprint equals any training "
                    "function fingerprint is excluded"},
        "lineage_fingerprint": "(repository key, fix commit, target-function fingerprint); "
                               "two candidates sharing any component are one lineage",
        "acquisition_mechanism": "scripts/run_repository_native_acquisition_pilot.py "
                                 "(frozen repository list, authenticated evidence, policy-A "
                                 "single-function diff, temporal and source-universe "
                                 "isolation, licence validation) followed by the v2.5 native "
                                 "environment builder and 3/3 official qualification",
        "selection": {
            "structural_only": ["repository not excluded", "Python, permissive licence",
                                "single changed function (policy A)", "official targeted test "
                                "fails 3/3 buggy, passes 3/3 fixed",
                                "environment reproducible", "buggy-side complexity tier"],
            "never": ["model outputs", "Atheris outcomes", "repair outcomes",
                      "validation or sealed records"],
            "minimum": {"targets": 80, "repositories": 10, "max_per_repository": 8,
                        "complexity_strata": ["simple", "moderate", "complex"],
                        "defect_families_min": 4},
            "atheris_subset": "structurally selected after freezing: targets whose probe is "
                              "eligible under adapter v5; reported separately",
            "underpowered_rule": "fewer than 80 targets or 10 repositories -> frozen as built "
                                 "and labelled underpowered (addendum 1 section 3)"},
        "freeze_before_training": ["target ids", "repositories and commits", "environments "
                                   "(locks)", "views", "prompts", "eligibility criteria",
                                   "hashes of all of these"],
        "usage_ledger": {"path": "results/sft_root_cause_v25_confirmation_usage_ledger.jsonl",
                         "row": ["utc", "actor", "purpose", "artifact", "artifact_sha256"],
                         "rule": "append-only; every read of panel targets, prompts or "
                                 "outcomes is recorded; any use for training, tuning or "
                                 "checkpoint selection invalidates the panel"}}
    status = publish_once(RECEIPT, spec)
    print(json.dumps({"status": status, "excluded_repositories": len(excluded),
                      "training_function_fingerprints": len(fps)}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
