"""v2.5 CPU program 2, Phase 6B/6C: freeze, BEFORE any outcome, the structural attempt order for
the remaining train repositories and the design of the existing-project fixes.

The order uses only structural facts: eligible fragments (stage1_r2 policy-valid plus the
Phase 6A recoverable rows), source dataset, interpreter, whether the project ships compiled
extensions (the builder never compiles or installs the project), and the number of lineages.
No generated-test, verification or environment outcome of these repositories exists yet.

    python scripts/v25_native_attempt_spec.py
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SPEC = "results/sft_root_cause_v25_native_attempt_spec.json"
R2 = "results/sft_root_cause/v25_corpus_stage1_r2/converted_candidates.jsonl"
RECOVERY = "results/sft_root_cause/v25_policy_recovery/policy_recovery_rows.jsonl"
COMPILED = {"matplotlib", "astropy"}            # C extensions: never built by the v2.5 builder
ATTEMPTED = {"django", "sympy", "thefuck"}
REMAINING = ("ansible", "astropy", "fastapi", "flask", "httpie", "matplotlib", "pylint",
             "sanic", "tornado", "youtube-dl")
# nearest installed CPython (3.7-3.12) per SWE-bench version spec (same rule as the canary spec)
SWEBENCH_INTERPRETERS = {"astropy": {"4.3": "3.9", "5.0": "3.9", "5.1": "3.9", "5.2": "3.9"},
                         "flask": {"2.3": "3.11"},
                         "matplotlib": {"3.4": "3.8", "3.5": "3.11", "3.6": "3.11",
                                        "3.7": "3.11"},
                         "pylint": {"*": "3.9"}}


def bugsinpy_python(project: str, bug: str) -> str:
    info = (ROOT / "data/BugsInPy_repo/projects" / project / "bugs" / bug / "bug.info") \
        .read_text(encoding="utf-8")
    for line in info.splitlines():
        if line.startswith("python_version"):
            v = line.split("=", 1)[1].strip().strip('"')
            major_minor = ".".join(v.split(".")[:2])
            return "3.7" if major_minor in ("3.6", "3.5") else major_minor   # 3.6 unavailable
    return "3.7"


def main() -> int:
    from scripts.native_rehearsal_rebuild_v22 import publish_once
    from scripts.v25_converted_corpus import access_receipt, install_audit, load_train_records
    install_audit()
    records = {r["id"]: r for r in load_train_records()}
    cands = [json.loads(l) for l in (ROOT / R2).read_text(encoding="utf-8").splitlines()]
    recovered = {r["index"] for r in map(json.loads, (ROOT / RECOVERY).read_text(
        encoding="utf-8").splitlines()) if r["category"].startswith("recoverable")}
    fragments, lineages, datasets, interp = Counter(), defaultdict(set), defaultdict(set), \
        defaultdict(Counter)
    for c in cands:
        if c["execution_mode"] == "function_assertion":
            continue
        if not (c["conversion"]["accepted"] or c["index"] in recovered):
            continue
        prov = records[c["id"]]["provenance"]
        p = c["project"]
        fragments[p] += 1
        lineages[p].add(prov["official_task_id"])
        datasets[p].add(c["dataset"])
        if c["dataset"] == "BugsInPy":
            interp[p][bugsinpy_python(p, str(prov["bug_id"]))] += 1
        else:
            m = SWEBENCH_INTERPRETERS.get(p, {})
            interp[p][m.get(str(prov.get("version"))) or m.get("*") or "unmapped"] += 1
    rows = []
    for p in REMAINING:
        rows.append({"project": p, "eligible_fragments": fragments.get(p, 0),
                     "lineages": len(lineages.get(p, ())),
                     "datasets": sorted(datasets.get(p, ())),
                     "interpreters": dict(interp.get(p, {})),
                     "compiled_extensions": p in COMPILED,
                     "estimated_build_cost": "high (C extensions not built: expected "
                                             "view_attestation_failed)" if p in COMPILED
                     else "low (pure Python)"})
    order = sorted(rows, key=lambda r: (r["eligible_fragments"] == 0, r["compiled_extensions"],
                                        -r["eligible_fragments"], r["project"]))
    spec = {
        "schema_version": "oneiros_v25_native_attempt_spec_v1",
        "frozen_before_outcomes": True,
        "order_rule": "projects with eligible fragments first; pure Python before compiled; "
                      "then more eligible fragments; then name",
        "attempt_order": [r["project"] for r in order], "projects": order,
        "already_attempted": sorted(ATTEMPTED),
        "existing_project_fixes": {
            "django_verification": "a Django settings layer (django test_sqlite-equivalent "
                                   "settings, django.setup() at interpreter start via a .pth "
                                   "file) in a SEPARATE copy of the qualified environment, "
                                   "recorded with its own lock; the candidate module, the "
                                   "static policy and the sandbox are unchanged; official/"
                                   "gold tests never enter the view",
            "view_rule_v2": "keep <pkg>/testing (and only that) when it is a package inside "
                            "the import-root package AND packaging metadata of that revision "
                            "ships it (literal dotted name, or find_packages/packages=find "
                            "without excluding it); test_*.py, *_test.py, conftest.py and "
                            "test/tests directories are still stripped at any depth; a "
                            "manifest check proves no official test file is in a view",
            "patches": "remaining 11 SWE-bench instances: exact sha256 equality with train "
                       "metadata only (row API, then PR-diff split); no relaxation",
            "test_support": "project tests.*/conftest imports stay rejected"},
        "repeatability": "environments built twice (identical lock); official buggy/fixed 3/3; "
                         "candidates verified in two runs; byte identity and normalised "
                         "semantic identity reported separately",
        "output_dirs": "results/sft_root_cause/v25_native_r2/<project>/ (fresh per project)",
        "access_audit": {k: access_receipt()[k] for k in ("passed", "opened")}}
    status = publish_once(SPEC, spec)
    print(json.dumps({"status": status, "order": spec["attempt_order"],
                      "projects": {r["project"]: (r["eligible_fragments"], r["lineages"],
                                                  r["interpreters"]) for r in order},
                      "sha256": hashlib.sha256((ROOT / SPEC).read_bytes()).hexdigest()},
                     indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
