"""v2.6 Phase 2: train-only failure funnel with one primary category per fragment.

Input: the diagnostic evidence (scripts/v26_funnel_diag_wsl.py) over the 163 eligible
repository fragments. Categories are assigned by FROZEN, evidence-based rules (no per-record
special cases). Primary owner of each category:
infrastructure | harness | conversion | context | oracle | candidate | positive.

    python scripts/v26_failure_funnel.py --diag results/sft_root_cause/v26_funnel/diagnostics.jsonl
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RECEIPT = "results/sft_root_cause_v26_failure_funnel.json"
STAGES = ("eligible", "environment_available", "target_localized", "import_context_valid",
          "test_collected", "test_executed", "target_reached", "fixed_passed", "buggy_failed",
          "kill")
TEST_SUPPORT_ROOTS = {"tests", "test", "testing", "conftest", "utils"}
ENV_OWNER = {"patch_unavailable": "infrastructure", "environment_install_failed":
             "infrastructure", "no_approved_interpreter": "infrastructure",
             "commit_not_exact": "infrastructure", "view_attestation_failed": "infrastructure",
             "not_stable_3_of_3": "infrastructure", "runner_error": "harness",
             "runner_incompatible": "harness", "view_contains_official_test": "harness",
             "no_environment_row": "infrastructure"}


def runtime_test_package(module: str) -> bool:
    """``pkg.test`` / ``pkg.testing`` below a real package (e.g. django.test) - not a
    top-level project test-support package."""
    parts = module.split(".")
    return len(parts) >= 2 and parts[1] in ("test", "testing") and parts[0] not in \
        TEST_SUPPORT_ROOTS


def categorise(r: dict) -> tuple:
    """-> (primary, owner, recoverable_by, secondary list)."""
    if r["stage"] == "environment":
        d = r.get("detail")
        return f"environment:{d}", ENV_OWNER.get(d, "infrastructure"), None, []
    if r["stage"] == "localization":
        if r.get("changed_functions"):
            return ("localization:first_change_outside_function", "harness",
                    "F2 any-changed-function target", [])
        return "localization:no_function_changed", "candidate", None, []
    cls = r["class"]
    ev = r["evidence"]["buggy"]
    missing = ev["missing_modules"] + [m.rsplit(".", 1)[0] for m in ev["cannot_import"]]
    if cls in ("semantic_kill", "crash_kill"):
        return f"positive:{cls}", "positive", None, []
    if cls == "fabricated_import":
        if missing and all(runtime_test_package(m) for m in missing):
            return ("import:runtime_test_package_stripped_by_view", "harness",
                    "F1 view rule keeps shipped test-free <pkg>/test packages", [])
        if any(m.split(".")[0] in TEST_SUPPORT_ROOTS for m in missing):
            return "import:project_test_support_required", "context", None, \
                [f"module:{m}" for m in missing]
        return "import:other_missing_module", "context", None, [f"module:{m}" for m in missing]
    if cls == "target_not_reached":
        loc, changed = r.get("localized") or "", r.get("changed_functions") or []
        outer = [c for c in changed if loc.startswith(c + ".")]
        if outer:
            return ("reach:nested_localization_outer_also_changed", "harness",
                    "F2 any-changed-function target", [])
        if r["project"] == "fastapi" or "__call__" in loc:
            return "reach:async_or_threaded_dispatch_untracked", "harness", None, []
        return "reach:changed_function_not_executed", "candidate", None, []
    if cls in ("no_tests_collected", "collection_failure", "skipped_or_xfail"):
        heads = json.dumps(ev.get("exceptions")) + str(ev.get("error_head"))
        if "uses no argument" in heads or "fixture" in heads or r["project"] == "thefuck":
            return f"collection:{cls}:project_fixture_required", "context", None, []
        return f"collection:{cls}", "harness", None, []
    if cls == "timeout":
        return "execution:timeout", "candidate", None, []
    if cls in ("pass_both",):
        return "oracle:passes_on_buggy", "oracle", None, []
    if cls == "fixed_side_failure":
        return "oracle:fails_on_fixed", "oracle", None, []
    return f"other:{cls}", "candidate", None, []


def funnel_flags(r: dict) -> dict:
    f = {s: False for s in STAGES}
    f["eligible"] = True
    if r["stage"] == "environment":
        return f
    f["environment_available"] = True
    if r["stage"] == "localization":
        return f
    f["target_localized"] = True
    cls = r["class"]
    if cls == "fabricated_import":
        return f
    f["import_context_valid"] = True
    if cls in ("no_tests_collected", "collection_failure"):
        return f
    f["test_collected"] = True
    if cls == "skipped_or_xfail":
        return f
    f["test_executed"] = True
    if cls in ("target_not_reached", "timeout"):
        return f
    f["target_reached"] = True
    f["fixed_passed"] = r.get("fixed_valid") or cls in ("semantic_kill", "crash_kill",
                                                         "pass_both")
    f["buggy_failed"] = cls in ("semantic_kill", "crash_kill")
    f["kill"] = cls in ("semantic_kill", "crash_kill")
    return f


def main(argv=None) -> int:
    from scripts.native_rehearsal_rebuild_v22 import publish_once
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diag", required=True)
    args = parser.parse_args(argv)
    rows = [json.loads(l) for l in (ROOT / args.diag).read_text(encoding="utf-8").splitlines()
            if l.strip()]
    out_rows, cats = [], defaultdict(list)
    for r in rows:
        primary, owner, fix, secondary = categorise(r)
        flags = funnel_flags(r)
        out_rows.append({"index": r["index"], "task": r["task"], "project": r["project"],
                         "primary": primary, "owner": owner, "recoverable_by": fix,
                         "secondary": secondary, "funnel": flags})
        cats[primary].append(r)
    stages = {s: sum(o["funnel"][s] for o in out_rows) for s in STAGES}
    n = len(out_rows)
    categories = {}
    for c, rs in sorted(cats.items(), key=lambda kv: -len(kv[1])):
        o = next(x for x in out_rows if x["primary"] == c)
        categories[c] = {"fragments": len(rs), "of": n, "owner": o["owner"],
                         "repositories": sorted({r["project"] for r in rs}),
                         "lineages": len({r["task"] for r in rs}),
                         "recoverable_by": o["recoverable_by"],
                         "recoverable_ceiling": len(rs) if o["recoverable_by"] else 0}
    by_owner = Counter(o["owner"] for o in out_rows)
    receipt = {
        "schema_version": "oneiros_v26_failure_funnel_v1",
        "diagnostics": {"path": args.diag, "sha256": hashlib.sha256(
            (ROOT / args.diag).read_bytes()).hexdigest()},
        "denominator_fragments": n,
        "funnel_cumulative": stages,
        "positives": {"semantic_kill": len(cats.get("positive:semantic_kill", [])),
                      "crash_kill": len(cats.get("positive:crash_kill", [])),
                      "note": "crash kills are kept separate from semantic (oracle) kills"},
        "by_owner": dict(by_owner), "categories": categories,
        "recovery_ceiling_from_generic_fixes": {
            "F1_view_rule_runtime_test_packages": sum(
                v["fragments"] for k, v in categories.items()
                if (v["recoverable_by"] or "").startswith("F1")),
            "F2_any_changed_function_target": sum(
                v["fragments"] for k, v in categories.items()
                if (v["recoverable_by"] or "").startswith("F2")),
            "note": "ceilings assume every unblocked fragment then verifies; the actual yield "
                    "must be re-observed"},
        "overfitting_risk": "F1 and F2 are repository-independent rules (packaging metadata and "
                            "AST change sets); no per-record exception exists",
        "rows": out_rows}
    status = publish_once(RECEIPT, receipt)
    print(json.dumps({"status": status, "funnel": stages, "by_owner": dict(by_owner),
                      "categories": {k: (v["fragments"], v["owner"], v["recoverable_by"])
                                     for k, v in categories.items()},
                      "ceiling": receipt["recovery_ceiling_from_generic_fixes"],
                      "sha256": hashlib.sha256((ROOT / RECEIPT).read_bytes()).hexdigest()},
                     indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
