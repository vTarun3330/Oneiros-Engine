"""Native-rehearsal receipt v2: denominator semantics, readiness flags, portable evidence.

Successor addendum to results/sft_root_cause_phase4_native_rehearsal_receipt_v1.json,
which is preserved byte-for-byte. No target is re-run and no number is re-measured:
every figure is re-derived from the tracked portable evidence bundle
(results/sft_root_cause_phase4_native_evidence_v1.json). Deterministic; CPU only.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness.atomic_publish import publish_file_atomically

OUTPUT = "results/sft_root_cause_phase4_native_rehearsal_receipt_v2.json"
V1 = "results/sft_root_cause_phase4_native_rehearsal_receipt_v1.json"
BUNDLE = "results/sft_root_cause_phase4_native_evidence_v1.json"
INFRASTRUCTURE = {"test_infrastructure_error", "environment_install_failed", "checkout_failed",
                  "clone_failed", "runner_error"}
Z = 1.959963984540054


def sha(rel: str) -> str:
    return hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()


def load(rel: str):
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))


def rate(k: int, n: int) -> dict:
    p = k / n
    centre = (p + Z * Z / (2 * n)) / (1 + Z * Z / n)
    half = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / (1 + Z * Z / n)
    return {"k": k, "n": n, "point": round(p, 4),
            "wilson_95": [round(max(0.0, centre - half), 4), round(min(1.0, centre + half), 4)]}


def build() -> dict:
    v1 = load(V1)
    per_target = load(BUNDLE)["facts"]["per_target"]
    total = len(per_target)
    infrastructure = [r for r in per_target if r["category"] in INFRASTRUCTURE]
    evaluable = [r for r in per_target if r["category"] not in INFRASTRUCTURE]
    qualified = [r for r in per_target if r["category"] == "natively_qualified"]
    safe = [r for r in qualified if r["safe_fixed_call_constructed"]]
    if any(r["environment_failure"] != (r["category"] in INFRASTRUCTURE) for r in per_target):
        raise SystemExit("REFUSED: environment_failure flag disagrees with the category")
    denominators = {
        "environment_success": {
            **rate(total - len(infrastructure), total),
            "meaning": "the native environment built and the official tests ran on both "
                       "revisions"},
        "semantic_qualification_among_evaluable": {
            **rate(len(qualified), len(evaluable)),
            "meaning": "among targets whose tests actually ran, the fraction whose official "
                       "regression tests distinguish buggy from fixed; infrastructure failures "
                       "are excluded from BOTH numerator and denominator"},
        "operational_end_to_end_qualified_throughput": {
            **rate(len(qualified), total),
            "meaning": "qualified targets per admitted target attempted, counting "
                       "infrastructure failures as attrition; used ONLY to project acquisition "
                       "cost"},
    }
    checks = {
        "v1_environment_success_equals_v2": v1["rates"]["environment_success"]["k"]
        == denominators["environment_success"]["k"],
        "v1_evaluable_equals_semantic": v1["rates"]["qualified_over_evaluable_admitted"]
        == {k: denominators["semantic_qualification_among_evaluable"][k]
            for k in ("k", "n", "point", "wilson_95")},
        "v1_all_admitted_equals_operational": v1["rates"]["qualified_over_all_admitted"]
        == {k: denominators["operational_end_to_end_qualified_throughput"][k]
            for k in ("k", "n", "point", "wilson_95")},
    }
    if not all(checks.values()):
        raise SystemExit(f"REFUSED: v2 denominators do not reproduce v1 figures: {checks}")
    return {
        "schema_version": "oneiros_sft_root_cause_phase4_native_rehearsal_v2",
        "supersedes": {"path": V1, "sha256": sha(V1), "modified": False,
                       "nature": "addendum: v1 numbers unchanged; semantics, readiness and "
                                 "portable evidence clarified"},
        "evidence_bundle": {"path": BUNDLE, "sha256": sha(BUNDLE)},
        "targets": total,
        "denominators": denominators,
        "reproduces_v1_figures": checks,
        "infrastructure_failures": {
            "targets": [{"key": r["key"], "category": r["category"]} for r in infrastructure],
            "semantic_role": "excluded: never a semantic negative and never a semantic positive",
            "training_role": "never a positive (or any) training example",
            "projection_role": "counted only as operational attrition when projecting total "
                               "acquisition cost (operational_end_to_end_qualified_throughput)"},
        "projection_rate_used": ("v1 projected acquisition cost with qualified/admitted = "
                                 "24/26, which IS the operational throughput; the projection "
                                 "is unchanged"),
        "readiness": {
            "native_environment_feasible": True,
            "fixed_input_ready": False,
            "choice_A_ready": False,
            "reason": (f"{len(qualified)} targets qualified natively, but "
                       f"{len(safe)}/{len(qualified)} produced a safe fixed call, so "
                       "repository-native targets cannot yet feed the fixed-input mediator"),
            "safe_fixed_call_over_qualified": rate(len(safe), len(qualified))},
        "dependency_resolution": {
            "recorded_in_v1": False,
            "limitation": load(BUNDLE)["dependency_resolution"]["limitation"]},
        "caveats": v1["caveats"],
        "model_calls": 0, "gpu_used": False, "training": False, "targets_rerun": 0,
    }


def main() -> int:
    receipt = build()
    data = (json.dumps(receipt, indent=1, sort_keys=True) + "\n").encode("utf-8")
    target = ROOT / OUTPUT
    if target.exists() and target.read_bytes() != data:
        raise SystemExit(f"REFUSED: {OUTPUT} exists with different bytes; write a successor")
    publish_file_atomically(target, data)
    print(json.dumps({"denominators": {k: (v["k"], v["n"], v["point"], v["wilson_95"])
                                       for k, v in receipt["denominators"].items()},
                      "readiness": receipt["readiness"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
