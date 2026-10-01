"""v2.5 CPU program 2, Phases 9 and 12: frozen bounded-acquisition manifest, its projection, and
the blocker. NO acquisition is run here (no GitHub credential is configured in this
environment; unauthenticated GitHub API access is limited to 60 requests per hour).

The projection uses only already-recorded rates: the earlier fresh confirmation pilot
(admitted fix commits per repository inspected), the v2.5 r2 native qualification rate, and the
r2 verified-positive rate per qualified lineage. It never uses a model outcome.

    python scripts/v25_acquisition_projection.py
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RECEIPT = "results/sft_root_cause_v25_acquisition_projection.json"
PILOT = "results/v4_3_repository_native_aprime_fresh_confirmation_pilot.json"
PARTITION = "results/sft_root_cause_v25_universe_partition.json"
VERIFY = "results/sft_root_cause_v25_repository_verification_r2.json"
DECISION = "results/sft_root_cause_v25_attainable_gate_decision.json"


def sha(rel: str) -> str:
    return hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()


def main() -> int:
    from scripts.native_rehearsal_rebuild_v22 import publish_once
    pilot = json.loads((ROOT / PILOT).read_text(encoding="utf-8"))["counts"]
    part = json.loads((ROOT / PARTITION).read_text(encoding="utf-8"))
    verify = json.loads((ROOT / VERIFY).read_text(encoding="utf-8"))
    decision = json.loads((ROOT / DECISION).read_text(encoding="utf-8"))
    admit_rate = pilot["unique_fix_commits_admitted"] / pilot["unique_repositories_inspected"]
    cats = verify["environment_categories"]
    attempted = sum(sum(c.values()) for c in cats.values()) + 24      # + thefuck (Phase 6)
    qualified = sum(c.get("qualified", 0) for c in cats.values()) + 24
    qual_rate = qualified / attempted
    pos_rate = verify["attainable_gate"]["repository_tests"] / decision["qualified_target_lineages"]
    train_pool = len(part["training_expansion_pool"])
    conf_pool = len(part["confirmation_only_pool"])
    shortfall = verify["attainable_gate"]["shortfall"]
    expected_admitted = train_pool * admit_rate
    expected_positive = expected_admitted * qual_rate * pos_rate
    panel_targets = conf_pool * admit_rate * qual_rate
    receipt = {
        "schema_version": "oneiros_v25_acquisition_projection_v1",
        "status": "NOT RUN: blocked (no GitHub credential); projection only",
        "blocker": "no GITHUB_TOKEN / gh authentication in this environment; unauthenticated "
                   "GitHub REST access (60 requests/hour) cannot mine fix commits at the needed "
                   "scale. The user must authenticate (e.g. `! gh auth login`) before any "
                   "acquisition; credentials are never printed or stored in receipts",
        "frozen_bounded_manifest": {
            "pool": "training_expansion_pool of " + PARTITION,
            "repositories": train_pool,
            "mechanism": "scripts/run_repository_native_acquisition_pilot.py (policy-A "
                         "single-function fixes, authenticated evidence, licence and "
                         "isolation checks) -> v2.5 native builder -> 3/3 qualification -> "
                         "pytest_module_v1 positives verified twice",
            "stop_rule": "stop once 150/8/60 and the caps hold; otherwise preserve and declare "
                         "the v2.5 training arm infeasible (never relax the gate)",
            "bound": "one pass over the pool; no second pool is consumed"},
        "recorded_rates": {"admitted_per_repository_inspected": round(admit_rate, 4),
                           "source": PILOT, "native_qualification_rate": round(qual_rate, 4),
                           "verified_unique_tests_per_qualified_lineage": round(pos_rate, 4)},
        "projection_training": {
            "expected_admitted_fix_commits": round(expected_admitted, 1),
            "expected_verified_positives": round(expected_positive, 1),
            "upper_bound_if_every_admitted_commit_became_a_positive": round(expected_admitted, 1),
            "shortfall": shortfall,
            "can_close_gate": expected_admitted >= shortfall["repository_tests"]},
        "projection_confirmation_panel": {
            "pool_repositories": conf_pool,
            "expected_qualified_targets": round(panel_targets, 1),
            "minimum": {"targets": 80, "repositories": 10},
            "expected_underpowered": panel_targets < 80},
        "inputs_sha256": {p: sha(p) for p in (PILOT, PARTITION, VERIFY, DECISION)},
        "conclusion": "even unblocked, one bounded pass over the 36-repository training pool is "
                      "projected to yield far fewer than the 122 missing repository tests; the "
                      "v2.5 training gate is projected infeasible with the frozen pools and "
                      "mechanism. The confirmation panel is projected underpowered.",
        "gate_relaxed": False, "acquisition_run": False}
    status = publish_once(RECEIPT, receipt)
    print(json.dumps({"status": status, **{k: receipt[k] for k in (
        "recorded_rates", "projection_training", "projection_confirmation_panel")},
        "sha256": sha(RECEIPT)}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
