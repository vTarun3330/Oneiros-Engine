"""v2.6 Phase 3: supersession receipt for the greedy Phase 4 structural-feasibility proof.

Links the historical greedy artifact (kept unchanged) to the exact MILP successor and records
the SCOPED decision states. Lists every tracked report whose wording overstated the greedy
result, with the correction that now applies to it.

    python scripts/v26_phase4_supersession.py
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RECEIPT = "results/sft_root_cause_v26_phase4_supersession.json"
OLD = "results/sft_root_cause_v26_structural_feasibility.json"
NEW = "results/sft_root_cause_v26_structural_feasibility_v2_exact.json"
STATES = ("CURRENT_MATERIALIZED_UNIVERSE_STRUCTURALLY_INFEASIBLE_FOR_150",
          "CURRENT_OBSERVED_CORPUS_NOT_TRAINING_READY", "CURRENT_GPU_TRAINING_NOT_AUTHORIZED",
          "TRAINING_EXPANSION_POOL_NOT_MATERIALIZED",
          "TRAINING_EXPANSION_POOL_FEASIBILITY_UNKNOWN")
FORBIDDEN_LABEL = "GLOBAL_STRICT_TRAINING_ARM_STRUCTURALLY_INFEASIBLE"


def sha(rel: str) -> str:
    return hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()


def main() -> int:
    from scripts.native_rehearsal_rebuild_v22 import publish_once
    old = json.loads((ROOT / OLD).read_text(encoding="utf-8"))
    new = json.loads((ROOT / NEW).read_text(encoding="utf-8"))
    a = new["universe_A_all_163_materialized_fragments"]["results"]
    states = new["states"]
    if not all(states.get(s) is True for s in STATES):
        raise SystemExit(f"REFUSED: exact successor does not support every scoped state: {states}")
    receipt = {
        "schema_version": "oneiros_v26_phase4_supersession_v1",
        "superseded": {"artifact": OLD, "sha256": sha(OLD), "kept_unchanged": True,
                       "method": "greedy selection sorted by canonical AST",
                       "status_of_its_numbers": "HEURISTIC_FEASIBLE_SET / LOWER_BOUND",
                       "overstatements": [
                           "its source called the greedy selection optimal",
                           "its 'maximum' lineages (139) was a witness count; the exact maximum "
                           f"is {a['max_lineages']['objective_value']}",
                           "its label STRICT_TRAINING_INFEASIBLE_UNDER_CURRENT_PROTOCOL read as "
                           "a claim about the whole protocol, including the unacquired 36-"
                           "repository expansion pool"]},
        "successor": {"artifact": NEW, "sha256": sha(NEW), "method": "binary MILP (HiGHS), "
                      "OPTIMAL / INFEASIBLE terminations only, independently validated witnesses",
                      "witness_file": new["witness_file"],
                      "constraint_manifest_sha256": new["constraint_manifest_sha256"],
                      "universe_A_max_tests": a["max_tests"]["objective_value"],
                      "universe_A_joint_150_8_60": a["joint_feasibility_150_8_60"]["status"],
                      "universe_B_max_tests": new["universe_B_qualified_environments_today"]
                      ["results"]["max_tests"]["objective_value"]},
        "states": {s: True for s in STATES},
        "not_used": FORBIDDEN_LABEL,
        "scope": new["scope"],
        "affected_reports": {
            OLD: "use the exact successor; greedy counts are lower bounds",
            "scripts/v26_structural_feasibility.py": "heuristic diagnostic only",
            "results/sft_root_cause_v26_phase0_reverification.json":
                "'13 existing train repositories' -> the 163 eligible fragments come from 12 "
                "repositories (pylint contributed no eligible fragment)",
            "results/sft_root_cause_v25_cpu2_preflight_decision.json":
                "'exhausting the 13 existing train repositories' -> 12 repositories with "
                "eligible fragments; conclusion unchanged (refused)"},
        "repository_count_reconciliation": {
            "train_repositories_with_records": 13,
            "repositories_with_eligible_fragments": 12,
            "difference": "pylint: its only repository fragment needs project test support"},
        "decision": "do not train on the 33 observed examples; GPU training not authorised; "
                    "next: repository-disjoint confirmation panel, then evaluation of existing "
                    "models"}
    status = publish_once(RECEIPT, receipt)
    print(json.dumps({"status": status, "states": receipt["states"],
                      "sha256": sha(RECEIPT)}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
