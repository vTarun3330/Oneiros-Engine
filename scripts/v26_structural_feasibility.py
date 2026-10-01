"""v2.6 Phase 4: deterministic structural feasibility of the strict 150/8/60 training gate.

Deterministic MAXIMA under the exact frozen selection rules (canonical-test deduplication,
<= 3 per function, <= 3 per lineage, <= 40 per repository), assuming every remaining candidate
SUCCEEDS - two universes:
  (a) all 163 eligible fragments of the 13 existing train repositories;
  (b) only fragments whose environment is qualified today (v2.6 a3).
Function identity: the verifier-side target when known; otherwise each lineage counts as its
own function (the most generous choice, so the result is a true maximum).
The 36-repository training-expansion pool has NO acquired candidates: its maximum is UNKNOWN
(no recorded quantity bounds it). The confirmation-only pool is reported separately and never
used to rescue the training gate.

    python scripts/v26_structural_feasibility.py
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RECEIPT = "results/sft_root_cause_v26_structural_feasibility.json"
DIR = "results/sft_root_cause/v25_native_r2"
DIAG = "results/sft_root_cause/v26_funnel/diagnostics.jsonl"
VERIFY = "results/sft_root_cause/v26_verify_a3/repository_verification_run1.jsonl"
CONSERVATION = "results/sft_root_cause_v26_selection_conservation.json"
PARTITION = "results/sft_root_cause_v25_universe_partition.json"
A3 = "results/sft_root_cause_v26_repository_verification_a3.json"
GATE = {"repository_tests": 150, "repositories": 8, "lineages": 60}
CAPS = {"per_function": 3, "per_lineage": 3, "per_repository": 40}


def sha(rel: str) -> str:
    return hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()


def jsonl(rel: str) -> list:
    return [json.loads(l) for l in (ROOT / rel).read_text(encoding="utf-8").splitlines()
            if l.strip()]


def maximum(rows: list) -> dict:
    """Greedy is optimal here: every cap is a per-group upper bound on independent rows, and a
    canonical duplicate never adds a test; take one row per canonical test, then caps."""
    seen, fn, lin, repo, kept = set(), Counter(), Counter(), Counter(), []
    for r in sorted(rows, key=lambda r: r["canonical"]):
        if r["canonical"] in seen or fn[r["function"]] >= CAPS["per_function"] or \
                lin[r["lineage"]] >= CAPS["per_lineage"] or \
                repo[r["repository"]] >= CAPS["per_repository"]:
            continue
        seen.add(r["canonical"])
        fn[r["function"]] += 1
        lin[r["lineage"]] += 1
        repo[r["repository"]] += 1
        kept.append(r)
    counts = {"repository_tests": len(kept), "repositories": len(repo),
              "lineages": len(lin)}
    return {"maximum": counts, "by_repository": dict(repo),
            "reachable": {k: counts[k] >= GATE[k] for k in GATE},
            "all_reachable": all(counts[k] >= GATE[k] for k in GATE)}


def main() -> int:
    from scripts.native_rehearsal_rebuild_v22 import publish_once
    cands = {c["index"]: c for c in jsonl(f"{DIR}/converted_candidates_r2r.jsonl")}
    targets = {t["task"]: t for t in jsonl(f"{DIR}/targets_a3.jsonl")}
    verdicts = {v["index"]: v for v in jsonl(VERIFY)}
    diag = {d["index"]: d for d in jsonl(DIAG)}
    rows = []
    for t in targets.values():
        for f in t["fragments"]:
            v = verdicts.get(f["index"], {})
            qual = v.get("qualname") or (diag.get(f["index"]) or {}).get("localized")
            rows.append({"index": f["index"], "repository": t["project"], "lineage": t["task"],
                         "canonical": cands[f["index"]]["conversion"]["identities"]
                         ["canonical_ast"],
                         "function": (t["project"], t["target_file"], qual) if qual
                         else (t["project"], t["task"], "unknown-function"),
                         "qualified_env": not str(v.get("status", "")).startswith("environment:")
                         and v.get("status") != "no_environment_row"})
    a = maximum(rows)
    b = maximum([r for r in rows if r["qualified_env"]])
    cons = json.loads((ROOT / CONSERVATION).read_text(encoding="utf-8"))["funnel"]["selected"]
    a3 = json.loads((ROOT / A3).read_text(encoding="utf-8"))
    part = json.loads((ROOT / PARTITION).read_text(encoding="utf-8"))
    label = ("STRICT_TRAINING_INFEASIBLE_UNDER_CURRENT_PROTOCOL" if not a["all_reachable"]
             else "STRICT_TRAINING_NOT_READY")
    receipt = {
        "schema_version": "oneiros_v26_structural_feasibility_v1",
        "gate": GATE, "caps": CAPS,
        "observed": {"selected_tests": cons["rows"], "repositories": cons["repositories"],
                     "lineages": cons["lineages"], "source": CONSERVATION},
        "qualified_environments_are_not_positives": {
            "qualified_environment_lineages": a3["qualified_environments"]["lineages"],
            "verified_positive_lineages": cons["lineages"]},
        "deterministic_maximum": {
            "a_all_163_eligible_fragments_existing_repositories": {
                "fragments": len(rows), **a,
                "assumption": "every eligible fragment verifies, including those whose "
                              "environment is an infrastructure failure today"},
            "b_qualified_environments_today": {
                "fragments": sum(r["qualified_env"] for r in rows), **b,
                "assumption": "every fragment with a qualified environment verifies"}},
        "overlap_with_selected": {"selected_rows_inside_universe_a": cons["rows"],
                                  "note": "the 33 selected rows are a subset of both universes; "
                                          "the maxima above already include them"},
        "training_expansion_pool": {"repositories": len(part["training_expansion_pool"]),
                                    "acquired_candidates": 0, "maximum": "UNKNOWN",
                                    "reason": "no acquisition has run (authentication pending); "
                                              "no recorded quantity bounds its yield"},
        "confirmation_only_pool": {"repositories": len(part["confirmation_only_pool"]),
                                   "use": "confirmation only; never counted toward training"},
        "decision": {"label": label, "chosen_option": "A",
                     "text": "the strict v2.6 training arm is closed as NOT READY / INFEASIBLE "
                             "with the available evidence: no SFT on the 33 examples; the gate "
                             "is not weakened; the 33 rows are preserved as verified evidence "
                             "and possible future replay material",
                     "project_status": "continuing: repository-disjoint confirmation panel and "
                                       "evaluation of existing models"},
        "inputs_sha256": {p: sha(p) for p in (DIAG, VERIFY, CONSERVATION, PARTITION, A3,
                                              f"{DIR}/targets_a3.jsonl")}}
    status = publish_once(RECEIPT, receipt)
    print(json.dumps({"status": status, "a": a, "b": b, "label": label,
                      "sha256": sha(RECEIPT)}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
