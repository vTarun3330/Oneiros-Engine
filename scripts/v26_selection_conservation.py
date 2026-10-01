"""v2.6 Phase 1: conservation funnel for the attempt-a3 repository selection (successor to the
a3 receipt, whose ``duplicates_ignored`` counted only canonical duplicates and hid the cap
exclusion).

Every input fragment ends in EXACTLY ONE outcome: selected, or excluded with one reason.
Selection order and rules are identical to scripts/v26_repository_receipt_a3.py (sorted by
canonical AST; exact duplicate -> per-function cap -> per-lineage cap -> per-repository cap).
Near-duplicate clusters are REPORTED (policy: reported, not removed). Unexplained attrition
fails closed.

    python scripts/v26_selection_conservation.py
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

DIR = "results/sft_root_cause/v25_native_r2"
VERIFY = "results/sft_root_cause/v26_verify_a3"
A3 = "results/sft_root_cause_v26_repository_verification_a3.json"
RECEIPT = "results/sft_root_cause_v26_selection_conservation.json"
SIDECAR = "results/sft_root_cause/v26_funnel/selection_outcomes.jsonl"
CAPS = {"per_function": 3, "per_lineage": 3, "per_repository": 40}


def sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def jsonl(p) -> list:
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]


def select(verified: list, cands: dict, targets: dict, caps: dict = CAPS) -> list:
    """-> [(row, outcome, detail)] in selection order; one outcome per row."""
    out, seen = [], {}
    per_fn, per_lin, per_repo = Counter(), Counter(), Counter()
    holder_fn = {}
    for v in sorted(verified, key=lambda v: cands[v["index"]]["conversion"]["identities"]
                    ["canonical_ast"]):
        canon = cands[v["index"]]["conversion"]["identities"]["canonical_ast"]
        fn = (v["project"], targets[v["task"]]["target_file"], v["qualname"])
        if canon in seen:
            out.append((v, "excluded:exact_duplicate", {"duplicate_of": seen[canon]}))
            continue
        if per_fn[fn] >= caps["per_function"]:
            out.append((v, "excluded:per_function_cap",
                        {"function": list(fn), "selected_rows_of_function": holder_fn[fn]}))
            continue
        if per_lin[v["task"]] >= caps["per_lineage"]:
            out.append((v, "excluded:per_lineage_cap", {"lineage": v["task"]}))
            continue
        if per_repo[v["project"]] >= caps["per_repository"]:
            out.append((v, "excluded:per_repository_cap", {"repository": v["project"]}))
            continue
        seen[canon] = v["index"]
        per_fn[fn] += 1
        per_lin[v["task"]] += 1
        per_repo[v["project"]] += 1
        holder_fn.setdefault(fn, []).append(v["index"])
        out.append((v, "selected", {}))
    return out


def stage(rows: list, cands: dict) -> dict:
    return {"rows": len(rows), "repositories": len({r["project"] for r in rows}),
            "issues": len({r["task"] for r in rows}), "lineages": len({r["task"] for r in rows}),
            "canonical_tests": len({cands[r["index"]]["conversion"]["identities"]
                                    ["canonical_ast"] for r in rows}),
            "semantic_kills": sum(r.get("class") == "semantic_kill" for r in rows),
            "crash_kills": sum(r.get("class") == "crash_kill" for r in rows)}


def main() -> int:
    from scripts.native_rehearsal_rebuild_v22 import publish_once
    root = ROOT / DIR
    cands = {c["index"]: c for c in jsonl(root / "converted_candidates_r2r.jsonl")}
    targets = {t["task"]: t for t in jsonl(root / "targets_a3.jsonl")}
    runs = [jsonl(ROOT / VERIFY / f"repository_verification_run{i}.jsonl") for i in (1, 2)]
    idx2 = {v["index"]: v for v in runs[1]}
    discovered = runs[0]
    qualified = [v for v in discovered if not str(v["status"]).startswith("environment:")
                 and v["status"] != "no_environment_row"]
    verified = [v for v in discovered if v["accepted"] and idx2.get(v["index"]) == v]
    outcomes = select(verified, cands, targets)
    selected = [v for v, o, _ in outcomes if o == "selected"]
    reasons = Counter(o for _, o, _ in outcomes)
    problems = []
    if len(outcomes) != len(verified) or len({v["index"] for v, _, _ in outcomes}) != len(verified):
        problems.append("a verified row has no unique outcome")
    a3 = json.loads((ROOT / A3).read_text(encoding="utf-8"))["observed_training_gate"]["observed"]
    if (len(selected), len({v["project"] for v in selected}),
            len({v["task"] for v in selected})) != (a3["repository_tests"], a3["repositories"],
                                                     a3["lineages"]):
        problems.append("selection does not reproduce the a3 receipt")
    near = Counter(cands[v["index"]]["conversion"]["identities"]["near_duplicate_cluster"]
                   for v in selected)
    sidecar = ROOT / SIDECAR
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps({"index": v["index"], "candidate_id": cands[v["index"]]["id"],
                         "task": v["task"], "project": v["project"], "qualname": v["qualname"],
                         "class": v.get("class"), "outcome": o, "detail": d,
                         "canonical_ast": cands[v["index"]]["conversion"]["identities"]
                         ["canonical_ast"]}, sort_keys=True) for v, o, d in outcomes]
    data = ("\n".join(lines) + "\n").encode("utf-8")
    if sidecar.exists() and sidecar.read_bytes() != data:
        raise SystemExit("REFUSED: sidecar exists with different content")
    sidecar.write_bytes(data)
    excluded = [{"index": v["index"], "candidate_id": cands[v["index"]]["id"], "task": v["task"],
                 "project": v["project"], "reason": o, **d}
                for v, o, d in outcomes if o != "selected"]
    receipt = {
        "schema_version": "oneiros_v26_selection_conservation_v1",
        "supersedes_wording_of": {"receipt": A3, "sha256": sha(ROOT / A3),
                                  "erratum": "duplicates_ignored=0 counted only canonical "
                                             "duplicates; one verified row was excluded by "
                                             "the per-function cap"},
        "funnel": {
            "discovered_fragments": stage(discovered, cands),
            "environment_qualified": stage(qualified, cands),
            "natively_verified_both_runs": stage(verified, cands),
            "pre_selection_eligible": stage(verified, cands),
            "excluded_exact_duplicates": reasons.get("excluded:exact_duplicate", 0),
            "excluded_near_duplicates": 0,
            "excluded_per_function_cap": reasons.get("excluded:per_function_cap", 0),
            "excluded_per_lineage_cap": reasons.get("excluded:per_lineage_cap", 0),
            "excluded_per_repository_cap": reasons.get("excluded:per_repository_cap", 0),
            "excluded_other": 0,
            "selected": stage(selected, cands)},
        "conservation": f"{len(verified)} verified = {len(selected)} selected + "
                        f"{len(verified) - len(selected)} excluded",
        "excluded_rows": excluded,
        "near_duplicate_policy": "reported, not removed",
        "near_duplicate_clusters_in_selection": {"clusters": len(near),
                                                 "multi_member": sum(n > 1 for n in near.values())},
        "caps": CAPS,
        "sidecar": {"path": SIDECAR, "sha256": hashlib.sha256(data).hexdigest()},
        "inputs_sha256": {"verification_run1": sha(ROOT / VERIFY / "repository_verification_run1.jsonl"),
                          "verification_run2": sha(ROOT / VERIFY / "repository_verification_run2.jsonl"),
                          "candidates": sha(root / "converted_candidates_r2r.jsonl"),
                          "targets": sha(root / "targets_a3.jsonl")},
        "problems": problems, "status": "PASS" if not problems else "FAIL"}
    if problems:
        raise SystemExit(f"REFUSED (fail closed): {problems}")
    status = publish_once(RECEIPT, receipt)
    print(json.dumps({"status": status, "conservation": receipt["conservation"],
                      "excluded": excluded, "selected": receipt["funnel"]["selected"],
                      "sha256": sha(ROOT / RECEIPT)}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
