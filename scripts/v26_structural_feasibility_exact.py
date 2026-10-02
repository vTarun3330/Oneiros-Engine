"""v2.6 Phase 1 (successor): EXACT structural feasibility of the frozen 150/8/60 training gate
for the currently MATERIALIZED universes, by binary MILP (harness/v26_exact_selection.py).

Supersedes the method (not the artifact) of scripts/v26_structural_feasibility.py, whose greedy
selection is only a feasible set (HEURISTIC_FEASIBLE_SET / LOWER_BOUND), not a maximum.

Universes (same candidate rows and identities as the greedy version):
  A - all 163 eligible fragments of the 12 repositories represented in the materialized corpus,
      assuming every candidate verifies;
  B - only fragments whose environment is qualified today (v2.6 a3).
The 36-repository training-expansion pool is NOT materialized: its maximum is UNKNOWN.

Per universe: direct joint feasibility (tests >= 150 AND repositories >= 8 AND lineages >= 60
under every cap), exact maxima of tests, repositories and lineages separately, and
lexicographic maxima (lineages, then repositories, subject to maximum tests). Only OPTIMAL or
INFEASIBLE terminations are accepted; every witness is independently validated.

    python scripts/v26_structural_feasibility_exact.py
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RECEIPT = "results/sft_root_cause_v26_structural_feasibility_v2_exact.json"
SUPERSEDED = "results/sft_root_cause_v26_structural_feasibility.json"
DIR = "results/sft_root_cause/v25_native_r2"
DIAG = "results/sft_root_cause/v26_funnel/diagnostics.jsonl"
VERIFY = "results/sft_root_cause/v26_verify_a3/repository_verification_run1.jsonl"
CONSERVATION = "results/sft_root_cause_v26_selection_conservation.json"
TIME_LIMIT = 600.0


def sha(rel) -> str:
    return hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()


def jsonl(rel) -> list:
    return [json.loads(l) for l in (ROOT / rel).read_text(encoding="utf-8").splitlines()
            if l.strip()]


def candidate_rows() -> list:
    cands = {c["index"]: c for c in jsonl(f"{DIR}/converted_candidates_r2r.jsonl")}
    targets = jsonl(f"{DIR}/targets_a3.jsonl")
    verdicts = {v["index"]: v for v in jsonl(VERIFY)}
    diag = {d["index"]: d for d in jsonl(DIAG)}
    rows = []
    for t in sorted(targets, key=lambda t: t["task"]):
        for f in t["fragments"]:
            v = verdicts.get(f["index"], {})
            qual = v.get("qualname") or (diag.get(f["index"]) or {}).get("localized")
            rows.append({"id": f"fragment:{f['index']}", "pool": "training",
                         "repository": t["project"], "lineage": t["task"],
                         "canonical": cands[f["index"]]["conversion"]["identities"]
                         ["canonical_ast"],
                         "function": json.dumps([t["project"], t["target_file"], qual]) if qual
                         else json.dumps([t["project"], t["task"], "unknown-function"]),
                         "qualified_env": not str(v.get("status", "")).startswith(
                             "environment:") and v.get("status") != "no_environment_row"})
    return rows


def analyse(rows: list) -> dict:
    from harness import v26_exact_selection as ex
    out, witnesses = {}, {}

    def run(name, **kw):
        res = ex.require_conclusive(ex.solve(rows, time_limit=TIME_LIMIT, **kw))
        rec = {k: res[k] for k in ("status", "objective_value", "mip_gap", "best_bound",
                                   "runtime_seconds", "constraints", "variables",
                                   "constraint_kinds") if k in res}
        if res["status"] == "OPTIMAL":
            problems = ex.validate_witness(rows, res["selected"])
            if problems:
                raise SystemExit(f"REFUSED: invalid witness for {name}: {problems}")
            rec.update(witness_counts=res["counts"],
                       witness_sha256=ex.witness_hash(rows, res["selected"]),
                       witness_valid=True)
            witnesses[name] = sorted(rows[i]["id"] for i in res["selected"])
        out[name] = rec
        return res
    run("joint_feasibility_150_8_60", objective="none", min_counts=ex.GATE)
    best = run("max_tests", objective="tests")
    run("max_repositories", objective="repositories")
    run("max_lineages", objective="lineages")
    run("lex_max_lineages_at_max_tests", objective="lineages",
        fix_tests=best["objective_value"])
    run("lex_max_repositories_at_max_tests", objective="repositories",
        fix_tests=best["objective_value"])
    greedy = ex.greedy_lower_bound(rows)
    out["heuristic_greedy"] = {"label": "HEURISTIC_FEASIBLE_SET / LOWER_BOUND",
                               "tests": len(greedy),
                               "valid": ex.validate_witness(rows, greedy) == []}
    return {"candidates": len(rows), "results": out}, witnesses


def main() -> int:
    import scipy
    from harness import v26_exact_selection as ex
    from scripts.native_rehearsal_rebuild_v22 import publish_once
    rows = candidate_rows()
    a, wa = analyse(rows)
    b, wb = analyse([r for r in rows if r["qualified_env"]])
    witness_file = ROOT / "results/sft_root_cause/v26_funnel/exact_witnesses.json"
    data = (json.dumps({"A": wa, "B": wb}, indent=1, sort_keys=True) + "\n").encode()
    if witness_file.exists() and witness_file.read_bytes() != data:
        raise SystemExit("REFUSED: witness file exists with different content")
    witness_file.write_bytes(data)
    manifest = {
        "rules": [
            {"type": "canonical_test_unique (exact-duplicate exclusion)", "bound": 1},
            {"type": "per_function_cap", "bound": ex.CAPS["per_function"]},
            {"type": "per_lineage_cap", "bound": ex.CAPS["per_lineage"]},
            {"type": "per_repository_cap", "bound": ex.CAPS["per_repository"]},
            {"type": "near_duplicates", "bound": None,
             "note": "policy: reported, not removed (no constraint)"},
            {"type": "membership", "bound": "training pool only; confirmation / validation / "
                                           "sealed rejected before the model is built"}],
        "source": {"harness/v26_exact_selection.py": sha("harness/v26_exact_selection.py"),
                   "docs/SFT_ROOT_CAUSE_NATIVE_GENERATED_TEST_PROTOCOL_V2_5_ADDENDUM_1.md":
                   sha("docs/SFT_ROOT_CAUSE_NATIVE_GENERATED_TEST_PROTOCOL_V2_5_ADDENDUM_1.md")},
        "affected_candidates": {"A": len(rows), "B": sum(r["qualified_env"] for r in rows)}}
    manifest_sha = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    reproduces = (a["results"]["max_tests"]["objective_value"],
                  b["results"]["max_tests"]["objective_value"]) == (146, 101)
    if not reproduces:
        raise SystemExit("STOP: exact maxima do not reproduce 146 / 101; reconcile first")
    joint_a = a["results"]["joint_feasibility_150_8_60"]["status"]
    receipt = {
        "schema_version": "oneiros_v26_structural_feasibility_v2_exact",
        "solver": {"name": "HiGHS via scipy.optimize.milp", "scipy": scipy.__version__,
                   "parameters": {"mip_rel_gap": 0.0, "presolve": True,
                                  "time_limit_seconds": TIME_LIMIT},
                   "accepted_terminations": sorted(ex.CONCLUSIVE)},
        "constraint_manifest": manifest, "constraint_manifest_sha256": manifest_sha,
        "universe_A_all_163_materialized_fragments": a,
        "universe_B_qualified_environments_today": b,
        "witness_file": {"path": "results/sft_root_cause/v26_funnel/exact_witnesses.json",
                         "sha256": hashlib.sha256(data).hexdigest()},
        "reproduces_independent_audit_146_101": reproduces,
        "supersedes_method_of": {"artifact": SUPERSEDED, "sha256": sha(SUPERSEDED),
                                 "method": "greedy (sorted by canonical); its counts are a "
                                           "HEURISTIC_FEASIBLE_SET / LOWER_BOUND, not maxima"},
        "states": {
            "CURRENT_MATERIALIZED_UNIVERSE_STRUCTURALLY_INFEASIBLE_FOR_150": joint_a == "INFEASIBLE",
            "CURRENT_OBSERVED_CORPUS_NOT_TRAINING_READY": True,
            "CURRENT_GPU_TRAINING_NOT_AUTHORIZED": True,
            "TRAINING_EXPANSION_POOL_NOT_MATERIALIZED": True,
            "TRAINING_EXPANSION_POOL_FEASIBILITY_UNKNOWN": True},
        "scope": "the exact maxima apply ONLY to the currently materialized 163-fragment "
                 "universe of 12 repositories (A) and its qualified subset (B); the 36-"
                 "repository training-expansion pool is unacquired and its feasibility is "
                 "UNKNOWN. This is not a global proof that strict training is impossible.",
        "inputs_sha256": {p: sha(p) for p in (DIAG, VERIFY, CONSERVATION,
                                              f"{DIR}/targets_a3.jsonl",
                                              f"{DIR}/converted_candidates_r2r.jsonl")}}
    status = publish_once(RECEIPT, receipt)
    print(json.dumps({"status": status,
                      "A": {k: (v.get("status"), v.get("objective_value"), v.get("mip_gap"))
                            for k, v in a["results"].items() if k != "heuristic_greedy"},
                      "A_greedy": a["results"]["heuristic_greedy"],
                      "B": {k: (v.get("status"), v.get("objective_value"))
                            for k, v in b["results"].items() if k != "heuristic_greedy"},
                      "B_greedy": b["results"]["heuristic_greedy"],
                      "sha256": sha(RECEIPT)}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
