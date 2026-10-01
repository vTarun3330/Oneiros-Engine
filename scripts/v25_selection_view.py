"""v2.5 Phase 4: unique-first selection VIEW over the verified stage-1 r2 pool (protocol v2.5
B.3-B.4, addendum 1 section 5). Not the final corpus: repository positives are still pending,
so this fixes only the synthetic auxiliary/replay component and reports what is achievable.

Rules (deterministic, outcome-free beyond the verification that defines a positive):
1. group verified rows by function (group_id); collapse identical canonical-AST modules;
   each canonical test carries the set of sibling mutants (records) it kills;
2. per function, greedy set cover: repeatedly take the canonical test killing the most
   still-uncovered mutants (ties: more mutants killed overall, then canonical hash), at most
   ``PER_FUNCTION`` canonical tests per function;
3. one row per selected canonical test (never repeated); its families are those of the mutants
   it kills; HumanEval, MBPP and curated mutations form ONE synthetic source group;
4. report record-, function- and unique-test-weighted distributions and the addendum caps.

    python scripts/v25_selection_view.py --dir results/sft_root_cause/v25_corpus_stage1_r2
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

PER_FUNCTION = 3                      # addendum 1, section 5
FAMILY_SHARE_CAP = 0.35
SYNTHETIC_GROUP = "synthetic_mutation_functions"
RECEIPT = "results/sft_root_cause_v25_selection_view.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def select_function(tests: dict) -> list:
    """tests: {canonical: set(record ids)} -> ordered selected canonical ids (set cover)."""
    uncovered = set().union(*tests.values()) if tests else set()
    chosen = []
    while uncovered and len(chosen) < PER_FUNCTION:
        best = max(sorted(tests), key=lambda c: (len(tests[c] & uncovered), len(tests[c]), c))
        if not tests[best] & uncovered:
            break
        chosen.append(best)
        uncovered -= tests[best]
    return chosen


def build(directory: Path) -> dict:
    cand = {c["index"]: c for c in map(json.loads, (directory / "converted_candidates.jsonl")
                                       .read_text(encoding="utf-8").splitlines())}
    runs = sorted(directory.glob("verification_run[0-9].jsonl"))
    if len(runs) != 2 or sha(runs[0]) != sha(runs[1]):
        raise SystemExit("REFUSED: two byte-identical repeatability runs are required")
    verified = [cand[v["index"]] for v in map(json.loads, runs[0].read_text(encoding="utf-8")
                                              .splitlines()) if v["accepted"]]
    synthetic = [c for c in verified if c["execution_mode"] == "function_assertion"]
    repository = [c for c in verified if c["execution_mode"] != "function_assertion"]
    by_function = defaultdict(lambda: defaultdict(set))
    meta = {}
    for c in synthetic:
        canon = c["conversion"]["identities"]["canonical_ast"]
        by_function[c["group_id"]][canon].add(c["id"])
        meta.setdefault(canon, {"module_sha256": c["conversion"]["module_sha256"],
                                "dataset": c["dataset"], "complexity": c["complexity"],
                                "families": set(), "near": c["conversion"]["identities"]
                                ["near_duplicate_cluster"]})
        meta[canon]["families"].add(c["bug_family"])
    selected, mutants_total, mutants_covered = [], 0, 0
    for function in sorted(by_function):
        tests = by_function[function]
        mutants = set().union(*tests.values())
        chosen = select_function(tests)
        mutants_total += len(mutants)
        mutants_covered += len(set().union(*(tests[c] for c in chosen)))
        for rank, canon in enumerate(chosen):
            m = meta[canon]
            selected.append({"function": function, "canonical_ast": canon, "rank": rank,
                             "module_sha256": m["module_sha256"], "dataset": m["dataset"],
                             "source_group": SYNTHETIC_GROUP, "complexity": m["complexity"],
                             "families": sorted(m["families"]),
                             "mutants_killed": sorted(tests[canon]),
                             "near_duplicate_cluster": m["near"]})
    fam_weight = Counter()
    for s in selected:                      # a test's weight is split across its families
        for f in s["families"]:
            fam_weight[f] += 1 / len(s["families"])
    n = len(selected)
    family_share = {f: round(w / n, 4) for f, w in fam_weight.most_common()}
    return {
        "inputs": {"converted_candidates_sha256": sha(directory / "converted_candidates.jsonl"),
                   "repeatability_run_sha256": sha(runs[0])},
        "rules": {"per_function_canonical_tests": PER_FUNCTION, "selection": "greedy mutant "
                  "set cover per function; ties by mutants killed then canonical hash",
                  "repetition": "none", "synthetic_source_group": SYNTHETIC_GROUP},
        "synthetic": {
            "verified_rows": len(synthetic),
            "functions": len(by_function),
            "unique_canonical_tests": len(meta),
            "selected_unique_tests": n,
            "mutants_total": mutants_total, "mutants_covered_by_selection": mutants_covered,
            "selected_per_function": dict(Counter(Counter(s["function"] for s in selected)
                                                  .values())),
            "distribution_row_weighted": {
                "dataset": dict(Counter(c["dataset"] for c in synthetic)),
                "complexity": dict(Counter(c["complexity"] for c in synthetic)),
                "bug_family": dict(Counter(c["bug_family"] for c in synthetic))},
            "distribution_function_weighted": {
                "dataset": dict(Counter({c["group_id"]: c["dataset"]
                                         for c in synthetic}.values())),
                "complexity": dict(Counter({c["group_id"]: c["complexity"]
                                            for c in synthetic}.values()))},
            "distribution_unique_test_weighted": {
                "dataset": dict(Counter(s["dataset"] for s in selected)),
                "complexity": dict(Counter(s["complexity"] for s in selected)),
                "family_share": family_share},
            "family_cap": {"cap": FAMILY_SHARE_CAP,
                           "over_cap": {f: v for f, v in family_share.items()
                                        if v > FAMILY_SHARE_CAP}},
            "near_duplicate_clusters_selected": len({s["near_duplicate_cluster"]
                                                     for s in selected}),
            "complexity_note": "small-function AST tiers; not repository-scale"},
        "repository": {"verified_rows": len(repository),
                       "status": "pending native environments (Phases 6-7)"},
        "mixture_view": {
            "replay_rule": "synthetic replay <= 1 per verified repository example (addendum 1)",
            "max_synthetic_rows_given_current_repository": len(repository),
            "addendum_minimum_repository_tests": 150,
            "training_gate": "FAILS until >= 150 canonical repository tests from >= 8 "
                             "repositories and >= 60 lineages are verified"},
        "selected": selected}


def main(argv=None) -> int:
    from scripts.native_rehearsal_rebuild_v22 import publish_once
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dir", required=True)
    args = parser.parse_args(argv)
    directory = ROOT / args.dir
    view = build(directory)
    selected = view.pop("selected")
    path = directory / "selection_view_synthetic.jsonl"
    path.write_bytes(("\n".join(json.dumps(s, sort_keys=True) for s in selected) + "\n")
                     .encode("utf-8"))
    view["selection_file_sha256"] = sha(path)
    view["script_sha256"] = sha(Path(__file__))
    status = publish_once(RECEIPT, {"schema_version": "oneiros_v25_selection_view_v1", **view})
    print(json.dumps({"status": status, "synthetic": {k: view["synthetic"][k] for k in (
        "verified_rows", "functions", "unique_canonical_tests", "selected_unique_tests",
        "mutants_total", "mutants_covered_by_selection", "selected_per_function",
        "family_cap")}, "repository": view["repository"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
