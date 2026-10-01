"""v2.6 accounting: every verified row ends in exactly one outcome; caps are not duplicates."""
from __future__ import annotations

import json
from pathlib import Path

from scripts import v26_selection_conservation as sc

ROOT = Path(__file__).resolve().parent.parent


def _world(specs):
    cands, targets, rows = {}, {}, []
    for i, (canon, fn, task, repo) in enumerate(specs):
        cands[i] = {"conversion": {"identities": {"canonical_ast": canon}}}
        targets[task] = {"target_file": "m.py"}
        rows.append({"index": i, "task": task, "project": repo, "qualname": fn,
                     "class": "semantic_kill"})
    return rows, cands, targets


def test_each_row_has_exactly_one_outcome_and_caps_are_not_duplicates():
    rows, cands, targets = _world([
        ("a", "f", "t1", "r"), ("a", "f", "t2", "r"),          # exact duplicate
        ("b", "f", "t3", "r"), ("c", "f", "t4", "r"),
        ("d", "f", "t5", "r"),                                   # 4th of function f -> cap
        ("e", "g", "t6", "r")])
    out = sc.select(rows, cands, targets)
    assert sorted(v["index"] for v, _, _ in out) == list(range(6))
    reasons = [o for _, o, _ in out]
    assert reasons.count("selected") == 4
    assert reasons.count("excluded:exact_duplicate") == 1
    assert reasons.count("excluded:per_function_cap") == 1
    capped = next(d for _, o, d in out if o == "excluded:per_function_cap")
    assert len(capped["selected_rows_of_function"]) == 3


def test_repository_and_lineage_caps():
    rows, cands, targets = _world([(f"c{i}", f"f{i}", "T", "r") for i in range(4)])
    out = sc.select(rows, cands, targets)
    assert [o for _, o, _ in out].count("excluded:per_lineage_cap") == 1
    rows, cands, targets = _world([(f"c{i}", f"f{i}", f"t{i}", "r") for i in range(3)])
    out = sc.select(rows, cands, targets, {**sc.CAPS, "per_repository": 2})
    assert [o for _, o, _ in out].count("excluded:per_repository_cap") == 1


def test_tracked_conservation_receipt_balances_34_to_33():
    r = json.loads((ROOT / sc.RECEIPT).read_text(encoding="utf-8"))
    f = r["funnel"]
    assert f["natively_verified_both_runs"]["rows"] == 34 and f["selected"]["rows"] == 33
    excluded = (f["excluded_exact_duplicates"] + f["excluded_near_duplicates"]
                + f["excluded_per_function_cap"] + f["excluded_per_lineage_cap"]
                + f["excluded_per_repository_cap"] + f["excluded_other"])
    assert f["pre_selection_eligible"]["rows"] - excluded == f["selected"]["rows"]
    assert f["excluded_per_function_cap"] == 1 and f["excluded_exact_duplicates"] == 0
    assert [x["task"] for x in r["excluded_rows"]] == ["bugsinpy::youtube-dl::26"]
    assert (f["selected"]["semantic_kills"], f["selected"]["crash_kills"]) == (25, 8)


def test_structural_maximum_applies_every_cap_and_dedup():
    from scripts.v26_structural_feasibility import maximum
    rows = [{"canonical": f"c{i}", "function": "F", "lineage": f"l{i}", "repository": "r"}
            for i in range(5)]                                     # one function: cap 3
    rows += [{"canonical": "c0", "function": "G", "lineage": "x", "repository": "r"}]  # dup
    rows += [{"canonical": f"d{i}", "function": f"g{i}", "lineage": "L", "repository": "s"}
             for i in range(5)]                                     # one lineage: cap 3
    out = maximum(rows)
    assert out["maximum"] == {"repository_tests": 6, "repositories": 2, "lineages": 4}
    assert out["all_reachable"] is False


def test_tracked_feasibility_is_below_the_gate_and_never_rescued_by_confirmation():
    r = json.loads((ROOT / "results/sft_root_cause_v26_structural_feasibility.json")
                   .read_text(encoding="utf-8"))
    a = r["deterministic_maximum"]["a_all_163_eligible_fragments_existing_repositories"]
    assert a["maximum"]["repository_tests"] < 150 and a["all_reachable"] is False
    assert r["decision"]["label"] == "STRICT_TRAINING_INFEASIBLE_UNDER_CURRENT_PROTOCOL"
    assert r["training_expansion_pool"]["maximum"] == "UNKNOWN"
    assert "never counted toward training" in r["confirmation_only_pool"]["use"]
