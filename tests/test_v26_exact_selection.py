"""Exact MILP selection: greedy counterexample, brute force agreement, order invariance,
thresholds, witness validation, inconclusive refusal and membership rejection."""
from __future__ import annotations

from itertools import combinations
import random

import pytest

from harness import v26_exact_selection as ex


def r(i, canon, fn, lin, repo, pool="training"):
    return {"id": f"c{i}", "canonical": canon, "function": fn, "lineage": lin,
            "repository": repo, "pool": pool}


def counterexample():
    # one canonical test available under F (row 0) or G (row 1); three more F-only tests
    # the duplicate sorts FIRST, so greedy (sorted by canonical) takes it under F
    return [r(0, "a_dup", "F", "l0", "R"), r(1, "a_dup", "G", "l1", "R"),
            r(2, "x", "F", "l2", "R"), r(3, "y", "F", "l3", "R"), r(4, "z", "F", "l4", "R")]


def test_greedy_is_suboptimal_and_the_milp_is_exact():
    rows = counterexample()
    greedy = ex.greedy_lower_bound(rows)
    assert len(greedy) == 3 and 0 in greedy                 # took the duplicate under F
    out = ex.solve(rows)
    assert out["status"] == "OPTIMAL" and out["objective_value"] == 4
    assert sorted(out["selected"]) == [1, 2, 3, 4]
    assert ex.validate_witness(rows, out["selected"]) == []


def brute(rows, caps=ex.CAPS):
    best = 0
    for k in range(len(rows), 0, -1):
        for sub in combinations(range(len(rows)), k):
            if not ex.validate_witness(rows, list(sub), caps):
                return k
    return best


@pytest.mark.parametrize("seed", range(25))
def test_milp_matches_exhaustive_enumeration(seed):
    rng = random.Random(seed)
    caps = {"per_function": 2, "per_lineage": 2, "per_repository": 3}
    rows = [r(i, f"k{rng.randrange(6)}", f"F{rng.randrange(3)}", f"L{rng.randrange(4)}",
              f"R{rng.randrange(2)}") for i in range(rng.randrange(4, 11))]
    out = ex.solve(rows, caps=caps)
    assert out["status"] == "OPTIMAL"
    assert out["objective_value"] == brute(rows, caps)
    assert ex.validate_witness(rows, out["selected"], caps) == []


@pytest.mark.parametrize("seed", range(5))
def test_order_permutation_does_not_change_the_optimum(seed):
    rng = random.Random(100 + seed)
    rows = [r(i, f"k{rng.randrange(12)}", f"F{rng.randrange(5)}", f"L{rng.randrange(6)}",
              f"R{rng.randrange(3)}") for i in range(30)]
    base = ex.solve(rows)["objective_value"]
    for _ in range(3):
        shuffled = rows[:]
        rng.shuffle(shuffled)
        assert ex.solve(shuffled)["objective_value"] == base


def test_all_constraints_bind_simultaneously():
    rows = [r(i, f"k{i}", "F", f"L{i // 2}", "R") for i in range(6)]       # function cap 3
    rows += [r(10 + i, f"m{i}", f"G{i}", "LL", "S") for i in range(5)]    # lineage cap 3
    rows += [r(20, "k0", "H", "Lx", "T")]                                  # duplicate canonical
    out = ex.solve(rows, caps={"per_function": 3, "per_lineage": 3, "per_repository": 40})
    assert out["objective_value"] == 7 and ex.validate_witness(rows, out["selected"]) == []


def gate_rows(tests, repos, lins):
    return [r(i, f"k{i}", f"F{i}", f"L{i % lins}", f"R{i % repos}") for i in range(tests)]


@pytest.mark.parametrize("tests, repos, lins, status", [
    (149, 8, 60, "INFEASIBLE"), (150, 8, 60, "OPTIMAL"), (151, 8, 60, "OPTIMAL"),
    (150, 7, 60, "INFEASIBLE"), (150, 9, 60, "OPTIMAL"),
    (150, 8, 59, "INFEASIBLE"), (150, 8, 61, "OPTIMAL")])
def test_joint_gate_threshold_boundaries(tests, repos, lins, status):
    out = ex.solve(gate_rows(tests, repos, lins), objective="none", min_counts=ex.GATE)
    assert out["status"] == status


def test_invalid_witnesses_are_rejected():
    rows = counterexample()
    assert ex.validate_witness(rows, [0, 1]) != []                 # duplicate canonical
    assert ex.validate_witness(rows, [0, 2, 3, 4]) != []           # function cap
    assert ex.validate_witness(rows, [2, 2]) != []                 # repeated index
    assert ex.validate_witness(rows, [99]) != []


def test_inconclusive_results_are_refused():
    for status in ("TIME_LIMIT", "NONZERO_GAP", "UNKNOWN", "OTHER"):
        with pytest.raises(ex.Inconclusive):
            ex.require_conclusive({"status": status})
    assert ex.require_conclusive({"status": "INFEASIBLE"})


@pytest.mark.parametrize("pool", ["confirmation", "validation", "sealed", "sealed_final"])
def test_protected_records_are_rejected_before_the_model(pool):
    rows = counterexample() + [r(9, "z", "Z", "lz", "RZ", pool=pool)]
    with pytest.raises(ValueError, match="non-training"):
        ex.solve(rows)


def test_heuristic_count_is_only_a_lower_bound():
    rows = counterexample()
    assert len(ex.greedy_lower_bound(rows)) <= ex.solve(rows)["objective_value"]


def test_tracked_exact_receipt_is_scoped_and_reproduces_the_audit():
    import json
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent
    r = json.loads((root / "results/sft_root_cause_v26_structural_feasibility_v2_exact.json")
                   .read_text(encoding="utf-8"))
    a = r["universe_A_all_163_materialized_fragments"]["results"]
    b = r["universe_B_qualified_environments_today"]["results"]
    assert (a["max_tests"]["objective_value"], b["max_tests"]["objective_value"]) == (146, 101)
    assert a["max_tests"]["status"] == "OPTIMAL" and a["max_tests"]["mip_gap"] == 0.0
    assert a["joint_feasibility_150_8_60"]["status"] == "INFEASIBLE"
    assert a["heuristic_greedy"]["label"].startswith("HEURISTIC")
    assert r["states"]["TRAINING_EXPANSION_POOL_FEASIBILITY_UNKNOWN"] is True
    s = json.loads((root / "results/sft_root_cause_v26_phase4_supersession.json")
                   .read_text(encoding="utf-8"))
    assert s["not_used"] == "GLOBAL_STRICT_TRAINING_ARM_STRUCTURALLY_INFEASIBLE"
    assert "GLOBAL_STRICT_TRAINING_ARM_STRUCTURALLY_INFEASIBLE" not in json.dumps(s["states"])
    assert s["repository_count_reconciliation"]["repositories_with_eligible_fragments"] == 12
