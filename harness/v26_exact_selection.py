"""Exact binary-MILP corpus selection under the frozen v2.5/v2.6 selection rules (SciPy/HiGHS).

One binary variable x_i per candidate. Constraints (all frozen rules of the corpus selector):
- canonical-test uniqueness / exact-duplicate exclusion: sum over each canonical test <= 1;
- per-function cap 3, per-lineage cap 3, per-repository cap 40;
- membership: a candidate of the confirmation, validation or sealed pools is rejected BEFORE the
  model is built (ValueError), never silently dropped;
- near duplicates: policy "reported, not removed" -> no constraint (recorded in the manifest).
Indicator variables y_r <= sum_{i in r} x_i and z_l <= sum_{i in l} x_i count selected
repositories and lineages exactly.

Only an OPTIMAL (zero gap) or INFEASIBLE termination supports a structural claim; a time limit,
nonzero gap or any other status is INCONCLUSIVE. Every witness is re-checked by an independent
validator that does not share code with the model builder.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import time
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

CAPS = {"per_function": 3, "per_lineage": 3, "per_repository": 40}
GATE = {"repository_tests": 150, "repositories": 8, "lineages": 60}
FORBIDDEN_POOLS = {"confirmation", "confirmation_only", "validation", "sealed", "sealed_final"}
CONCLUSIVE = {"OPTIMAL", "INFEASIBLE"}


class Inconclusive(RuntimeError):
    """The solver did not prove optimality or infeasibility."""


def check_membership(rows: Sequence[Mapping]) -> None:
    bad = [r.get("id") for r in rows if str(r.get("pool", "training")).lower() in FORBIDDEN_POOLS]
    if bad:
        raise ValueError(f"REFUSED: non-training records offered to the selector: {bad[:5]}")


def _groups(rows: Sequence[Mapping], key: str) -> Dict:
    g = defaultdict(list)
    for i, r in enumerate(rows):
        g[r[key]].append(i)
    return g


def solve(rows: Sequence[Mapping], objective: str = "tests",
          min_counts: Optional[Mapping[str, int]] = None,
          fix_tests: Optional[int] = None, caps: Mapping[str, int] = CAPS,
          time_limit: float = 600.0) -> Dict:
    """``rows``: dicts with canonical, function, lineage, repository (and optional pool, id).
    objective: tests | repositories | lineages | none (pure feasibility)."""
    import numpy as np
    from scipy.optimize import Bounds, LinearConstraint, milp
    check_membership(rows)
    n = len(rows)
    repos, lins = _groups(rows, "repository"), _groups(rows, "lineage")
    rkeys, lkeys = sorted(repos, key=str), sorted(lins, key=str)
    nv = n + len(rkeys) + len(lkeys)
    A, lo, hi, kinds = [], [], [], []

    def row(coeffs, lb, ub, kind):
        v = np.zeros(nv)
        for j, c in coeffs:
            v[j] += c
        A.append(v)
        lo.append(lb)
        hi.append(ub)
        kinds.append(kind)
    for key, cap, kind in (("canonical", 1, "canonical_unique"),
                           ("function", caps["per_function"], "per_function"),
                           ("lineage", caps["per_lineage"], "per_lineage"),
                           ("repository", caps["per_repository"], "per_repository")):
        for members in _groups(rows, key).values():
            if len(members) > cap:
                row([(i, 1) for i in members], -np.inf, cap, kind)
    for k, r in enumerate(rkeys):                   # y_r <= sum x
        row([(n + k, 1)] + [(i, -1) for i in repos[r]], -np.inf, 0, "repository_indicator")
    for k, l in enumerate(lkeys):                   # z_l <= sum x
        row([(n + len(rkeys) + k, 1)] + [(i, -1) for i in lins[l]], -np.inf, 0,
            "lineage_indicator")
    x_idx = [(i, 1) for i in range(n)]
    y_idx = [(n + k, 1) for k in range(len(rkeys))]
    z_idx = [(n + len(rkeys) + k, 1) for k in range(len(lkeys))]
    mins = dict(min_counts or {})
    if "repository_tests" in mins:
        row(x_idx, mins["repository_tests"], np.inf, "min_tests")
    if "repositories" in mins:
        row(y_idx, mins["repositories"], np.inf, "min_repositories")
    if "lineages" in mins:
        row(z_idx, mins["lineages"], np.inf, "min_lineages")
    if fix_tests is not None:
        row(x_idx, fix_tests, fix_tests, "fix_tests")
    c = np.zeros(nv)
    target = {"tests": x_idx, "repositories": y_idx, "lineages": z_idx}.get(objective, [])
    for j, _ in target:
        c[j] = -1.0
    started = time.time()
    res = milp(c=c, integrality=np.ones(nv), bounds=Bounds(0, 1),
               constraints=LinearConstraint(np.array(A), lo, hi) if A else (),
               options={"time_limit": time_limit, "mip_rel_gap": 0.0, "presolve": True,
                        "disp": False})
    runtime = round(time.time() - started, 3)
    status = {0: "OPTIMAL", 1: "TIME_LIMIT", 2: "INFEASIBLE", 3: "UNBOUNDED",
              4: "OTHER"}.get(res.status, "UNKNOWN")
    gap = getattr(res, "mip_gap", None)
    if status == "OPTIMAL" and gap not in (None, 0, 0.0) and gap > 1e-12:
        status = "NONZERO_GAP"
    out = {"status": status, "objective": objective, "runtime_seconds": runtime,
           "mip_gap": gap, "best_bound": getattr(res, "mip_dual_bound", None),
           "time_limit": time_limit, "constraints": len(A), "variables": nv,
           "constraint_kinds": {k: kinds.count(k) for k in sorted(set(kinds))}}
    if status == "OPTIMAL":
        x = [int(round(v)) for v in res.x[:n]]
        sel = [i for i in range(n) if x[i]]
        out.update(selected=sel, objective_value=int(round(-res.fun)) if target else 0,
                   counts=counts([rows[i] for i in sel]))
    return out


def counts(selected: Iterable[Mapping]) -> Dict[str, int]:
    selected = list(selected)
    return {"repository_tests": len(selected),
            "repositories": len({r["repository"] for r in selected}),
            "lineages": len({r["lineage"] for r in selected})}


def validate_witness(rows: Sequence[Mapping], selected: Sequence[int],
                     caps: Mapping[str, int] = CAPS) -> List[str]:
    """Independent check (no shared code with the model builder)."""
    problems = []
    if len(set(selected)) != len(selected):
        problems.append("index selected twice")
    if any(i < 0 or i >= len(rows) for i in selected):
        return problems + ["index out of range"]
    tally = {"canonical": {}, "function": {}, "lineage": {}, "repository": {}}
    for i in selected:
        r = rows[i]
        if str(r.get("pool", "training")).lower() in FORBIDDEN_POOLS:
            problems.append(f"non-training record {i}")
        for k in tally:
            tally[k][r[k]] = tally[k].get(r[k], 0) + 1
    limits = {"canonical": 1, "function": caps["per_function"],
              "lineage": caps["per_lineage"], "repository": caps["per_repository"]}
    for k, t in tally.items():
        over = [g for g, n in t.items() if n > limits[k]]
        if over:
            problems.append(f"{k} limit exceeded: {over[:3]}")
    return problems


def require_conclusive(result: Mapping) -> Mapping:
    if result["status"] not in CONCLUSIVE:
        raise Inconclusive(f"INCONCLUSIVE: solver status {result['status']}")
    return result


def witness_hash(rows: Sequence[Mapping], selected: Sequence[int]) -> str:
    ids = sorted(str(rows[i].get("id", i)) for i in selected)
    return hashlib.sha256(json.dumps(ids).encode()).hexdigest()


def greedy_lower_bound(rows: Sequence[Mapping], caps: Mapping[str, int] = CAPS) -> List[int]:
    """The v2.6 Phase 4 heuristic (sorted by canonical, take when caps allow). A feasible set
    and therefore a LOWER BOUND - never a maximum."""
    seen, tally, out = set(), defaultdict(int), []
    for i in sorted(range(len(rows)), key=lambda i: str(rows[i]["canonical"])):
        r = rows[i]
        if r["canonical"] in seen or tally[("f", r["function"])] >= caps["per_function"] or \
                tally[("l", r["lineage"])] >= caps["per_lineage"] or \
                tally[("r", r["repository"])] >= caps["per_repository"]:
            continue
        seen.add(r["canonical"])
        for k in (("f", r["function"]), ("l", r["lineage"]), ("r", r["repository"])):
            tally[k] += 1
        out.append(i)
    return out
