"""v2.7 Phase 5: exact optimality certificate for a frozen confirmation panel.

The admissible pool is every target that passed all per-target admission checks (i.e. the
selected targets plus those excluded only by a SELECTION constraint: per-repository cap,
function-fingerprint or lineage duplication). The selection constraints are separable by
repository except fingerprint/lineage uniqueness, so the maximum panel size is computed exactly
as a maximum bipartite-free set packing per repository by exhaustive search when small, and the
global upper bound is sum_r min(cap, distinct lineages and fingerprints in r). The certificate
states whether the frozen selection attains that maximum.

    python scripts/v27_panel_optimality.py --panel results/<panel receipt>.json --out <json>
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import itertools
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SELECTION_REASONS = ("per_repository_cap", "duplicate_function_fingerprint", "duplicate_lineage")


def max_unique(items: list, cap: int) -> int:
    """Largest subset with distinct fingerprints and distinct lineages, at most ``cap``."""
    best = 0
    for size in range(min(cap, len(items)), 0, -1):
        for combo in itertools.combinations(items, size):
            if len({c["fingerprint"] for c in combo}) == size and \
                    len({c["lineage"] for c in combo}) == size:
                return size
    return best


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--panel", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    panel_path = ROOT / args.panel
    panel = json.loads(panel_path.read_text(encoding="utf-8"))
    gate = json.loads((ROOT / panel["gate"]["path"]).read_text(encoding="utf-8"))
    cap = int(panel["panel_status"]["minimum"]["max_per_repository"])
    selected = panel["targets"]
    pool = defaultdict(list)
    for t in selected:
        pool[t["repository"]].append({"key": t["target_id"], "fingerprint": t["function_fingerprint"],
                                      "lineage": t["lineage"]})
    selection_excluded = [e for e in panel["exclusions"]
                          if e["reason"].split(":")[0] in SELECTION_REASONS]
    qual = {t["key"]: t for t in json.loads((ROOT / "results/sft_root_cause/v27_confirmation/"
                                             f"{panel['name']}_qualification_manifest.json")
                                            .read_text(encoding="utf-8"))["targets"]}
    for e in selection_excluded:
        t = qual[e["key"]]
        pool[t["repository"]].append({"key": e["key"], "fingerprint": t["function_fingerprint"],
                                      "lineage": t["lineage"]})
    per_repo = {r: {"admissible": len(v), "maximum": max_unique(v, cap),
                    "selected": sum(1 for t in selected if t["repository"] == r)}
                for r, v in sorted(pool.items())}
    maximum = sum(v["maximum"] for v in per_repo.values())
    minimum = panel["panel_status"]["minimum"]
    out = {
        "schema_version": "oneiros_v27_panel_optimality_v1",
        "panel": {"path": args.panel, "sha256": hashlib.sha256(panel_path.read_bytes()).hexdigest()},
        "method": "exact: per-repository exhaustive maximum of distinct-fingerprint, "
                  "distinct-lineage subsets under the cap; repositories are independent",
        "admissible_pool": sum(len(v) for v in pool.values()),
        "selection_exclusions": len(selection_excluded),
        "per_repository": per_repo,
        "exact_maximum_targets": maximum,
        "exact_maximum_repositories": sum(1 for v in per_repo.values() if v["maximum"]),
        "selected_targets": len(selected),
        "selection_is_optimal": len(selected) == maximum,
        "target_80_feasible_in_materialized_pool": maximum >= int(minimum["targets"]),
        "gate_minimum": minimum,
        "frozen_selection_order": gate.get("selection_order") or gate.get("order"),
        "scope_note": "infeasibility holds for this materialized r4 pool only; it says nothing "
                      "about the ceiling of unmaterialized repositories",
    }
    (ROOT / args.out).write_text(json.dumps(out, indent=1, sort_keys=True) + "\n",
                                 encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("admissible_pool", "exact_maximum_targets",
                                         "selected_targets", "selection_is_optimal",
                                         "target_80_feasible_in_materialized_pool")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
