"""Summarise the post-hoc exploratory mechanical-repair matrix (R0-R4) with the SAME class sets,
fixed-validity rule and Kill@k definition as the frozen v2.4 analysis. R0 is the v2.4 primary
execution. Exploratory only: no significance, no confirmation claim. Written once.

    python scripts/repair_ceiling_summarise.py
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

from scripts import native_generated_tests_analyse as an
from scripts.native_rehearsal_rebuild_v22 import publish_once

OUT = "results/sft_root_cause_native_v24_repair_ceiling_results.json"
DIR = "results/sft_root_cause/native_v24_repair_ceiling"
V24 = "results/sft_root_cause/native_v24_execution/results_primary_whole_module.jsonl"
V24_SHA = "e40c912d23eccb3fb8a6b0cd06c11e27a48052582f4e60d1f1507ce7471a0d55"
DEFINITION = "results/sft_root_cause_native_v24_repair_ceiling_definition.json"
CLASSES = ("syntax_failure", "policy_refused", "over_limits", "collection_failure",
           "fabricated_import", "no_tests_collected", "skipped_or_xfail", "timeout",
           "target_not_reached", "pass_both", "fixed_side_failure", "semantic_kill",
           "crash_kill", "nondeterminism", "environment_failure")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(rel: str) -> list:
    return [json.loads(l) for l in (ROOT / rel).read_text(encoding="utf-8").splitlines()]


def metrics(rows: list, targets: list) -> dict:
    out = {}
    for arm in an.ARMS:
        grid = {(r["seed"], r["target_key"], r["slot"]): r for r in rows if r["arm"] == arm}
        if len(grid) != len(targets) * len(an.SEEDS) * an.SLOTS:
            raise SystemExit(f"REFUSED: incomplete grid for {arm}")
        classes = [r["class"] for r in grid.values()]
        kill = lambda s, t, k: float(any(grid[(s, t, i)]["class"] in an.KILLS  # noqa: E731
                                         for i in range(k)))
        entry = {"candidates": len(classes),
                 "parsed": sum(c not in an.PARSED_FAIL for c in classes),
                 "collected": sum(c not in an.COLLECT_FAIL for c in classes),
                 "executed": sum(c not in an.EXEC_FAIL for c in classes),
                 "reached": sum(c not in an.REACH_FAIL for c in classes),
                 "fixed_valid": sum(an.fixed_valid(r) for r in grid.values()),
                 "semantic_kills": classes.count("semantic_kill"),
                 "crash_kills": classes.count("crash_kill"),
                 "classes": {c: classes.count(c) for c in CLASSES if c in classes}}
        for k in (1, 4, 8):
            per_target = [sum(kill(s, t, k) for s in an.SEEDS) / len(an.SEEDS) for t in targets]
            entry[f"kill_at_{k}_percent"] = round(sum(per_target) / len(targets) * 100, 3)
        killed = sorted(t for t in targets if any(kill(s, t, an.SLOTS) for s in an.SEEDS))
        entry["unique_bugs_killed"] = len(killed)
        entry["killed_targets"] = killed
        entry["fixed_valid_targets"] = len({t for (s, t, i), r in grid.items()
                                            if an.fixed_valid(r)})
        entry["completion_limit_syntax_failures"] = sum(
            1 for r in grid.values() if r["class"] == "syntax_failure"
            and (r.get("generation") or {}).get("hit_completion_limit") is True)
        out[arm] = entry
    return out


def main() -> int:
    if sha(ROOT / V24) != V24_SHA:
        raise SystemExit("REFUSED: v2.4 primary execution differs")
    run = json.loads((ROOT / DIR / "run_summary.json").read_text(encoding="utf-8"))
    for c, digest in run["files"].items():
        if sha(ROOT / DIR / f"results_{c}.jsonl") != digest:
            raise SystemExit(f"REFUSED: {c} results differ from the run summary")
    v24 = load(V24)
    targets = sorted({r["target_key"] for r in v24})
    table = {"R0": metrics(v24, targets)}
    for c in ("R1", "R2", "R3", "R4"):
        rows = load(f"{DIR}/results_{c}.jsonl")
        if any(r.get("class") == "environment_failure" for r in rows):
            raise SystemExit(f"REFUSED: environment failures in {c} (canary failed)")
        table[c] = metrics(rows, targets)
    fields = ("parsed", "collected", "executed", "reached", "fixed_valid", "semantic_kills",
              "crash_kills", "unique_bugs_killed", "kill_at_8_percent")
    delta = {c: {arm: {f: round(table[c][arm][f] - table["R0"][arm][f], 3) for f in fields}
                 for arm in an.ARMS} for c in ("R1", "R2", "R3", "R4")}
    receipt = {"schema_version": "oneiros_repair_ceiling_results_v1",
               "study": "posthoc_exploratory_mechanical_repair_ceiling",
               "label": "exploratory diagnostic; R0 is the v2.4 primary result; no "
                        "significance, promotion or confirmation claim",
               "definition_sha256": sha(ROOT / DEFINITION),
               "summariser_sha256": sha(Path(__file__)),
               "inputs": {V24: V24_SHA, **{f"{DIR}/results_{c}.jsonl": d
                                           for c, d in run["files"].items()}},
               "execution": {k: run[k] for k in ("executed", "reused_v24_rows",
                                                 "reused_same_module")},
               "targets": len(targets), "metrics": table, "delta_vs_R0": delta}
    status = publish_once(OUT, receipt)
    print(json.dumps({"status": status, **{c: {arm: {f: table[c][arm][f] for f in fields}
                                                   for arm in an.ARMS} for c in table}},
                     indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
