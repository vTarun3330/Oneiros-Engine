"""Phase 3C analysis: 7B vs 1.5B on the frozen fixed-input items (CPU only).

Applies the frozen answer-rate gate first; accuracy is interpreted only if it
passes.  Primary contrasts C1 (A4) and C2 (A0), 7B minus 1.5B base, with Holm
correction on bootstrap two-sided p-values.  Paired cluster bootstrap over
semantic groups, 10,000 resamples, seed 20260928, one look.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness.atomic_publish import publish_file_atomically
from harness.fixed_input_probe import CONTROL_LEVEL, LEVELS, extract_answer, score_answers
from harness.parallel_execution import map_jobs

DESIGN_3A = "results/sft_root_cause_phase3a_design_receipt.json"
DESIGN = "results/sft_root_cause_phase3c_design_receipt.json"
RECEIPT = "results/sft_root_cause_phase3c_result_receipt.json"
SOURCES = {"base": "results/sft_root_cause/phase3a", "arm_a_431": "results/sft_root_cause/phase3a",
           "qwen7b_base": "results/sft_root_cause/phase3c"}
ALL_LEVELS = (*LEVELS, CONTROL_LEVEL)
RESAMPLES, SEED = 10_000, 20260928


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def score(job):
    function, record, generations = job
    rows = []
    for item in function["items"]:
        for arm, gens in generations.items():
            for level in ALL_LEVELS:
                key = f"{function['record_id']}::{item['input_kind']}::{level}"
                g = gens.get(key)
                a = extract_answer(g["output"], level, item["call"]) if g else \
                    {"answer": None, "method": "missing"}
                rows.append({"arm": arm, "level": level, "call": item["call"],
                             "answer": a["answer"], "method": a["method"],
                             "group_id": function["group_id"], "cohort": function["cohort"],
                             "tier": function["complexity_tier"]})
    for row, verdict in zip(rows, score_answers(rows, record["reference_code"])):
        row.update(verdict)
    return rows


def contrast(rows, groups, a, b, select=lambda r: True):
    idx = {g: i for i, g in enumerate(groups)}
    sums = np.zeros((2, len(groups), 2))
    for r in rows:
        for k, (arm, level) in enumerate((a, b)):
            if r["arm"] == arm and r["level"] == level and select(r):
                sums[k, idx[r["group_id"]], 0] += r["correct"]
                sums[k, idx[r["group_id"]], 1] += 1
    rng = np.random.default_rng(SEED)
    draws = rng.integers(0, len(groups), size=(RESAMPLES, len(groups)))
    w = np.zeros((RESAMPLES, len(groups)))
    np.add.at(w, (np.repeat(np.arange(RESAMPLES), len(groups)), draws.ravel()), 1)
    boot = [(w @ sums[k]) for k in range(2)]
    diff = (boot[1][:, 0] / boot[1][:, 1] - boot[0][:, 0] / boot[0][:, 1]) * 100
    point = [sums[k][:, 0].sum() / sums[k][:, 1].sum() for k in range(2)]
    low, high = np.percentile(diff, [2.5, 97.5])
    p = min(1.0, 2 * min((diff <= 0).mean(), (diff >= 0).mean()))
    return {"a": {"cell": list(a), "accuracy": round(point[0], 4)},
            "b": {"cell": list(b), "accuracy": round(point[1], 4)},
            "difference_points": round((point[1] - point[0]) * 100, 3),
            "ci95_points": [round(float(low), 3), round(float(high), 3)],
            "bootstrap_p_two_sided": round(float(p), 5)}


def region(c):
    low = c["ci95_points"][0]
    return ("supported" if low >= 5 else "directional" if low > 0 else
            "reversed" if c["ci95_points"][1] < 0 else "not_improved")


def main() -> int:
    design3a = json.loads((ROOT / DESIGN_3A).read_text(encoding="utf-8"))
    cohort_path = ROOT / design3a["cohort"]["path"]
    if sha(cohort_path) != design3a["cohort"]["sha256"]:
        raise SystemExit("REFUSED: cohort changed")
    cohort = json.loads(cohort_path.read_text(encoding="utf-8"))
    ids = {f["record_id"] for f in cohort["functions"]}
    records = {r["id"]: r for r in json.loads((ROOT / design3a["record_shard"]["path"])
                                              .read_text(encoding="utf-8")) if r["id"] in ids}
    generations, completions = {}, {}
    for arm, directory in SOURCES.items():
        completion = json.loads((ROOT / directory / f"completion_{arm}.json")
                                .read_text(encoding="utf-8"))
        path = ROOT / completion["output"]
        if sha(path) != completion["output_sha256"]:
            raise SystemExit(f"REFUSED: {arm} generations changed")
        completions[arm] = completion
        generations[arm] = {json.loads(l)["key"]: json.loads(l)
                            for l in path.read_text(encoding="utf-8").splitlines()}
    rows = [r for chunk in map_jobs(score, [(f, records[f["record_id"]], generations)
                                            for f in cohort["functions"]]) for r in chunk]
    groups = sorted({r["group_id"] for r in rows})

    def cell(arm, level):
        sel = [r for r in rows if r["arm"] == arm and r["level"] == level]
        answered = [r for r in sel if r["answer"]]
        return {"items": len(sel), "answer_rate": round(len(answered) / len(sel), 4),
                "accuracy": round(sum(r["correct"] for r in sel) / len(sel), 4),
                "methods": dict(Counter(r["method"] for r in sel))}

    cells = {f"{arm}/{level}": cell(arm, level) for arm in SOURCES for level in ALL_LEVELS}
    gate_rows = {level: {"qwen7b": cells[f"qwen7b_base/{level}"]["answer_rate"],
                         "base_1_5b": cells[f"base/{level}"]["answer_rate"]} for level in LEVELS}
    strict = sum(1 for r in rows if r["arm"] == "qwen7b_base" and r["level"] == CONTROL_LEVEL
                 and r["method"] == "strict") / cells[f"qwen7b_base/{CONTROL_LEVEL}"]["items"]
    gate = {"per_level": gate_rows, "control_strict_rate": round(strict, 4),
            "passed": all(v["qwen7b"] >= v["base_1_5b"] - 0.02 for v in gate_rows.values())
            and strict >= 0.95}
    receipt = {"schema_version": "oneiros_sft_root_cause_phase3c_result_v1", "phase": "3C",
               "command": "python scripts/analyze_capacity_probe.py",
               "design_receipt": {"path": DESIGN, "sha256": sha(ROOT / DESIGN)},
               "generation_completions": completions, "cells": cells, "gate": gate}
    if gate["passed"]:
        c1 = contrast(rows, groups, ("base", "A4"), ("qwen7b_base", "A4"))
        c2 = contrast(rows, groups, ("base", "A0"), ("qwen7b_base", "A0"))
        ordered = sorted([("C1", c1), ("C2", c2)], key=lambda x: x[1]["bootstrap_p_two_sided"])
        holm = {}
        for rank, (name, c) in enumerate(ordered):
            holm[name] = min(1.0, c["bootstrap_p_two_sided"] * (2 - rank))
        for name, c in (("C1", c1), ("C2", c2)):
            c["holm_p"] = round(max(holm[name], *(holm[n] for n, _ in ordered[
                :[x[0] for x in ordered].index(name)])), 5) if name != ordered[0][0] \
                else round(holm[name], 5)
            c["region"] = region(c)
        secondary = {
            "7b_minus_arm_a_A0": contrast(rows, groups, ("arm_a_431", "A0"), ("qwen7b_base", "A0")),
            "7b_minus_arm_a_A4": contrast(rows, groups, ("arm_a_431", "A4"), ("qwen7b_base", "A4")),
            "7b_A4_minus_A0": contrast(rows, groups, ("qwen7b_base", "A0"), ("qwen7b_base", "A4")),
            "7b_A3_minus_A0": contrast(rows, groups, ("qwen7b_base", "A0"), ("qwen7b_base", "A3")),
            "C1_by_cohort": {c: contrast(rows, groups, ("base", "A4"), ("qwen7b_base", "A4"),
                                         (lambda co: lambda r: r["cohort"] == co)(c))
                             for c in ("exposed", "unexposed")},
            "C2_by_tier": {t: contrast(rows, groups, ("base", "A0"), ("qwen7b_base", "A0"),
                                       (lambda ti: lambda r: r["tier"] == ti)(t))
                           for t in ("simple", "moderate")}}
        h3 = {"supported": "H3 supported at model scale: 1.5B execution reasoning is "
                           "capacity-limited",
              "directional": "H3 directional",
              "not_improved": "H3 weakened: scaling to 7B does not relieve the ceiling",
              "reversed": "H3 weakened: 7B is worse"}[c1["region"]]
        c2_note = ("C1 improves but C2 does not: A0 information (specification) is the binding "
                   "limit at 7B" if c1["region"] in ("supported", "directional")
                   and c2["region"] not in ("supported", "directional") else None)
        receipt.update({"primary": {"C1_A4": c1, "C2_A0": c2}, "secondary": secondary,
                        "decision": {"H3": h3, "C2_note": c2_note}})
    else:
        receipt["decision"] = {"H3": "gate failed: no capacity conclusion; H3 unchanged"}
    publish_file_atomically(ROOT / RECEIPT, (json.dumps(receipt, indent=1) + "\n")
                            .encode("utf-8"))
    print(json.dumps({"gate": gate, "cells": {k: (v["answer_rate"], v["accuracy"])
                                              for k, v in cells.items()}}, indent=1))
    print(json.dumps({k: receipt.get(k) for k in ("primary", "decision")}, indent=1))
    if "secondary" in receipt:
        print(json.dumps({k: (v["difference_points"], v["ci95_points"]) if "ci95_points" in v
                          else {kk: (vv["difference_points"], vv["ci95_points"])
                                for kk, vv in v.items()}
                          for k, v in receipt["secondary"].items()}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
