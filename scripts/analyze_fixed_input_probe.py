"""Phase 3A analysis: score the fixed-input generations and apply the frozen rules.

CPU only.  Reads the frozen cohort (hash-checked against the design receipt),
the record shard (for the reference, used only to score), and the two
generation files (hash-checked against their completion receipts).  Scoring
runs ``assert <call> == (<answer>)`` on the reference in the restricted worker.
Intervals: paired cluster bootstrap over semantic groups (one function per
group, both items of a function share its cluster), 10,000 resamples, seed
20260928.  One look.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness.atomic_publish import publish_file_atomically
from harness.fixed_input_probe import CONTROL_LEVEL, LEVELS, extract_answer, score_answers
from harness.parallel_execution import map_jobs

DESIGN = "results/sft_root_cause_phase3a_design_receipt.json"
GEN = "results/sft_root_cause/phase3a"
RECEIPT = "results/sft_root_cause_phase3a_result_receipt.json"
ARMS = ("base", "arm_a_431")
RESAMPLES, SEED = 10_000, 20260928
ALL_LEVELS = (*LEVELS, CONTROL_LEVEL)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_rows():
    design = json.loads((ROOT / DESIGN).read_text(encoding="utf-8"))
    cohort_path = ROOT / design["cohort"]["path"]
    if sha(cohort_path) != design["cohort"]["sha256"]:
        raise SystemExit("REFUSED: cohort changed")
    cohort = json.loads(cohort_path.read_text(encoding="utf-8"))
    ids = {f["record_id"] for f in cohort["functions"]}
    records = {r["id"]: r for r in json.loads((ROOT / design["record_shard"]["path"])
                                              .read_text(encoding="utf-8")) if r["id"] in ids}
    generations, completions = {}, {}
    for arm in ARMS:
        completion = json.loads((ROOT / GEN / f"completion_{arm}.json").read_text(encoding="utf-8"))
        path = ROOT / completion["output"]
        if sha(path) != completion["output_sha256"]:
            raise SystemExit(f"REFUSED: {arm} generations changed after completion")
        completions[arm] = completion
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        keys = [r["key"] for r in rows]
        if len(keys) != len(set(keys)):
            raise SystemExit(f"REFUSED: duplicate generation keys in {arm}")
        generations[arm] = {r["key"]: r for r in rows}
    return design, cohort, records, generations, completions


def score_function(job):
    function, record, generations = job["function"], job["record"], job["generations"]
    out = []
    for item in function["items"]:
        for arm in ARMS:
            for level in ALL_LEVELS:
                key = f"{function['record_id']}::{item['input_kind']}::{level}"
                generation = generations[arm].get(key)
                answer = (extract_answer(generation["output"], level, item["call"])
                          if generation else {"answer": None, "method": "missing"})
                out.append({"key": key, "arm": arm, "level": level, "call": item["call"],
                            "answer": answer["answer"], "method": answer["method"],
                            "hit_token_limit": bool(generation and generation["hit_token_limit"]),
                            "cohort": function["cohort"], "input_kind": item["input_kind"],
                            "group_id": function["group_id"],
                            "complexity_tier": function["complexity_tier"],
                            "bug_family": function["bug_family"],
                            "sft_target": item["call_in_arm_a_sft_target"]})
    verdicts = score_answers(out, record["reference_code"])
    for row, verdict in zip(out, verdicts):
        row.update(verdict)
    return out


def boot(rows, select, contrast, groups):
    """Paired cluster bootstrap of a contrast between two (arm, level) cells."""
    index = {g: i for i, g in enumerate(groups)}
    cells = {}
    for name, (arm, level) in contrast.items():
        sums = np.zeros((len(groups), 2))
        for r in rows:
            if r["arm"] == arm and r["level"] == level and select(r):
                sums[index[r["group_id"]], 0] += r["correct"]
                sums[index[r["group_id"]], 1] += 1
        cells[name] = sums
    rng = np.random.default_rng(SEED)
    draws = rng.integers(0, len(groups), size=(RESAMPLES, len(groups)))
    weights = np.zeros((RESAMPLES, len(groups)))
    np.add.at(weights, (np.repeat(np.arange(RESAMPLES), len(groups)), draws.ravel()), 1)
    rates = {}
    for name, sums in cells.items():
        b = weights @ sums
        with np.errstate(invalid="ignore", divide="ignore"):
            rates[name] = b[:, 0] / b[:, 1]
    a, b = list(contrast)
    diff = (rates[b] - rates[a]) * 100
    diff = diff[~np.isnan(diff)]
    point = {n: float(s[:, 0].sum() / s[:, 1].sum()) if s[:, 1].sum() else None
             for n, s in cells.items()}
    low, high = np.percentile(diff, [2.5, 97.5])
    return {"cells": {n: {"accuracy": round(point[n], 4) if point[n] is not None else None,
                          "correct": int(cells[n][:, 0].sum()), "items": int(cells[n][:, 1].sum())}
                      for n in cells},
            "difference_points": round((point[b] - point[a]) * 100, 3),
            "ci95_points": [round(float(low), 3), round(float(high), 3)],
            "improves": bool(low > 0), "supported": bool(low >= 5.0)}


def interaction(rows, groups, level="A0"):
    """(SFT-base on exposed) minus (SFT-base on unexposed), clusters resampled per cohort."""
    rng = np.random.default_rng(SEED)
    out = {}
    per = {}
    for cohort in ("exposed", "unexposed"):
        gs = sorted({r["group_id"] for r in rows if r["cohort"] == cohort})
        idx = {g: i for i, g in enumerate(gs)}
        s = np.zeros((len(gs), 2, 2))
        for r in rows:
            if r["cohort"] == cohort and r["level"] == level:
                s[idx[r["group_id"]], ARMS.index(r["arm"]), 0] += r["correct"]
                s[idx[r["group_id"]], ARMS.index(r["arm"]), 1] += 1
        draws = rng.integers(0, len(gs), size=(RESAMPLES, len(gs)))
        w = np.zeros((RESAMPLES, len(gs)))
        np.add.at(w, (np.repeat(np.arange(RESAMPLES), len(gs)), draws.ravel()), 1)
        b = np.einsum("rg,gak->rak", w, s)
        per[cohort] = (b[:, 1, 0] / b[:, 1, 1] - b[:, 0, 0] / b[:, 0, 1]) * 100
        out[cohort] = float((s[:, 1, 0].sum() / s[:, 1, 1].sum()
                             - s[:, 0, 0].sum() / s[:, 0, 1].sum()) * 100)
    diff = per["exposed"] - per["unexposed"]
    low, high = np.percentile(diff, [2.5, 97.5])
    return {"level": level, "sft_minus_base_exposed": round(out["exposed"], 3),
            "sft_minus_base_unexposed": round(out["unexposed"], 3),
            "interaction_points": round(out["exposed"] - out["unexposed"], 3),
            "ci95_points": [round(float(low), 3), round(float(high), 3)]}


def main() -> int:
    started = time.time()
    design, cohort, records, generations, completions = load_rows()
    jobs = [{"function": f, "record": records[f["record_id"]], "generations": generations}
            for f in cohort["functions"]]
    rows = [row for chunk in map_jobs(score_function, jobs) for row in chunk]
    groups = sorted({r["group_id"] for r in rows})
    ledger = "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows).encode("utf-8")
    publish_file_atomically(ROOT / GEN / "scored.jsonl", ledger)

    def cell_stats(select):
        out = {}
        for arm in ARMS:
            for level in ALL_LEVELS:
                sel = [r for r in rows if r["arm"] == arm and r["level"] == level and select(r)]
                answered = [r for r in sel if r["answer"]]
                out[f"{arm}/{level}"] = {
                    "items": len(sel),
                    "accuracy_per_requested": round(sum(r["correct"] for r in sel) / len(sel), 4)
                    if sel else None,
                    "answer_rate": round(len(answered) / len(sel), 4) if sel else None,
                    "accuracy_per_answered": round(sum(r["correct"] for r in answered)
                                                   / len(answered), 4) if answered else None,
                    "token_limit_hits": sum(r["hit_token_limit"] for r in sel),
                    "nonanswers": len(sel) - len(answered),
                    "extraction_methods": dict(Counter(r["method"] for r in sel))}
        return out

    strata = {"all": lambda r: True,
              "exposed": lambda r: r["cohort"] == "exposed",
              "unexposed": lambda r: r["cohort"] == "unexposed",
              "exposed_upstream_sft_target": lambda r: r["cohort"] == "exposed"
              and r["input_kind"] == "upstream",
              "exposed_novel": lambda r: r["cohort"] == "exposed" and r["input_kind"] == "novel",
              "unexposed_upstream": lambda r: r["cohort"] == "unexposed"
              and r["input_kind"] == "upstream",
              "unexposed_novel": lambda r: r["cohort"] == "unexposed"
              and r["input_kind"] == "novel"}
    for tier in ("simple", "moderate"):
        strata[f"tier_{tier}"] = (lambda t: lambda r: r["complexity_tier"] == t)(tier)
    descriptive = {name: cell_stats(sel) for name, sel in strata.items()}

    sft_vs_base = {name: {level: boot(rows, sel, {"base": ("base", level),
                                                  "sft": ("arm_a_431", level)}, groups)
                          for level in ALL_LEVELS}
                   for name, sel in strata.items()}
    level_contrasts = {arm: {f"{level}_minus_A0": boot(rows, strata["all"],
                                                        {"A0": (arm, "A0"),
                                                         level: (arm, level)}, groups)
                             for level in ("A3", "A4")}
                       for arm in ARMS}
    inter = interaction(rows, groups)
    families = Counter()
    for f in cohort["functions"]:
        families[(f["bug_family"], f["cohort"])] += 1
    big = sorted({fam for (fam, _), n in families.items()
                  if families[(fam, "exposed")] >= 10 and families[(fam, "unexposed")] >= 10})
    sensitivity = {fam: {c: boot(rows, (lambda fa, co: lambda r: r["bug_family"] == fa
                                        and r["cohort"] == co)(fam, c),
                                 {"base": ("base", "A0"), "sft": ("arm_a_431", "A0")}, groups)
                         for c in ("exposed", "unexposed")} for fam in big}

    # --- frozen decision rules ---------------------------------------------------------------
    exposed, unexposed = sft_vs_base["exposed"]["A0"], sft_vs_base["unexposed"]["A0"]
    control_all = sft_vs_base["all"][CONTROL_LEVEL]
    a0_all = sft_vs_base["all"]["A0"]
    schema_dependent = (np.sign(a0_all["difference_points"])
                        != np.sign(control_all["difference_points"])
                        and a0_all["difference_points"] != 0)
    if not exposed["improves"]:
        h4 = ("weakened", "SFT does not improve fixed-input output prediction even on exposed "
              "training functions: weaken distribution shift; strengthen objective/capacity")
    elif not unexposed["improves"]:
        h4 = ("memorisation_supported", "SFT improves exposed but not lineage-disjoint "
              "unexposed functions: memorisation/exposure failure")
    else:
        h4 = ("open", "SFT improves both train cohorts; no adequate external cohort exists, so "
              "distribution shift cannot be supported or rejected")
    base_levels = level_contrasts["base"]
    a3_helps = any(level_contrasts[a]["A3_minus_A0"]["improves"]
                   and level_contrasts[a]["A3_minus_A0"]["difference_points"] >= 5 for a in ARMS)
    a4_helps = any(level_contrasts[a]["A4_minus_A0"]["improves"] for a in ARMS)
    if a3_helps:
        h1 = "non-revealing missing context matters (A3 raises accuracy by >= 5 points)"
    elif a4_helps:
        h1 = ("A0 is under-specified for value prediction: only the fixed implementation (A4) "
              "raises accuracy")
    else:
        h1 = "neither A3 nor A4 improves over A0: capacity/reasoning more likely limiting"
    a4_base = descriptive["all"]["base/A4"]["accuracy_per_requested"]
    receipt = {
        "schema_version": "oneiros_sft_root_cause_phase3a_result_v1",
        "phase": "3A", "command": "python scripts/analyze_fixed_input_probe.py",
        "design_receipt": {"path": DESIGN, "sha256": sha(ROOT / DESIGN)},
        "generation_completions": completions,
        "scored_ledger": {"path": f"{GEN}/scored.jsonl",
                          "sha256": hashlib.sha256(ledger).hexdigest(), "rows": len(rows)},
        "descriptive": descriptive,
        "sft_minus_base": sft_vs_base,
        "level_contrasts": level_contrasts,
        "exposure_interaction_A0": inter,
        "family_sensitivity_A0": sensitivity,
        "decisions": {
            "schema_dependent_A0": bool(schema_dependent),
            "H4": {"verdict": h4[0], "why": h4[1],
                   "exposed_A0": exposed, "unexposed_A0": unexposed},
            "H1_H3": {"verdict": h1,
                      "base_A4_accuracy": a4_base,
                      "execution_reasoning_limited_even_with_fixed_code": bool(
                          a4_base is not None and a4_base < 0.5)},
        },
        "duration_seconds": round(time.time() - started, 1),
    }
    publish_file_atomically(ROOT / RECEIPT, (json.dumps(receipt, indent=1) + "\n")
                            .encode("utf-8"))
    for name in ("all", "exposed", "unexposed", "exposed_upstream_sft_target", "exposed_novel",
                 "unexposed_upstream", "unexposed_novel"):
        line = "  ".join(f"{lv}: {sft_vs_base[name][lv]['cells']['base']['accuracy']:.3f}->"
                         f"{sft_vs_base[name][lv]['cells']['sft']['accuracy']:.3f} "
                         f"{sft_vs_base[name][lv]['ci95_points']}" for lv in ALL_LEVELS)
        print(f"{name:30s} {line}")
    print(json.dumps(receipt["decisions"], indent=1, default=str)[:3000])
    print(json.dumps(level_contrasts, indent=1)[:2000])
    print(json.dumps(inter))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
