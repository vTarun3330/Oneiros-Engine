"""Phase 3A analysis v3 (CPU only; v1 and v2 receipts are preserved unchanged).

Repairs relative to v2:
* exact item pairing (``harness.probe_statistics.paired_cells``) in every contrast;
* route-exact H4 (``decide_h4``);
* reproduction is checked field by field against every reproducible v1 statistic and
  every v2 interval, and the rescored rows get a canonical keyed SHA-256; v1 kept no
  row-level hash, so the claim is "all retained v1 aggregate statistics reproduced";
* the design's declared multiplicity family ("Holm across the primary H4 and H1
  contrasts") is implemented with paired semantic-group sign-flip tests;
* corrected A3/A4 labels; H1 is not decided from A3/A4.

Retained generations only; no model call, no GPU.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness.atomic_publish import publish_file_atomically
from harness.fixed_input_probe import CONTROL_LEVEL, CORRECTED_LEVEL_LABELS
from harness.fixed_input_rescoring import (
    ALL_LEVELS, DESIGN_3A, load_inputs, row_identity, score_function, sha_file,
)
from harness.parallel_execution import map_jobs
from harness.probe_statistics import (
    H4_STRATA, RESAMPLES, SEED, bootstrap_contrast, bootstrap_interaction, decide_h4, holm,
    schema_dependent, sign_flip_test,
)

V1 = "results/sft_root_cause_phase3a_result_receipt.json"
V2 = "results/sft_root_cause_phase3a_result_receipt_v2.json"
RECEIPT = "results/sft_root_cause_phase3a_result_receipt_v3.json"
BASE, SFT = "base", "arm_a_431"
EXPECTED_ROWS = 240 * 2 * 2 * len(ALL_LEVELS)
SOURCES = ["scripts/analyze_fixed_input_probe_v3.py", "harness/probe_statistics.py",
           "harness/fixed_input_rescoring.py", "harness/fixed_input_probe.py",
           "harness/safe_execution.py", "engine/test_generation_prompt.py"]

STRATA = {
    "all": lambda r: True,
    "exposed": lambda r: r["cohort"] == "exposed",
    "unexposed": lambda r: r["cohort"] == "unexposed",
    "exposed_upstream_sft_target": lambda r: r["cohort"] == "exposed"
    and r["input_kind"] == "upstream",
    "exposed_novel": lambda r: r["cohort"] == "exposed" and r["input_kind"] == "novel",
    "unexposed_upstream": lambda r: r["cohort"] == "unexposed" and r["input_kind"] == "upstream",
    "unexposed_novel": lambda r: r["cohort"] == "unexposed" and r["input_kind"] == "novel",
    "tier_simple": lambda r: r["complexity_tier"] == "simple",
    "tier_moderate": lambda r: r["complexity_tier"] == "moderate",
}
assert set(H4_STRATA) <= set(STRATA)

#: The declared Holm family, operationalised: the H4 rule contrasts (SFT - base at A0
#: on exposed and unexposed) and the H1 rule contrasts (A3 - A0 and A4 - A0, each arm).
DECLARED_FAMILY = {
    "H4_exposed_A0_sft_minus_base": ((BASE, "A0"), (SFT, "A0"), "exposed"),
    "H4_unexposed_A0_sft_minus_base": ((BASE, "A0"), (SFT, "A0"), "unexposed"),
    "H1_base_A3_minus_A0": ((BASE, "A0"), (BASE, "A3"), "all"),
    "H1_base_A4_minus_A0": ((BASE, "A0"), (BASE, "A4"), "all"),
    "H1_arm_a_A3_minus_A0": ((SFT, "A0"), (SFT, "A3"), "all"),
    "H1_arm_a_A4_minus_A0": ((SFT, "A0"), (SFT, "A4"), "all"),
}


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def cell_stats(rows, select):
    out = {}
    for arm in (BASE, SFT):
        for level in ALL_LEVELS:
            sel = [r for r in rows if r["arm"] == arm and r["level"] == level and select(r)]
            answered = [r for r in sel if r["answer"]]
            correct = sum(r["correct"] for r in sel)
            out[f"{arm}/{level}"] = {
                "items": len(sel), "correct": correct,
                "accuracy_per_requested": round(correct / len(sel), 4),
                "answer_rate": round(len(answered) / len(sel), 4),
                "accuracy_per_answered": (round(sum(r["correct"] for r in answered)
                                                / len(answered), 4) if answered else None),
                "token_limit_hits": sum(r["hit_token_limit"] for r in sel),
                "nonanswers": len(sel) - len(answered),
                "extraction_methods": dict(Counter(r["method"] for r in sel))}
    return out


def reconcile(descriptive, v1, v2, strata):
    """Every reproducible v1 aggregate field and every v2 interval, compared exactly."""
    compared, mismatches = 0, []
    for name, cells in v1["descriptive"].items():
        for key, old in cells.items():
            new = descriptive[name][key]
            for field in ("items", "accuracy_per_requested", "answer_rate",
                          "accuracy_per_answered", "token_limit_hits", "nonanswers",
                          "extraction_methods"):
                compared += 1
                if new[field] != old[field]:
                    mismatches.append(("v1", name, key, field, old[field], new[field]))
            implied = old["accuracy_per_requested"] * old["items"]
            if math.isclose(implied, round(implied), abs_tol=0.02):
                compared += 1
                if round(implied) != new["correct"]:
                    mismatches.append(("v1", name, key, "correct", round(implied),
                                       new["correct"]))
    for name, block in v1["sft_minus_base"].items():
        for level, old in block.items():
            for arm_label, cell in (("base", old["cells"]["base"]), ("sft", old["cells"]["sft"])):
                arm = BASE if arm_label == "base" else SFT
                new = descriptive[name][f"{arm}/{level}"]
                compared += 2
                if (cell["correct"], cell["items"]) != (new["correct"], new["items"]):
                    mismatches.append(("v1_contrast_cells", name, level, arm_label,
                                       (cell["correct"], cell["items"]),
                                       (new["correct"], new["items"])))
    for name, old in v2["strata"].items():
        for part in ("prefill", "answer"):
            compared += 2
            if old[part]["difference_points"] != strata[name][part]["difference_points"]:
                mismatches.append(("v2", name, part, "difference_points",
                                   old[part]["difference_points"],
                                   strata[name][part]["difference_points"]))
            if old[part]["ci95_points"] != strata[name][part]["ci95_points"]:
                mismatches.append(("v2", name, part, "ci95_points", old[part]["ci95_points"],
                                   strata[name][part]["ci95_points"]))
        old_i = old["schema_interaction_answer_minus_prefill"]
        new_i = strata[name]["schema_interaction_answer_minus_prefill"]
        compared += 2
        for field in ("interaction_points", "ci95_points"):
            if old_i[field] != new_i[field]:
                mismatches.append(("v2", name, "interaction", field, old_i[field], new_i[field]))
    return {"fields_compared": compared, "mismatches": mismatches}


def main() -> int:
    started = time.time()
    inputs = load_inputs((BASE, SFT))
    jobs = [{"function": f, "record": inputs["records"][f["record_id"]],
             "generations": inputs["generations"]} for f in inputs["cohort"]["functions"]]
    rows = [row for chunk in map_jobs(score_function, jobs) for row in chunk]
    identity = row_identity(rows)
    if identity["rows"] != EXPECTED_ROWS or identity["duplicates"]:
        raise SystemExit(f"REFUSED: unexpected row identity {identity}")

    descriptive = {name: cell_stats(rows, sel) for name, sel in STRATA.items()}
    strata = {}
    for name, select in STRATA.items():
        prefill = bootstrap_contrast(rows, (BASE, "A0"), (SFT, "A0"), select)
        answer = bootstrap_contrast(rows, (BASE, CONTROL_LEVEL), (SFT, CONTROL_LEVEL), select)
        strata[name] = {
            "prefill": prefill, "answer": answer,
            "schema_interaction_answer_minus_prefill": bootstrap_interaction(
                rows, ((BASE, "A0"), (SFT, "A0")),
                ((BASE, CONTROL_LEVEL), (SFT, CONTROL_LEVEL)), select),
            "schema_dependent": schema_dependent(prefill["difference_points"],
                                                 answer["difference_points"])}
    v1 = json.loads((ROOT / V1).read_text(encoding="utf-8"))
    v2 = json.loads((ROOT / V2).read_text(encoding="utf-8"))
    reconciliation = reconcile(descriptive, v1, v2, strata)
    if reconciliation["mismatches"]:
        raise SystemExit(f"REFUSED: v3 does not reconcile: {reconciliation['mismatches'][:5]}")

    family = {}
    for name, (a, b, stratum) in DECLARED_FAMILY.items():
        family[name] = {**bootstrap_contrast(rows, a, b, STRATA[stratum]),
                        "sign_flip": sign_flip_test(rows, a, b, STRATA[stratum])}
    adjusted = holm({n: f["sign_flip"]["monte_carlo_p"] for n, f in family.items()})
    for name, f in family.items():
        f["holm_adjusted_monte_carlo_p"] = round(adjusted[name], 5)
    h4 = decide_h4(strata)
    within_arm = {arm: {f"{level}_minus_A0": bootstrap_contrast(rows, (arm, "A0"), (arm, level))
                        for level in ("A3", "A4")} for arm in (BASE, SFT)}

    design = inputs["design"]
    receipt = {
        "schema_version": "oneiros_sft_root_cause_phase3a_result_v3",
        "phase": "3A", "version": 3,
        "command": "python scripts/analyze_fixed_input_probe_v3.py",
        "branch": git("branch", "--show-current"), "starting_commit": git("rev-parse", "HEAD"),
        "model_calls": 0, "gpu_used": False, "training": False,
        "restricted_splits_accessed": "none (reads the frozen cohort, the train shard and "
                                      "retained Phase 3A generations only)",
        "provenance": {
            "supersedes": {V1: sha_file(V1), V2: sha_file(V2)},
            "v1_v2_modified": False,
            "design_receipt": {DESIGN_3A: sha_file(DESIGN_3A)},
            "cohort": {design["cohort"]["path"]: sha_file(design["cohort"]["path"])},
            "record_shard": {design["record_shard"]["path"]:
                             sha_file(design["record_shard"]["path"])},
            "generations": inputs["completions"],
            "sources": {p: sha_file(p) for p in SOURCES},
            "historical_analyzers_not_rerunnable": (
                "the v1/v2 analyzers build rows without item identity; the repaired pairing "
                "checks refuse them, which is the defect v3 fixes")},
        "rescoring": {**identity, "expected_rows": EXPECTED_ROWS,
                      "reproduction": "all retained v1 aggregate statistics reproduced "
                                      "(v1 kept no row-level hash, so exact row-level "
                                      "reproduction is not claimed); all v2 contrasts and "
                                      "intervals reproduced",
                      "reconciliation": reconciliation},
        "level_labels": CORRECTED_LEVEL_LABELS,
        "level_notes": {"A1": "not run", "A2": "not applicable (no repository context)"},
        "inference": {
            "intervals": (f"paired cluster bootstrap over semantic group_id on exactly paired "
                          f"items, {RESAMPLES} resamples, seed {SEED}; per-contrast 95% "
                          "percentile intervals, not multiplicity-adjusted"),
            "tests": (f"paired semantic-group sign-flip, {RESAMPLES} resamples, Monte Carlo "
                      "p = (extreme + 1) / (resamples + 1)"),
            "declared_multiplicity": design["inference"]["multiplicity"],
            "family_operationalisation": list(DECLARED_FAMILY),
            "note": ("the design names the family as 'the primary H4 and H1 contrasts' without "
                     "listing them; this operationalisation is stated here, before use in any "
                     "decision, and changes no decision because H4 is decided by the frozen "
                     "interval rules and H1 is not decided from A3/A4")},
        "descriptive": descriptive,
        "strata": strata,
        "declared_family_zero_effect_tests": family,
        "level_contrasts_within_arm": within_arm,
        "decisions": {
            "H4": h4,
            "H1": {"status": "open",
                   "why": ("A1 and A2 were not run; A3 shows only that this oracle-derived "
                           "type/length hint was insufficient; A4 reveals the complete fixed "
                           "implementation and is not an identifying manipulation of public "
                           "specification quality")},
            "A3": "this particular oracle-derived type/length hint did not measurably improve "
                  "accuracy",
            "A4": "access to the correct implementation improves fixed-input prediction",
        },
        "duration_seconds": round(time.time() - started, 1),
    }
    publish_file_atomically(ROOT / RECEIPT, (json.dumps(receipt, indent=1, sort_keys=True)
                                             + "\n").encode("utf-8"))
    for name in STRATA:
        s = strata[name]
        i = s["schema_interaction_answer_minus_prefill"]
        print(f"{name:28s} prefill {s['prefill']['difference_points']:+7.3f} "
              f"{s['prefill']['ci95_points']}  answer {s['answer']['difference_points']:+7.3f} "
              f"{s['answer']['ci95_points']}  DiD {i['interaction_points']:+7.3f} "
              f"{i['ci95_points']}  flip={s['schema_dependent']}")
    for name, f in family.items():
        print(f"{name:32s} {f['difference_points']:+7.3f} {f['ci95_points']} "
              f"holm={f['holm_adjusted_monte_carlo_p']}  {f['sign_flip']['report']}")
    print(json.dumps({"H4": h4, "rows": identity,
                      "fields_compared": reconciliation["fields_compared"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
