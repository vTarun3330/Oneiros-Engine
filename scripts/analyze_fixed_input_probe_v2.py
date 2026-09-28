"""Phase 3A analysis v2 (CPU only; corrects v1, which is preserved unchanged).

Corrections relative to results/sft_root_cause_phase3a_result_receipt.json:

* the frozen schema-control rule is applied to the exact strata H4 reads
  (exposed and unexposed), not only to the pooled stratum;
* for every stratum: SFT-base under the prefilled assertion schema, SFT-base
  under the ANSWER schema, and their difference-in-differences with a
  semantic-group-clustered interval;
* A3 is labelled what it is: a diagnostic, oracle-derived partial behavioural
  hint (correct type and, where sized, length); A4 is access to the full fixed
  implementation; neither is described as specification sufficiency.

Scoring reuses v1's scorer on the same hash-checked generations; the v2 cell
accuracies must reproduce v1's exactly or the run is refused.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness.atomic_publish import publish_file_atomically
from harness.fixed_input_probe import CONTROL_LEVEL
from harness.parallel_execution import map_jobs
from harness.probe_statistics import (
    H4_STRATA, bootstrap_contrast, bootstrap_interaction, decide_h4, schema_dependent,
)
from scripts.analyze_fixed_input_probe import load_rows, score_function

V1 = "results/sft_root_cause_phase3a_result_receipt.json"
RECEIPT = "results/sft_root_cause_phase3a_result_receipt_v2.json"
BASE, SFT = "base", "arm_a_431"

STRATA = {
    "all": lambda r: True,
    "exposed": lambda r: r["cohort"] == "exposed",
    "unexposed": lambda r: r["cohort"] == "unexposed",
    "exposed_upstream_sft_target": lambda r: r["cohort"] == "exposed"
    and r["input_kind"] == "upstream",
    "exposed_novel": lambda r: r["cohort"] == "exposed" and r["input_kind"] == "novel",
    "unexposed_upstream": lambda r: r["cohort"] == "unexposed" and r["input_kind"] == "upstream",
    "unexposed_novel": lambda r: r["cohort"] == "unexposed" and r["input_kind"] == "novel",
}
assert set(H4_STRATA) <= set(STRATA)

LEVEL_LABELS = {
    "A0": "production information (buggy code under test and record specification)",
    "A3": ("DIAGNOSTIC, oracle-derived partial behavioural hint: the correct result's type "
           "and, where sized, its correct length; does not expose the complete value; "
           "non-deployable; prohibited from training and confirmation"),
    "A4": ("DIAGNOSTIC: access to the complete fixed implementation instead of the buggy one; "
           "non-deployable; prohibited from training and confirmation"),
    CONTROL_LEVEL: "A0 information with the one-line 'ANSWER: <literal>' schema (no prefill)",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    started = time.time()
    design, cohort, records, generations, completions = load_rows()
    jobs = [{"function": f, "record": records[f["record_id"]], "generations": generations}
            for f in cohort["functions"]]
    rows = [row for chunk in map_jobs(score_function, jobs) for row in chunk]

    v1 = json.loads((ROOT / V1).read_text(encoding="utf-8"))
    mismatches = []
    for name, cells in v1["descriptive"].items():
        if name not in STRATA:
            continue
        for key, stats in cells.items():
            arm, level = key.split("/")
            sel = [r for r in rows if r["arm"] == arm and r["level"] == level
                   and STRATA[name](r)]
            accuracy = round(sum(r["correct"] for r in sel) / len(sel), 4)
            if accuracy != stats["accuracy_per_requested"]:
                mismatches.append((name, key, accuracy, stats["accuracy_per_requested"]))
    if mismatches:
        raise SystemExit(f"REFUSED: re-scoring does not reproduce v1: {mismatches[:5]}")

    strata = {}
    for name, select in STRATA.items():
        prefill = bootstrap_contrast(rows, (BASE, "A0"), (SFT, "A0"), select)
        answer = bootstrap_contrast(rows, (BASE, CONTROL_LEVEL), (SFT, CONTROL_LEVEL), select)
        interaction = bootstrap_interaction(
            rows, ((BASE, "A0"), (SFT, "A0")), ((BASE, CONTROL_LEVEL), (SFT, CONTROL_LEVEL)),
            select)
        strata[name] = {
            "prefill": prefill, "answer": answer,
            "schema_interaction_answer_minus_prefill": interaction,
            "schema_dependent": schema_dependent(prefill["difference_points"],
                                                 answer["difference_points"])}
    h4 = decide_h4(strata)
    levels = {arm: {f"{level}_minus_A0": bootstrap_contrast(rows, (arm, "A0"), (arm, level))
                    for level in ("A3", "A4")} for arm in (BASE, SFT)}
    sft_minus_base_by_level = {level: bootstrap_contrast(rows, (BASE, level), (SFT, level))
                               for level in ("A0", "A3", "A4", CONTROL_LEVEL)}
    receipt = {
        "schema_version": "oneiros_sft_root_cause_phase3a_result_v2",
        "phase": "3A", "version": 2,
        "command": "python scripts/analyze_fixed_input_probe_v2.py",
        "supersedes": {"path": V1, "sha256": sha(ROOT / V1), "modified": False},
        "model_calls": 0, "gpu_used": False,
        "design_receipt": {"path": "results/sft_root_cause_phase3a_design_receipt.json",
                           "sha256": sha(ROOT / "results/sft_root_cause_phase3a_design_receipt.json")},
        "generation_completions": completions,
        "rescoring_reproduces_v1": True,
        "level_labels": LEVEL_LABELS,
        "inference": {"interval": "paired cluster bootstrap over semantic group_id, 10,000 "
                                  "resamples, seed 20260928", "p_values": "none reported"},
        "strata": strata,
        "sft_minus_base_by_level": sft_minus_base_by_level,
        "level_contrasts_within_arm": levels,
        "decisions": {
            "H4": h4,
            "schema_rule": ("frozen: an SFT effect that changes sign between the prefilled "
                            "assertion schema and the ANSWER schema is schema-dependent and not "
                            "interpreted; v2 applies it to the exact strata the H4 rules read"),
            "A3": "this particular type/length hint did not measurably improve accuracy",
            "A4": ("access to the correct implementation improves fixed-input prediction; it "
                   "does not independently establish natural-language specification "
                   "insufficiency"),
        },
        "duration_seconds": round(time.time() - started, 1),
    }
    publish_file_atomically(ROOT / RECEIPT, (json.dumps(receipt, indent=1) + "\n")
                            .encode("utf-8"))
    for name, s in strata.items():
        i = s["schema_interaction_answer_minus_prefill"]
        print(f"{name:30s} prefill {s['prefill']['difference_points']:+7.3f} "
              f"{s['prefill']['ci95_points']}  answer {s['answer']['difference_points']:+7.3f} "
              f"{s['answer']['ci95_points']}  DiD {i['interaction_points']:+7.3f} "
              f"{i['ci95_points']}  flip={s['schema_dependent']}")
    print(json.dumps(h4, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
