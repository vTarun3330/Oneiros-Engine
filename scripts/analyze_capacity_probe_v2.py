"""Phase 3C analysis v2 (CPU only; corrects v1, which is preserved unchanged).

Corrections relative to results/sft_root_cause_phase3c_result_receipt.json:

* v1 reported a bootstrap tail proportion as ``bootstrap_p_two_sided: 0.0``.
  v2 removes it and uses a paired semantic-group sign-flip randomisation test,
  p = (extreme + 1) / (resamples + 1), Holm-adjusted across the two frozen
  primary contrasts; intervals remain the predeclared clustered bootstrap;
* the interpretation is a model-scale association within the Qwen2.5-Coder
  family, not a causal parameter-count or adapter-capacity result;
* A3/A4 are labelled as diagnostic oracle-derived hint / fixed-implementation
  access.

Re-scores the same hash-checked generations with v1's scorer; cell accuracies
must reproduce v1 exactly.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness.atomic_publish import publish_file_atomically
from harness.fixed_input_probe import CONTROL_LEVEL, LEVELS
from harness.parallel_execution import map_jobs
from harness.probe_statistics import bootstrap_contrast, holm, sign_flip_test
from scripts.analyze_capacity_probe import DESIGN, DESIGN_3A, SOURCES, score
from scripts.analyze_fixed_input_probe_v2 import LEVEL_LABELS

V1 = "results/sft_root_cause_phase3c_result_receipt.json"
RECEIPT = "results/sft_root_cause_phase3c_result_receipt_v2.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
        completions[arm] = {"output": completion["output"], "sha256": completion["output_sha256"],
                            "model": completion["model"], "revision": completion["revision"]}
        generations[arm] = {json.loads(l)["key"]: json.loads(l)
                            for l in path.read_text(encoding="utf-8").splitlines()}
    rows = [r for chunk in map_jobs(score, [(f, records[f["record_id"]], generations)
                                            for f in cohort["functions"]]) for r in chunk]
    v1 = json.loads((ROOT / V1).read_text(encoding="utf-8"))
    for key, cell in v1["cells"].items():
        arm, level = key.split("/")
        sel = [r for r in rows if r["arm"] == arm and r["level"] == level]
        if round(sum(r["correct"] for r in sel) / len(sel), 4) != cell["accuracy"]:
            raise SystemExit(f"REFUSED: re-scoring does not reproduce v1 cell {key}")

    primary = {"C1_A4": (("base", "A4"), ("qwen7b_base", "A4")),
               "C2_A0": (("base", "A0"), ("qwen7b_base", "A0"))}
    results = {}
    for name, (a, b) in primary.items():
        results[name] = {**bootstrap_contrast(rows, a, b), "sign_flip": sign_flip_test(rows, a, b)}
    adjusted = holm({n: r["sign_flip"]["p_two_sided"] for n, r in results.items()})
    for name, r in results.items():
        r["holm_p"] = adjusted[name]
        r["holm_report"] = (f"Holm p <= {adjusted[name]:.5f} (bound from the smallest attainable "
                            "finite-resample p)" if r["sign_flip"]["extreme"] == 0
                            else f"Holm p = {adjusted[name]:.5f}")
        low = r["ci95_points"][0]
        r["region"] = ("clears_5_points" if low >= 5 else "positive" if low > 0
                       else "not_improved")
    secondary = {
        "7b_minus_arm_a_A0": bootstrap_contrast(rows, ("arm_a_431", "A0"), ("qwen7b_base", "A0")),
        "7b_minus_arm_a_A4": bootstrap_contrast(rows, ("arm_a_431", "A4"), ("qwen7b_base", "A4")),
        "7b_A4_minus_A0": bootstrap_contrast(rows, ("qwen7b_base", "A0"), ("qwen7b_base", "A4")),
        "7b_A3_minus_A0": bootstrap_contrast(rows, ("qwen7b_base", "A0"), ("qwen7b_base", "A3")),
        "7b_minus_1_5b_answer_schema": bootstrap_contrast(
            rows, ("base", CONTROL_LEVEL), ("qwen7b_base", CONTROL_LEVEL)),
    }
    receipt = {
        "schema_version": "oneiros_sft_root_cause_phase3c_result_v2",
        "phase": "3C", "version": 2, "command": "python scripts/analyze_capacity_probe_v2.py",
        "supersedes": {"path": V1, "sha256": sha(ROOT / V1), "modified": False},
        "model_calls": 0, "gpu_used": False,
        "design_receipt": {"path": DESIGN, "sha256": sha(ROOT / DESIGN)},
        "generation_artifacts": completions,
        "rescoring_reproduces_v1": True,
        "gate": v1["gate"],
        "level_labels": LEVEL_LABELS,
        "inference": {"interval": "paired cluster bootstrap over semantic group_id, 10,000 "
                                  "resamples, seed 20260928",
                      "test": "paired semantic-group sign-flip, 10,000 resamples, "
                              "p = (extreme + 1) / (resamples + 1)",
                      "multiplicity": "Holm across C1 and C2"},
        "primary": results,
        "secondary": secondary,
        "interpretation": {
            "H3": {"status": "strengthened", "qualifier": "model_scale_association",
                   "statement": ("Within the Qwen2.5-Coder family, the 7B checkpoint is "
                                 "substantially better than the 1.5B checkpoint on fixed-input "
                                 "output prediction."),
                   "not_established": ("that parameter count or adapter capacity is the causal "
                                       "ceiling: the pretrained checkpoints differ in scale and "
                                       "learned representation together; adapter rank and full "
                                       "fine-tuning remain untested")},
            "A4_at_7b": ("access to the correct implementation improves fixed-input prediction, "
                         "especially at 7B; this does not independently establish "
                         "natural-language specification insufficiency"),
            "A3_at_7b": "this particular type/length hint did not measurably improve accuracy",
        },
    }
    publish_file_atomically(ROOT / RECEIPT, (json.dumps(receipt, indent=1) + "\n")
                            .encode("utf-8"))
    for name, r in results.items():
        print(name, r["difference_points"], r["ci95_points"], r["sign_flip"]["report"],
              r["holm_report"], r["region"])
    for name, r in secondary.items():
        print(name, r["difference_points"], r["ci95_points"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
