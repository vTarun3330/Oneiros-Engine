"""Freeze the post-hoc exploratory mechanical-repair diagnostic BEFORE any repaired candidate is
executed: the transform rules and source hash, the exact v2.4 inputs, and outcome-free
applicability counts (how many retained candidates each condition changes or refuses - computed
from module text and prompts only). Written once.

    python scripts/repair_ceiling_definition.py
"""
from __future__ import annotations

import collections
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import repair_ceiling_transforms as rt
from scripts.native_rehearsal_rebuild_v22 import publish_once

OUT = "results/sft_root_cause_native_v24_repair_ceiling_definition.json"
COND = "primary_whole_module"
JOB = "results/sft_root_cause_native_v24_rehearsal_job_v5.json"
GEN = "results/sft_root_cause/native_v24_generations"
INPUTS = {f"{GEN}/base/generations_{COND}_base.jsonl":
          "e71367acefdbad5a422aba3790376454f04831d80c4b5dec6ac1429e31eb9f44",
          f"{GEN}/sft/generations_{COND}_sft.jsonl":
          "5af755e9f947175a1996037865e1c96023faf36e78cd87678be066367d6e59d9",
          "results/sft_root_cause/native_v24_execution/results_primary_whole_module.jsonl":
          "e40c912d23eccb3fb8a6b0cd06c11e27a48052582f4e60d1f1507ce7471a0d55",
          JOB: "53948495758839c55fa911fd3a59f57d9cc4541954fa5c9c83c7a0ff3857f817"}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    for rel, want in INPUTS.items():
        if sha(ROOT / rel) != want:
            raise SystemExit(f"REFUSED: {rel} differs from the completed v2.4 artifact")
    prompts = {i["target_key"]: i["prompt"] for i in
               json.loads((ROOT / JOB).read_text(encoding="utf-8"))[COND]["items"]}
    applicability = {}
    for arm in ("base", "sft"):
        rows = [json.loads(l) for l in (ROOT / f"{GEN}/{arm}/generations_{COND}_{arm}.jsonl")
                .read_text(encoding="utf-8").splitlines()]
        for c in rt.CONDITIONS:
            results = [rt.apply(c, cand["module"], prompts[r["target_key"]])
                       for r in rows for cand in r["candidates"]]
            applicability[f"{arm}:{c}"] = {
                "candidates": len(results), "changed": sum(r["changed"] for r in results),
                "applied": dict(collections.Counter(s for r in results for s in r["applied"])),
                "not_applied": rt.refusal_counts(results)}
    receipt = {"schema_version": "oneiros_repair_ceiling_definition_v1",
               "study": rt.STUDY,
               "label": "post-hoc exploratory diagnostic; NOT a primary metric; cannot "
                        "establish confirmation-level improvement",
               "transforms": rt.definition(),
               "transforms_sha256": sha(ROOT / "scripts/repair_ceiling_transforms.py"),
               "executor": {"script": "scripts/repair_ceiling_execute_wsl.py",
                            "sha256": sha(ROOT / "scripts/repair_ceiling_execute_wsl.py"),
                            "frozen_v24_executor_sha256":
                                sha(ROOT / "scripts/native_generated_tests_execute_wsl.py")},
               "inputs": INPUTS,
               "outcome_free_applicability": applicability,
               "primary_diagnostic_outcome": "fixed-valid semantic execution, then confirmed "
                                             "kills; parse improvement alone is not success",
               "decision_rules": [
                   "R4 substantially restores fixed validity AND semantic kills -> mechanical "
                   "output-contract mismatch is a major bottleneck",
                   "R4 restores fixed validity but not kills -> mechanical validity plus "
                   "semantic oracle weakness",
                   "training targets mostly assertion fragments / no imports -> "
                   "training/evaluation contract mismatch",
                   "training targets aligned but SFT omits imports -> forgetting / prompt "
                   "incompatibility / truncation",
                   "repair barely changes fixed validity -> prioritise context/input/oracle"]}
    status = publish_once(OUT, receipt)
    print(json.dumps({"status": status, "applicability": {k: v["changed"] for k, v in
                                                          applicability.items()}}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
