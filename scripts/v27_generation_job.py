"""v2.7 Phase 8 input: the generation job and cohort manifest for the repository-native model
evaluation (CPU only; the frozen base TOKENIZER is used to count prompt tokens, no model weights
are loaded).

Cohort = the 29 NATIVE-EXECUTABLE targets of the frozen 34-target confirmation panel (native
target reach on both revisions AND the model-free sandbox control), taken from the committed
successor common-subset receipt. It is independent of the Atheris adapter. The five policy
exclusions stay in the panel and are recorded here, outside the generation cohort.

Each prompt is rebuilt by the real v2.5 job builder (harness/native_generated_test_job_v25:
prompt builder, permitted view, leakage scan against the verifier-only material, exact chat
token fit) and must reproduce the frozen panel's model-visible prompt hash exactly.

    python scripts/v27_generation_job.py --job results/<job>.json --manifest results/<m>.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness import native_generated_test_job_v25 as jobs  # noqa: E402
from scripts.native_generated_tests_generate import CONTRACT  # noqa: E402
from scripts.native_generation_io import cohort_fields  # noqa: E402

D = "results/sft_root_cause/v27_confirmation"
PANEL = "results/sft_root_cause_v27_confirmation_panel_r4.json"
SUBSET = "results/sft_root_cause_v27_common_subset_r4_v2.json"
VISIBLE = f"{D}/r4_panel/model_visible_bundle.json"
PREP = f"{D}/r4_prep/records.jsonl"
GEN_PREP = f"{D}/r4_panel/generation_prep_records.jsonl"


def sha(rel) -> str:
    return hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()


def refuse(cond: bool, msg: str) -> None:
    if cond:
        raise SystemExit("REFUSED: " + msg)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--job", required=True)
    ap.add_argument("--manifest", required=True)
    args = ap.parse_args(argv)
    for rel in (args.job, args.manifest):
        refuse((ROOT / rel).exists(), f"{rel} exists (never overwritten)")
    panel = json.loads((ROOT / PANEL).read_text(encoding="utf-8"))
    subset = json.loads((ROOT / SUBSET).read_text(encoding="utf-8"))
    refuse(subset["inputs_sha256"][PANEL] != sha(PANEL), "subset receipt binds another panel")
    cohort = subset["sets"]["native_executable"]
    refuse(len(cohort) != 29, f"expected 29 native-executable targets, found {len(cohort)}")
    refuse(hashlib.sha256("\n".join(sorted(cohort)).encode()).hexdigest()
           != subset["native_executable_ids_sha256"], "cohort ID hash mismatch")
    panel_targets = {t["target_id"]: t for t in panel["targets"]}
    refuse(not set(cohort) <= set(panel_targets), "cohort is not a subset of the frozen panel")
    visible = json.loads((ROOT / VISIBLE).read_text(encoding="utf-8"))
    refuse(sha(VISIBLE) != panel["model_visible_bundle"]["sha256"], "visible bundle changed")
    prep_lines = {json.loads(l)["key"]: l for l in
                  (ROOT / PREP).read_bytes().decode("utf-8").splitlines() if l.strip()}
    count = jobs.chat_token_counter(CONTRACT["base_model"], CONTRACT["base_revision"])
    targets = []
    for key in sorted(cohort):
        rec = json.loads(prep_lines[key])
        refuse(rec["category"] != "requalified", f"{key} not requalified")
        bv = json.loads((ROOT / f"{D}/r4_prep/buggy_view/{rec['tag']}.json")
                        .read_text(encoding="utf-8"))
        vf = json.loads((ROOT / f"{D}/r4_prep/verifier/{rec['tag']}.json")
                        .read_text(encoding="utf-8"))
        targets.append({"dto": bv["dto"], "buggy_source": bv["buggy_source"], "verifier": vf})
    job_file = jobs.build_job_file(targets, count, ROOT,
                                   purpose="v2.7 repository-native confirmation evaluation "
                                           "(29 native-executable targets of the frozen panel)")
    job = job_file[jobs.CONDITION]
    refuse(bool(job["refused"]), f"job builder refused targets: {job['refused']}")
    for item in job["items"]:
        refuse(item["prompt_sha256"] != visible[item["target_key"]]["prompt_sha256"],
               f"prompt for {item['target_key']} differs from the frozen visible bundle")
    jobs.validate_job_file(job_file, jobs.CONDITION, ROOT)
    job_bytes = (json.dumps(job_file, indent=1, sort_keys=True) + "\n").encode("utf-8")
    (ROOT / args.job).write_bytes(job_bytes)
    # requalification records: byte-identical lines of exactly the cohort (resolve_prep rule)
    (ROOT / GEN_PREP).write_bytes(("\n".join(prep_lines[k] for k in sorted(cohort)) + "\n")
                                  .encode("utf-8"))
    qualified = [{"key": k, "repository": panel_targets[k]["repository"]} for k in sorted(cohort)]
    fields = cohort_fields(qualified, args.job, job_bytes, jobs.CONDITION, {})
    excluded = sorted(set(panel_targets) - set(cohort))
    manifest = {
        "schema_version": "oneiros_v27_generation_manifest_v1",
        "nature": "repository-native confirmation evaluation (exploratory, underpowered)",
        "study_mode": "confirmation_exploratory",
        "role": "CONFIRMATION_ONLY", "training_prohibited": True,
        **fields,
        "targets": qualified,
        "requalification_records": {"path": GEN_PREP, "sha256": sha(GEN_PREP),
                                    "derived_from": {"path": PREP, "sha256": sha(PREP)}},
        "panel": {"path": PANEL, "sha256": sha(PANEL), "targets": len(panel_targets)},
        "subset_receipt": {"path": SUBSET, "sha256": sha(SUBSET)},
        "panel_policy_exclusions": [
            {"target_key": k, "reason": next(r["exclusion_reason"] for r in subset["targets"]
                                             if r["target_id"] == k)} for k in excluded],
        "atheris_independent": True,
    }
    (ROOT / args.manifest).write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n",
                                      encoding="utf-8")
    print(json.dumps({"items": len(job["items"]), "refused": len(job["refused"]),
                      "job_sha256": job["job_sha256"], "job_file_sha256": sha(args.job),
                      "manifest_sha256": sha(args.manifest), "expected": fields["expected"],
                      "prompt_tokens_max": max(i["prompt_tokens"] for i in job["items"]),
                      "excluded": excluded}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
