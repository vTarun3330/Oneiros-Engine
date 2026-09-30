"""Amendment v2.4 successor job (v5) and manifest (v7).

Job items and prompt bytes are REUSED from the frozen v2.3 job v4 only after proving them
unchanged: the v4 file and manifest v6 hashes equal those bound by preflight v2.3; job_sha256
recomputes; every item's prompt hashes to its record; an independent rebuild of every prompt
from the buggy-side views (builder v2) reproduces every item byte for byte. Manifest v7 adds
the engineering-only study mode, the inherited coverage gate, the exact preparation binding,
the sequential/exclusive launch contract and the v2.4 output directories. Outputs are new
versioned paths written once; an identical rebuild is a verified reproduction.

    python scripts/native_rehearsal_rebuild_v24.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.native_rehearsal_rebuild_v22 import publish_once
from scripts.native_rehearsal_revalidate import LOCAL, load, sha

PROTOCOLS = tuple(f"docs/SFT_ROOT_CAUSE_NATIVE_GENERATED_TEST_PROTOCOL_{v}.md"
                  for v in ("V2", "V2_1", "V2_2", "V2_3", "V2_4"))
CONDITION = "primary_whole_module"
V23_JOB = "results/sft_root_cause_native_v23_rehearsal_job_v4.json"
V23_MANIFEST = "results/sft_root_cause_native_v23_rehearsal_manifest_v6.json"
V23_PREFLIGHT = "results/sft_root_cause_native_generated_tests_preflight_v2_3.json"
JOB = "results/sft_root_cause_native_v24_rehearsal_job_v5.json"
MANIFEST = "results/sft_root_cause_native_v24_rehearsal_manifest_v7.json"
GENERATIONS = "results/sft_root_cause/native_v24_generations"
EXCLUSIVE_KEY = "native_v24_generation"
LAUNCH_CONTRACT = {
    "order": ["base", "sft"], "exclusive_key": EXCLUSIVE_KEY,
    "reservation": "atomic os.mkdir of runs/.exclusive/<sha256(key)> taken before any "
                   "inspection or run-directory creation; token-owned; renewed by the "
                   "supervisor; released by its owner only after the final status; stale "
                   "recovery only after every recorded process is proven dead",
    "sft_requires_verified_base": True, "allow_concurrent": False}
LIVE_VIEW_POLICY = {
    "algorithm": "sha256(json.dumps({relative_posix_path: sha256(file)}, sort_keys=True))",
    "symlinks": "refused", "missing_or_unreadable": "refused",
    "executor": "both revisions of all generation targets, before any write",
    "atheris": "both revisions of all qualified targets, before the contract is written"}
ENGINEERING_GATE_REQUIREMENTS = [
    "coverage_gate_passed", "repository_gate_passed", "stage_receipts_gate_passed",
    "canaries_gate_passed", "artifact_integrity_gate_passed",
    "unexplained_failures_gate_passed"]
ANALYSIS_CONTRACT = {
    "study_mode": "engineering_dress_rehearsal",
    "required_inputs": ["manifest", "job", "prep", "preflight_v2_4", "execution_contract",
                        "execution_results", "base_and_sft_generations",
                        "atheris_contract_and_results_together_when_compared"],
    "engineering_gate": "all of engineering_gate_requirements; otherwise every arm "
                        "comparison is suppressed"}


def main(argv=None) -> int:
    argparse.ArgumentParser(description=__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter).parse_args(argv)
    from harness.acquisition_receipt import ProtectedAccessMonitor
    from harness.native_generated_test_prompt import build_prompt
    from scripts.native_generation_io import cohort_fields, contract_sha, sha256_bytes
    from scripts.native_rehearsal_prepare_wsl import tag_of
    ProtectedAccessMonitor.install(ROOT)
    mark = ProtectedAccessMonitor.mark()
    problems = []
    v6, pre = load(V23_MANIFEST), load(V23_PREFLIGHT)
    v4_bytes = (ROOT / V23_JOB).read_bytes()
    if pre["inputs"].get(V23_JOB) != sha256_bytes(v4_bytes) or \
            pre["inputs"].get(V23_MANIFEST) != sha(V23_MANIFEST):
        problems.append("job v4 / manifest v6 differ from preflight v2.3")
    if v6["job"]["file_sha256"] != sha256_bytes(v4_bytes):
        problems.append("job v4 differs from manifest v6")
    v4 = json.loads(v4_bytes.decode("utf-8"))
    job = v4[CONDITION]
    if contract_sha({"items": job["items"]}) != job["job_sha256"]:
        problems.append("job v4 job_sha256 does not recompute")
    for item in job["items"]:
        if sha256_bytes(item["prompt"].encode("utf-8")) != item["prompt_sha256"]:
            problems.append(f"prompt hash {item['target_key']}")
        view = json.loads((ROOT / LOCAL / "buggy_view" / f"{tag_of(item['target_key'])}.json")
                          .read_text(encoding="utf-8"))
        again = build_prompt(view["dto"], view["buggy_source"])
        if again["prompt"] != item["prompt"] or again["prompt_sha256"] != item["prompt_sha256"]:
            problems.append(f"prompt not reproduced {item['target_key']}")
    if problems:
        raise SystemExit(f"REFUSED: v2.3 job cannot be reused unchanged: {problems[:5]}")
    job_file = {"schema_version": "oneiros_native_v24_generation_job_v5",
                "protocols": list(PROTOCOLS),
                "reused_from": {"path": V23_JOB, "file_sha256": sha256_bytes(v4_bytes),
                                "proof": "every item and prompt hash unchanged; prompts "
                                         "independently rebuilt byte for byte"},
                **{k: v4[k] for k in ("builder_version", "builder_sha256", "scanner_version",
                                      "scanner_sha256")},
                CONDITION: job}
    job_bytes = (json.dumps(job_file, indent=1, sort_keys=True) + "\n").encode("utf-8")
    status = {JOB: publish_once(JOB, job_file)}
    detail = {e["target_key"]: {k: v for k, v in e.items()
                                if k not in ("target_key", "stage", "reasons")}
              for e in v6["pre_generation_exclusions"]}
    fields = cohort_fields(v6["targets"], JOB, job_bytes, CONDITION, detail)
    qualified = len(fields["qualified_targets"])
    manifest = {
        "schema_version": "oneiros_native_v24_rehearsal_manifest_v7",
        "nature": v6["nature"], "study_mode": "engineering_dress_rehearsal",
        "claims": {"root_cause": False, "generalization": False, "sft_benefit": False,
                   "atheris_superiority": False, "model_selection_or_promotion": False,
                   "confirmation_statistics": False, "training_use": False},
        "protocols": {p: sha(p) for p in PROTOCOLS},
        "supersedes": {"manifest": V23_MANIFEST, "manifest_sha256": sha(V23_MANIFEST),
                       "preflight": V23_PREFLIGHT, "preflight_sha256": sha(V23_PREFLIGHT)},
        **{k: v6[k] for k in ("source_manifest", "requalification_records", "isolation",
                              "prompt_records", "per_target", "targets", "kept_targets",
                              "repositories", "rehearsal_rule", "builder_version",
                              "builder_sha256", "scanner_version", "scanner_sha256")},
        **fields,
        "coverage_gate": {"min_fraction_of_qualified": 0.90, "min_repositories": 5,
                          "required_eligible": -(-9 * qualified // 10),
                          "rule": "generated AND infrastructure-eligible in both arms"},
        "generation_outputs": {"base": f"{GENERATIONS}/base", "sft": f"{GENERATIONS}/sft"},
        "launch_contract": LAUNCH_CONTRACT, "live_view_policy": LIVE_VIEW_POLICY,
        "engineering_gate_requirements": ENGINEERING_GATE_REQUIREMENTS,
        "analysis_contract": ANALYSIS_CONTRACT,
        "telemetry": {"schema": "oneiros_native_generation_telemetry_v2",
                      "generator_version": "oneiros_native_generated_tests_generate_v4"},
        "counts": {"qualified": qualified, "generation": len(fields["generation_targets"]),
                   "pre_generation_excluded": len(fields["pre_generation_exclusions"])},
        "admitted_to_training": False, "gpu_used": False,
    }
    evidence = ProtectedAccessMonitor.evidence(mark)
    manifest["protected_access_audit"] = {k: v for k, v in evidence.items() if k != "opens_checked"}
    status[MANIFEST] = publish_once(MANIFEST, manifest)
    print(json.dumps({"status": status, "counts": manifest["counts"],
                      "expected": manifest["expected"], "job_sha256": fields["job"]["job_sha256"],
                      "coverage_gate": manifest["coverage_gate"],
                      "protected_paths_opened": evidence["protected_paths_opened"]}, indent=1))
    ok = manifest["counts"] == {"qualified": 24, "generation": 23, "pre_generation_excluded": 1} \
        and manifest["expected"]["candidates_total"] == 1104 \
        and manifest["coverage_gate"]["required_eligible"] == 22
    return 0 if ok and not evidence["protected_paths_opened"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
