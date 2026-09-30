"""Amendment v2.3 successor job (v4) and manifest (v6) with an explicit generation cohort.

Prompt bytes are REUSED from the frozen v2.2 job v3 only after proving them unchanged:
  1. job v3's file hash equals the one bound by manifest v5 and preflight v2.2b;
  2. every v3 item's prompt hashes to its recorded prompt_sha256 and job_sha256 recomputes;
  3. an independent rebuild of every kept target's prompt from the buggy-side views (builder
     v2) reproduces every v3 item byte for byte;
  4. the only v3 refusal is the recorded pre-generation sequence overflow, whose token counts
     are taken from the hash-bound v2.2 prompt records.
Outputs are new versioned paths, written once; an identical rebuild is a verified reproduction.

    python scripts/native_rehearsal_rebuild_v23.py
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

PROTOCOLS = ("docs/SFT_ROOT_CAUSE_NATIVE_GENERATED_TEST_PROTOCOL_V2.md",
             "docs/SFT_ROOT_CAUSE_NATIVE_GENERATED_TEST_PROTOCOL_V2_1.md",
             "docs/SFT_ROOT_CAUSE_NATIVE_GENERATED_TEST_PROTOCOL_V2_2.md",
             "docs/SFT_ROOT_CAUSE_NATIVE_GENERATED_TEST_PROTOCOL_V2_3.md")
CONDITION = "primary_whole_module"
V22_JOB = "results/sft_root_cause_native_v22_rehearsal_job_v3.json"
V22_MANIFEST = "results/sft_root_cause_native_v22_rehearsal_manifest_v5.json"
V22_PROMPTS = "results/sft_root_cause_native_v22_prompt_records.json"
V22_PREFLIGHT = "results/sft_root_cause_native_generated_tests_preflight_v2_2b.json"
JOB = "results/sft_root_cause_native_v23_rehearsal_job_v4.json"
MANIFEST = "results/sft_root_cause_native_v23_rehearsal_manifest_v6.json"
GENERATIONS = "results/sft_root_cause/native_v23_generations"


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
    v5 = load(V22_MANIFEST)
    preflight = load(V22_PREFLIGHT)
    v3_bytes = (ROOT / V22_JOB).read_bytes()
    if sha256_bytes(v3_bytes) != v5["job"]["sha256"] or \
            sha256_bytes(v3_bytes) != preflight["job"]["file_sha256"]:
        problems.append("job v3 differs from manifest v5 / preflight v2.2b")
    if sha(V22_PROMPTS) != v5["prompt_records"]["sha256"]:
        problems.append("prompt records differ from manifest v5")
    v3 = json.loads(v3_bytes.decode("utf-8"))
    job = v3[CONDITION]
    if contract_sha({"items": job["items"]}) != job["job_sha256"]:
        problems.append("job v3 job_sha256 does not recompute")
    for item in job["items"]:
        if sha256_bytes(item["prompt"].encode("utf-8")) != item["prompt_sha256"]:
            problems.append(f"prompt hash {item['target_key']}")
    rebuilt = {}
    for key in v5["kept_targets"]:
        view = json.loads((ROOT / LOCAL / "buggy_view" / f"{tag_of(key)}.json")
                          .read_text(encoding="utf-8"))
        rebuilt[key] = build_prompt(view["dto"], view["buggy_source"])
    for item in job["items"]:
        again = rebuilt.get(item["target_key"])
        if again is None or again["prompt"] != item["prompt"] or \
                again["prompt_sha256"] != item["prompt_sha256"]:
            problems.append(f"prompt not reproduced {item['target_key']}")
    records = {r["target_key"]: r for r in load(V22_PROMPTS)["rows"]}
    refused = {r["target_key"]: r["reasons"] for r in job["refused"]}
    if list(refused.values()) != [["sequence_overflow"]] or len(refused) != 1:
        problems.append(f"unexpected v3 refusals {refused}")
    if problems:
        raise SystemExit(f"REFUSED: v2.2 job cannot be reused unchanged: {problems[:5]}")
    detail = {k: {"prompt_tokens": records[k]["v22_prompt_tokens"],
                  "target_source_tokens": records[k]["target_source_tokens"],
                  "prompt_token_limit": load(V22_PROMPTS)["prompt_token_limit"],
                  "truncated": False, "recovered": False,
                  "note": "pre-generation sequence-overflow exclusion; not a model failure"}
              for k in refused}
    job_file = {"schema_version": "oneiros_native_v23_generation_job_v4",
                "protocols": list(PROTOCOLS),
                "reused_from": {"path": V22_JOB, "file_sha256": sha256_bytes(v3_bytes),
                                "proof": "every item and prompt hash unchanged; prompts "
                                         "independently rebuilt byte for byte"},
                **{k: v3[k] for k in ("builder_version", "builder_sha256", "scanner_version",
                                      "scanner_sha256")},
                CONDITION: job}
    job_bytes = (json.dumps(job_file, indent=1, sort_keys=True) + "\n").encode("utf-8")
    status = {JOB: publish_once(JOB, job_file)}
    fields = cohort_fields(v5["targets"], JOB, job_bytes, CONDITION, detail)
    manifest = {
        "schema_version": "oneiros_native_v23_rehearsal_manifest_v6",
        "nature": "ENGINEERING DRESS REHEARSAL ONLY; not confirmation, generalisation or "
                  "model-selection evidence",
        "protocols": {p: sha(p) for p in PROTOCOLS},
        "supersedes": {"manifest": V22_MANIFEST, "manifest_sha256": sha(V22_MANIFEST),
                       "preflight": V22_PREFLIGHT, "preflight_sha256": sha(V22_PREFLIGHT)},
        **{k: v5[k] for k in ("source_manifest", "requalification_records", "isolation",
                              "prompt_records", "per_target", "targets", "kept_targets",
                              "repositories", "rehearsal_rule", "builder_version",
                              "builder_sha256", "scanner_version", "scanner_sha256")},
        **fields,
        "generation_outputs": {"base": f"{GENERATIONS}/base", "sft": f"{GENERATIONS}/sft"},
        "counts": {"qualified": len(fields["qualified_targets"]),
                   "generation": len(fields["generation_targets"]),
                   "pre_generation_excluded": len(fields["pre_generation_exclusions"])},
        "admitted_to_training": False, "gpu_used": False,
    }
    evidence = ProtectedAccessMonitor.evidence(mark)
    manifest["protected_access_audit"] = {k: v for k, v in evidence.items() if k != "opens_checked"}
    status[MANIFEST] = publish_once(MANIFEST, manifest)
    print(json.dumps({"status": status, "counts": manifest["counts"],
                      "expected": manifest["expected"], "job_sha256": fields["job"]["job_sha256"],
                      "excluded": fields["pre_generation_exclusions"],
                      "protected_paths_opened": evidence["protected_paths_opened"]}, indent=1))
    ok = manifest["counts"] == {"qualified": 24, "generation": 23, "pre_generation_excluded": 1} \
        and manifest["expected"]["candidates_total"] == 1104
    return 0 if ok and not evidence["protected_paths_opened"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
