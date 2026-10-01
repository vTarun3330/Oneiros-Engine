"""v2.5 CPU program 2, Phase 14: the pilot preflight decision. A green v2.5 pilot preflight is
produced ONLY if the pilot universe has >= 8 repositories and >= 60 qualified target lineages, a
successful pilot could project to 150 verified repository positives within the 10 GPU-hour
ceiling, and every identity is frozen. Otherwise a precise refusal is written. No GPU
authorisation is created and nothing is launched.

    python scripts/v25_cpu2_preflight_decision.py --suite results/sft_root_cause/<suite>.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RECEIPT = "results/sft_root_cause_v25_cpu2_preflight_decision.json"
RECEIPTS = ("results/sft_root_cause_v25_cpu2_phase0_reverification.json",
            "results/sft_root_cause_v25_cpu2_archive_manifest.json",
            "results/sft_root_cause_v25_cpu2_accounting_erratum.json",
            "results/sft_root_cause_v25_sft_data_dryrun_r3.json",
            "results/sft_root_cause_v25_policy_recovery_audit.json",
            "results/sft_root_cause_v25_native_attempt_spec.json",
            "results/sft_root_cause_v25_repository_verification_r2.json",
            "results/sft_root_cause_v25_attainable_gate_decision.json",
            "results/sft_root_cause_v25_universe_partition.json",
            "results/sft_root_cause_v25_acquisition_projection.json",
            "results/sft_root_cause_v25_power_simulation.json",
            "results/sft_root_cause_v25_atheris_readiness.json",
            "results/sft_root_cause_v25_cpu2_archive_manifest_r2.json",
            "docs/SFT_ROOT_CAUSE_NATIVE_GENERATED_TEST_PROTOCOL_V2_5_ADDENDUM_2.md")


def sha(rel: str) -> str:
    return hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()


def main(argv=None) -> int:
    from scripts.native_rehearsal_rebuild_v22 import publish_once
    from scripts.native_generated_tests_generate import (GENERATOR_VERSION, adapter_sha256,
                                                         model_identity)
    from harness import native_launch_gate as gate
    from harness.native_generated_test_job_v25 import JOB_SCHEMA, builder_identity
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", required=True)
    args = parser.parse_args(argv)
    suite = json.loads((ROOT / args.suite).read_text(encoding="utf-8"))
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                          text=True).stdout.strip()
    decision = json.loads((ROOT / "results/sft_root_cause_v25_attainable_gate_decision.json")
                          .read_text(encoding="utf-8"))
    projection = json.loads((ROOT / "results/sft_root_cause_v25_acquisition_projection.json")
                            .read_text(encoding="utf-8"))
    repos, lineages = decision["qualified_repositories"], decision["qualified_target_lineages"]
    universe_ok = repos >= 8 and lineages >= 60
    blockers = []
    if not universe_ok:
        blockers.append(f"pilot universe has {repos} repositories with qualified environments "
                        f"(< 8 required); {lineages} qualified target lineages")
    blockers.append("150 verified repository positives are not reachable: 28 verified unique "
                    "tests from 3 repositories after exhausting the 13 existing train "
                    "repositories (shortfall 122 / 5 / 32)")
    blockers.append("bounded train-only acquisition NOT RUN: no GitHub credential configured; "
                    "even unblocked it projects <= 10.6 admitted fix commits from the 36-"
                    "repository training pool (" + str(projection["projection_training"]) + ")")
    blockers.append("confirmation panel not acquired (same credential blocker); projected "
                    "~11.7 qualified targets from the 70-repository confirmation pool (< 80): "
                    "underpowered by addendum 1")
    blockers.append("final mixed corpus not built: the frozen 150/8/60 training gate fails")
    out = {
        "schema_version": "oneiros_v25_cpu2_preflight_decision_v1",
        "decision": "REFUSED: no v2.5 pilot preflight produced; DO NOT AUTHORIZE PILOT",
        "blockers": blockers,
        "preflight_conditions": {"universe_8_repositories_60_lineages": universe_ok,
                                 "projected_150_positives_within_10_gpu_hours": False,
                                 "all_identities_frozen": False},
        "source_commit": head,
        "full_suite": {"receipt": args.suite, "sha256": sha(args.suite),
                       "source_commit": suite["source_commit"], "passed": suite["passed"],
                       "failed": suite["failed"], "skipped": suite["skipped"]},
        "identities": {"generator_version": GENERATOR_VERSION, "job_schema": JOB_SCHEMA,
                       "prompt_builder": builder_identity(ROOT),
                       "preflight_schema": gate.PREFLIGHT_SCHEMA,
                       "authorization_schema": gate.AUTH_SCHEMA,
                       "protocol_sha256": {p: sha(p) for p in gate.PROTOCOL_FILES},
                       "model": model_identity(), "adapter_manifest_sha256": adapter_sha256()},
        "wiring_verified_by_tests": {
            "job_builder_uses_v25_prompt": "tests/test_native_generated_job_v25.py",
            "trainer_consumes_v25_pairs": "tests/test_v25_sft_data.py"},
        "canaries_not_rerun": "sandbox, native-environment and Atheris canaries were not rerun "
                              "for a freeze because the preflight is structurally refused; "
                              "Atheris v5 canary receipt v4 and the Phase 13 ABI readiness "
                              "stand",
        "receipts_sha256": {r: sha(r) for r in RECEIPTS},
        "gpu_authorization_created": False, "anything_launched": False,
        "validation_or_sealed_access": False,
        "claims": {"sft_improvement": False, "generalization": False,
                   "root_cause_established": False, "atheris_superiority": False,
                   "confirmation_ready": False, "training_ready": False}}
    status = publish_once(RECEIPT, out)
    print(json.dumps({"status": status, "decision": out["decision"],
                      "blockers": len(blockers), "sha256": sha(RECEIPT)}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
