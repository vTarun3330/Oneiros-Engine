"""Quarantine/explanation ledger for the v2.4 cycle (amendment v2.4 G: no unexplained failure).

Every artifact quarantined during the v2.4 cycle is recorded with its hash, the reason, the
corrective change and the receipt that supersedes it; superseded (not failed) receipts are
recorded for the audit trail. Written once to a tracked, versioned path (ledger v2 succeeds
v1 after the Atheris resume-validation fix; v1 is preserved unchanged); an identical rebuild is
a verified reproduction. Explained development failures never count as unexplained failures.

    python scripts/native_quarantine_ledger_v24.py
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

from scripts.native_rehearsal_rebuild_v22 import publish_once

LEDGER_V1 = "results/sft_root_cause_native_v24_quarantine_ledger.json"
LEDGER = "results/sft_root_cause_native_v24_quarantine_ledger_v2.json"
PREFLIGHT_V1 = "results/sft_root_cause_native_generated_tests_preflight_v2_4.json"
FINAL_PIPELINE = "results/sft_root_cause/native_v24_canaries/pipeline_receipt.json"
QUARANTINED = (
    ("results/sft_root_cause/quarantine/v24_pipeline_attempt1/pipeline_receipt.json",
     "the synthetic cohort manifest did not declare the requalification records required by "
     "v2.4 C, so the executor correctly refused (execution_complete=false)",
     "scripts/native_pipeline_synthetic.py now declares the exact preparation file and hash "
     "(commit a26b72f)"),
    ("results/sft_root_cause/quarantine/v24_pipeline_attempt2/pipeline_receipt.json",
     "the synthetic analysis check still expected arm-comparison metrics that the v2.4 "
     "coverage/repository gate correctly suppresses for the 2-target, 1-repository toy",
     "the synthetic check asserts the gate failure and the suppression (commit a26b72f); "
     "later strengthened to assert every engineering subgate"),
    ("results/sft_root_cause/quarantine/v24_pipeline_attempt3/pipeline_receipt.json",
     "the strengthened synthetic check expected coverage_gate_passed=false, but the 2-target "
     "toy has 2 of 2 qualified targets eligible, so coverage correctly passes; only the "
     "repository and synthetic stage-receipt subgates fail (the analysis was correct)",
     "the synthetic check expects coverage=true, repositories=false (commit 397d2c1)"),
    ("results/sft_root_cause/quarantine/v24_pipeline_attempt4/pipeline_receipt.json",
     "passed every check but was written with platform (CRLF) line endings, so its bytes "
     "would not reproduce on checkout",
     "native_pipeline_synthetic.finish() writes UTF-8 LF bytes (follow-up commit after "
     "397d2c1)"),
)
CONFORMANCE = ("superseded by the v2.4 implementation-conformance fix (live views, Atheris "
               "eligibility, analysis binding, complete gate, atomic launch); preserved in Git "
               "history")
RESUME = ("superseded by the Atheris resume-validation fix (one shared row validator for resume "
          "and the final loader; source change after this receipt); preserved in Git history")
SUPERSEDED = (
    ("results/sft_root_cause/native_v24_canaries/pipeline_receipt.json",
     "92a02d0b70141cf09a339e48a17d9cb678754aac333a02d394dae8a767fc39c3", "13b60da", CONFORMANCE),
    ("results/sft_root_cause/native_v24_canaries/canary_receipt_v2.json",
     "d5d41625ceb6ad42ab0bbf298514d963220db8ce2374b6d7adf5d890e63e48c9", "13b60da", CONFORMANCE),
    ("results/sft_root_cause/native_v24_canaries/atheris_canary_receipt_v3.json",
     "3f8fa959404fcc193ce20c6823fb4bfced46bc601a659fb48a8686b7acf99950", "13b60da", CONFORMANCE),
    ("results/sft_root_cause/native_v24_canaries/pipeline_receipt.json",
     "9c9dd2799b7eff6b4ccbf3e044fd3baecdfd02b21049b7a73bd0a2cf3beeb017", "eeeaced", RESUME),
    ("results/sft_root_cause/native_v24_canaries/canary_receipt_v2.json",
     "6a949b02a3558bebfdbbaf1b07c7b8a1feca65492f20c1f11cc2aa05ae238d38", "eeeaced", RESUME),
    ("results/sft_root_cause/native_v24_canaries/atheris_canary_receipt_v3.json",
     "cf58b29016039f4f22fed950af8bb8eda096c8edfc96ee4cbdea0dd7ced2dd4f", "eeeaced", RESUME),
)
# immutable predecessors kept at their own paths (never overwritten, never used for launch)
RETIRED = (
    (LEDGER_V1, "2fb360e67ced6e12e057074ce73e5d3be65fc03e3498da92e6ae65ec92370cdf", "c6077bc",
     "ledger v1: its superseded_by hashes name the pre-fix pipeline receipt; succeeded by v2"),
    ("results/sft_root_cause/native_v24_full_suite.json",
     "32ffd4545f33f36e2685db9a1f4e2797b92bd48b12f917ae00654f72a282a597", "c6077bc",
     "full-suite receipt for pre-fix source caa0012 (immutable); succeeded by "
     "native_v24_full_suite_r2.json"),
    (PREFLIGHT_V1, "c5479bb371f6c8ecbfaaa06491c3bde0f14db46e4908b1d654b3d257963104cd", "78bf3ef",
     "preflight bound to pre-fix source caa0012; stale after the Atheris resume fix; no "
     "authorisation was ever bound to it; succeeded by preflight_v2_4_r2"),
)


def sha(rel: str) -> str:
    return hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()


def main(argv=None) -> int:
    argparse.ArgumentParser(description=__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter).parse_args(argv)
    final = {"path": FINAL_PIPELINE, "sha256": sha(FINAL_PIPELINE)}
    entries = [{"artifact": rel, "sha256": sha(rel), "reason": reason,
                "corrective_change": change, "superseded_by": final}
               for rel, reason, change in QUARANTINED]
    ledger = {
        "schema_version": "oneiros_native_quarantine_ledger_v1",
        "cycle": "amendment v2.4",
        "entries": entries,
        "supersedes_ledger": {"path": LEDGER_V1, "sha256": sha(LEDGER_V1)},
        "superseded_receipts": [
            {"artifact": rel, "sha256": digest, "committed_in": commit, "reason": reason,
             "superseded_by": {"path": rel, "sha256": sha(rel)}}
            for rel, digest, commit, reason in SUPERSEDED],
        "retired_immutable_artifacts": [
            {"artifact": rel, "sha256": digest, "committed_in": commit, "reason": reason,
             "still_present_and_unchanged": sha(rel) == digest}
            for rel, digest, commit, reason in RETIRED],
        "control_plane_failures": [
            "server-side command classifier returned no verdict for repeated git add/status "
            "requests; the commands never executed (not repository failures)"],
    }
    status = publish_once(LEDGER, ledger)
    print(json.dumps({"status": status, "entries": len(entries),
                      "superseded_receipts": len(SUPERSEDED)}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
