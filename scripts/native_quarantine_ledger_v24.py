"""Quarantine/explanation ledger for the v2.4 cycle (amendment v2.4 G: no unexplained failure).

Every artifact quarantined during the v2.4 cycle is recorded with its hash, the reason, the
corrective change and the receipt that supersedes it; superseded (not failed) receipts are
recorded for the audit trail. Written once to a tracked path; an identical rebuild is a
verified reproduction. Explained development failures never count as unexplained failures.

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

LEDGER = "results/sft_root_cause_native_v24_quarantine_ledger.json"
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
SUPERSEDED = (
    ("results/sft_root_cause/native_v24_canaries/pipeline_receipt.json",
     "92a02d0b70141cf09a339e48a17d9cb678754aac333a02d394dae8a767fc39c3"),
    ("results/sft_root_cause/native_v24_canaries/canary_receipt_v2.json",
     "d5d41625ceb6ad42ab0bbf298514d963220db8ce2374b6d7adf5d890e63e48c9"),
    ("results/sft_root_cause/native_v24_canaries/atheris_canary_receipt_v3.json",
     "3f8fa959404fcc193ce20c6823fb4bfced46bc601a659fb48a8686b7acf99950"),
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
        "superseded_receipts": [
            {"artifact": rel, "sha256": digest, "committed_in": "13b60da",
             "reason": "superseded by the v2.4 implementation-conformance fix (live views, "
                       "Atheris eligibility, analysis binding, complete gate, atomic launch); "
                       "preserved in Git history",
             "superseded_by": {"path": rel, "sha256": sha(rel)}}
            for rel, digest in SUPERSEDED],
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
