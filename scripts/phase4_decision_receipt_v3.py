"""Decision receipt v3 (2026-09-29): corrected Choice A status; Choice B preflight v1 superseded.

Additive successor to decision receipts v1 and v2 (both unchanged). Corrections:

* Choice A: the argument-capture pilot found usable literal fixed calls for 3/24
  development targets (not 0/24, which was the source-literal extraction), in 1/8
  repositories, all in python-humanize. The Wilson interval is DESCRIPTIVE: the targets
  are clustered and purposively selected, not an iid sample. Choice A stays not ready,
  isolation-v6 revalidation stays required and mass acquisition stays refused.
* Choice B: the unversioned preflight receipt (bound to commit 0bf1f39) is preserved
  byte-for-byte as HISTORICAL/SUPERSEDED. Its launch guard cannot progress past the first
  tracked result (a clean-tree check plus a no-change-since-preflight diff check), so a
  stage-aware lifecycle and a versioned preflight v2 replace it before any GPU use.

Deterministic; launches nothing.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness.atomic_publish import publish_file_atomically

OUTPUT = "results/sft_root_cause_decision_receipt_2026-09-29_v3.json"
SOURCES = {
    "decision_v1": "results/sft_root_cause_decision_receipt_2026-09-29_v1.json",
    "decision_v2": "results/sft_root_cause_decision_receipt_2026-09-29_v2.json",
    "argument_capture_manifest": "results/sft_root_cause_phase4_argument_capture_manifest_v1.json",
    "argument_capture_receipt": "results/sft_root_cause_phase4_argument_capture_receipt_v1.json",
    "native_rehearsal_v2": "results/sft_root_cause_phase4_native_rehearsal_receipt_v2.json",
    "choice_b_preflight_v1": "results/sft_root_cause_phase4_choice_b_preflight_receipt.json",
    "choice_b_split_v2": "results/sft_root_cause_phase4_choice_b_split_v2.json",
}


def sha(rel: str) -> str:
    return hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()


def load(rel: str):
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))


def main() -> int:
    v2 = load(SOURCES["decision_v2"])
    capture = load(SOURCES["argument_capture_receipt"])
    manifest = load(SOURCES["argument_capture_manifest"])
    preflight = load(SOURCES["choice_b_preflight_v1"])
    repo_of = {t["key"]: t["repository"] for t in manifest["targets"]}
    usable = [t["key"] for t in capture["per_target"] if t["short_verified_oracle"] > 0]
    repos = sorted(set(repo_of.values()))
    usable_repos = sorted({repo_of[k] for k in usable})
    receipt = {
        "schema_version": "oneiros_sft_root_cause_decision_v3", "date": "2026-09-29",
        "supersedes": {"path": SOURCES["decision_v2"], "modified": False},
        "launches_nothing": True,
        "sources": {name: {"path": rel, "sha256": sha(rel)} for name, rel in SOURCES.items()},
        "state_of_evidence": {**v2["state_of_evidence"], "root_cause_established": False,
                              "generalization_established": False},
        "choice_A": {
            **v2["choice_A"],
            "fixed_call_yield": {
                "source_literal_extraction_v1": "0/24 qualified targets (native rehearsal v1)",
                "runtime_argument_capture_pilot": {
                    "targets_with_usable_fixed_call": f"{len(usable)}/{len(capture['per_target'])}",
                    "repositories_with_usable_fixed_call": f"{len(usable_repos)}/{len(repos)}",
                    "usable_repositories": usable_repos,
                    "all_successes_in_one_repository": len(usable_repos) == 1,
                    "totals": capture["totals"],
                    "wilson_95_targets": capture["qualified_targets_with_a_usable_fixed_call"][
                        "wilson_95"],
                    "interval_status": ("DESCRIPTIVE ONLY: targets are clustered by repository "
                                        "and purposively selected development targets, not an "
                                        "iid sample of any population; no population-confidence "
                                        "claim"),
                    "dominant_failures": dict(Counter(capture["rejection_families"])
                                              .most_common(3))}},
            "readiness": {"native_environment_feasible": True, "fixed_input_ready": False,
                          "choice_A_ready": False},
            "isolation_v6_revalidation": "required before any target is used; not performed",
            "mass_acquisition": "REFUSED"},
        "choice_B": {
            **v2["choice_B"],
            "preflight_v1": {
                "path": SOURCES["choice_b_preflight_v1"], "sha256": sha(SOURCES["choice_b_preflight_v1"]),
                "bound_commit": preflight["git"]["commit"],
                "status": "HISTORICAL / SUPERSEDED; preserved byte-for-byte; never launched",
                "reasons": [
                    "lifecycle-stale: HEAD moved past the bound commit, so its guard refuses even "
                    "though every bound source hash still matches",
                    "lifecycle deadlock: the guard requires a clean tree AND no change since the "
                    "preflight other than the receipt, so an unignored stage result blocks every "
                    "later stage whether uncommitted or committed",
                    "evaluation integrity gaps: duplicate generation keys overwritten by dict "
                    "construction; Kill@8 analysed on an intersection of function rows; no "
                    "immutable contract for resumable ignored generation files; promised metrics "
                    "incomplete"]},
            "status": "not launched; to be re-frozen under a stage-aware lifecycle (preflight v2)"},
        "recommendation": {
            "decision": ("fix the Choice B lifecycle and evaluation integrity, re-freeze a v2 "
                         "preflight, pass a bounded GPU integration smoke, then run the single "
                         "predeclared screen; keep Choice A at CPU-only mechanistic piloting"),
            "not_launched": ["Choice B training", "Choice A acquisition"]},
    }
    publish_file_atomically(ROOT / OUTPUT, (json.dumps(receipt, indent=1, sort_keys=True)
                                            + "\n").encode("utf-8"))
    print(json.dumps(receipt["choice_A"]["fixed_call_yield"]["runtime_argument_capture_pilot"],
                     indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
