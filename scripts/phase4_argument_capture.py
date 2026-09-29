"""Phase 8 argument-capture pilot: manifest (before) and receipt (after). CPU only.

    manifest  freeze the pilot targets: the natively QUALIFIED v1 rehearsal targets, with
              their difference-exposing official tests, from tracked evidence only
    run       print the WSL command (the runner is scripts/native_argument_capture_wsl.py)
    receipt   summarise the WSL records into a tracked receipt with portable evidence

Development targets only (the A-prime retrospective). No mass acquisition, no model, no
GPU. A target that yields a usable fixed call is NOT thereby admitted to Choice A: it must
first be revalidated under isolation v6.
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
from harness.evidence_bundle import embed

NATIVE_MANIFEST = "results/sft_root_cause_phase4_native_rehearsal_manifest_v1.json"
NATIVE_BUNDLE = "results/sft_root_cause_phase4_native_evidence_v1.json"
MANIFEST = "results/sft_root_cause_phase4_argument_capture_manifest_v1.json"
RECEIPT = "results/sft_root_cause_phase4_argument_capture_receipt_v1.json"
RAW = "results/sft_root_cause/phase4_argument_capture"
Z = 1.959963984540054


def sha(rel: str) -> str:
    return hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()


def load(rel: str):
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))


def write_once(rel: str, payload: dict) -> None:
    data = (json.dumps(payload, indent=1, sort_keys=True) + "\n").encode("utf-8")
    if (ROOT / rel).exists() and (ROOT / rel).read_bytes() != data:
        raise SystemExit(f"REFUSED: {rel} exists with different bytes")
    publish_file_atomically(ROOT / rel, data)


def manifest() -> int:
    bundle = load(NATIVE_BUNDLE)
    records = {}
    for line in bundle["files"]["records.jsonl"]["text"].splitlines():
        if line.strip():
            row = json.loads(line)
            records[row["key"]] = row
    targets = []
    for target in load(NATIVE_MANIFEST)["targets"]:
        record = records[target["key"]]
        if record["category"] != "natively_qualified":
            continue
        targets.append({k: target[k] for k in ("key", "repository", "repository_url",
                                               "buggy_commit", "fixed_commit", "target",
                                               "target_file", "regression_test_files")}
                       | {"difference_exposing_tests": record["difference_exposing_tests"]})
    write_once(MANIFEST, {
        "schema_version": "oneiros_phase4_argument_capture_manifest_v1",
        "pool": "natively qualified v1 rehearsal targets (A-prime DEVELOPMENT retrospective)",
        "inputs": {NATIVE_MANIFEST: sha(NATIVE_MANIFEST), NATIVE_BUNDLE: sha(NATIVE_BUNDLE)},
        "targets": sorted(targets, key=lambda t: t["key"]),
        "isolation_v6_revalidated": False,
        "mass_acquisition": False, "model_calls": 0, "gpu_used": False})
    print(f"{len(targets)} qualified targets")
    return 0


def wilson(k: int, n: int) -> list:
    if not n:
        return [None, None]
    p = k / n
    centre = (p + Z * Z / (2 * n)) / (1 + Z * Z / n)
    half = Z * (p * (1 - p) / n + Z * Z / (4 * n * n)) ** 0.5 / (1 + Z * Z / n)
    return [round(max(0.0, centre - half), 4), round(min(1.0, centre + half), 4)]


def receipt() -> int:
    records_rel = f"{RAW}/records.jsonl"
    rows = [json.loads(l) for l in (ROOT / records_rel).read_text(encoding="utf-8").splitlines()
            if l.strip()]
    targets = load(MANIFEST)["targets"]
    if sorted(r["key"] for r in rows) != sorted(t["key"] for t in targets):
        raise SystemExit("REFUSED: records do not cover exactly the manifest targets")
    totals = Counter()
    kinds, rejections, per_target = Counter(), Counter(), []
    for r in sorted(rows, key=lambda r: r["key"]):
        capture = r.get("capture") or {}
        calls = r.get("calls") or []
        counts = {
            "observed": capture.get("observed", 0),
            "serializable": r.get("calls_serializable", 0),
            "replayed": len(calls),
            "replayable_both": sum(c["replayable_both"] for c in calls),
            "fixed_short_literal": sum(c["fixed_short_literal"] for c in calls),
            "short_verified_oracle": sum(c["short_verified_oracle"] for c in calls)}
        totals.update(counts)
        kinds[capture.get("kind") or "unresolved"] += 1
        rejections.update(capture.get("rejected") or {})
        per_target.append({"key": r["key"], "category": r["category"],
                           "target_kind": capture.get("kind"),
                           "resolve_error": capture.get("resolve_error"),
                           "wanted_tests_reaching_target":
                               capture.get("wanted_tests_reaching_target"),
                           **counts, "wall_seconds": r.get("wall_seconds"),
                           "dependencies_recorded": bool((r.get("environment") or {})
                                                         .get("dependencies"))})
    usable = sum(t["short_verified_oracle"] > 0 for t in per_target)
    rejection_families = Counter()
    for reason, count in rejections.items():
        rejection_families[reason.split(":")[0]] += count
    summary = load(f"{RAW}/run_summary.json") if (ROOT / RAW / "run_summary.json").exists() \
        else {}
    payload = {
        "schema_version": "oneiros_phase4_argument_capture_receipt_v1",
        "manifest": {"path": MANIFEST, "sha256": sha(MANIFEST)},
        "method": ("profile-hook capture on the fixed revision during the difference-exposing "
                   "official tests; literal-only serialisation; independent replay on both "
                   "revisions"),
        "targets": len(per_target),
        "totals": dict(totals),
        "qualified_targets_with_a_usable_fixed_call": {
            "k": usable, "n": len(per_target), "wilson_95": wilson(usable, len(per_target))},
        "categories": dict(Counter(t["category"] for t in per_target)),
        "target_kinds": dict(kinds),
        "rejection_reasons": dict(rejections.most_common()),
        "rejection_families": dict(rejection_families.most_common()),
        "per_target": per_target,
        "limits": summary.get("limits"), "wall_seconds": summary.get("wall_seconds"),
        "portable_evidence": {"records.jsonl": embed(ROOT, records_rel, [])},
        "isolation_v6_revalidated": False,
        "choice_A_admission": ("none: any target with a usable call must be revalidated under "
                               "isolation v6 before it is used for Choice A work"),
        "model_exposure": "none; fixed code and gold tests are verifier-only",
        "mass_acquisition": False, "model_calls": 0, "gpu_used": False,
    }
    write_once(RECEIPT, payload)
    print(json.dumps({k: payload[k] for k in ("totals", "categories", "target_kinds",
                                              "rejection_families",
                                              "qualified_targets_with_a_usable_fixed_call")},
                     indent=1))
    return 0


def main(argv=None) -> int:
    command = (argv or sys.argv[1:] or ["manifest"])[0]
    if command == "manifest":
        return manifest()
    if command == "receipt":
        return receipt()
    raise SystemExit(f"unknown command {command}")


if __name__ == "__main__":
    raise SystemExit(main())
