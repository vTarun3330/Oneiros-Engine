"""Choice A receiver-aware replay pilot (v2): frozen manifest and result receipt. CPU only.

    manifest  freeze the pilot targets: EXACTLY the 24 targets of the v1 argument-capture
              manifest (checked, not re-selected)
    receipt   summarise the WSL records and apply the predeclared feasibility gate

Protocol: docs/SFT_ROOT_CAUSE_CHOICE_A_RECEIVER_REPLAY_PROTOCOL_V2.md. Retrospective
feasibility study only; nothing is admitted to training; no model; no GPU.
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
from harness.evidence_bundle import PRIVATE_PATTERNS, embed, file_problems

V1_MANIFEST = "results/sft_root_cause_phase4_argument_capture_manifest_v1.json"
MANIFEST = "results/sft_root_cause_phase4_receiver_capture_manifest_v2.json"
RECEIPT = "results/sft_root_cause_phase4_receiver_capture_receipt_v2.json"
RAW = "results/sft_root_cause/phase4_receiver_capture"
PROTOCOL = "docs/SFT_ROOT_CAUSE_CHOICE_A_RECEIVER_REPLAY_PROTOCOL_V2.md"
GATE = {"min_usable_targets": 8, "targets": 24, "min_repositories": 4, "repositories": 8,
        "max_repository_share": 0.40}
Z = 1.959963984540054


def sha(rel: str) -> str:
    return hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()


def load(rel: str):
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))


def write_once(rel: str, payload) -> None:
    data = (json.dumps(payload, indent=1, sort_keys=True) + "\n").encode("utf-8")
    if (ROOT / rel).exists() and (ROOT / rel).read_bytes() != data:
        raise SystemExit(f"REFUSED: {rel} exists with different bytes")
    publish_file_atomically(ROOT / rel, data)


def wilson(k: int, n: int) -> list:
    p = k / n
    centre = (p + Z * Z / (2 * n)) / (1 + Z * Z / n)
    half = Z * (p * (1 - p) / n + Z * Z / (4 * n * n)) ** 0.5 / (1 + Z * Z / n)
    return [round(max(0.0, centre - half), 4), round(min(1.0, centre + half), 4)]


def manifest() -> int:
    from harness.acquisition_receipt import ProtectedAccessMonitor
    ProtectedAccessMonitor.install(ROOT)
    mark = ProtectedAccessMonitor.mark()
    v1 = load(V1_MANIFEST)
    targets = v1["targets"]
    if len(targets) != GATE["targets"] or len({t["repository"] for t in targets}) != \
            GATE["repositories"]:
        raise SystemExit("REFUSED: the declared pilot set is not 24 targets in 8 repositories")
    evidence = ProtectedAccessMonitor.evidence(mark)
    write_once(MANIFEST, {
        "schema_version": "oneiros_receiver_capture_manifest_v2",
        "identical_to": {"path": V1_MANIFEST, "sha256": sha(V1_MANIFEST)},
        "targets_sha256": hashlib.sha256(json.dumps(targets, sort_keys=True).encode())
        .hexdigest(),
        "targets": targets, "protocol": {"path": PROTOCOL, "sha256": sha(PROTOCOL)},
        "feasibility_gate": GATE, "nature": "retrospective feasibility study; not efficacy",
        "protected_access_audit": {k: v for k, v in evidence.items() if k != "opens_checked"},
        "mass_acquisition": False, "model_calls": 0, "gpu_used": False})
    print(f"{len(targets)} targets frozen")
    return 0


def receipt() -> int:
    records_rel = f"{RAW}/records.jsonl"
    rows = [json.loads(l) for l in (ROOT / records_rel).read_text(encoding="utf-8").splitlines()
            if l.strip()]
    frozen = load(MANIFEST)
    keys = [t["key"] for t in frozen["targets"]]
    if sorted(r["key"] for r in rows) != sorted(keys) or len(rows) != len(keys):
        raise SystemExit("REFUSED: records do not cover exactly the frozen targets once each")
    repo_of = {t["key"]: t["repository"] for t in frozen["targets"]}
    per_target, totals, statuses, rejections = [], Counter(), Counter(), Counter()
    for r in sorted(rows, key=lambda r: r["key"]):
        capture = r.get("capture") or {}
        calls = r.get("calls") or []
        counts = {"observed": capture.get("observed", 0),
                  "recipes_registered": capture.get("recipes_registered", 0),
                  "captured": r.get("calls_captured", 0), "replayed": len(calls),
                  "constructor_or_factory_calls": sum(bool(c["recipe_kind"]) for c in calls),
                  "deterministic_differential": sum(c["status"] in ("usable",
                                                                    "fixed_not_literal_value",
                                                                    "fixed_value_too_long")
                                                    for c in calls),
                  "usable": sum(c["usable"] for c in calls)}
        totals.update(counts)
        statuses.update(c["status"] for c in calls)
        rejected = capture.get("rejected") or {}
        rejections.update(rejected)
        per_target.append({
            "key": r["key"], "repository": repo_of[r["key"]], "category": r["category"],
            "failure": r.get("failure"), "target_kind": capture.get("kind"),
            "resolve_error": capture.get("resolve_error"), **counts,
            "call_statuses": dict(Counter(c["status"] for c in calls)),
            "rejection_reasons": rejected,
            "mutated_or_opaque_receivers": sum(v for k, v in rejected.items()
                                               if "mutated" in k or "opaque_state" in k),
            "wall_seconds": r.get("wall_seconds"),
            "dependencies_recorded": bool((r.get("environment") or {}).get("dependencies"))})
    usable = [t for t in per_target if t["usable"] > 0]
    by_repo = Counter(t["repository"] for t in usable)
    largest = max(by_repo.values()) / len(usable) if usable else None
    checks = {"usable_targets": len(usable) >= GATE["min_usable_targets"],
              "repositories": len(by_repo) >= GATE["min_repositories"],
              "max_repository_share": largest is not None
              and largest <= GATE["max_repository_share"]}
    passed = all(checks.values())
    reason_families = Counter()
    for reason, count in rejections.items():
        reason_families[reason.split(":")[0]] += count
    summary = load(f"{RAW}/run_summary.json")
    evidence = {"records.jsonl": embed(ROOT, records_rel, []),
                "run_summary.json": embed(ROOT, f"{RAW}/run_summary.json", [])}
    problems = [p for name, e in evidence.items()
                for p in file_problems(ROOT, name, e, PRIVATE_PATTERNS)]
    if problems:
        raise SystemExit(f"REFUSED: evidence problems {problems}")
    payload = {
        "schema_version": "oneiros_receiver_capture_receipt_v2",
        "manifest": {"path": MANIFEST, "sha256": sha(MANIFEST)},
        "protocol": {"path": PROTOCOL, "sha256": sha(PROTOCOL)},
        "nature": "retrospective feasibility study on development targets; not efficacy",
        "totals": dict(totals), "call_statuses": dict(statuses),
        "rejection_reason_families": dict(reason_families.most_common()),
        "rejection_reasons": dict(rejections.most_common()),
        "categories": dict(Counter(t["category"] for t in per_target)),
        "environment_failures": [{"key": t["key"], "failure": t["failure"]} for t in per_target
                                 if t["category"] in ("environment_failure", "runner_error")],
        "usable_targets": {"k": len(usable), "n": len(per_target),
                           "wilson_95_descriptive": wilson(len(usable), len(per_target))},
        "usable_repositories": {"k": len(by_repo), "n": GATE["repositories"],
                                "wilson_95_descriptive": wilson(len(by_repo),
                                                                GATE["repositories"]),
                                "per_repository": dict(sorted(by_repo.items()))},
        "largest_repository_share": round(largest, 4) if largest is not None else None,
        "interval_status": ("DESCRIPTIVE ONLY: clustered, purposively selected, non-iid "
                            "development targets"),
        "feasibility_gate": {"thresholds": GATE, "checks": checks, "passed": passed},
        "branch": "A" if passed else "B",
        "per_target": per_target, "wall_seconds": summary.get("wall_seconds"),
        "network_isolation": summary.get("network_isolation"),
        "portable_evidence": evidence,
        "v1_limitation_found": ("while building v2, a stale-bytecode hazard was found: an "
                                "editable finder rewritten with the same size in the same "
                                "second can keep importing the previous revision; v2 disables "
                                "bytecode writes, clears finder bytecode after each switch and "
                                "refuses a replay whose module is not under the expected "
                                "revision. v1 argument-capture replays had no such guard, so "
                                "v1 'no difference' outcomes may include false negatives."),
        "admitted_to_training": False, "isolation_v6_revalidated": False,
        "mass_acquisition": False, "model_calls": 0, "gpu_used": False,
    }
    write_once(RECEIPT, payload)
    print(json.dumps({k: payload[k] for k in ("totals", "usable_targets", "usable_repositories",
                                              "largest_repository_share", "feasibility_gate",
                                              "branch", "categories")}, indent=1))
    return 0


def main(argv=None) -> int:
    command = (argv or sys.argv[1:] or ["manifest"])[0]
    return {"manifest": manifest, "receipt": receipt}[command]()


if __name__ == "__main__":
    raise SystemExit(main())
