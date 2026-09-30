"""Isolation-v6 revalidation + engineering manifest v4 + primary generation job (CPU only).

Amendment v2.1. Inputs are the frozen 24-target manifest, the FORMAL re-qualification records
exported from WSL (results/sft_root_cause/native_v21_rehearsal/), and the retained evidence
sidecars. Steps, each with its own inputs:
  1. isolation v6: recompute ``check_candidate`` for all 24 targets against the verified frozen
     reference universe; cached in a receipt that binds the source manifest, the evidence
     pool, the isolation implementation and the reference-universe receipt;
  2. keep = isolation-v6 admissible AND formally requalified; everything else is reported,
     never silently kept or replaced;
  3. prompts from buggy_view/ only; 4. leakage scan with verifier/ plus sidecar issue text;
  5. tokenizer sequence fit (refuse, never truncate) -> primary job;
  6. gates: rehearsal rule (20-30 targets, >= 5 repositories) and job coverage
     (>= 20 targets and >= 90% of kept targets in the primary job, no unrecorded loss).
Exit status is non-zero if any gate fails.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness.atomic_publish import publish_file_atomically

V2_MANIFEST = "results/sft_root_cause_phase4_receiver_capture_manifest_v2.json"
POOL = "results/v4_3_repository_native_a_prime_retrospective.json"
STORE = "data/repository_native/a_prime_retrospective/objects"
BUNDLE = "results/next_direction_bundle"
ISOLATION_SOURCE = "harness/repository_isolation.py"
LOCAL = "results/sft_root_cause/native_v21_rehearsal"
ISOLATION = "results/sft_root_cause_native_v21_isolation_v6.json"
MANIFEST = "results/sft_root_cause_native_v21_rehearsal_manifest_v4.json"
JOB = "results/sft_root_cause_native_v21_rehearsal_job_v2.json"
PROTOCOLS = ("docs/SFT_ROOT_CAUSE_NATIVE_GENERATED_TEST_PROTOCOL_V2.md",
             "docs/SFT_ROOT_CAUSE_NATIVE_GENERATED_TEST_PROTOCOL_V2_1.md")
RULE = {"min_targets": 20, "max_targets": 30, "min_repositories": 5}
COVERAGE = {"min_job_targets": 20, "min_fraction_of_kept": 0.90}


# Amendment v2.2: re-running would overwrite immutable v2.1 evidence.
ALLOW_HISTORICAL_RERUN = False


def sha(rel: str) -> str:
    return hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()


def load(rel: str):
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))


def isolation_v6(targets) -> tuple:
    from harness.atomic_publish import read_current_bundle
    from harness.github_acquisition import ContentStore
    from harness.repository_isolation import (
        ISOLATION_VERSION, build_reference_universe, candidate_from_sidecar, check_candidate,
        freeze_reference_universe)
    bundle = read_current_bundle(ROOT / BUNDLE)
    receipt = json.loads(bundle["files"]["reference_universe_receipt.json"])
    frozen = freeze_reference_universe(build_reference_universe(ROOT), receipt, root=ROOT)
    admitted = {a["key"]: a for a in load(POOL)["admitted"]}
    store = ContentStore(ROOT / STORE)
    records, issues = {}, {}
    for t in targets:
        entry = admitted.get(t["key"])
        if entry is None:
            records[t["key"]] = {"admissible": False, "reasons": ["not in the evidence pool"]}
            continue
        candidate = candidate_from_sidecar(json.loads(store.get_raw(entry["sidecar_address"])))
        record = check_candidate(candidate, frozen)
        records[t["key"]] = {"admissible": record["admissible"], "reasons": record["reasons"],
                             "record_sha256": record["record_sha256"],
                             "pre_v6_record_sha256": entry["record_sha256"]}
        try:
            body = json.loads(candidate.issue_metadata.body or b"{}")
            issues[t["key"]] = f"{body.get('title') or ''}\n{body.get('body') or ''}"
        except ValueError:
            issues[t["key"]] = ""
    return {"isolation_version": ISOLATION_VERSION,
            "reference_universe_sha256": frozen.receipt_sha256, "records": records}, issues


def cached_isolation(targets) -> tuple:
    from harness.source_identity import canonical_sha256
    binding = {"source_manifest_sha256": sha(V2_MANIFEST), "pool_sha256": sha(POOL),
               "isolation_source_sha256": canonical_sha256(ROOT / ISOLATION_SOURCE)}
    if (ROOT / ISOLATION).exists():
        stored = load(ISOLATION)
        if {k: stored.get(k) for k in binding} != binding:
            raise SystemExit("REFUSED: the isolation receipt is bound to different inputs or "
                             "a different isolation implementation; write a successor")
        return stored["isolation"], stored["issue_text"]
    iso, issues = isolation_v6(targets)
    publish_file_atomically(ROOT / ISOLATION, (json.dumps({
        "schema_version": "oneiros_native_v21_isolation_v6_revalidation", **binding,
        "isolation": iso, "issue_text": issues}, indent=1, sort_keys=True) + "\n").encode("utf-8"))
    return iso, issues


def main() -> int:
    if not ALLOW_HISTORICAL_RERUN:
        raise SystemExit("REFUSED: superseded by scripts/native_rehearsal_rebuild_v22.py (amendment v2.2); the v2.1 artifacts "
                         "this script writes are immutable historical evidence")
    from harness.acquisition_receipt import ProtectedAccessMonitor
    from harness.native_generated_test_leakage import scan
    from harness.native_generated_test_prompt import PromptRefused, build_prompt
    from scripts.native_rehearsal_prepare_wsl import FINAL, tag_of
    ProtectedAccessMonitor.install(ROOT)
    mark = ProtectedAccessMonitor.mark()
    targets = load(V2_MANIFEST)["targets"]
    iso, issues = cached_isolation(targets)
    history = {}
    for line in (ROOT / LOCAL / "records.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            history.setdefault(row["key"], []).append(row)
    per_target, kept = [], []
    for t in targets:
        v6 = iso["records"][t["key"]]
        rows = history.get(t["key"], [])
        q = rows[-1] if rows else {"category": "missing"}
        ok = v6["admissible"] and q.get("category") == "requalified"
        per_target.append({
            "key": t["key"], "repository": t["repository"],
            "isolation_v6_admissible": v6["admissible"], "isolation_v6_reasons": v6["reasons"],
            "requalification": q.get("category"), "attempts": len(rows),
            "final": q.get("category") in FINAL, "failure": q.get("failure"),
            "qualification": q.get("qualification"),
            "environment_lock": q.get("environment_lock"), "interpreter": q.get("interpreter"),
            "view_manifest_sha256": q.get("view_manifest_sha256"), "kept": ok})
        if ok:
            kept.append(t)
    repos = sorted({t["repository"] for t in kept})
    rule_ok = RULE["min_targets"] <= len(kept) <= RULE["max_targets"] and \
        len(repos) >= RULE["min_repositories"]
    sealed, builder_refused = [], []
    for t in kept:
        view = json.loads((ROOT / LOCAL / "buggy_view" / f"{tag_of(t['key'])}.json")
                          .read_text(encoding="utf-8"))
        try:
            sealed.append(build_prompt(view["dto"], view["buggy_source"]))
        except PromptRefused as exc:
            builder_refused.append({"target_key": t["key"], "reason": f"prompt_refused: {exc}"})
    scans = {}
    for s in sealed:
        verifier = json.loads((ROOT / LOCAL / "verifier" / f"{tag_of(s['target_key'])}.json")
                              .read_text(encoding="utf-8"))
        verifier["issue_text"] = issues.get(s["target_key"], "")
        scans[s["target_key"]] = scan(s, verifier)
    import transformers
    from engine.test_generation_prompt import format_chat_prompt
    from scripts.native_generated_tests_generate import CONTRACT, build_job, sequence_fit
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        CONTRACT["base_model"], revision=CONTRACT["base_revision"], local_files_only=True)

    def count(text: str) -> int:
        return len(tokenizer(format_chat_prompt(tokenizer, text), add_special_tokens=False)
                   ["input_ids"])
    job = build_job(sealed, scans, sequence_fit(sealed, count))
    evidence = ProtectedAccessMonitor.evidence(mark)
    publish_file_atomically(ROOT / JOB, (json.dumps(
        {"schema_version": "oneiros_native_v21_generation_job_v2", "protocols": list(PROTOCOLS),
         "primary_whole_module": job}, indent=1, sort_keys=True) + "\n").encode("utf-8"))
    items = {i["target_key"] for i in job["items"]}
    refused = job["refused"] + builder_refused
    accounted = items | {r["target_key"] for r in refused}
    coverage_ok = (len(items) >= COVERAGE["min_job_targets"]
                   and len(items) >= COVERAGE["min_fraction_of_kept"] * len(kept)
                   and accounted == {t["key"] for t in kept})
    manifest = {
        "schema_version": "oneiros_native_v21_rehearsal_manifest_v4",
        "nature": "ENGINEERING DRESS REHEARSAL ONLY; not confirmation, generalisation or "
                  "model-selection evidence",
        "source_manifest": {"path": V2_MANIFEST, "sha256": sha(V2_MANIFEST)},
        "protocols": {p: sha(p) for p in PROTOCOLS},
        "requalification_records": {"path": f"{LOCAL}/records.jsonl",
                                    "sha256": sha(f"{LOCAL}/records.jsonl")},
        "isolation": {"receipt": ISOLATION, "receipt_sha256": sha(ISOLATION),
                      **{k: iso[k] for k in ("isolation_version", "reference_universe_sha256")}},
        "superseded_preliminary": "results/sft_root_cause/native_v2_rehearsal (Python 3.13; "
                                  "engineering evidence only; preserved)",
        "per_target": per_target, "kept_targets": [t["key"] for t in kept], "targets": kept,
        "kept": len(kept), "of": len(targets), "repositories": len(repos),
        "rehearsal_rule": {**RULE, "passed": rule_ok},
        "job": {"path": JOB, "sha256": sha(JOB), "items": len(items),
                "refused": refused, "coverage_rule": COVERAGE, "coverage_passed": coverage_ok,
                "max_prompt_tokens": max((e["prompt_tokens"] for e in
                                          job["sequence_fit"]["admitted"]), default=None)},
        "excluded": [p for p in per_target if not p["kept"]],
        "leakage_refusals": {k: v["reasons"] for k, v in scans.items() if not v["ok"]},
        "protected_access_audit": {k: v for k, v in evidence.items() if k != "opens_checked"},
        "admitted_to_training": False, "gpu_used": False,
    }
    publish_file_atomically(ROOT / MANIFEST, (json.dumps(manifest, indent=1, sort_keys=True)
                                              + "\n").encode("utf-8"))
    print(json.dumps({k: manifest[k] for k in ("kept", "of", "repositories", "rehearsal_rule",
                                               "leakage_refusals")}
                     | {"job": {k: manifest["job"][k] for k in ("items", "refused",
                                                                "coverage_passed",
                                                                "max_prompt_tokens")},
                        "excluded": [(p["key"][5:50], p["isolation_v6_admissible"],
                                      p["requalification"], str(p["failure"])[:80])
                                     for p in manifest["excluded"]]}, indent=1))
    return 0 if rule_ok and coverage_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
