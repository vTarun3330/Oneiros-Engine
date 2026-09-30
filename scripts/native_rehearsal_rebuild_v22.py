"""Amendment v2.2 prompt/scan/sequence-fit/job rebuild for the 24-target rehearsal (CPU only).

Inputs (all unchanged historical evidence): the frozen 24-target manifest, the isolation-v6
receipt (bound to its inputs and implementation), the formal re-qualification records and the
separated buggy/verifier views exported from WSL. Builder v2 and scanner v2 are applied
uniformly to every kept target. Every refusal reason is recorded (multi-label); coverage
counts unique admitted targets.

Outputs are NEW versioned paths. An existing output is never overwritten: an identical
rebuild is reported as a verified reproduction, a different one refuses.

    python scripts/native_rehearsal_rebuild_v22.py
Exit status is non-zero if the rehearsal rule or the coverage gate fails.
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
from scripts.native_rehearsal_revalidate import (COVERAGE, ISOLATION, LOCAL, RULE, V2_MANIFEST,
                                                 cached_isolation, load, sha)

PROTOCOLS = ("docs/SFT_ROOT_CAUSE_NATIVE_GENERATED_TEST_PROTOCOL_V2.md",
             "docs/SFT_ROOT_CAUSE_NATIVE_GENERATED_TEST_PROTOCOL_V2_1.md",
             "docs/SFT_ROOT_CAUSE_NATIVE_GENERATED_TEST_PROTOCOL_V2_2.md")
V21_JOB = "results/sft_root_cause_native_v21_rehearsal_job_v2.json"
JOB = "results/sft_root_cause_native_v22_rehearsal_job_v3.json"
MANIFEST = "results/sft_root_cause_native_v22_rehearsal_manifest_v5.json"
PROMPTS = "results/sft_root_cause_native_v22_prompt_records.json"
NEVER_INCLUDED = ("issue_or_pr_text", "fixed_code", "patches", "official_tests",
                  "expected_outputs", "execution_outcomes", "leakage_verdicts",
                  "helper_bodies", "target_specific_exceptions")


def publish_once(rel: str, payload: dict) -> str:
    """Write a new artifact; an identical existing one is a verified reproduction."""
    data = (json.dumps(payload, indent=1, sort_keys=True) + "\n").encode("utf-8")
    path = ROOT / rel
    if path.exists():
        if path.read_bytes() != data:
            raise SystemExit(f"REFUSED: {rel} exists with different content; "
                             "write a successor version instead of overwriting")
        return "verified_reproduction"
    publish_file_atomically(path, data)
    return "written"


def main() -> int:
    from harness.acquisition_receipt import ProtectedAccessMonitor
    from harness.native_generated_test_leakage import SCANNER_VERSION, scan
    from harness.native_generated_test_prompt import BUILDER_VERSION, PromptRefused, build_prompt
    from harness.source_identity import canonical_sha256
    from scripts.native_rehearsal_prepare_wsl import tag_of
    ProtectedAccessMonitor.install(ROOT)
    mark = ProtectedAccessMonitor.mark()
    targets = load(V2_MANIFEST)["targets"]
    iso, issues = cached_isolation(targets)
    history: dict = {}
    for line in (ROOT / LOCAL / "records.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            history.setdefault(row["key"], []).append(row)
    kept, per_target = [], []
    for t in targets:
        v6 = iso["records"][t["key"]]
        q = (history.get(t["key"]) or [{"category": "missing"}])[-1]
        ok = v6["admissible"] and q.get("category") == "requalified"
        per_target.append({"key": t["key"], "repository": t["repository"],
                           "isolation_v6_admissible": v6["admissible"],
                           "requalification": q.get("category"), "kept": ok})
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
            builder_refused.append({"target_key": t["key"], "reason": str(exc)})
    scans = {}
    for s in sealed:
        verifier = json.loads((ROOT / LOCAL / "verifier" / f"{tag_of(s['target_key'])}.json")
                              .read_text(encoding="utf-8"))
        verifier["issue_text"] = issues.get(s["target_key"], "")
        scans[s["target_key"]] = scan(s, verifier)

    import transformers
    from engine.test_generation_prompt import format_chat_prompt
    from scripts.native_generated_tests_generate import (CONTRACT, build_job, refusal_accounting,
                                                         sequence_fit)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        CONTRACT["base_model"], revision=CONTRACT["base_revision"], local_files_only=True)

    def count(text: str) -> int:
        return len(tokenizer(format_chat_prompt(tokenizer, text), add_special_tokens=False)
                   ["input_ids"])

    def raw(text: str) -> int:
        return len(tokenizer(text, add_special_tokens=False)["input_ids"])
    fit = sequence_fit(sealed, count)
    job = build_job(sealed, scans, fit, builder_refused)
    accounting = refusal_accounting([t["key"] for t in kept], job)
    old = {e["target_key"]: e["prompt_tokens"]
           for part in ("admitted", "refused")
           for e in load(V21_JOB)["primary_whole_module"]["sequence_fit"][part]}
    tokens = {e["target_key"]: e["prompt_tokens"] for part in ("admitted", "refused")
              for e in fit[part]}
    reasons = {r["target_key"]: r["reasons"] for r in job["refused"]}
    rows = []
    for s in sealed:
        key = s["target_key"]
        rows.append({
            "target_key": key, "v21_prompt_tokens": old.get(key),
            "v22_prompt_tokens": tokens[key], "target_source_tokens": raw(s["target_source"]),
            "fits": tokens[key] <= CONTRACT["prompt_token_limit"],
            "leakage_ok": scans[key]["ok"], "leakage_reasons": scans[key]["reasons"],
            "admitted": key not in reasons, "refusal_reasons": reasons.get(key, []),
            "retained_components": s["retained_components"], "omitted": s["omitted"],
            "never_included": list(NEVER_INCLUDED), "truncated": False,
            "view_sha256": s["view_sha256"], "prompt_sha256": s["prompt_sha256"],
            "buggy_source_sha256": s["buggy_source_sha256"],
            "builder_version": s["builder_version"], "scanner_version": scans[key]["scanner_version"]})
    for b in builder_refused:
        rows.append({"target_key": b["target_key"], "admitted": False,
                     "refusal_reasons": ["prompt_refused"], "detail": b["reason"]})
    rows.sort(key=lambda r: r["target_key"])
    items = {i["target_key"] for i in job["items"]}
    coverage_ok = (len(items) >= COVERAGE["min_job_targets"]
                   and len(items) >= COVERAGE["min_fraction_of_kept"] * len(kept)
                   and not accounting["unaccounted"] and not accounting["unexpected"]
                   and not accounting["admitted_and_refused"]
                   and all(tokens[k] <= CONTRACT["prompt_token_limit"] for k in items)
                   and all(scans[k]["ok"] for k in items))
    evidence = ProtectedAccessMonitor.evidence(mark)
    versions = {"builder_version": BUILDER_VERSION, "scanner_version": SCANNER_VERSION,
                "builder_sha256": canonical_sha256(ROOT / "harness/native_generated_test_prompt.py"),
                "scanner_sha256": canonical_sha256(ROOT / "harness/native_generated_test_leakage.py")}
    status = {}
    status[PROMPTS] = publish_once(PROMPTS, {
        "schema_version": "oneiros_native_v22_prompt_records", **versions,
        "prompt_token_limit": CONTRACT["prompt_token_limit"], "rows": rows})
    status[JOB] = publish_once(JOB, {
        "schema_version": "oneiros_native_v22_generation_job_v3", "protocols": list(PROTOCOLS),
        **versions, "primary_whole_module": job})
    manifest = {
        "schema_version": "oneiros_native_v22_rehearsal_manifest_v5",
        "nature": "ENGINEERING DRESS REHEARSAL ONLY; not confirmation, generalisation or "
                  "model-selection evidence",
        "source_manifest": {"path": V2_MANIFEST, "sha256": sha(V2_MANIFEST)},
        "protocols": {p: sha(p) for p in PROTOCOLS},
        "requalification_records": {"path": f"{LOCAL}/records.jsonl",
                                    "sha256": sha(f"{LOCAL}/records.jsonl")},
        "isolation": {"receipt": ISOLATION, "receipt_sha256": sha(ISOLATION),
                      **{k: iso[k] for k in ("isolation_version", "reference_universe_sha256")}},
        "superseded": {"v21_job": V21_JOB, "v21_job_sha256": sha(V21_JOB),
                       "note": "preserved unchanged; the v2.1 red preflight correctly caught "
                               "the failed coverage gate under the v2.1 implementation"},
        **versions,
        "prompt_records": {"path": PROMPTS, "sha256": sha(PROMPTS)},
        "per_target": per_target, "kept_targets": [t["key"] for t in kept], "targets": kept,
        "kept": len(kept), "of": len(targets), "repositories": len(repos),
        "rehearsal_rule": {**RULE, "passed": rule_ok},
        "job": {"path": JOB, "sha256": sha(JOB), "job_sha256": job["job_sha256"],
                "items": len(items), "refused": job["refused"], "accounting": accounting,
                "coverage_rule": COVERAGE, "coverage_passed": coverage_ok,
                "max_prompt_tokens": max((tokens[k] for k in items), default=None),
                "truncated_prompts": 0},
        "protected_access_audit": {k: v for k, v in evidence.items() if k != "opens_checked"},
        "admitted_to_training": False, "gpu_used": False,
    }
    status[MANIFEST] = publish_once(MANIFEST, manifest)
    print(json.dumps({"status": status, "kept": len(kept), "repositories": len(repos),
                      "rule": rule_ok, "coverage_passed": coverage_ok,
                      "accounting": accounting, "max_prompt_tokens": manifest["job"]["max_prompt_tokens"],
                      "protected_paths_opened": evidence["protected_paths_opened"]}, indent=1))
    return 0 if rule_ok and coverage_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
