"""Phase 3 correction addendum v2 (2026-09-28): the authoritative current interpretation.

Additive and versioned.  It never modifies the first correction receipt
(results/sft_root_cause_phase3_interpretation_correction_2026-09-28.json), the v1/v2/v3
result receipts, cohorts, generations or design receipts.  It writes:

* the addendum receipt (stable path, deterministic content: it depends only on the
  immutable inputs and the v3 receipts, never on the ledger or state it updates);
* the hypothesis ledger: current statuses, superseding links on historical evidence,
  one new evidence entry per hypothesis (never duplicated);
* the loop state: one unambiguous ``phase3_results.current`` section and history
  sections for v1, v2 and the commit-2e7ddba correction; historical log entries are
  marked, not deleted.

Publication validates everything first, then replaces ledger, state and receipt as a
unit: if any replacement fails, files already replaced are restored and the receipt is
not left published.  Running it again produces byte-identical files.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any

ROOT = Path(__file__).resolve().parent.parent

ADDENDUM = "results/sft_root_cause_phase3_interpretation_correction_addendum_2026-09-28_v2.json"
FIRST_CORRECTION = "results/sft_root_cause_phase3_interpretation_correction_2026-09-28.json"
LEDGER = "results/sft_root_cause_hypotheses.json"
STATE = "results/sft_root_cause_state.json"
RECEIPTS = {
    "v1": {"3A": "results/sft_root_cause_phase3a_result_receipt.json",
           "3C": "results/sft_root_cause_phase3c_result_receipt.json"},
    "v2": {"3A": "results/sft_root_cause_phase3a_result_receipt_v2.json",
           "3C": "results/sft_root_cause_phase3c_result_receipt_v2.json"},
    "v3": {"3A": "results/sft_root_cause_phase3a_result_receipt_v3.json",
           "3C": "results/sft_root_cause_phase3c_result_receipt_v3.json"},
}
EVIDENCE_PHASE = "3-correction-addendum-v2"

CURRENT = {
    "H1": {"status": "open", "qualifier": None,
           "claim": "The prompt may not contain enough information to deduce the correct value.",
           "why": ("A1 and A2 were not run; A3 shows only that this specific oracle-derived "
                   "type/length hint was insufficient; A4 reveals the complete fixed "
                   "implementation and is not an identifying manipulation of public "
                   "specification quality. Current evidence neither supports nor rejects "
                   "natural-language specification insufficiency.")},
    "H2": {"status": "strengthened", "qualifier": "not_supported",
           "why": ("Phase 2 showed input-choice and assertion-form changes. Whether SFT improves "
                   "fixed-input value prediction is schema-dependent and unresolved (Phase 3A "
                   "v3). Causal support still requires a valid Phase 4 objective intervention.")},
    "H3": {"status": "strengthened", "qualifier": "model_scale_association_only",
           "subclaims": {
               "model_scale_association": {
                   "status": "strengthened",
                   "statement": ("Within the Qwen2.5-Coder family, 7B beats 1.5B on this "
                                 "fixed-input panel (A0 +11.46, CI [6.46, 16.67]; A4 +24.79, "
                                 "CI [19.38, 30.42]; recomputed answer-rate gate passed).")},
               "adapter_or_parameter_capacity_causal": {
                   "status": "open",
                   "statement": ("LoRA rank, adapter capacity, full fine-tuning and parameter "
                                 "count as causal factors are untested; the two checkpoints "
                                 "differ in scale and learned representation together.")}},
           "why": "7B beats 1.5B on this panel; no causal parameter-count or capacity claim."},
    "H4": {"status": "open", "qualifier": "schema_dependent_exposure_result",
           "why": ("On exposed functions the SFT effect is -1.67 points under assertion-prefill "
                   "and +5.00 under the ANSWER schema; the route-exact rule stops at the exposed "
                   "schema check. Distribution shift is untested: no external diagnostic "
                   "cohort exists.")},
    "H5": {"status": "weakened", "qualifier": "narrowed_syntactic_structural_scope",
           "why": "Syntactic/structural contract degradation only; semantic validity unresolved."},
    "H6": {"status": "weakened", "qualifier": "narrowed_progressive_degradation_scope",
           "why": ("Progressive degradation after step 100 only; a rapid early shift or "
                   "forgetting remains open.")},
    "H7": {"status": "strengthened", "qualifier": "uncertainty_power_diagnosis",
           "why": ("An uncertainty/power diagnosis (Kill@8 +3.43, clustered CI [-4.03, 10.88]); "
                   "not proof of a positive final effect.")},
}

#: Historical ledger evidence superseded by this addendum: (hypothesis, phase) -> reason.
SUPERSEDED = {
    ("H1", "3A"): "described A3 as non-revealing context",
    ("H1", "3C"): "read A4 as specification insufficiency; 'specification became the larger "
                  "gap at 7B'",
    ("H1", "3-correction"): "set H1 strengthened; H1 is open",
    ("H2", "3A"): "'SFT did not improve output prediction even on its training inputs'",
    ("H2", "3-correction"): "replaced by the addendum wording (strengthened, not supported)",
    ("H3", "3A"): "'1.5B execution reasoning is limited' framed as a capacity finding",
    ("H3", "3C"): "'1.5B execution reasoning is capacity-limited' (causal claim)",
    ("H3", "3-correction"): "reported Holm 'p <= 0.0002' as a bound; replaced by v3 Monte Carlo "
                            "wording and explicit subclaims",
    ("H4", "3A"): "'H4 weakened' from a schema-dependent exposed result",
    ("H4", "3-correction"): "replaced by the route-exact v3 decision (same status: open)",
}

WITHDRAWN_STATEMENTS = [
    "SFT did not improve output prediction even on its training functions.",
    "H4 was weakened (Phase 3A).",
    "A3 is non-revealing context.",
    "A0/specification insufficiency follows from A4.",
    "1.5B execution semantics were causally capacity-limited.",
    "Specification became the larger gap at 7B.",
    "Bootstrap p-values were zero (Phase 3C v1: bootstrap_p_two_sided 0.0, holm_p 0.0).",
    "H1 strengthened (first correction receipt, commit 2e7ddba).",
    "Holm 'p <= 0.0002' as a bound (first correction receipt).",
]


def _sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _dump(obj: Any) -> bytes:
    return (json.dumps(obj, indent=1, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")


def build_receipt(root: Path) -> bytes:
    def sha(path):
        return _sha_bytes((root / path).read_bytes())
    receipt = {
        "schema_version": "oneiros_sft_root_cause_phase3_correction_addendum_v2",
        "date": "2026-09-28",
        "model_calls": 0, "gpu_used": False, "training": False,
        "restricted_splits_accessed": "none",
        "supersedes_interpretation_of": {FIRST_CORRECTION: sha(FIRST_CORRECTION)},
        "first_correction_modified": False,
        "current_result_receipts": {k: {"path": p, "sha256": sha(p)}
                                    for k, p in RECEIPTS["v3"].items()},
        "history_result_receipts": {version: {k: {"path": p, "sha256": sha(p)}
                                              for k, p in paths.items()}
                                    for version, paths in RECEIPTS.items() if version != "v3"},
        "hypotheses_current": CURRENT,
        "terminal_support": "none: no hypothesis is terminally supported",
        "superseded_ledger_evidence": [{"hypothesis": h, "phase": p, "reason": r}
                                       for (h, p), r in sorted(SUPERSEDED.items())],
        "withdrawn_statements": WITHDRAWN_STATEMENTS,
        "numerical_changes": ("none in any primary or stratum estimate; four exploratory 3C "
                              "subgroup intervals move by <= 0.3 points (subgroup-only "
                              "resampling, documented in the 3C v3 receipt)"),
    }
    return _dump(receipt)


def apply_ledger(ledger: dict, receipt_path: str, receipt_sha: str) -> dict:
    ledger = copy.deepcopy(ledger)
    for h in ledger["hypotheses"]:
        for e in h["evidence"]:
            reason = SUPERSEDED.get((h["id"], str(e["phase"])))
            if reason:
                e["withdrawn"] = True
                e.setdefault("withdrawn_by", receipt_path)
                e["superseded_by"] = receipt_path
                e["superseded_by_sha256"] = receipt_sha
                e["superseded_reason"] = reason
                e.setdefault("status_after_as_recorded", e["status_after"])
        current = CURRENT[h["id"]]
        h["status"], h["qualifier"] = current["status"], current["qualifier"]
        if "subclaims" in current:
            h["subclaims"] = current["subclaims"]
        if not any(e["phase"] == EVIDENCE_PHASE for e in h["evidence"]):
            h["evidence"].append({"phase": EVIDENCE_PHASE, "receipt": receipt_path,
                                  "receipt_sha256": receipt_sha,
                                  "status_after": current["status"],
                                  "qualifier": current["qualifier"], "why": current["why"]})
    ledger["terminal_support"] = "none"
    return ledger


def apply_state(state: dict, root: Path, receipt_path: str, receipt_sha: str) -> dict:
    def sha(path):
        return _sha_bytes((root / path).read_bytes())
    state = copy.deepcopy(state)
    phase3 = state["phases"][3]
    legacy_keys = [k for k in list(phase3) if k.startswith(("phase3a_result", "phase3c_result"))
                   or k == "result_receipts"]
    if legacy_keys:
        phase3.setdefault("history_as_recorded_before_v3", {})
        for key in legacy_keys:
            phase3["history_as_recorded_before_v3"][key] = phase3.pop(key)
    state["phase3_results"] = {
        "current": {"3A": {"path": RECEIPTS["v3"]["3A"], "sha256": sha(RECEIPTS["v3"]["3A"])},
                    "3C": {"path": RECEIPTS["v3"]["3C"], "sha256": sha(RECEIPTS["v3"]["3C"])},
                    "interpretation": {"path": receipt_path, "sha256": receipt_sha}},
        "history": {
            "v1": {k: {"path": p, "sha256": sha(p)} for k, p in RECEIPTS["v1"].items()},
            "v2": {k: {"path": p, "sha256": sha(p)} for k, p in RECEIPTS["v2"].items()},
            "commit_2e7ddba_correction": {"path": FIRST_CORRECTION,
                                          "sha256": sha(FIRST_CORRECTION)}},
    }
    events = [e.get("event") for e in state["log"]]
    own = "phase 3 correction addendum v2"
    # Only entries recorded BEFORE this addendum are history; later entries (e.g. the
    # Phase 4 redesign) are left untouched so a rerun stays a no-op.
    cutoff = events.index(own) if own in events else len(state["log"])
    for entry in state["log"][:cutoff]:
        entry["historical"] = True
        if str(entry.get("event", "")).startswith(("phase 3", "phase 3A", "phase 3C")):
            entry["superseded_by"] = receipt_path
    if not any(e.get("event") == "phase 3 correction addendum v2" for e in state["log"]):
        state["log"].append({
            "utc": "2026-09-28", "event": "phase 3 correction addendum v2",
            "receipt": receipt_path, "receipt_sha256": receipt_sha,
            "finding": ("H1 open; H2 strengthened (not supported); H3 strengthened as a "
                        "model-scale association only; H4 open (schema-dependent exposure "
                        "result); no hypothesis terminally supported"),
            "current": "phase3_results.current"})
    return state


def validate(ledger: dict, state: dict, root: Path, receipt_path: str,
             receipt_bytes: bytes) -> None:
    allowed = set(ledger["status_values"])
    receipt_sha = _sha_bytes(receipt_bytes)
    for h in ledger["hypotheses"]:
        if h["status"] not in allowed:
            raise ValueError(f"{h['id']} status {h['status']!r} undeclared")
        for sub in (h.get("subclaims") or {}).values():
            if sub["status"] not in allowed:
                raise ValueError(f"{h['id']} subclaim status undeclared")
        phases = [e["phase"] for e in h["evidence"]]
        if len(phases) != len(set(phases)):
            raise ValueError(f"{h['id']} has duplicate evidence entries")
        for e in h["evidence"]:
            if e["status_after"] not in allowed:
                raise ValueError(f"{h['id']} evidence status {e['status_after']!r} undeclared")
            if e.get("superseded_by"):
                if e["superseded_by"] != receipt_path or e["superseded_by_sha256"] != receipt_sha:
                    raise ValueError(f"{h['id']} superseding link does not resolve")
            linked = e.get("receipt")
            if linked and linked != receipt_path and not (root / linked).exists():
                raise ValueError(f"{h['id']} evidence receipt {linked} missing")
    if CURRENT["H1"]["status"] != "open":
        raise ValueError("H1 must be open")
    current = state["phase3_results"]["current"]
    for key in ("3A", "3C"):
        if current[key]["path"] != RECEIPTS["v3"][key]:
            raise ValueError("current pointers must reference v3")
        if _sha_bytes((root / current[key]["path"]).read_bytes()) != current[key]["sha256"]:
            raise ValueError("current pointer hash mismatch")
    for key in ("phase3a_result_receipt", "phase3c_result_receipt", "result_receipts"):
        if key in state["phases"][3]:
            raise ValueError(f"legacy top-level pointer {key} remains")


def publish_many(root: Path, files: dict[str, bytes]) -> None:
    """Replace several files as a unit; on any failure restore what was already replaced."""
    staged, originals, done = {}, {}, []
    try:
        for rel, data in files.items():
            target = root / rel
            originals[rel] = target.read_bytes() if target.exists() else None
            handle, temp = tempfile.mkstemp(dir=target.parent, prefix=target.name + ".")
            with os.fdopen(handle, "wb") as out:
                out.write(data)
            staged[rel] = temp
        for rel in files:
            os.replace(staged[rel], root / rel)
            done.append(rel)
    except BaseException:
        for rel in done:
            if originals[rel] is None:
                (root / rel).unlink(missing_ok=True)
            else:
                (root / rel).write_bytes(originals[rel])
        raise
    finally:
        for temp in staged.values():
            if os.path.exists(temp):
                os.unlink(temp)


def run(root: Path = ROOT) -> dict[str, str]:
    receipt_bytes = build_receipt(root)
    receipt_sha = _sha_bytes(receipt_bytes)
    existing = root / ADDENDUM
    if existing.exists() and existing.read_bytes() != receipt_bytes:
        raise SystemExit("REFUSED: a different addendum already exists at the stable path")
    ledger = apply_ledger(json.loads((root / LEDGER).read_text(encoding="utf-8")), ADDENDUM,
                          receipt_sha)
    state = apply_state(json.loads((root / STATE).read_text(encoding="utf-8")), root, ADDENDUM,
                        receipt_sha)
    validate(ledger, state, root, ADDENDUM, receipt_bytes)
    # Ledger and state first, receipt last: a failure never leaves a published receipt
    # whose ledger/state updates are missing.
    publish_many(root, {LEDGER: _dump(ledger), STATE: _dump(state), ADDENDUM: receipt_bytes})
    return {"addendum": ADDENDUM, "sha256": receipt_sha}


def main() -> int:
    print(json.dumps(run(), indent=1))
    ledger = json.loads((ROOT / LEDGER).read_text(encoding="utf-8"))
    for h in ledger["hypotheses"]:
        print(h["id"], h["status"], h.get("qualifier"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
