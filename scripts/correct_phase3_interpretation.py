"""Dated Phase 3 interpretation-correction receipt (2026-09-28), CPU only.

Writes results/sft_root_cause_phase3_interpretation_correction_2026-09-28.json
and applies it to the hypothesis ledger and the loop state.  Nothing is deleted:

* the Phase 3A/3C v1 receipts, generations, completion receipts and cohort are
  hashed and left unchanged;
* ledger evidence that recorded an undeclared status or a withdrawn claim keeps
  its text and gains ``withdrawn``/``withdrawn_by`` (an undeclared status is
  kept verbatim in ``status_after_as_recorded`` and ``status_after`` is mapped to
  the declared value that replaces it);
* corrected evidence is appended.
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

OUTPUT = "results/sft_root_cause_phase3_interpretation_correction_2026-09-28.json"
LEDGER = "results/sft_root_cause_hypotheses.json"
STATE = "results/sft_root_cause_state.json"
V1 = {"3A": "results/sft_root_cause_phase3a_result_receipt.json",
      "3C": "results/sft_root_cause_phase3c_result_receipt.json"}
V2 = {"3A": "results/sft_root_cause_phase3a_result_receipt_v2.json",
      "3C": "results/sft_root_cause_phase3c_result_receipt_v2.json"}
UNCHANGED = ["results/sft_root_cause_phase3a_cohort.json",
             "results/sft_root_cause_phase3a_design_receipt.json",
             "results/sft_root_cause_phase3c_design_receipt.json",
             "results/sft_root_cause/phase3a/generations_base.jsonl",
             "results/sft_root_cause/phase3a/generations_arm_a_431.jsonl",
             "results/sft_root_cause/phase3c/generations_qwen7b_base.jsonl",
             "results/sft_root_cause/phase3a/completion_base.json",
             "results/sft_root_cause/phase3a/completion_arm_a_431.json",
             "results/sft_root_cause/phase3c/completion_qwen7b_base.json"]

#: (hypothesis, evidence phase) -> why the entry is withdrawn.
WITHDRAWN = {
    ("H4", "3A"): "exposure verdict violated the frozen schema-control rule (exposed stratum "
                  "flips sign between schemas)",
    ("H2", "3A"): "its fixed-input claim ('no improvement ... including literal training "
                  "targets') is schema-dependent in the exposed strata",
    ("H1", "3A"): "described A3 as non-revealing context; A3 is an oracle-derived partial hint",
    ("H1", "3C"): "undeclared status 'scale_dependent'; treated A4 as specification evidence",
    ("H3", "3C"): "undeclared status 'supported_at_model_scale'; a checkpoint-scale contrast "
                  "does not establish parameter-count or adapter capacity",
}
#: Declared status that replaces an undeclared recorded one.
STATUS_MAP = {"scale_dependent": "strengthened", "supported_at_model_scale": "strengthened"}

CORRECTED = {
    "H1": {"status": "strengthened", "qualifier": "scale_qualification",
           "why": ("A3 (oracle-derived type/length hint) did not measurably help at 1.5B or 7B: "
                   "this particular hint was insufficient, nothing more. A4 (access to the fixed "
                   "implementation) helps modestly at 1.5B and strongly at 7B (+19.4, CI [14.8, "
                   "24.2]); that shows access to the correct implementation improves prediction, "
                   "not that the natural-language specification is under-specified. A1/A2 were "
                   "not run, so richer public context remains untested.")},
    "H2": {"status": "strengthened", "qualifier": None,
           "why": ("Rests on Phase 2 (input choice and assertion form moved). Whether SFT "
                   "improves fixed-input output prediction is schema-dependent and unresolved "
                   "(Phase 3A v2).")},
    "H3": {"status": "strengthened", "qualifier": "model_scale_association",
           "why": ("Within the Qwen2.5-Coder family, 7B beats 1.5B on fixed-input prediction: A0 "
                   "+11.46 [6.46, 16.67], A4 +24.79 [19.38, 30.42], answer-rate gate passed, "
                   "sign-flip Holm p <= 0.0002. The checkpoints differ in scale and learned "
                   "representation together; adapter rank and full fine-tuning are untested.")},
    "H4": {"status": "open", "qualifier": "schema_dependent_exposure_result",
           "why": ("On exposed functions, the estimated SFT effect changes from -1.67 points under "
                   "assertion-prefill to +5.00 points under the ANSWER schema. Therefore the "
                   "exposure result is answer-schema dependent and cannot currently support or "
                   "weaken memorization or distribution shift. Distribution shift stays open: no "
                   "external diagnostic cohort exists.")},
}


def sha(path: str) -> str:
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def main() -> int:
    v2 = {k: json.loads((ROOT / p).read_text(encoding="utf-8")) for k, p in V2.items()}
    strata = v2["3A"]["strata"]
    receipt = {
        "schema_version": "oneiros_sft_root_cause_phase3_correction_v1",
        "date": "2026-09-28",
        "model_calls": 0, "gpu_used": False, "training": False,
        "v1_receipts_unchanged": {k: {"path": p, "sha256": sha(p)} for k, p in V1.items()},
        "v2_receipts": {k: {"path": p, "sha256": sha(p)} for k, p in V2.items()},
        "unchanged_artifacts": {p: sha(p) for p in UNCHANGED},
        "defects_corrected": [
            "3A v1 checked the frozen schema-control rule only on the pooled stratum but decided "
            "H4 from the exposed stratum, where the SFT effect flips sign",
            "ledger statuses 'scale_dependent' and 'supported_at_model_scale' were not in "
            "status_values",
            "3C capacity claim stated as causal although two different pretrained checkpoints "
            "were compared",
            "A3 was described as non-revealing context; it is an oracle-derived partial hint",
            "A4 was read as evidence of specification insufficiency; it gives access to the full "
            "fixed implementation",
            "3C v1 reported bootstrap tail proportions as p = 0.0",
        ],
        "withdrawn_claims": [
            "SFT did not improve output prediction even on its training functions.",
            "H4 is weakened (based on Phase 3A).",
            "A3 is non-revealing context / non-revealing missing context does not matter.",
            "A0 is under-specified for value prediction (from A4 alone).",
            "H3 supported: 1.5B execution reasoning is capacity-limited.",
            "bootstrap_p_two_sided: 0.0 and holm_p: 0.0 (Phase 3C v1).",
        ],
        "schema_dependence_by_stratum": {
            name: {"prefill": s["prefill"]["difference_points"],
                   "answer": s["answer"]["difference_points"],
                   "interaction": s["schema_interaction_answer_minus_prefill"]["interaction_points"],
                   "interaction_ci95": s["schema_interaction_answer_minus_prefill"]["ci95_points"],
                   "schema_dependent": s["schema_dependent"]} for name, s in strata.items()},
        "relabels": {
            "A3": ("diagnostic, oracle-derived partial behavioural hint containing the correct "
                   "result type and, where sized, its correct length; does not expose the "
                   "complete value; non-deployable; prohibited from training and confirmation"),
            "A4": "diagnostic access to the complete fixed implementation; non-deployable",
            "note": ("the frozen v1 cohort and design receipts keep their original wording so their "
                     "hashes stay valid; this receipt supersedes that wording"),
        },
        "corrected_hypotheses": {h: {k: v for k, v in c.items()} for h, c in CORRECTED.items()},
        "unchanged_hypotheses": {"H5": "weakened (syntactic/structural contract only)",
                                 "H6": "weakened (progressive degradation after step 100 only)",
                                 "H7": "strengthened"},
        "corrected_interpretation": [
            "SFT clearly changes input choice and assertion form.",
            "Whether it improves fixed-output prediction on exposed training inputs is "
            "schema-dependent and unresolved.",
            "Qwen 7B is substantially stronger than Qwen 1.5B on fixed-input prediction; this is a "
            "model-scale association, not isolated proof of adapter capacity.",
            "Partial oracle information consisting only of type/length does not help.",
            "Full access to the fixed implementation helps, particularly at 7B, but does not by "
            "itself establish natural-language specification insufficiency.",
            "No root cause is yet terminally supported.",
        ],
    }
    publish_file_atomically(ROOT / OUTPUT, (json.dumps(receipt, indent=1) + "\n").encode("utf-8"))
    digest = sha(OUTPUT)

    ledger = json.loads((ROOT / LEDGER).read_text(encoding="utf-8"))
    allowed = set(ledger["status_values"])
    for h in ledger["hypotheses"]:
        for e in h["evidence"]:
            why = WITHDRAWN.get((h["id"], str(e["phase"])))
            if why and not e.get("withdrawn"):
                e["withdrawn"], e["withdrawn_by"], e["withdrawn_reason"] = True, OUTPUT, why
            if e["status_after"] not in allowed:
                e["status_after_as_recorded"] = e["status_after"]
                e["status_after"] = STATUS_MAP[e["status_after"]]
        fix = CORRECTED.get(h["id"])
        if fix and not any(e["phase"] == "3-correction" for e in h["evidence"]):
            h["status"] = fix["status"]
            h["qualifier"] = fix["qualifier"]
            h["evidence"].append({"phase": "3-correction", "receipt": OUTPUT,
                                  "receipt_sha256": digest, "status_after": fix["status"],
                                  "qualifier": fix["qualifier"], "why": fix["why"]})
    bad = [(h["id"], h["status"]) for h in ledger["hypotheses"] if h["status"] not in allowed]
    if bad:
        raise SystemExit(f"REFUSED: undeclared statuses remain {bad}")
    (ROOT / LEDGER).write_text(json.dumps(ledger, indent=1), encoding="utf-8")

    state = json.loads((ROOT / STATE).read_text(encoding="utf-8"))
    phase3 = state["phases"][3]
    phase3["result_receipts"] = {
        "current": {k: {"path": p, "sha256": sha(p)} for k, p in V2.items()},
        "history_v1": {k: {"path": p, "sha256": sha(p), "superseded_by": V2[k]}
                       for k, p in V1.items()},
        "correction": {"path": OUTPUT, "sha256": digest}}
    if not any(entry.get("event", "").startswith("phase 3 correction") for entry in state["log"]):
        state["log"].append({
            "utc": "2026-09-28", "event": "phase 3 correction (CPU only, v2 analyses)",
            "correction_receipt": OUTPUT, "correction_sha256": digest,
            "finding": "H4 open (schema-dependent exposure result); H1/H3 strengthened with "
                       "qualifiers; no root cause terminally supported",
            "next_decision": "Phase 4 data preflight and design draft; training awaits "
                             "authorisation"})
    (ROOT / STATE).write_text(json.dumps(state, indent=1), encoding="utf-8")
    print(json.dumps({"receipt": OUTPUT, "sha256": digest,
                      "statuses": {h["id"]: [h["status"], h.get("qualifier")]
                                   for h in ledger["hypotheses"]}}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
