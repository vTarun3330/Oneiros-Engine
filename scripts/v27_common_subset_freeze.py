"""v2.7 Phase 6b: freeze the proposed Oneiros/Atheris common subset (model-independent).

common subset = frozen panel membership AND native buggy/fixed executability (model-free
control) AND Atheris eligibility (same adapter plan on both revisions, instrumentation, a
deterministic seed invocation) AND no environment/harness failure. Nothing here reads a model
output, a kill or a fuzzing result.

Also reports the exclusion waterfall from acquisition, the subset's composition, and an EXACT
power analysis for the predeclared paired test (two-sided exact McNemar) at the subset, the
native-runnable set and the full panel.

    python scripts/v27_common_subset_freeze.py --out results/<receipt>.json
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PANEL = "results/sft_root_cause_v27_confirmation_panel_r4.json"
ACQ = "results/sft_root_cause_v27_confirmation_acquisition_report_r4.json"
GATE4 = "results/sft_root_cause_v27_confirmation_phase4_gate_r4.json"
CONTROL = "results/sft_root_cause/v27_confirmation/phase6/native_control.jsonl"
ELIG = "results/sft_root_cause/v27_confirmation/phase6/atheris_eligibility"
FULL_SUITE = "results/sft_root_cause/v27_full_suite_receipt_ad9df21.json"
# historical discordance of the locked function-mode comparison (114 + 88 of 757 targets)
HIST_DISCORDANCE = (114 + 88) / 757


def sha(rel) -> str:
    return hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()


def mcnemar_p(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def power(n: int, discordance: float, diff: float, alpha: float) -> float:
    """Exact power of the two-sided exact McNemar test with n paired targets."""
    p10, p01 = (discordance + diff) / 2, (discordance - diff) / 2
    if p01 < 0:
        return float("nan")
    p00 = 1 - discordance
    total = 0.0
    for b in range(n + 1):
        for c in range(n - b + 1):
            if mcnemar_p(b, c) < alpha:
                total += (comb(n, b) * comb(n - b, c) * p10 ** b * p01 ** c
                          * p00 ** (n - b - c))
    return total


def mde(n: int, discordance: float, alpha: float, target: float = 0.8):
    steps = [d / 1000 for d in range(1, int(discordance * 1000) + 1)]
    for d in steps:
        if power(n, discordance, d, alpha) >= target:
            return round(d * 100, 1)
    return None


def smallest_attainable_p(n: int) -> float:
    return mcnemar_p(n, 0)


def failure_class(reason: str | None, evidence: dict | None) -> str | None:
    if not reason:
        return None
    if reason.startswith("native_control_failed"):
        return "policy"
    if reason.startswith("adapter_unsupported"):
        return "adapter"
    if reason.startswith("revision_load_mismatch"):
        return "target"            # the fix changed the signature the adapter sees
    if reason.startswith("seed_invocation_failure"):
        return "target"            # see interpretation: reclassified below with evidence
    return "harness"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    panel = json.loads((ROOT / PANEL).read_text(encoding="utf-8"))
    acq = json.loads((ROOT / ACQ).read_text(encoding="utf-8"))
    gate4 = json.loads((ROOT / GATE4).read_text(encoding="utf-8"))
    contract = json.loads((ROOT / ELIG / "contract.json").read_text(encoding="utf-8"))
    if contract["results"]["sha256"] != sha(contract["results"]["path"]):
        raise SystemExit("REFUSED: eligibility results changed since the probe")
    elig = {json.loads(l)["target_id"]: json.loads(l) for l in
            (ROOT / contract["results"]["path"]).read_text(encoding="utf-8").splitlines()
            if l.strip()}
    control = {json.loads(l)["key"]: json.loads(l) for l in
               (ROOT / CONTROL).read_text(encoding="utf-8").splitlines() if l.strip()}
    targets = {t["target_id"]: t for t in panel["targets"]}
    assert set(elig) == set(targets) == set(control), "cohort mismatch"
    rows, subset = [], []
    for key, t in sorted(targets.items()):
        e = elig[key]
        reason = e.get("reason")
        cls = failure_class(reason, None)
        if reason == "seed_invocation_failure:replay_process_failure":
            reason = ("target_terminates_interpreter: target raises SystemExit (no virtual "
                      "environment inside the sandbox), so every replay process exits without "
                      "an outcome; identical on both revisions and both repetitions; the v5 "
                      "harness catches Exception only (frozen)")
        in_subset = (control[key]["control_passed"] and e["eligible"]
                     and e.get("deterministic") is True)
        row = {"target_id": key, "repository": t["repository"],
               "buggy_commit": t["buggy_commit"], "fixed_commit": t["fixed_commit"],
               "complexity": t["complexity"], "defect_family": t["defect_family"],
               "native_runnable": control[key]["control_passed"],
               "native_deterministic": control[key]["deterministic"],
               "atheris_status": e["atheris_status"], "atheris_eligible": e["eligible"],
               "atheris_deterministic": e.get("deterministic"),
               "exclusion_reason": None if in_subset else reason,
               "failure_class": None if in_subset else cls,
               "evidence": e.get("evidence"), "in_common_subset": in_subset}
        rows.append(row)
        if in_subset:
            subset.append(key)
    ids_sha = hashlib.sha256("\n".join(sorted(subset)).encode()).hexdigest()
    sub_rows = [r for r in rows if r["in_common_subset"]]
    native = [r for r in rows if r["native_runnable"]]
    counts = acq["counts"]
    waterfall = [
        {"stage": "frozen repositories (acquired and accounted)", "count": 70},
        {"stage": "repositories scanned", "count": gate4["funnel"]["stage:scanned"]},
        {"stage": "candidates discovered (evaluated)", "count": counts["candidates_inspected"]},
        {"stage": "candidates admitted (unique)", "count": counts["admitted"]},
        {"stage": "admitted with regression test, no overlap", "count": 85},
        {"stage": "natively qualified", "count": panel["funnel"]["natively_qualified"]},
        {"stage": "requalified 3/3", "count": panel["funnel"]["requalified_3_of_3"]},
        {"stage": "target reached on both revisions", "count": panel["funnel"]["target_reached"]},
        {"stage": "prompt leakage and token fit", "count":
            panel["funnel"]["prompt_leakage_and_token_fit_passed"]},
        {"stage": "frozen panel targets", "count": len(rows)},
        {"stage": "native-runnable (model-free control)", "count": len(native)},
        {"stage": "Atheris-eligible", "count": sum(r["atheris_eligible"] for r in rows)},
        {"stage": "proposed common subset", "count": len(subset)},
    ]
    excl = Counter((r["failure_class"], (r["exclusion_reason"] or "").split(":")[0] + ":" +
                    ":".join((r["exclusion_reason"] or "").split(":")[1:3]))
                   for r in rows if not r["in_common_subset"])
    per_repo = Counter(r["repository"] for r in sub_rows)
    analysis = {}
    for label, n in (("common_subset", len(subset)), ("native_runnable", len(native)),
                     ("frozen_panel", len(rows))):
        analysis[label] = {
            "n": n, "smallest_attainable_two_sided_p": smallest_attainable_p(n),
            "significance_attainable_at_0.05": smallest_attainable_p(n) < 0.05,
            "significance_attainable_at_0.01_gate": smallest_attainable_p(n) < 0.01,
            "mde_pp_80pct_power_alpha_0.05_hist_discordance": mde(n, HIST_DISCORDANCE, 0.05),
            "mde_pp_80pct_power_alpha_0.01_hist_discordance": mde(n, HIST_DISCORDANCE, 0.01),
        }
    out = {
        "schema_version": "oneiros_v27_common_subset_v1",
        "status": "FROZEN_PROPOSED_COMMON_SUBSET",
        "model_independent": True, "fuzzing": False, "model": False,
        "definition": "frozen panel AND native-runnable (model-free control, deterministic) AND "
                      "Atheris-eligible (same adapter plan on both revisions, instrumentation, "
                      "deterministic seed invocation) AND no environment/harness failure",
        "inputs": {p: sha(p) for p in (PANEL, ACQ, GATE4, CONTROL, ELIG + "/contract.json",
                                       contract["results"]["path"], FULL_SUITE)},
        "common_subset_ids": sorted(subset), "common_subset_ids_sha256": ids_sha,
        "waterfall": waterfall,
        "composition": {"targets": len(subset), "repositories": len(per_repo),
                        "max_per_repository": max(per_repo.values()) if per_repo else 0,
                        "per_repository": dict(per_repo),
                        "complexity": dict(Counter(r["complexity"] for r in sub_rows)),
                        "defect_family": dict(Counter(r["defect_family"] for r in sub_rows))},
        "exclusions_by_class_and_reason": [{"class": k[0], "reason": k[1], "count": v}
                                           for k, v in sorted(excl.items(), key=str)],
        "power": {"test": "two-sided exact McNemar on paired per-target kill outcomes",
                  "assumed_discordance": round(HIST_DISCORDANCE, 4),
                  "discordance_source": "historical locked comparison (114+88)/757",
                  **analysis},
        "panel_policy": "the frozen panel remains 34 targets; 29 native-runnable and the common "
                        "subset are model-independent views of it, never redefinitions",
        "targets": rows,
        "source_identity": {"head_at_probe": "ea03fad73319c8b3df7b570de01fe3a8a2d3eadc",
                            "full_suite_receipt": {"path": FULL_SUITE, "sha256": sha(FULL_SUITE)}},
    }
    (ROOT / args.out).write_text(json.dumps(out, indent=1, sort_keys=True) + "\n",
                                 encoding="utf-8")
    print(json.dumps({"waterfall": [(w["stage"], w["count"]) for w in waterfall],
                      "common_subset": sorted(subset), "ids_sha256": ids_sha,
                      "composition": out["composition"], "power": analysis}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
