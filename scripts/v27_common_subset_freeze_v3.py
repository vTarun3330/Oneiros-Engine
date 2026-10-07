"""v2.7 Phase 6b successor v3: freeze the Atheris adapter-covered descriptive subset from the
committed v3 evidence bundle, RECOMPUTING meaningful fuzzability from the committed per-seed
stage records instead of trusting stored Booleans. Supersedes v2 (preserved unchanged).

v3 adds to v2:
- eligibility, determinism, fuzz-input consumption, target entry, receiver/argument building and
  interpreter-terminating outcomes are recomputed from ``atheris_v2_structured_runs`` with the
  probe's own decision function (scripts/v27_atheris_eligibility_v2_wsl.py:evaluate); any
  disagreement with the stored flags refuses the freeze;
- the historical planning discordance (114 gained, 88 lost, 757 paired functions) is read from
  its committed source receipt at a fixed JSON pointer and bound by SHA-256;
- every statistical sentence is derived from the actual receipt counts;
- the output is never overwritten.

    python scripts/v27_common_subset_freeze_v3.py --out results/<receipt>.json
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PANEL = "results/sft_root_cause_v27_confirmation_panel_r4.json"
ACQ = "results/sft_root_cause_v27_confirmation_acquisition_report_r4.json"
GATE4 = "results/sft_root_cause_v27_confirmation_phase4_gate_r4.json"
BUNDLE = "results/sft_root_cause_v27_phase6_evidence_bundle_v3.json"
PREVIOUS = "results/sft_root_cause_v27_common_subset_r4_v2.json"
HISTORICAL = "results/sft_root_cause_phase2_decomposition_receipt.json"
HISTORICAL_POINTER = ("primary_locked_validation", "function_transitions")
FLAGS = ("receiver_built", "arguments_built", "target_invoked", "consumes_fuzz_input",
         "deterministic", "v5_terminating", "eligible_for_fuzzing", "reason", "failure_class")


def _load_module(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


v2 = _load_module("v27_freeze_v2", "scripts/v27_common_subset_freeze_v2.py")
probe = _load_module("v27_probe_v2", "scripts/v27_atheris_eligibility_v2_wsl.py")
refuse, sha, load, stats = v2.refuse, v2.sha, v2.load, v2.stats


def historical_discordance() -> dict:
    data = load(HISTORICAL)
    node = data
    for part in HISTORICAL_POINTER:
        refuse(part not in node, f"historical receipt lacks {'.'.join(HISTORICAL_POINTER)}")
        node = node[part]
    gained, lost, n = node["killed"]["gained"], node["killed"]["lost"], node["paired_functions"]
    refuse(not (isinstance(gained, int) and isinstance(lost, int) and isinstance(n, int)
                and 0 < gained + lost <= n), "historical transition counts malformed")
    return {"gained": gained, "lost": lost, "paired": n,
            "discordance": (gained + lost) / n,
            "source": {"path": HISTORICAL, "sha256": sha(HISTORICAL),
                       "json_pointer": "/" + "/".join(HISTORICAL_POINTER) + "/{killed,paired_functions}"}}


def runs_for_evaluate(structured: dict) -> dict:
    """Committed per-seed records in the probe's own run format (the outcome is represented by
    its kind and full SHA-256, which preserves the probe's equality semantics)."""
    out = {}
    for label, runs in structured.items():
        out[label] = []
        for r in runs:
            if not r["normal_exit"]:
                out[label].append({"normal_exit": False, "result": None})
                continue
            seeds = [{"receiver_built": s["receiver_built"], "arguments_built": s["arguments_built"],
                      "consumed_bytes": s["consumed_bytes"], "target_invoked": s["target_invoked"],
                      "outcome": [s["outcome_kind"], s["outcome_sha256"]]} for s in r["seeds"]]
            out[label].append({"normal_exit": True, "result": {"seeds": seeds}})
    return out


def recompute(atheris: dict, structured: dict | None) -> dict:
    """Recompute every eligibility flag; refuse on any disagreement with the stored values."""
    if not atheris.get("adapter_covered"):
        refuse(atheris.get("eligible_for_fuzzing") is True,
               "eligible_for_fuzzing without adapter coverage")
        return {"eligible_for_fuzzing": False, "recomputed": False}
    refuse(structured is None, "adapter-covered target without committed per-seed evidence")
    decision = probe.evaluate(runs_for_evaluate(structured))
    stored = {k: atheris.get(k) for k in FLAGS}
    fresh = {k: decision.get(k) for k in FLAGS}
    differ = sorted(k for k in FLAGS if stored[k] != fresh[k])
    refuse(bool(differ), f"stored eligibility flags disagree with the committed evidence: {differ}")
    return {**fresh, "recomputed": True}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    out_path = ROOT / args.out
    refuse(out_path.exists(), f"{args.out} exists (accepted receipts are never overwritten)")
    panel, acq, gate4, bundle = load(PANEL), load(ACQ), load(GATE4), load(BUNDLE)
    refuse(bundle.get("schema_version") != "oneiros_v27_phase6_evidence_bundle_v3",
           "not a v3 evidence bundle")
    refuse(bundle["panel"]["sha256"] != sha(PANEL), "evidence bundle binds a different panel")
    refuse(gate4.get("status") != "PASS", "Phase 4 acquisition gate did not pass")
    targets = {t["target_id"]: t for t in panel["targets"]}
    ev = {t["target_id"]: t for t in bundle["targets"]}
    refuse(set(ev) != set(targets), "evidence bundle does not cover exactly the frozen panel")
    accounted = gate4["checks"]["all_frozen_repositories_accounted"]["detail"]
    refuse(bool(accounted["unaccounted"]), "Phase 4 left repositories unaccounted")
    rows = []
    for key in sorted(targets):
        t, e = targets[key], ev[key]
        a = e["atheris_v2"]
        fresh = recompute(a, e.get("atheris_v2_structured_runs"))
        reach_ok = e["native_reach"]["reach"] == "reach_verified"
        sandbox_ok = e["sandbox_control"]["passed"] and e["sandbox_control"]["deterministic"]
        native = reach_ok and sandbox_ok
        covered = native and bool(a["adapter_covered"])
        fuzzable = covered and fresh["eligible_for_fuzzing"] is True
        if not reach_ok:
            reason, cls = "native_target_not_reached", "target"
        elif not sandbox_ok or not fuzzable:
            reason, cls = a["reason"], a["failure_class"]
        else:
            reason, cls = None, None
        rows.append({"target_id": key, "repository": t["repository"],
                     "buggy_commit": t["buggy_commit"], "fixed_commit": t["fixed_commit"],
                     "complexity": t["complexity"], "defect_family": t["defect_family"],
                     "native_reach_verified": reach_ok, "sandbox_compatible": sandbox_ok,
                     "native_executable": native, "adapter_covered": covered,
                     **{k: fresh.get(k) for k in ("receiver_built", "arguments_built",
                                                  "target_invoked", "consumes_fuzz_input",
                                                  "deterministic", "v5_terminating")},
                     "eligibility_recomputed_from_evidence": fresh["recomputed"],
                     "eligible_for_fuzzing": fuzzable, "exclusion_reason": reason,
                     "failure_class": cls, "evidence_sha256": e["atheris_v2_evidence_sha256"]})
    native = [r for r in rows if r["native_executable"]]
    covered = [r for r in rows if r["adapter_covered"]]
    fuzz = [r for r in rows if r["eligible_for_fuzzing"]]
    ids = sorted(r["target_id"] for r in fuzz)
    native_ids = sorted(r["target_id"] for r in native)
    previous = load(PREVIOUS)
    refuse(ids != previous["sets"]["fuzzable"] or native_ids != previous["sets"]["native_executable"],
           "recomputed sets differ from the accepted v2 conclusion")
    f = panel["funnel"]
    waterfall = [
        ("frozen repositories (Phase 4 gate)", accounted["frozen"]),
        ("repositories scanned", gate4["funnel"]["stage:scanned"]),
        ("candidates evaluated", acq["counts"]["candidates_inspected"]),
        ("candidates admitted (unique)", acq["counts"]["admitted"]),
        ("admitted with regression test, no overlap", f["with_regression_test_and_no_overlap"]),
        ("natively qualified", f["natively_qualified"]),
        ("requalified 3/3", f["requalified_3_of_3"]),
        ("target reached on both revisions", f["target_reached"]),
        ("prompt leakage and token fit", f["prompt_leakage_and_token_fit_passed"]),
        ("frozen panel targets", len(rows)),
        ("native-executable (reach-verified AND sandbox-compatible)", len(native)),
        ("Atheris adapter-covered", len(covered)),
        ("meaningfully fuzzable (adapter-covered descriptive subset)", len(fuzz))]
    hist = historical_discordance()
    disc = hist["discordance"]

    def repos(rs):
        return Counter(r["repository"] for r in rs)
    s_fuzz, s_native, s_panel = stats(len(fuzz), disc), stats(len(native), disc), stats(len(rows), disc)
    s_cluster = stats(len(repos(native)), disc)

    def sentence(label, s):
        sig = ("significance at 0.01 is mathematically possible" if s["significance_possible_at_0.01"]
               else "significance at 0.05 is mathematically impossible"
               if not s["significance_possible_at_0.05"]
               else "significance at 0.05 but not 0.01 is possible")
        p80 = ("80% power at alpha 0.01 is unattainable under the assumed discordance"
               if s["mde_pp_80pct_power_alpha_0.01"] is None
               else f"80% power at alpha 0.01 needs >= {s['mde_pp_80pct_power_alpha_0.01']} pp")
        return f"{label} (n={s['n']}): {sig}; {p80}"
    wording = "; ".join([sentence("native-executable", s_native), sentence("frozen panel", s_panel),
                         sentence("fuzzable subset", s_fuzz),
                         sentence("one target per repository", s_cluster)])
    result = {
        "schema_version": "oneiros_v27_common_subset_v3",
        "status": "FROZEN_ADAPTER_COVERED_DESCRIPTIVE_SUBSET",
        "label": "adapter-covered descriptive subset - NOT representative of repository testing",
        "supersedes": {"path": PREVIOUS, "sha256": sha(PREVIOUS), "conclusion_unchanged": True,
                       "corrections": ["eligibility recomputed from committed per-seed evidence",
                                       "historical discordance bound to its source receipt",
                                       "statistical wording derived from counts",
                                       "receipts never overwritten"]},
        "model_independent": True, "fuzzing": False, "model": False,
        "inputs_sha256": {p: sha(p) for p in (PANEL, ACQ, GATE4, BUNDLE, PREVIOUS, HISTORICAL)},
        "waterfall": [{"stage": s, "count": n} for s, n in waterfall],
        "sets": {"native_executable": native_ids,
                 "adapter_covered": sorted(r["target_id"] for r in covered), "fuzzable": ids,
                 "zero_input_invocable": sorted(r["target_id"] for r in rows
                                                if r["failure_class"] == "zero_input")},
        "fuzzable_ids_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest(),
        "native_executable_ids_sha256": hashlib.sha256("\n".join(native_ids).encode()).hexdigest(),
        "composition": {label: {"targets": len(rs), "repositories": len(repos(rs)),
                                "max_per_repository": max(repos(rs).values()) if rs else 0,
                                "per_repository": dict(sorted(repos(rs).items())),
                                "complexity": dict(Counter(r["complexity"] for r in rs)),
                                "defect_family": dict(Counter(r["defect_family"] for r in rs))}
                        for label, rs in (("native_executable", native),
                                          ("adapter_covered", covered), ("fuzzable", fuzz))},
        "exclusions": [{"class": k[0], "reason": k[1], "count": v} for k, v in sorted(
            Counter((r["failure_class"], r["exclusion_reason"]) for r in rows
                    if not r["eligible_for_fuzzing"]).items(), key=str)],
        "statistics": {
            "test": "two-sided exact McNemar on paired per-target outcomes (planning only)",
            "historical_discordance": hist,
            "assumption_note": "a PLANNING SENSITIVITY borrowed from the historical synthetic/"
                               "function-mode comparison; repository-native discordance is unknown",
            "fuzzable_subset": s_fuzz, "native_executable": s_native, "frozen_panel": s_panel,
            "repository_clustering": {
                "native_executable_repositories": len(repos(native)),
                "largest_repository_share": round(max(repos(native).values()) / len(native), 3),
                "cluster_sensitivity_one_target_per_repository": s_cluster},
            "wording": wording},
        "targets": rows,
    }
    out_path.write_bytes((json.dumps(result, indent=1, sort_keys=True) + "\n").encode("utf-8"))
    print(json.dumps({"fuzzable": ids, "native_executable": len(native_ids),
                      "recomputed": sum(r["eligibility_recomputed_from_evidence"] for r in rows),
                      "historical": {k: hist[k] for k in ("gained", "lost", "paired")},
                      "wording": wording, "sha256": sha(args.out)}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
