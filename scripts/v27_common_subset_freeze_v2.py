"""v2.7 Phase 6b successor: freeze the Atheris ADAPTER-COVERED DESCRIPTIVE SUBSET from COMMITTED
evidence only (reproducible from a clean clone). Supersedes v1 (results preserved) which used
git-ignored inputs, literal waterfall counts, assert-based checks, reinterpreted a process
failure without exit evidence, and counted a zero-input target as fuzzable.

Model-independent: panel membership, native target reach (official tests entered the target on
both revisions), the model-free sandbox control, and the Atheris v2 probe. No model output, kill
or fuzzing result is read.

Sets (all views of the frozen 34-target panel, never redefinitions):
- native_executable: reach-verified on both revisions AND sandbox control passed (deterministic);
- adapter_covered: native_executable AND the v5 adapter builds a plan identical on both revisions;
- fuzzable (primary descriptive subset): adapter_covered AND meaningfully fuzzable
  (receiver/arguments built from fuzz bytes, target entered, deterministic, no interpreter exit).

    python scripts/v27_common_subset_freeze_v2.py --out results/<receipt>.json
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
BUNDLE = "results/sft_root_cause_v27_phase6_evidence_bundle_v2.json"
PROVISIONAL = "results/sft_root_cause_v27_common_subset_r4.json"
HIST_B, HIST_C, HIST_N = 114, 88, 757          # historical locked function-mode comparison


class Refused(SystemExit):
    pass


def refuse(condition: bool, message: str) -> None:
    if condition:
        raise Refused("REFUSED: " + message)


def sha(rel) -> str:
    return hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()


def load(rel) -> dict:
    path = ROOT / rel
    refuse(not path.is_file(), f"required committed input missing: {rel}")
    return json.loads(path.read_text(encoding="utf-8"))


def mcnemar_p(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def power(n: int, discordance: float, diff: float, alpha: float) -> float:
    p10, p01 = (discordance + diff) / 2, (discordance - diff) / 2
    if p01 < 0:
        return float("nan")
    p00, total = 1 - discordance, 0.0
    for b in range(n + 1):
        for c in range(n - b + 1):
            if mcnemar_p(b, c) < alpha:
                total += comb(n, b) * comb(n - b, c) * p10 ** b * p01 ** c * p00 ** (n - b - c)
    return total


def mde(n: int, discordance: float, alpha: float, target: float = 0.8):
    for d in range(1, int(discordance * 1000) + 1):
        if power(n, discordance, d / 1000, alpha) >= target:
            return round(d / 10, 1)
    return None


def stats(n: int, discordance: float) -> dict:
    floor = mcnemar_p(n, 0)
    return {"n": n, "smallest_attainable_two_sided_p": floor,
            "significance_possible_at_0.05": floor < 0.05,
            "significance_possible_at_0.01": floor < 0.01,
            "mde_pp_80pct_power_alpha_0.05": mde(n, discordance, 0.05),
            "mde_pp_80pct_power_alpha_0.01": mde(n, discordance, 0.01),
            "max_effect_pp_at_assumed_discordance": round(discordance * 100, 1)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    panel, acq, gate4, bundle = load(PANEL), load(ACQ), load(GATE4), load(BUNDLE)
    refuse(bundle["panel"]["sha256"] != sha(PANEL), "evidence bundle binds a different panel")
    refuse(gate4.get("status") != "PASS", "Phase 4 acquisition gate did not pass")
    targets = {t["target_id"]: t for t in panel["targets"]}
    ev = {t["target_id"]: t for t in bundle["targets"]}
    refuse(set(ev) != set(targets), "evidence bundle does not cover exactly the frozen panel")
    refuse(len(targets) != panel["funnel"]["selected"], "panel targets disagree with its funnel")
    frozen_repos = gate4["checks"]["all_frozen_repositories_accounted"]["detail"]["frozen"]
    refuse(gate4["checks"]["all_frozen_repositories_accounted"]["detail"]["unaccounted"],
           "Phase 4 left repositories unaccounted")
    rows = []
    for key in sorted(targets):
        t, e = targets[key], ev[key]
        reach_ok = e["native_reach"]["reach"] == "reach_verified"
        sandbox_ok = e["sandbox_control"]["passed"] and e["sandbox_control"]["deterministic"]
        a = e["atheris_v2"]
        native = reach_ok and sandbox_ok
        covered = native and bool(a["adapter_covered"])
        fuzzable = covered and bool(a["eligible_for_fuzzing"])
        if not reach_ok:
            reason, cls = "native_target_not_reached", "target"
        elif not sandbox_ok:
            reason, cls = a["reason"], a["failure_class"]          # recorded policy exclusion
        else:
            reason, cls = (None, None) if fuzzable else (a["reason"], a["failure_class"])
        rows.append({"target_id": key, "repository": t["repository"],
                     "buggy_commit": t["buggy_commit"], "fixed_commit": t["fixed_commit"],
                     "complexity": t["complexity"], "defect_family": t["defect_family"],
                     "native_reach_verified": reach_ok, "sandbox_compatible": sandbox_ok,
                     "native_executable": native, "adapter_covered": covered,
                     "receiver_built": a.get("receiver_built"),
                     "arguments_built": a.get("arguments_built"),
                     "target_invoked": a.get("target_invoked"),
                     "consumes_fuzz_input": a.get("consumes_fuzz_input"),
                     "deterministic": a.get("deterministic"),
                     "eligible_for_fuzzing": fuzzable, "exclusion_reason": reason,
                     "failure_class": cls,
                     "evidence_sha256": e["atheris_v2_evidence_sha256"]})
    native = [r for r in rows if r["native_executable"]]
    covered = [r for r in rows if r["adapter_covered"]]
    fuzz = [r for r in rows if r["eligible_for_fuzzing"]]
    zero = [r["target_id"] for r in rows if r["failure_class"] == "zero_input"]
    ids = sorted(r["target_id"] for r in fuzz)
    f = panel["funnel"]
    waterfall = [
        ("frozen repositories (Phase 4 gate)", frozen_repos),
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
        ("meaningfully fuzzable (adapter-covered descriptive subset)", len(fuzz)),
    ]
    refuse(acq["counts"]["admitted"] > acq["counts"]["candidates_inspected"], "funnel inverted")
    exclusions = sorted(Counter((r["failure_class"], r["exclusion_reason"]) for r in rows
                                if not r["eligible_for_fuzzing"]).items(), key=str)
    disc = (HIST_B + HIST_C) / HIST_N

    def repos(rs):
        return Counter(r["repository"] for r in rs)

    out = {
        "schema_version": "oneiros_v27_common_subset_v2",
        "status": "FROZEN_ADAPTER_COVERED_DESCRIPTIVE_SUBSET",
        "label": "adapter-covered descriptive subset - NOT representative of repository testing",
        "supersedes": {"path": PROVISIONAL, "sha256": sha(PROVISIONAL),
                       "status": "PROVISIONAL (preserved unchanged)",
                       "corrections": ["probe v1 lost child exit evidence; v1 freeze rewrote a "
                                       "process failure into a SystemExit conclusion manually",
                                       "zero-input target counted as fuzzable",
                                       "single-seed determinism (missed random jitter)",
                                       "literal waterfall counts and assert-based checks",
                                       "git-ignored inputs; reproduction test skipped"]},
        "model_independent": True, "fuzzing": False, "model": False,
        "inputs_sha256": {p: sha(p) for p in (PANEL, ACQ, GATE4, BUNDLE)},
        "waterfall": [{"stage": s, "count": n} for s, n in waterfall],
        "sets": {"native_executable": sorted(r["target_id"] for r in native),
                 "adapter_covered": sorted(r["target_id"] for r in covered),
                 "fuzzable": ids, "zero_input_invocable": sorted(zero)},
        "fuzzable_ids_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest(),
        "native_executable_ids_sha256": hashlib.sha256("\n".join(
            sorted(r["target_id"] for r in native)).encode()).hexdigest(),
        "composition": {
            label: {"targets": len(rs), "repositories": len(repos(rs)),
                    "max_per_repository": max(repos(rs).values()) if rs else 0,
                    "per_repository": dict(sorted(repos(rs).items())),
                    "complexity": dict(Counter(r["complexity"] for r in rs)),
                    "defect_family": dict(Counter(r["defect_family"] for r in rs))}
            for label, rs in (("native_executable", native), ("adapter_covered", covered),
                              ("fuzzable", fuzz))},
        "exclusions": [{"class": k[0], "reason": k[1], "count": v} for k, v in exclusions],
        "statistics": {
            "test": "two-sided exact McNemar on paired per-target outcomes",
            "assumed_discordance": round(disc, 4),
            "assumption_note": "a PLANNING SENSITIVITY borrowed from the historical synthetic/"
                               "function-mode locked comparison (114+88 of 757); repository-"
                               "native discordance is unknown",
            "fuzzable_subset": stats(len(fuzz), disc),
            "native_executable": stats(len(native), disc),
            "frozen_panel": stats(len(rows), disc),
            "repository_clustering": {
                "native_executable_repositories": len(repos(native)),
                "largest_repository_share": round(max(repos(native).values()) / len(native), 3)
                if native else None,
                "cluster_sensitivity_one_target_per_repository": stats(len(repos(native)), disc),
                "note": "targets within a repository are not independent; the per-repository "
                        "bound is the conservative effective sample size and repository-level "
                        "descriptive summaries must accompany any per-target test"},
            "wording": "at n=29 and n=34 significance at 0.01 is mathematically possible; what is "
                       "unattainable is 80% power at alpha=0.01 under the assumed discordance; "
                       "the fuzzable subset is descriptive only",
        },
        "targets": rows,
        "source_identity": "bound by the Stage 2 receipt (source commit + full-suite receipt)",
    }
    (ROOT / args.out).write_text(json.dumps(out, indent=1, sort_keys=True) + "\n",
                                 encoding="utf-8")
    print(json.dumps({"waterfall": waterfall, "fuzzable": ids,
                      "fuzzable_ids_sha256": out["fuzzable_ids_sha256"],
                      "zero_input": zero,
                      "composition": out["composition"]["fuzzable"],
                      "exclusions": out["exclusions"], "statistics": out["statistics"]},
                     indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
