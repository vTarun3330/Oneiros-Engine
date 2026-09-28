"""Phase 2 of the SFT root-cause loop: decompose base -> SFT by pipeline stage.

Reads only the Phase 1 ledgers (hash-checked against the Phase 1 receipt).
Primary comparison: locked validation, base vs arm A checkpoint 431 (paired on
the same 757 functions).  Secondary: the arm A training trajectory on
ablation_dev, every checkpoint vs step 0.  All intervals are paired cluster
bootstraps over semantic groups (docs/SFT_ROOT_CAUSE_PROTOCOL.md).
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness.atomic_publish import publish_file_atomically
from harness.causal_decomposition import (
    BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED, category_distribution, function_transitions,
    paired_comparison,
)
from harness.causal_ledger import TERMINAL_CATEGORIES

PHASE1 = "results/sft_root_cause_phase1_ledger_receipt.json"
RECEIPT = "results/sft_root_cause_phase2_decomposition_receipt.json"
LOCKED_RECEIPT = "results/v4_2_locked_validation_result_receipt.json"
QUARANTINED_FIRST_RUN = {
    "path": "results/sft_root_cause/quarantine/phase2_first_run_SUPERSEDED.json",
    "sha256": "f4465e58f7887d46c40bf4065315608bb944500b8375d48a56f198839690070f",
    "why_superseded": [
        "its H6 rule tested 'below step 0 at every step >= 100', not the frozen signature "
        "'declines with steps'; the data show a drop by step 100 and then a plateau, which the "
        "frozen signature treats as weakening",
        "it lacked form-stratified oracle accuracy: SFT moved ~99.5% of assertions to exact "
        "equality, so the unstratified conditional oracle rate mixes value prediction with "
        "assertion form"]}
TRAJECTORY_METRICS = ("kill_at_8", "p_reference_valid", "p_disc_given_execute",
                      "p_correct_oracle_given_disc", "p_correct_oracle_given_disc_unique",
                      "p_correct_oracle_given_disc_equality",
                      "p_correct_oracle_given_disc_equality_unique",
                      "equality_oracle_given_policy_valid",
                      "duplicate_rate_given_parse", "unique_call_signatures_per_parsed",
                      "function_disc_recall_at_8", "test_function_shape_given_policy_valid")
STAGE_METRICS = ("p_parse", "p_policy_valid_given_parse", "p_execute_given_parse",
                 "p_disc_given_execute", "p_correct_oracle_given_disc",
                 "p_correct_oracle_given_disc_equality", "equality_oracle_given_policy_valid",
                 "p_reference_valid",
                 "p_kill_given_reference_valid", "kill_at_1", "kill_at_4", "kill_at_8")


def load_ledgers() -> tuple[dict, dict[str, list[dict]]]:
    receipt = json.loads((ROOT / PHASE1).read_text(encoding="utf-8"))
    ledgers = {}
    for arm, info in receipt["ledgers"].items():
        data = (ROOT / info["path"]).read_bytes()
        if hashlib.sha256(data).hexdigest() != info["sha256"]:
            raise SystemExit(f"REFUSED: ledger {arm} does not match the Phase 1 receipt")
        ledgers[arm] = [json.loads(line) for line in data.decode("utf-8").splitlines()]
    return receipt, ledgers


def power(kill: dict, clusters: int) -> dict:
    se = kill["bootstrap_se_points"]
    observed = kill["difference_points"]
    # Supported needs lower bound >= 3: observed - 1.96 se >= 3.
    needed_se = (observed - 3.0) / 1.96 if observed > 3.0 else None
    return {
        "kill_at_8_difference_points": observed,
        "clustered_se_points": se,
        "minimum_detectable_effect_80pct_power_points": round(2.8 * se, 3),
        "clusters": clusters,
        "se_needed_for_supported_at_observed_effect": (round(needed_se, 4)
                                                        if needed_se else None),
        "cluster_multiplier_needed_for_supported_at_observed_effect": (
            round((se / needed_se) ** 2, 1) if needed_se else None),
        "note": ("a 'supported' Kill@8 claim needs the lower 95% bound >= +3 points; at the "
                 "observed effect that requires the multiplier above on independent clusters, "
                 "assuming the effect is real and SE scales as 1/sqrt(clusters)"),
    }


def by_dataset(base: list[dict], sft: list[dict]) -> dict:
    out = {}
    for dataset in sorted({r["dataset"] for r in base}):
        a = [r for r in base if r["dataset"] == dataset]
        b = [r for r in sft if r["dataset"] == dataset]
        comparison = paired_comparison(a, b)
        out[dataset] = {"functions": len({r["record_id"] for r in a}),
                        "clusters": comparison["clusters"],
                        "metrics": {m: comparison["metrics"][m] for m in STAGE_METRICS}}
    return out


def verdicts(primary: dict, trajectory: dict, plateau: dict, power_block: dict) -> dict:
    m = primary["metrics"]
    d = {name: m[name]["difference_points"] for name in m}
    ci = {name: m[name]["ci95_points"] for name in m}
    out = {}
    contract_ok = d["p_parse"] > -1 and d["p_execute_given_parse"] > -1 \
        and d["p_policy_valid_given_parse"] > -1
    eq = m["p_correct_oracle_given_disc_equality"]
    form = m["equality_oracle_given_policy_valid"]
    out["H5"] = {
        "status": "weakened" if contract_ok else "open",
        "why": (f"parse {d['p_parse']:+.2f}, policy {d['p_policy_valid_given_parse']:+.2f}, "
                f"execute|parse {d['p_execute_given_parse']:+.2f} points: no contract stage "
                "falls by more than 1 point. The form DID change, toward exact-equality "
                f"oracles ({form['a']:.3f} -> {form['b']:.3f} of policy-valid candidates), "
                "i.e. stronger, not degraded, tests"),
    }
    disc_up = ci["p_disc_given_execute"][0] > 0
    out["H2"] = {
        "status": "strengthened" if disc_up and eq["ci95_points"][1] < 5.0 else "open",
        "why": (f"P(disc|execute) {d['p_disc_given_execute']:+.2f} points, CI "
                f"{ci['p_disc_given_execute']}. The frozen signature (P(correct oracle|disc) "
                f"falls) holds unstratified: {d['p_correct_oracle_given_disc']:+.2f}, CI "
                f"{ci['p_correct_oracle_given_disc']}. Stratified by assertion form (added after "
                f"the first look), exact-value accuracy on discriminating inputs moves "
                f"{eq['difference_points']:+.2f}, CI {eq['ci95_points']}: SFT neither measurably "
                "damaged nor improved value prediction (an improvement of >= 5 points is "
                "excluded). Imitation transferred assertion form and input choice, not the "
                "value. Causal support needs a Phase 4 single-factor arm"),
    }
    k = m["kill_at_8"]
    out["H7"] = {
        "status": ("strengthened" if k["ci95_points"][0] < 0 and k["ci95_points"][0] < 3
                   < k["ci95_points"][1] else "open"),
        "why": (f"Kill@8 {k['difference_points']:+.2f} points, clustered CI {k['ci95_points']} "
                f"over {primary['clusters']} semantic clusters; minimum detectable effect at 80% "
                f"power is {power_block['minimum_detectable_effect_80pct_power_points']} points"),
    }
    steps = sorted(trajectory, key=int)
    unstrat = plateau["p_correct_oracle_given_disc"]
    strat = plateau["p_correct_oracle_given_disc_equality"]
    declining = unstrat["ci95_points"][1] < 0 or strat["ci95_points"][1] < 0
    oracle_path = ", ".join(
        f"{s}: {trajectory[s]['p_correct_oracle_given_disc']['difference_points']:+.1f}"
        for s in steps)
    form_path = ", ".join(
        f"{s}: {trajectory[s]['equality_oracle_given_policy_valid']['b']:.2f}" for s in steps)
    out["H6"] = {
        "status": "open" if declining else "weakened",
        "why": ("frozen signature: the conditional oracle rate declines WITH STEPS. Step 431 vs "
                f"step 100 on ablation_dev: unstratified {unstrat['difference_points']:+.2f} "
                f"{unstrat['ci95_points']}, equality-stratified {strat['difference_points']:+.2f} "
                f"{strat['ci95_points']}. Versus step 0 the unstratified rate drops by step 100 "
                f"and then plateaus ({oracle_path}), tracking the equality-form share "
                f"({form_path})"),
    }
    for hypothesis in ("H1", "H3", "H4"):
        out[hypothesis] = {"status": "open",
                           "why": "not identifiable from retained outputs; needs a Phase 3 probe"}
    return out


def main() -> int:
    started = time.time()
    phase1, ledgers = load_ledgers()
    base, sft = ledgers["locked_val_base"], ledgers["locked_val_armA_431"]
    primary = paired_comparison(base, sft)
    locked = json.loads((ROOT / LOCKED_RECEIPT).read_text(encoding="utf-8"))["arms"]
    reconciliation = {
        "kill_at_8": [primary["metrics"]["kill_at_8"]["a"], primary["metrics"]["kill_at_8"]["b"],
                      locked["base_control"]["metrics"]["k8"],
                      locked["arm_a_checkpoint_431"]["metrics"]["k8"]],
        "reference_valid": [primary["metrics"]["p_reference_valid"]["a"],
                            primary["metrics"]["p_reference_valid"]["b"],
                            locked["base_control"]["metrics"]["ref"],
                            locked["arm_a_checkpoint_431"]["metrics"]["ref"]],
    }
    reconciled = all(math.isclose(v[0], v[2], abs_tol=1e-6) and math.isclose(v[1], v[3],
                                                                             abs_tol=1e-6)
                     for v in reconciliation.values())
    if not reconciled:
        raise SystemExit(f"REFUSED: ledger does not reconcile with the locked receipt "
                         f"{reconciliation}")
    trajectory_full = {}
    step0 = ledgers["monitor_step_000"]
    for arm, rows in ledgers.items():
        if arm.startswith("monitor_step_") and arm != "monitor_step_000":
            comparison = paired_comparison(step0, rows)
            trajectory_full[str(int(arm.rsplit("_", 1)[1]))] = {
                m: comparison["metrics"][m] for m in TRAJECTORY_METRICS}
    trajectory_clusters = paired_comparison(step0, step0, resamples=10)["clusters"]
    plateau_full = paired_comparison(ledgers["monitor_step_100"], ledgers["monitor_step_431"])
    plateau = {m: plateau_full["metrics"][m] for m in TRAJECTORY_METRICS}
    power_block = power(primary["metrics"]["kill_at_8"], primary["clusters"])
    receipt = {
        "schema_version": "oneiros_sft_root_cause_phase2_receipt_v1",
        "phase": 2,
        "objective": "locate the stage(s) at which SFT improves or regresses",
        "command": "python scripts/decompose_sft_pipeline.py",
        "protocol": "docs/SFT_ROOT_CAUSE_PROTOCOL.md",
        "inputs": {"phase1_receipt": PHASE1,
                   "phase1_receipt_sha256": hashlib.sha256(
                       (ROOT / PHASE1).read_bytes()).hexdigest(),
                   "ledgers": {arm: info["sha256"] for arm, info in phase1["ledgers"].items()}},
        "model_calls": 0, "gpu_used": False,
        "inference": {"method": "paired cluster bootstrap over semantic group_id",
                      "resamples": BOOTSTRAP_RESAMPLES, "seed": BOOTSTRAP_SEED,
                      "difference": "SFT minus base, in percentage points"},
        "reconciliation_with_locked_receipt": {"values_[ledger_base, ledger_sft, "
                                               "receipt_base, receipt_sft]": reconciliation,
                                               "exact": reconciled},
        "primary_locked_validation": {
            "clusters": primary["clusters"],
            "stage_chain": {m: primary["metrics"][m] for m in STAGE_METRICS},
            "all_metrics": primary["metrics"],
            "function_transitions": function_transitions(base, sft),
            "terminal_categories": {
                "base": category_distribution(base, TERMINAL_CATEGORIES),
                "arm_a_431": category_distribution(sft, TERMINAL_CATEGORIES)},
            "by_dataset": by_dataset(base, sft),
        },
        "trajectory_ablation_dev_vs_step0": {
            "clusters": trajectory_clusters,
            "terminal_categories": {arm: category_distribution(rows, TERMINAL_CATEGORIES)
                                    for arm, rows in ledgers.items()
                                    if arm.startswith("monitor_step_")},
            "differences": trajectory_full,
            "step_431_vs_step_100": plateau},
        "power": power_block,
    }
    receipt["hypothesis_updates"] = verdicts(primary, trajectory_full, plateau, power_block)
    receipt["post_hoc_additions"] = {
        "form_stratified_metrics": ["p_correct_oracle_given_disc_equality",
                                    "p_correct_oracle_given_disc_equality_unique",
                                    "p_correct_oracle_given_disc_non_equality"],
        "status": "exploratory refinement added after the first Phase 2 look; reported "
                  "alongside, never instead of, the frozen unstratified mediator",
        "quarantined_first_run": QUARANTINED_FIRST_RUN}
    receipt["duration_seconds"] = round(time.time() - started, 1)
    publish_file_atomically(ROOT / RECEIPT, (json.dumps(receipt, indent=1) + "\n")
                            .encode("utf-8"))
    chain = receipt["primary_locked_validation"]["stage_chain"]
    for name, metric in chain.items():
        print(f"{name:34s} {metric['a']:.4f} -> {metric['b']:.4f}  "
              f"{metric['difference_points']:+7.2f} {metric['ci95_points']}  {metric['region']}")
    print(json.dumps(receipt["hypothesis_updates"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
