"""Frozen analysis for the execution-dose-and-retention experiment.

Written, tested and committed before the treatment is trained.  Thresholds,
interval methods, stopping rules and the result-dependent path are constants
here and in docs/EXECUTION_DOSE_RETENTION_PROTOCOL.md; neither may change once
any evaluation exists.

Two gates, both required:

Mechanism (97-item train-derived panel, frozen control vs 25% treatment)
    * primary outcome: lenient semantic correctness, per requested item;
    * in BOTH conditions (intended output, shown code's actual output):
      paired gain >= +5 pp and Newcombe (1998, method 10) 90% lower bound >= 0;
    * format guard: strict answer rate on intended output may not fall more
      than 2 pp;  strict metrics are reported but never count as a gain;
    * no generation may hit the 128-token completion limit.

Retention (613-record train-derived canonical Kill@8 panel)
    * treatment - control Kill@8, paired by record, 90% percentile interval
      from a 10,000-replicate lineage-cluster bootstrap;
    * pass when the lower bound >= -3 pp; fail when the upper bound < -3 pp;
      anything else is inconclusive and does not pass.

Every number here is train-derived: not generalisation, not model selection,
not a final-test result.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness.execution_dose import cluster_bootstrap_difference
from scripts.diagnose_execution_trace_pilot import exact_gain_upper, newcombe_paired

SCHEMA = "oneiros_execution_dose_analysis_v1"
CONDITIONS = ("intended_output", "shown_actual_output")
MIN_GAIN_PP = 5.0
MAX_STRICT_ANSWER_REGRESSION_PP = 2.0
RETENTION_MARGIN_PP = 3.0
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20260924
MECHANISM_ITEMS = 97
RETENTION_RECORDS = 613
BOUND_MECHANISM_FIELDS = ("model", "model_revision", "pilot_development_sha256",
                          "items", "conditions", "decoding")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def exact_mcnemar_p(gained: int, lost: int) -> float:
    """Two-sided exact McNemar p-value on the discordant pairs."""
    discordant = gained + lost
    if discordant == 0:
        return 1.0
    from scipy.stats import binom
    return min(1.0, 2 * float(binom.cdf(min(gained, lost), discordant, 0.5)))


def paired_comparison(pairs: list[tuple[bool, bool]]) -> dict[str, Any]:
    n = len(pairs)
    gained = sum(1 for a, b in pairs if b and not a)
    lost = sum(1 for a, b in pairs if a and not b)
    low, high = newcombe_paired(pairs)
    return {"n": n, "first_correct": sum(a for a, _ in pairs),
            "second_correct": sum(b for _, b in pairs),
            "difference_pp": 100 * (gained - lost) / n,
            "newcombe90_low_pp": low, "newcombe90_high_pp": high,
            "gained": gained, "lost": lost,
            "exact_mcnemar_p": exact_mcnemar_p(gained, lost),
            "exact_one_sided_95_gain_upper_pp": exact_gain_upper(gained, n)}


def load_mechanism(path: Path, arm: str) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("status") != "complete" or data.get("arm") != arm:
        raise ValueError(f"invalid mechanism artifact for {arm}: {path}")
    return data


def correct_by_item(data: dict[str, Any], condition: str) -> dict[str, bool]:
    return {row["record_id"]: row["lenient"]["verdict"] == "correct"
            for row in data["detail"][condition]}


def mechanism_section(control: dict, treatment: dict, reference: dict | None) -> dict:
    for field in BOUND_MECHANISM_FIELDS:
        if json.dumps(control.get(field), sort_keys=True) != json.dumps(
                treatment.get(field), sort_keys=True):
            raise ValueError(f"mechanism artifacts differ on frozen field {field}")
    if control["items"] != MECHANISM_ITEMS:
        raise ValueError("mechanism panel is not the frozen 97 items")
    comparisons, reference_comparisons, format_report = {}, {}, {}
    for condition in CONDITIONS:
        first, second = correct_by_item(control, condition), correct_by_item(treatment, condition)
        if sorted(first) != sorted(second) or len(first) != MECHANISM_ITEMS:
            raise ValueError(f"mechanism panel mismatch in {condition}")
        ids = sorted(first)
        comparisons[condition] = paired_comparison([(first[i], second[i]) for i in ids])
        if reference is not None:
            base = correct_by_item(reference, condition)
            reference_comparisons[condition] = paired_comparison(
                [(base[i], second[i]) for i in ids])
        format_report[condition] = {
            arm: {key: data["summary"][condition].get(key) for key in (
                "strict_accuracy_per_requested", "strict_answer_rate",
                "lenient_accuracy_per_requested", "completion_limit_hits")}
            for arm, data in (("control", control), ("treatment", treatment))}
    strict_change = 100 * (treatment["summary"]["intended_output"]["strict_answer_rate"]
                           - control["summary"]["intended_output"]["strict_answer_rate"])
    checks = {}
    for condition in CONDITIONS:
        checks[f"{condition}_gain_at_least_5pp"] = (
            comparisons[condition]["difference_pp"] >= MIN_GAIN_PP)
        checks[f"{condition}_newcombe90_lower_at_least_0"] = (
            comparisons[condition]["newcombe90_low_pp"] >= 0.0)
    checks["strict_answer_rate_noninferior_2pp"] = (
        strict_change >= -MAX_STRICT_ANSWER_REGRESSION_PP)
    checks["no_completion_limit_hits"] = all(
        treatment["summary"][condition]["completion_limit_hits"] == 0
        for condition in CONDITIONS)
    return {
        "primary_outcome": "lenient semantic correctness per requested item",
        "treatment_minus_control": comparisons,
        "treatment_minus_12pct_ordered_trace_descriptive": reference_comparisons or None,
        "format_metrics_not_semantic": format_report,
        "strict_answer_rate_change_pp": strict_change,
        "checks": checks,
        "passed": all(checks.values()),
    }


def retention_verdict(low_pp: float, high_pp: float) -> str:
    """Noninferiority at the frozen margin; only "pass" passes."""
    if low_pp >= -RETENTION_MARGIN_PP:
        return "pass"
    if high_pp < -RETENTION_MARGIN_PP:
        return "fail"
    return "inconclusive"


def load_retention(envelope_path: Path, arm: str, panel: dict) -> dict[str, Any]:
    envelope = json.loads(envelope_path.read_text(encoding="utf-8"))
    if envelope.get("status") != "complete" or envelope.get("arm") != arm:
        raise ValueError(f"invalid retention envelope for {arm}")
    if envelope.get("panel_record_ids_sha256") != panel["record_ids_sha256"]:
        raise ValueError(f"retention {arm} ran on a different panel")
    result_path = ROOT / envelope["rehearsal_result"]["path"]
    if sha256(result_path) != envelope["rehearsal_result"]["sha256"]:
        raise ValueError(f"retention {arm} result hash mismatch")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if not result["raw_output_integrity"]["complete"]:
        raise ValueError(f"retention {arm} raw-output integrity failed")
    killed = {str(item["record_id"]): bool(item["killed"])
              for item in result["function_results"]}
    if sorted(killed) != sorted(panel["record_ids"]) or len(killed) != RETENTION_RECORDS:
        raise ValueError(f"retention {arm} does not cover exactly the frozen panel")
    return {"envelope": envelope, "result": result, "killed": killed}


def retention_section(control: dict, treatment: dict, base: dict | None,
                      panel: dict) -> dict:
    settings = {arm: json.dumps(data["result"]["frozen_settings"], sort_keys=True)
                for arm, data in (("control", control), ("treatment", treatment))}
    if len(set(settings.values())) != 1:
        raise ValueError("retention runs used different generation settings")
    lineages = panel["record_lineages"]

    def compare(first: dict, second: dict) -> dict[str, Any]:
        ids = sorted(first["killed"])
        pairs = [(first["killed"][i], second["killed"][i]) for i in ids]
        clustered = cluster_bootstrap_difference(
            [(lineages[i], first["killed"][i], second["killed"][i]) for i in ids],
            replicates=BOOTSTRAP_REPLICATES, seed=BOOTSTRAP_SEED)
        return {"cluster_bootstrap90": clustered,
                "record_level_newcombe90": paired_comparison(pairs)}

    primary = compare(control, treatment)
    verdict = retention_verdict(primary["cluster_bootstrap90"]["low_pp"],
                                primary["cluster_bootstrap90"]["high_pp"])

    def headline(data: dict) -> dict[str, Any]:
        result = data["result"]
        return {"kill_at_k": {k: result["kill_at_k"][k]["rate"] for k in result["kill_at_k"]},
                "pass_at_8": result["pass_at_k"]["8"]["rate"],
                "prompt_budget_failures": sum(bool(item.get("prompt_budget_failure"))
                                              for item in result["function_results"]),
                "failure_taxonomy": result["failure_taxonomy"],
                "adapter_sha256": data["envelope"].get("adapter_sha256")}

    arms = {"control": headline(control), "treatment": headline(treatment)}
    descriptive = {}
    if base is not None:
        arms["base"] = headline(base)
        descriptive = {"control_minus_base": compare(base, control),
                       "treatment_minus_base": compare(base, treatment)}
    return {
        "primary_outcome": "canonical Kill@8 per record, paired",
        "margin_pp": RETENTION_MARGIN_PP,
        "treatment_minus_control": primary,
        "descriptive": descriptive or None,
        "arms": arms,
        "verdict": verdict,
        "passed": verdict == "pass",
    }


def decide(mechanism: dict, retention: dict) -> dict[str, Any]:
    if mechanism["passed"] and retention["passed"]:
        outcome = "mechanism_supported_at_25pct_dose"
        next_step = ("stop and request explicit authorization before opening the "
                     "100-lineage confirmation panel; nothing is opened automatically")
    elif mechanism["passed"]:
        outcome = f"mechanism_gain_with_retention_{retention['verdict']}"
        next_step = ("stop; the gain is not accepted because test-generation retention "
                     "did not pass. Any follow-up needs a new explicit decision")
    else:
        excluded = all(mechanism["treatment_minus_control"][c]["newcombe90_high_pp"] < MIN_GAIN_PP
                       for c in CONDITIONS)
        outcome = ("null_at_25pct_dose_5pp_gain_excluded" if excluded
                   else "null_at_25pct_dose_inconclusive")
        next_step = ("stop this line; do not escalate the dose, add epochs, change the "
                     "mixture or re-threshold without a new explicit decision")
    return {"outcome": outcome, "next_permitted_step": next_step,
            "promotion_permitted": False, "confirmation_opening_permitted": False}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-mechanism", type=Path, default=ROOT / "results"
                        / "v4_3_execution_supervision_v1" / "eval_control.json")
    parser.add_argument("--treatment-mechanism", type=Path, default=ROOT / "results"
                        / "v4_3_execution_dose_v1" / "eval_dose_treatment.json")
    parser.add_argument("--reference-12pct", type=Path, default=ROOT / "results"
                        / "v4_3_execution_trace_v1" / "eval_ordered_trace.json")
    parser.add_argument("--retention-control", type=Path, default=ROOT / "results"
                        / "v4_3_execution_dose_v1" / "retention_control.json")
    parser.add_argument("--retention-treatment", type=Path, default=ROOT / "results"
                        / "v4_3_execution_dose_v1" / "retention_dose_treatment.json")
    parser.add_argument("--retention-base", type=Path, default=ROOT / "results"
                        / "v4_3_execution_dose_v1" / "retention_base.json")
    parser.add_argument("--panel", type=Path, default=ROOT / "results"
                        / "v4_3_execution_dose_retention_panel.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results"
                        / "v4_3_execution_dose_analysis.json")
    args = parser.parse_args(argv)

    required = (args.control_mechanism, args.treatment_mechanism,
                args.retention_control, args.retention_treatment)
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        # The analysis never runs on a partial set: that would let one gate's
        # result influence whether the other is measured.
        print(f"REFUSED: required artifacts missing: {missing}")
        return 2
    panel = json.loads(args.panel.read_text(encoding="utf-8"))
    control_mechanism = load_mechanism(args.control_mechanism, "control")
    treatment_mechanism = load_mechanism(args.treatment_mechanism, "dose_treatment")
    reference = (load_mechanism(args.reference_12pct, "ordered_trace")
                 if args.reference_12pct.exists() else None)
    mechanism = mechanism_section(control_mechanism, treatment_mechanism, reference)
    retention = retention_section(
        load_retention(args.retention_control, "control", panel),
        load_retention(args.retention_treatment, "dose_treatment", panel),
        load_retention(args.retention_base, "base", panel)
        if args.retention_base.exists() else None,
        panel)
    inputs = {name: {"path": path.relative_to(ROOT).as_posix(), "sha256": sha256(path)}
              for name, path in (
                  ("control_mechanism", args.control_mechanism),
                  ("treatment_mechanism", args.treatment_mechanism),
                  ("reference_12pct", args.reference_12pct),
                  ("retention_control", args.retention_control),
                  ("retention_treatment", args.retention_treatment),
                  ("retention_base", args.retention_base),
                  ("panel", args.panel)) if path.exists()}
    report = {
        "schema_version": SCHEMA,
        "label": "train-derived mechanism and retention diagnostic; not generalisation, "
                 "not model selection, not a final-test result",
        "predeclared": {
            "mechanism_min_gain_pp_each_condition": MIN_GAIN_PP,
            "mechanism_interval": "Newcombe (1998) method 10 paired score, 90%, lower >= 0",
            "max_strict_answer_regression_pp": MAX_STRICT_ANSWER_REGRESSION_PP,
            "retention_margin_pp": RETENTION_MARGIN_PP,
            "retention_interval": (f"lineage-cluster percentile bootstrap, 90%, "
                                   f"{BOOTSTRAP_REPLICATES} replicates, seed {BOOTSTRAP_SEED}"),
        },
        "inputs": inputs,
        "mechanism": mechanism,
        "retention": retention,
        "decision": decide(mechanism, retention),
    }
    args.output.write_bytes((json.dumps(report, indent=2) + "\n").encode("utf-8"))
    print(json.dumps({"mechanism_passed": mechanism["passed"],
                      "retention_verdict": retention["verdict"],
                      "decision": report["decision"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
