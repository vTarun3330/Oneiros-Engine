"""Phases 2-3 of the execution-dose experiment: choose a dose, build the arm.

Evaluates both predeclared designs (25% and 50% execution examples) against
the same hard constraints, applies the predeclared selection rule, and builds
exactly one treatment arm from the frozen control.  The control itself is not
rebuilt or retrained: its frozen adapter is the comparison.

Hard constraints for every design (see docs/EXECUTION_DOSE_RETENTION_PROTOCOL.md):

* execution rows come only from the Phase-1 census pool (verified, train
  lineages, trace <= 1,024 tokens), one call per record, unique targets;
* the tier mix reproduces the failed 12% ordered-trace arm, so the dose
  changes and the content mix does not;
* at most ``MAX_LINEAGE_CAP`` rows per lineage (the O1 dataset's own cap),
  using the smallest cap that fills; no bug family above 35% of the rows;
* the representation is the ordered trace, byte-for-byte the builder the
  12% arm used;
* balanced replay: exact (source x tier) removal quotas, bounded family loss,
  every dataset, tier, family, execution mode and the real-repository rows kept;
* no prompt, completion or sequence budget is exceeded, nothing is truncated.

Selection rule: the largest dose meeting every hard constraint.  Token-mass
levers, applied in a fixed order: (1) per record, the call with the shortest
verified ordered trace; (2) within each replay stratum, the longest canonical
completions are replaced first.  The residual mass difference is reported, not
hidden.

CPU only.  Opens the train-derived artifacts of the earlier pilots and the
Phase-1 pool; nothing from validation, ablation_dev, test, sealed-final or the
confirmation lineages.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import sys
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness.execution_dose import (
    ARM_SIZE, DOSE_DESIGNS, dose_summary, largest_remainder, plan_replay,
    replay_balance_report, select_execution_rows, stratum,
)
from harness.execution_supervision import format_output_prediction_chat_prompt
from harness.execution_supervision_sidecar import sha256_file, stable_rank
from scripts.build_execution_trace_ab import build_pair_prompt, event_completion
from scripts.preflight_execution_supervision_ab import MODEL_NAME, MODEL_REVISION

SCHEMA = "oneiros_execution_dose_ab_v1"
SOURCE_DIR = ROOT / "results" / "v4_3_execution_supervision_v1"
TRACE_DIR = ROOT / "results" / "v4_3_execution_trace_v1"
MAX_LINEAGE_CAP = 4
FAMILY_SHARE_CAP = 0.35
FUNCTION_PROMPT_TOKENS = 1024
AUXILIARY_COMPLETION_TOKENS = 1024
MAX_SEQUENCE_TOKENS = 3072
SUPERVISION_KIND = "ordered_execution_trace_and_value_pair"


def reference_tier_mix(trace_manifest: dict, sidecar: list[dict]) -> dict[str, int]:
    """Tier counts of the 12% ordered-trace arm's 122 execution rows."""
    eligible = [row for row in sidecar if row.get("trace_training_eligible") is True]
    if len(eligible) != trace_manifest["replacement_examples"]:
        raise ValueError("12% arm reference rows do not match its manifest")
    return dict(Counter(str(row["complexity_tier"]) for row in eligible))


def one_call_per_record(pool: list[dict], tokens: dict[str, int]) -> list[dict]:
    """Lever 1: per record, the verified call with the shortest ordered trace."""
    by_record: dict[str, list[dict]] = defaultdict(list)
    for row in pool:
        by_record[str(row["record_id"])].append(row)
    return [min(rows, key=lambda row: (tokens[_key(row)], stable_rank(
        "dose_call", row["record_id"], row["call_expression"])))
        for _, rows in sorted(by_record.items())]


def _key(row: dict) -> str:
    return f"{row['record_id']}\x1f{row['call_expression']}\x1f{row['trace_completion_sha256']}"


def evaluate_design(name: str, count: int, candidates: list[dict], control: list[dict],
                    control_tokens: list[int], exec_tokens: dict[str, int],
                    tier_mix: dict[str, int]) -> dict[str, Any]:
    tier_quotas = largest_remainder(count, tier_mix)
    report: dict[str, Any] = {"design": name, "execution_examples": count,
                              "tier_quotas": tier_quotas, "hard_constraints": {}}
    selected = None
    for cap in range(1, MAX_LINEAGE_CAP + 1):
        try:
            selected = select_execution_rows(
                candidates, count, lineage_cap=cap, tier_quotas=tier_quotas,
                family_cap=int(FAMILY_SHARE_CAP * count),
                order_key=lambda row: stable_rank("dose_select", row["record_id"]))
        except ValueError as exc:
            report.setdefault("lineage_cap_attempts", {})[cap] = str(exc)
            continue
        report["lineage_cap"] = cap
        break
    report["hard_constraints"]["execution_rows_fill_under_lineage_cap_4"] = selected is not None
    if selected is None:
        report["feasible"] = False
        return report
    try:
        removed = plan_replay(control, count, lambda position: (
            -control_tokens[position], stable_rank("dose_replay", position)))
    except ValueError as exc:
        report["hard_constraints"]["balanced_replay"] = False
        report["replay_error"] = str(exc)
        report["feasible"] = False
        return report
    balance = replay_balance_report(control, removed)
    report["hard_constraints"]["balanced_replay"] = balance["no_category_removed"]
    report["hard_constraints"]["unique_records"] = (
        len({row["record_id"] for row in selected}) == count)
    report["hard_constraints"]["unique_targets"] = (
        len({row["trace_completion_sha256"] for row in selected}) == count)
    report["dose"] = dose_summary(control_tokens, removed,
                                  [exec_tokens[_key(row)] for row in selected])
    report["feasible"] = all(report["hard_constraints"].values())
    report["_selected"], report["_removed"], report["_balance"] = selected, removed, balance
    return report


def _counts(rows, field) -> dict[str, int]:
    return dict(sorted(Counter(str(row[field]) for row in rows).items()))


def _median_by_stratum(rows: list[dict], tokens: list[int],
                       positions: list[int]) -> dict[str, float]:
    groups: dict[str, list[int]] = defaultdict(list)
    for position in positions:
        groups["/".join(stratum(rows[position]))].append(tokens[position])
    return {key: statistics.median(values) for key, values in sorted(groups.items())}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results"
                        / "v4_3_execution_dose_v1")
    parser.add_argument("--design-report", type=Path, default=ROOT / "results"
                        / "v4_3_execution_dose_design.json")
    parser.add_argument("--manifest-copy", type=Path, default=ROOT / "results"
                        / "v4_3_execution_dose_dataset_manifest.json")
    parser.add_argument("--census", type=Path, default=ROOT / "results"
                        / "v4_3_execution_dose_census.json")
    args = parser.parse_args(argv)

    census = json.loads(args.census.read_text(encoding="utf-8"))
    pool_path = ROOT / census["outputs"]["pool"]["path"]
    if sha256_file(pool_path) != census["outputs"]["pool"]["sha256"]:
        raise SystemExit("REFUSED: census pool hash mismatch")
    if any(census["leakage"][key] for key in census["leakage"] if key != "splits_opened"):
        raise SystemExit("REFUSED: census records a protected access")
    source_manifest = json.loads((SOURCE_DIR / "manifest.json").read_text(encoding="utf-8"))
    control_path = SOURCE_DIR / "arm_a.control.json"
    if sha256_file(control_path) != source_manifest["arm_a_sha256"]:
        raise SystemExit("REFUSED: frozen control arm hash mismatch")
    trace_manifest = json.loads((TRACE_DIR / "manifest.json").read_text(encoding="utf-8"))
    sidecar_path = SOURCE_DIR / "train.execution.json"
    if sha256_file(sidecar_path) != source_manifest["sidecar_sha256"]:
        raise SystemExit("REFUSED: 12% arm sidecar hash mismatch")
    split = json.loads((SOURCE_DIR / "lineage_split.json").read_text(encoding="utf-8"))
    train_lineages = set(split["train_lineages"])
    held_out = set(split["pilot_development_lineages"]) | set(
        split["unopened_confirmation_lineages"])

    control = json.loads(control_path.read_text(encoding="utf-8"))
    pool = [row for row in json.loads(pool_path.read_text(encoding="utf-8"))
            if row["trace_training_eligible"] is True]
    if any(row["function_lineage"] not in train_lineages or row["function_lineage"] in held_out
           or row.get("evaluation_split") != "train" or row.get("verified") is not True
           for row in pool):
        raise SystemExit("REFUSED: pool row outside verified train lineages")
    tier_mix = reference_tier_mix(trace_manifest, json.loads(
        sidecar_path.read_text(encoding="utf-8")))

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, revision=MODEL_REVISION, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def completion_tokens(text: str) -> int:
        return len(tokenizer(text + tokenizer.eos_token, add_special_tokens=False)["input_ids"])

    control_tokens = [completion_tokens(str(row["completion"])) for row in control]
    exec_tokens = {_key(row): completion_tokens(event_completion(row, ordered=True))
                   for row in pool}
    candidates = one_call_per_record(pool, exec_tokens)

    designs = {name: evaluate_design(name, count, candidates, control, control_tokens,
                                     exec_tokens, tier_mix)
               for name, count in DOSE_DESIGNS.items()}
    feasible = [name for name in DOSE_DESIGNS if designs[name]["feasible"]]
    if not feasible:
        raise SystemExit("REFUSED: no dose design meets the hard constraints")
    chosen = max(feasible, key=lambda name: DOSE_DESIGNS[name])
    design = designs[chosen]
    selected, removed = design["_selected"], design["_removed"]

    # Deterministic assignment of execution rows to the replaced positions.
    selected = sorted(selected, key=lambda row: stable_rank("dose_assign", row["record_id"]))
    treatment = [dict(row) for row in control]
    budget_report: list[dict[str, int]] = []
    for position, evidence in zip(removed, selected):
        prompt = build_pair_prompt(evidence, ordered=True)
        completion = event_completion(evidence, ordered=True)
        prompt_tokens = len(tokenizer(format_output_prediction_chat_prompt(tokenizer, prompt),
                                      add_special_tokens=False)["input_ids"])
        target_tokens = completion_tokens(completion)
        if (prompt_tokens > FUNCTION_PROMPT_TOKENS or target_tokens > AUXILIARY_COMPLETION_TOKENS
                or prompt_tokens + target_tokens > MAX_SEQUENCE_TOKENS):
            raise SystemExit(f"REFUSED: budget exceeded at position {position}")
        budget_report.append({"prompt": prompt_tokens, "completion": target_tokens})
        treatment[position] = {
            "record_id": evidence["record_id"],
            "function_lineage": evidence["function_lineage"],
            "source_dataset": evidence["source_dataset"],
            "bug_family": evidence["bug_family"],
            "complexity_tier": evidence["complexity_tier"],
            "execution_mode": "function_assertion",
            "task_kind": "execution_output_prediction",
            "supervision_kind": SUPERVISION_KIND,
            "execution_supervision_candidate_identity": evidence["candidate_identity"],
            "call_expression": evidence["call_expression"],
            "prompt": prompt, "completion": completion,
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "completion_sha256": hashlib.sha256(completion.encode("utf-8")).hexdigest(),
            "replaced_control_record_id": control[position]["record_id"],
        }

    removed_set = set(removed)
    kept = [row for position, row in enumerate(control) if position not in removed_set]
    execution_rows = [treatment[position] for position in removed]
    training_lineages = {str(row["function_lineage"]) for row in treatment}
    training_ids = {str(row["record_id"]) for row in treatment}
    panel = json.loads((SOURCE_DIR / "pilot_development.execution.json").read_text(
        encoding="utf-8"))
    confirmation = json.loads((SOURCE_DIR / "unopened_confirmation.ids.json").read_text(
        encoding="utf-8"))
    leakage_checks = {
        "no_pilot_or_confirmation_lineage_in_treatment": not (training_lineages & held_out),
        "no_confirmation_lineage_in_treatment": not (
            training_lineages & set(confirmation["function_lineages"])),
        "no_mechanism_panel_record_in_treatment": not (
            training_ids & {str(row["record_id"]) for row in panel}),
        "every_execution_row_verified_train": all(
            row["function_lineage"] in train_lineages for row in execution_rows),
        "canonical_rows_byte_identical_to_control": all(
            treatment[position] == control[position]
            for position in range(ARM_SIZE) if position not in removed_set),
    }
    if not all(leakage_checks.values()):
        raise SystemExit(f"REFUSED: leakage checks failed: {leakage_checks}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    arm_path = args.output_dir / "arm_dose_treatment.json"
    arm_path.write_bytes((json.dumps(treatment, indent=2, ensure_ascii=False) + "\n")
                         .encode("utf-8"))
    execution_tokens = [entry["completion"] for entry in budget_report]
    manifest = {
        "schema_version": SCHEMA,
        "label": "train-only treatment arm; the frozen control is reused unchanged",
        "evaluation_split": "train",
        "chosen_design": chosen,
        "examples": ARM_SIZE,
        "execution_examples": len(removed),
        "replay_examples": len(kept),
        "replacement_positions": removed,
        "replaced_control_record_ids": [control[p]["record_id"] for p in removed],
        "execution_record_ids": [row["record_id"] for row in execution_rows],
        "supervision_kind": SUPERVISION_KIND,
        "representation": "scripts/build_execution_trace_ab.py ordered=True (unchanged)",
        "lineage_cap": design["lineage_cap"],
        "dose": design["dose"],
        "execution_rows": {
            "unique_records": len({row["record_id"] for row in execution_rows}),
            "unique_lineages": len({row["function_lineage"] for row in execution_rows}),
            "unique_completions": len({row["completion_sha256"] for row in execution_rows}),
            "by_source": _counts(execution_rows, "source_dataset"),
            "by_complexity_tier": _counts(execution_rows, "complexity_tier"),
            "by_bug_family": _counts(execution_rows, "bug_family"),
            "max_rows_per_lineage": max(Counter(
                row["function_lineage"] for row in execution_rows).values()),
            "prompt_tokens": {"total": sum(e["prompt"] for e in budget_report),
                              "min": min(e["prompt"] for e in budget_report),
                              "max": max(e["prompt"] for e in budget_report)},
            "completion_tokens": {"total": sum(execution_tokens),
                                  "min": min(execution_tokens),
                                  "median": statistics.median(execution_tokens),
                                  "max": max(execution_tokens)},
        },
        "replay_balance": design["_balance"],
        "replay_length_shift": {
            "note": ("lever 2 replaces the longest canonical completions within each "
                     "stratum; medians of supervised completion tokens"),
            "control_median_by_stratum": _median_by_stratum(
                control, control_tokens, list(range(ARM_SIZE))),
            "kept_median_by_stratum": _median_by_stratum(
                control, control_tokens, [p for p in range(ARM_SIZE) if p not in removed_set]),
        },
        "treatment_by_source": _counts(treatment, "source_dataset"),
        "treatment_by_task_kind": _counts(treatment, "task_kind"),
        "real_repository_rows": {
            "control": sum(r["source_dataset"] in {"SWE-bench Verified", "BugsInPy"}
                           for r in control),
            "treatment": sum(r["source_dataset"] in {"SWE-bench Verified", "BugsInPy"}
                             for r in treatment)},
        "leakage_checks": leakage_checks,
        "treatment_arm_sha256": sha256_file(arm_path),
        "control_arm_sha256": source_manifest["arm_a_sha256"],
        "control_adapter_training_result": "results/v4_3_execution_supervision_v1/"
                                           "arm_a_training_result.json",
        "pilot_development_sha256": source_manifest["pilot_development_sha256"],
        "unopened_confirmation_ids_sha256": source_manifest["unopened_confirmation_ids_sha256"],
        "census_sha256": sha256_file(args.census),
        "pool_sha256": census["outputs"]["pool"]["sha256"],
        "model": MODEL_NAME, "model_revision": MODEL_REVISION,
        "sealed_final_test_accessed": False, "ablation_dev_accessed": False,
        "validation_accessed": False, "test_accessed": False, "confirmation_opened": False,
        "source_files_sha256": {relative: sha256_file(ROOT / relative) for relative in (
            "scripts/build_execution_dose_ab.py", "harness/execution_dose.py",
            "scripts/build_execution_trace_ab.py", "harness/execution_supervision.py",
            "engine/sft_trainer.py")},
    }
    manifest_bytes = (json.dumps(manifest, indent=2) + "\n").encode("utf-8")
    (args.output_dir / "manifest.json").write_bytes(manifest_bytes)
    args.manifest_copy.write_bytes(manifest_bytes)

    public = {name: {key: value for key, value in report.items() if not key.startswith("_")}
              for name, report in designs.items()}
    design_report = {
        "schema_version": "oneiros_execution_dose_design_v1",
        "label": "CPU design selection from train supply only; no model outcome was used",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "reference_12pct_arm": {
            "execution_examples": trace_manifest["replacement_examples"],
            "execution_example_share": round(trace_manifest["replacement_examples"] / ARM_SIZE, 6),
            "tier_mix": tier_mix,
        },
        "selection_rule": "largest dose meeting every hard constraint",
        "designs": public,
        "feasible_designs": feasible,
        "chosen_design": chosen,
        "dataset_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "census_sha256": sha256_file(args.census),
    }
    args.design_report.write_bytes((json.dumps(design_report, indent=2) + "\n").encode("utf-8"))
    print(json.dumps({"feasible": feasible, "chosen": chosen, "dose": design["dose"],
                      "lineage_cap": design["lineage_cap"],
                      "execution_rows": manifest["execution_rows"],
                      "leakage_checks": leakage_checks,
                      "treatment_arm_sha256": manifest["treatment_arm_sha256"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
