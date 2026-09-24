"""Feasibility and construction of a token-matched replay control (CPU only).

The frozen control has 105,457 supervised tokens; the treatment has 127,108.
A comparator that differs from the treatment *only* in the kind of
supervision must have the treatment's token mass and no execution-output
supervision.  This script asks whether such a control exists, honestly, at
three predeclared absolute tolerances (1%, 2%, 5% of the treatment's mass).

Construction (fixed before any result):

* the 768 shared replay rows are the treatment's own, byte-identical;
* each of the 256 intervention positions holds either the frozen control's
  original row or an unused canonical train row from the *identical*
  (source, complexity tier, execution mode, bug family) cell, so every
  representation count equals the frozen control's exactly;
* swaps are taken largest-gain first (fewest rows changed) with one best-fit
  swap to close the gap; nothing is duplicated, padded or invented;
* candidates are the dataset's own gold tests (the frozen control's
  canonical-candidate builder), never model outputs, from train lineages
  outside the pilot-development (mechanism and retention panels) and
  confirmation lineages, under the frozen control's lineage caps, token
  limits and prompt-compaction feasibility.

The primary control is the tightest feasible tolerance.  If none is feasible,
nothing is written except the negative report.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import statistics
import sys
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness.execution_dose import MATCH_TOLERANCES, plan_token_matched_swaps, stratum
from harness.execution_supervision_sidecar import sha256_file, stable_rank
from harness.source_identity import canonical_sha256
from scripts.build_execution_supervision_dataset import (
    MAX_SEQUENCE_TOKENS, OUTPUT_COMPLETION_TOKEN_LIMIT, PROMPT_TOKEN_LIMIT,
    TRACE_COMPLETION_TOKEN_LIMIT, _canonical_candidates,
)
from scripts.census_execution_dose_pool import CORPUS, corpus_disjointness_problems
from scripts.preflight_execution_supervision_ab import MODEL_NAME, MODEL_REVISION

SCHEMA = "oneiros_execution_dose_matched_control_v1"
SOURCE_DIR = ROOT / "results" / "v4_3_execution_supervision_v1"
DOSE_DIR = ROOT / "results" / "v4_3_execution_dose_v1"
ARM_FILE = "arm_matched_control.json"
REPOSITORY_SOURCES = {"SWE-bench Verified", "BugsInPy"}


def cell(row: dict) -> tuple[str, str, str, str]:
    return (str(row["source_dataset"]), str(row["complexity_tier"]),
            str(row["execution_mode"]), str(row["bug_family"]))


def normalised(text: str) -> str:
    """Whitespace-insensitive form used for the near-duplicate check."""
    return hashlib.sha256(re.sub(r"\s+", " ", text).strip().encode("utf-8")).hexdigest()


def lineage_cap(row: dict) -> int:
    # The frozen control builder's caps (_select_shared).
    return 64 if row["source_dataset"] in REPOSITORY_SOURCES else 4


def shares(rows: list[dict], field) -> dict[str, float]:
    counts = Counter(field(row) for row in rows)
    return {key: round(value / len(rows), 6) for key, value in sorted(counts.items())}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=ROOT / "results"
                        / "v4_3_execution_dose_matched_control.json")
    args = parser.parse_args(argv)

    problems = corpus_disjointness_problems(CORPUS)
    if problems:
        raise SystemExit("REFUSED: " + "; ".join(problems))
    manifest = json.loads((ROOT / "results" / "v4_3_execution_dose_dataset_manifest.json")
                          .read_text(encoding="utf-8"))
    treatment_path = DOSE_DIR / "arm_dose_treatment.json"
    control_path = SOURCE_DIR / "arm_a.control.json"
    if sha256_file(treatment_path) != manifest["treatment_arm_sha256"]:
        raise SystemExit("REFUSED: treatment arm hash mismatch")
    if sha256_file(control_path) != manifest["control_arm_sha256"]:
        raise SystemExit("REFUSED: frozen control arm hash mismatch")
    treatment = json.loads(treatment_path.read_text(encoding="utf-8"))
    control = json.loads(control_path.read_text(encoding="utf-8"))
    positions = list(manifest["replacement_positions"])
    split = json.loads((SOURCE_DIR / "lineage_split.json").read_text(encoding="utf-8"))
    forbidden = set(split["pilot_development_lineages"]) | set(
        split["unopened_confirmation_lineages"])
    panel = json.loads((ROOT / "results" / "v4_3_execution_dose_retention_panel.json")
                       .read_text(encoding="utf-8"))
    mechanism = json.loads((SOURCE_DIR / "pilot_development.execution.json")
                           .read_text(encoding="utf-8"))
    panel_lineages = set(panel["record_lineages"].values()) | {
        str(row["function_lineage"]) for row in mechanism}
    if not panel_lineages <= forbidden:
        raise SystemExit("REFUSED: an evaluation-panel lineage is outside the forbidden set")

    from transformers import AutoTokenizer
    from engine.prompt_budget import PromptBudgetError, compact_unified_user_prompt
    from engine.test_generation_prompt import format_chat_prompt
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, revision=MODEL_REVISION,
                                              trust_remote_code=True)

    def tokens(text: str) -> int:
        return len(tokenizer(text + tokenizer.eos_token, add_special_tokens=False)["input_ids"])

    control_tokens = [tokens(str(row["completion"])) for row in control]
    control_total = sum(control_tokens)
    target_total = manifest["dose"]["treatment_supervised_tokens"]

    records = {str(record["id"]): record for record in __import__(
        "harness.corpus_view", fromlist=["load_development_split"]
    ).load_development_split(CORPUS, "train", include_excluded=True)}
    arm_hashes = {hashlib.sha256(str(row["completion"]).encode()).hexdigest()
                  for row in control + treatment}
    arm_normalised = {normalised(str(row["completion"])) for row in control + treatment}
    rejected: Counter = Counter()
    candidates: list[dict] = []
    for row in _canonical_candidates(records, forbidden):
        if row["function_lineage"] in forbidden:
            rejected["forbidden_lineage"] += 1
            continue
        digest = hashlib.sha256(row["completion"].encode()).hexdigest()
        if digest in arm_hashes:
            rejected["exact_duplicate_of_arm_row"] += 1
            continue
        if normalised(row["completion"]) in arm_normalised:
            rejected["near_duplicate_of_arm_row"] += 1
            continue
        count = tokens(row["completion"])
        repository = str(row["execution_mode"]).startswith("repository_")
        if count > (TRACE_COMPLETION_TOKEN_LIMIT if repository else OUTPUT_COMPLETION_TOKEN_LIMIT):
            rejected["completion_token_limit"] += 1
            continue
        prompt_limit = min(2048 if repository else PROMPT_TOKEN_LIMIT,
                           MAX_SEQUENCE_TOKENS - count)
        try:
            compact_unified_user_prompt(tokenizer, row["canonical_prompt"], prompt_limit,
                                        format_chat_prompt)
        except PromptBudgetError:
            rejected["prompt_budget"] += 1
            continue
        candidates.append({**row, "tokens": count, "cell": cell(row),
                           "id": f"{row['record_id']}#{row['position']}",
                           "normalised": normalised(row["completion"])})

    base_rows = {position: control[position] for position in positions}
    base_tokens = {position: control_tokens[position] for position in positions}
    base_cells = {position: cell(row) for position, row in base_rows.items()}
    fixed_lineages = Counter(str(row["function_lineage"]) for index, row in enumerate(control)
                             if index not in set(positions))
    base_lineages = Counter(str(row["function_lineage"]) for row in base_rows.values())

    def admissible_factory():
        def admissible(candidate: dict, position: int, state: dict) -> bool:
            lineages = fixed_lineages + base_lineages
            for swapped, chosen in state["swaps"].items():
                lineages[str(base_rows[swapped]["function_lineage"])] -= 1
                lineages[str(chosen["function_lineage"])] += 1
            # The row being replaced leaves the arm before the cap is tested.
            lineages[str(base_rows[position]["function_lineage"])] -= 1
            if any(chosen["normalised"] == candidate["normalised"]
                   for chosen in state["swaps"].values()):
                return False
            return lineages[str(candidate["function_lineage"])] < lineage_cap(candidate)
        return admissible

    def build_arm(swaps: dict[int, dict]) -> list[dict]:
        arm = [dict(row) for row in treatment]
        for position in positions:
            source = swaps.get(position)
            if source is None:
                arm[position] = dict(control[position])
            else:
                arm[position] = {
                    "record_id": source["record_id"],
                    "function_lineage": source["function_lineage"],
                    "source_dataset": source["source_dataset"],
                    "bug_family": source["bug_family"],
                    "complexity_tier": source["complexity_tier"],
                    "execution_mode": source["execution_mode"],
                    "task_kind": "test_generation",
                    "prompt": source["canonical_prompt"],
                    "completion": source["completion"],
                    "prompt_sha256": hashlib.sha256(
                        source["canonical_prompt"].encode()).hexdigest(),
                    "completion_sha256": hashlib.sha256(
                        source["completion"].encode()).hexdigest(),
                    "matched_control_swap_for_record_id": control[position]["record_id"],
                }
        return arm

    from engine.sft_trainer import OneirosSFTTrainer
    from scripts.preflight_execution_dose_ab import TRAINER_KWARGS
    from scripts.preflight_execution_supervision_ab import LEARNING_RATE, _data_points

    def prepared(arm: list[dict]) -> dict[str, Any]:
        trainer = OneirosSFTTrainer(model_name=MODEL_NAME, model_revision=MODEL_REVISION,
                                    output_dir=ROOT / "checkpoints" / "preflight_matched",
                                    learning_rate=LEARNING_RATE, **TRAINER_KWARGS)
        trainer.tokenizer = tokenizer
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        dataset = trainer.prepare_dataset(_data_points(arm))
        stats = dict(trainer.dataset_stats)
        stats["supervised_tokens"] = sum(len(row["input_ids"]) - int(row["completion_start"])
                                         for row in dataset)
        stats["prepared_examples"] = len(dataset)
        return stats

    results: dict[str, Any] = {}
    arms: dict[str, list[dict]] = {}
    frozen_stats = prepared(control)
    for tolerance in MATCH_TOLERANCES:
        label = f"{round(tolerance * 100)}pct"
        plan = plan_token_matched_swaps(
            base_tokens, base_cells, candidates, current_total=control_total,
            target_total=target_total, tolerance=tolerance,
            admissible=admissible_factory(),
            order_key=lambda position, candidate: stable_rank(
                "matched_swap", position, candidate["id"]))
        swaps = plan["swaps"]
        arm = build_arm(swaps)
        stats = prepared(arm)
        swapped = [arm[position] for position in swaps]
        originals = [control[position] for position in swaps]
        lineages = Counter(str(row["function_lineage"]) for row in arm)
        entry = {
            "tolerance": tolerance,
            "feasible": plan["feasible"] and stats["supervised_tokens"] == plan["total"]
            and stats["prepared_examples"] == 1024 and not stats["dropped_overlong_examples"],
            "examples": len(arm),
            "execution_output_examples": sum(row["task_kind"] != "test_generation"
                                             for row in arm),
            "supervised_tokens": stats["supervised_tokens"],
            "treatment_supervised_tokens": target_total,
            "ratio_to_treatment": round(stats["supervised_tokens"] / target_total, 6),
            "ratio_to_frozen_control": round(stats["supervised_tokens"] / control_total, 6),
            "rows_swapped_from_frozen_control": len(swaps),
            "swapped_by_source": dict(Counter(r["source_dataset"] for r in swapped)),
            "swapped_completion_tokens": {
                "original_total": sum(tokens(r["completion"]) for r in originals),
                "replacement_total": sum(tokens(r["completion"]) for r in swapped)},
            "shares": {
                "source": shares(arm, lambda r: r["source_dataset"]),
                "complexity_tier": shares(arm, lambda r: r["complexity_tier"]),
                "bug_family": shares(arm, lambda r: r["bug_family"]),
                "execution_mode": shares(arm, lambda r: r["execution_mode"]),
                "repository": shares(arm, lambda r: "repository"
                                     if r["source_dataset"] in REPOSITORY_SOURCES
                                     else "function"),
            },
            "representation_identical_to_frozen_control": all(
                Counter(map(key, arm)) == Counter(map(key, control))
                for key in (cell, stratum)),
            "lineages": len(lineages),
            "max_rows_per_function_lineage": max(Counter(
                str(r["function_lineage"]) for r in arm
                if r["source_dataset"] not in REPOSITORY_SOURCES).values()),
            "duplicate_checks": {
                "exact_duplicate_completions": len(arm) - len(
                    {r["completion_sha256"] for r in arm}),
                "near_duplicate_completions": len(arm) - len(
                    {normalised(r["completion"]) for r in arm}),
                "evaluation_panel_lineages_present": len(
                    {r["function_lineage"] for r in arm} & panel_lineages),
            },
            "trainer_preparation": {key: stats[key] for key in (
                "prepared_examples", "dropped_overlong_examples", "malformed_prompt_examples",
                "prompt_compacted_examples", "prompt_truncated_examples",
                "support_units_dropped", "code_units_dropped", "task_kind_counts")},
            "distribution_shift": {
                "median_completion_tokens_by_stratum": {
                    "frozen_control": _medians(control, [tokens(r["completion"])
                                                         for r in control]),
                    "matched_control": _medians(arm, [tokens(r["completion"]) for r in arm]),
                },
            },
        }
        results[label] = entry
        arms[label] = arm

    def upper_bound(swappable) -> int:
        """Conflict-free ceiling: per cell, longest candidates onto shortest rows.

        Ignores lineage caps and candidate reuse across cells, so the true
        maximum can only be lower.
        """
        pool: dict[Any, list[int]] = defaultdict(list)
        for candidate in candidates:
            pool[candidate["cell"]].append(int(candidate["tokens"]))
        rows: dict[Any, list[int]] = defaultdict(list)
        for position in swappable:
            rows[cell(control[position])].append(control_tokens[position])
        total = control_total
        for key, lengths in rows.items():
            for current, offered in zip(sorted(lengths), sorted(pool[key], reverse=True)):
                total += max(0, offered - current)
        return total

    strict_bound = upper_bound(positions)
    relaxed_bound = upper_bound(range(len(control)))
    bounds = {
        "intervention_positions_only": {
            "max_supervised_tokens": strict_bound,
            "ratio_to_treatment": round(strict_bound / target_total, 6),
            "valid_comparator": True,
            "note": "the only construction in which control and treatment differ solely "
                    "at the 256 intervention positions"},
        "any_position": {
            "max_supervised_tokens": relaxed_bound,
            "ratio_to_treatment": round(relaxed_bound / target_total, 6),
            "valid_comparator": False,
            "note": "a canonical swap at a shared replay position must appear in both arms "
                    "to keep them identical there, which adds equal mass to both and "
                    "cannot close the gap; applied to the control alone it changes the "
                    "replay and the arms no longer differ only by the intervention"},
        "minimum_required_at_5pct": round(target_total * (1 - max(MATCH_TOLERANCES))),
    }

    feasible = [label for label in results if results[label]["feasible"]]
    primary = feasible[0] if feasible else None
    output: dict[str, Any] = {
        "schema_version": SCHEMA,
        "label": "CPU feasibility of a token-matched replay control; train-only; no model "
                 "outcome used",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "target_supervised_tokens": target_total,
        "frozen_control_supervised_tokens": control_total,
        "frozen_control_trainer_preparation": {key: frozen_stats[key] for key in (
            "prompt_compacted_examples", "prompt_truncated_examples",
            "support_units_dropped", "code_units_dropped", "malformed_prompt_examples",
            "dropped_overlong_examples")},
        "candidate_pool": {"admissible": len(candidates), "rejected": dict(rejected),
                           "cells_with_candidates": len({c["cell"] for c in candidates})},
        "construction": ("treatment's 768 replay rows + at each of the 256 intervention "
                         "positions the frozen control's row or an unused gold test from the "
                         "identical source/tier/mode/family cell; fewest swaps"),
        "tolerances": results,
        "upper_bounds": bounds,
        "feasible_tolerances": feasible,
        "primary_tolerance": primary,
        "leakage": {"splits_opened": ["train"], "validation_accessed": False,
                    "ablation_dev_accessed": False, "test_accessed": False,
                    "sealed_final_test_accessed": False, "confirmation_opened": False},
        "source_files_sha256": {relative: canonical_sha256(ROOT / relative) for relative in (
            "scripts/build_execution_dose_matched_control.py", "harness/execution_dose.py",
            "scripts/build_execution_supervision_dataset.py", "engine/sft_trainer.py")},
    }
    if primary is not None:
        arm = arms[primary]
        arm_path = DOSE_DIR / ARM_FILE
        arm_path.write_bytes((json.dumps(arm, indent=2, ensure_ascii=False) + "\n")
                             .encode("utf-8"))
        differing = [index for index, (left, right) in enumerate(zip(arm, treatment))
                     if left != right]
        output["primary_arm"] = {
            "path": arm_path.relative_to(ROOT).as_posix(), "sha256": sha256_file(arm_path),
            "differs_from_treatment_only_at_intervention_positions":
                set(differing) <= set(positions),
            "positions_differing_from_treatment": len(differing),
            "swapped_positions": [i for i in positions if arm[i] != control[i]],
        }
    args.report.write_bytes((json.dumps(output, indent=2) + "\n").encode("utf-8"))
    print(json.dumps({"feasible": feasible, "primary": primary, "upper_bounds": {
                          key: value["ratio_to_treatment"] for key, value in bounds.items()
                          if isinstance(value, dict)},
                      "summary": {label: {key: entry[key] for key in (
                          "feasible", "supervised_tokens", "ratio_to_treatment",
                          "rows_swapped_from_frozen_control")}
                          for label, entry in results.items()}}, indent=2))
    return 0


def _medians(rows: list[dict], counts: list[int]) -> dict[str, float]:
    groups: dict[str, list[int]] = defaultdict(list)
    for row, count in zip(rows, counts):
        groups["/".join(stratum(row))].append(count)
    return {key: statistics.median(values) for key, values in sorted(groups.items())}


if __name__ == "__main__":
    raise SystemExit(main())
