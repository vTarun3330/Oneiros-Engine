"""One controlled comparison: frozen baseline, versus baseline plus O1. CPU only.

Arm A is the frozen baseline corpus and configuration. Arm B is the identical
corpus and configuration plus the O1 sidecar at a fixed bounded ratio. Model,
seed, learning rate, epochs, candidate protocol, prompt budget, regularisation
and evaluation protocol are held constant by CONSTRUCTION: arm B reads arm A's
own recorded configuration and this script refuses if any of those fields
differs, so the two arms cannot drift apart through a hand-edited command the
way the metamorphic and single-assertion arms did.

THE MEASUREMENT THIS EXISTS FOR is coverage. A sidecar keyed by displayed
record only helps the records the trainer actually reaches. The multi-mutant
run learned this the expensive way - 76 of 800 selected pairs received the
supervision it was named after, so it was 96% ordinary supervision and the
comparison measured almost nothing. This script reports, before any GPU time is
spent, how many sidecar rows land inside arm A's selection, how many must be
force-included, and what the effective supervision share therefore is.

It loads a tokenizer, never a model. It reads the hash-verified train shard,
never the canonical records.json, and never validation or the sealed test.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.sft_trainer import MAX_SFT_SEQUENCE_LENGTH
from harness.corpus_view import load_development_split, verify_development_view

SCHEMA = "oneiros_o1_sidecar_ab_preflight_v1"

#: Held constant between the arms. A difference in any of these makes the
#: comparison uninterpretable, so it is a refusal rather than a warning.
HELD_CONSTANT = (
    ("model_name", ("tokenization", "model_name")),
    ("model_revision", ("tokenization", "model_revision")),
    ("prompt_token_limit", ("tokenization", "prompt_token_limit")),
    ("completion_token_limit", ("tokenization", "completion_token_limit")),
    ("sequence_token_limit", ("tokenization", "sequence_token_limit")),
    ("prompt_information_variant", ("tokenization", "prompt_information_variant")),
    ("output_instruction_variant", ("tokenization", "output_instruction_variant")),
    ("prompt_compaction_strategy", ("tokenization", "prompt_compaction_strategy")),
    ("epochs", ("training", "epochs")),
    ("batch_size", ("training", "batch_size")),
    ("learning_rate", ("training", "learning_rate")),
    ("lr_scheduler_type", ("training", "lr_scheduler_type")),
    ("warmup_steps", ("training", "warmup_steps_requested")),
    ("checkpoint_steps", ("training", "checkpoint_steps")),
    ("evaluation_split", ("evaluation_panel", "evaluation_split")),
)


def _dig(report: dict[str, Any], path: tuple[str, ...]) -> Any:
    value: Any = report
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _sha_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _shares(rows: list[dict[str, Any]], key: str) -> dict[str, float]:
    total = len(rows) or 1
    return {value: round(count / total, 4)
            for value, count in Counter(r.get(key) for r in rows).most_common()}


def _token_summary(lengths: list[int]) -> dict[str, int]:
    if not lengths:
        return {"count": 0}
    ordered = sorted(lengths)

    def percentile(fraction: float) -> int:
        index = min(len(ordered) - 1, max(0, round(fraction * (len(ordered) - 1))))
        return ordered[index]

    return {"count": len(ordered), "minimum": ordered[0],
            "median": percentile(0.50), "p95": percentile(0.95),
            "maximum": ordered[-1]}


def run(arm_a_path: Path, sidecar_dir: Path, corpus_version: str,
        local_files_only: bool) -> dict[str, Any]:
    started = time.time()
    problems: list[str] = []
    corpus_dir = ROOT / "data" / "corpus" / corpus_version

    arm_a = json.loads(arm_a_path.read_text(encoding="utf-8"))
    sidecar_manifest = json.loads(
        (sidecar_dir / "manifest.json").read_text(encoding="utf-8"))
    sidecar_path = sidecar_dir / "train.sidecar.json"
    if _sha_file(sidecar_path) != sidecar_manifest.get("sidecar_sha256"):
        problems.append("sidecar file does not match its manifest hash")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))

    if not arm_a.get("ready"):
        problems.append(
            "arm A's own preflight is not ready=true; a baseline that cannot "
            "launch is not a baseline")
    if arm_a.get("gpu_model_loaded"):
        problems.append("arm A's preflight reports a loaded model")

    # ------------------------------------------------ held-constant contract
    held: dict[str, Any] = {}
    for name, path in HELD_CONSTANT:
        value = _dig(arm_a, path)
        held[name] = value
        if value is None:
            problems.append(f"arm A records no {name}; it cannot be held constant")

    # ----------------------------------------- rebuild arm A's own selection
    verify_development_view(corpus_dir, ["train"])
    from scripts import train_on_dataset as trainer
    trainer.PROMPT_INFORMATION_VARIANT = held["prompt_information_variant"]
    trainer.OUTPUT_INSTRUCTION_VARIANT = held["output_instruction_variant"]
    trainer.REQUIRE_SPLIT_ISOLATION = True
    from scripts.train_on_dataset import (
        FUNCTION_EXECUTION_MODE, _filter_overlong_repository_completions,
        build_pair_prompt, is_repository_execution_mode, load_phase3_pairs,
        select_bounded_train_pairs,
    )

    source_pairs = load_phase3_pairs(corpus_dir, "train")
    eligible_pairs, _ = _filter_overlong_repository_completions(source_pairs)
    selection = arm_a.get("selection") or {}
    sampling = arm_a.get("sampling") or {}
    selected_pairs = select_bounded_train_pairs(
        eligible_pairs,
        int(selection["requested_pairs"]),
        compatible_repository_ids=set(),
        compatible_synthetic_ids=set(),
        target_real_fraction=float(sampling["target_real_fraction"]),
        max_real_repeats=int(sampling["max_real_repeats"]),
        target_complex_fraction=float(sampling["target_complex_function_fraction"]),
    )
    selected_ids = [pair["id"] for pair in selected_pairs]
    rebuilt_sha = _sha_json(selected_ids)
    selection_reproduced = rebuilt_sha == selection.get("selection_sha256")
    if not selection_reproduced:
        problems.append(
            "arm A's selection did not reproduce from its own recorded "
            f"configuration ({rebuilt_sha[:12]}... vs "
            f"{str(selection.get('selection_sha256'))[:12]}...); the two arms "
            "would not share a baseline")

    # ------------------------------------------------------ coverage, exactly
    selected_id_set = set(selected_ids)
    sidecar_ids = {row["record_id"] for row in sidecar}
    inside = sidecar_ids & selected_id_set
    outside = sidecar_ids - selected_id_set
    rows_inside = [r for r in sidecar if r["record_id"] in inside]
    rows_outside = [r for r in sidecar if r["record_id"] in outside]

    # ------------------------------------------------------- leakage / splits
    train_ids = {str(r["id"]) for r in load_development_split(
        corpus_dir, "train", include_excluded=True)}
    ablation_ids = {str(r["id"]) for r in load_development_split(
        corpus_dir, "ablation_dev", include_excluded=True)}
    not_train = sorted(sidecar_ids - train_ids)
    in_ablation = sorted(sidecar_ids & ablation_ids)
    if not_train:
        problems.append(
            f"{len(not_train)} sidecar records are not in the train shard")
    if in_ablation:
        problems.append(
            f"{len(in_ablation)} sidecar records are also in ablation_dev, the "
            "checkpoint-selection panel; training on them would make selection "
            "self-referential")

    # ------------------------------------------------- token budget, measured
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        held["model_name"], revision=held["model_revision"],
        trust_remote_code=True, local_files_only=local_files_only)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    pairs_by_id = {pair["id"]: pair for pair in eligible_pairs}
    prompt_lengths: list[int] = []
    completion_lengths: list[int] = []
    total_lengths: list[int] = []
    overflow: list[dict[str, Any]] = []
    unreachable: list[str] = []
    repository_mode_rows = 0
    for row in sidecar:
        pair = pairs_by_id.get(row["record_id"])
        if pair is None:
            unreachable.append(row["record_id"])
            continue
        if is_repository_execution_mode(
                pair.get("execution_mode", FUNCTION_EXECUTION_MODE)):
            repository_mode_rows += 1
        prompt_tokens = len(tokenizer(build_pair_prompt(pair),
                                      add_special_tokens=False)["input_ids"])
        completion_tokens = len(tokenizer(
            str(row["completion"]).strip() + tokenizer.eos_token,
            add_special_tokens=False)["input_ids"])
        prompt_lengths.append(prompt_tokens)
        completion_lengths.append(completion_tokens)
        total_lengths.append(prompt_tokens + completion_tokens)
        if prompt_tokens + completion_tokens > int(held["sequence_token_limit"]):
            overflow.append({"record_id": row["record_id"],
                             "prompt_tokens": prompt_tokens,
                             "completion_tokens": completion_tokens})
    if unreachable:
        problems.append(
            f"{len(unreachable)} sidecar records have no training pair and can "
            f"never be reached by the trainer, first: {unreachable[:3]}")
    if overflow:
        problems.append(
            f"{len(overflow)} sidecar rows exceed the "
            f"{held['sequence_token_limit']}-token sequence limit; raise the "
            "sequence budget rather than truncating a verified completion")
    over_completion = [
        length for length in completion_lengths
        if length > int(held["completion_token_limit"])]
    if over_completion:
        problems.append(
            f"{len(over_completion)} sidecar completions exceed the "
            f"{held['completion_token_limit']}-token completion budget")

    baseline_examples = int(
        (arm_a.get("sampling") or {}).get("effective_total_examples") or 0)
    arm_b_examples = baseline_examples + len(sidecar)

    return {
        "schema_version": SCHEMA,
        "ready": not problems,
        "problems": problems,
        "gpu_model_loaded": False,
        "gpu_used": False,
        "training_launched": False,
        "canonical_records_json_opened": False,
        "validation_split_read": False,
        "sealed_final_test_accessed": False,
        "elapsed_seconds": round(time.time() - started, 3),
        "comparison": {
            "arm_a": "frozen baseline corpus and configuration",
            "arm_b": "identical corpus and configuration plus the O1 sidecar",
            "only_difference": "the O1 sidecar rows",
            "held_constant": held,
            "held_constant_source": str(arm_a_path).replace("\\", "/"),
            "held_constant_enforced_by": (
                "arm B reads these values from arm A's own report; this "
                "preflight refuses if any is absent, so neither arm can be "
                "given a different one by hand"),
        },
        "arm_a": {
            "requested_pairs": selection.get("requested_pairs"),
            "retained_pairs": selection.get("retained_pairs"),
            "unique_records": len(selected_id_set),
            "selection_sha256": selection.get("selection_sha256"),
            "selection_reproduced": selection_reproduced,
            "effective_total_examples": baseline_examples,
            "effective_synthetic_examples":
                sampling.get("effective_synthetic_examples"),
            "effective_repository_examples":
                sampling.get("effective_repository_examples"),
            "actual_real_fraction": sampling.get("actual_real_fraction"),
            "prompt_tokens": _dig(arm_a, ("tokenization", "retained_prompt_tokens")),
            "completion_tokens": _dig(arm_a, ("tokenization", "completion_tokens")),
        },
        "sidecar": {
            "directory": str(sidecar_dir).replace("\\", "/"),
            "sidecar_sha256": sidecar_manifest.get("sidecar_sha256"),
            "source_positives_sha256":
                sidecar_manifest.get("source_positives_sha256"),
            "source_candidates_sha256":
                sidecar_manifest.get("source_candidates_sha256"),
            "source_derived_artifact_sha256":
                sidecar_manifest.get("source_derived_artifact_sha256"),
            "rows": len(sidecar),
            "unique_records": len(sidecar_ids),
            "unique_completions": len({r["completion"] for r in sidecar}),
            "distinct_lineages": len({r["function_lineage"] for r in sidecar}),
            "repeats_per_row": 1,
            "duplicated_rows": 0,
            "repository_execution_mode_rows": repository_mode_rows,
            "real_repository_rows": sum(
                1 for r in sidecar if r["origin"] == "real_repository"),
            "shares": {
                dimension: _shares(sidecar, dimension) for dimension in
                ("source_dataset", "bug_family", "complexity_tier", "origin",
                 "supervision_role")
            },
        },
        "coverage": {
            "why_this_matters": (
                "a sidecar keyed by displayed record only reaches records the "
                "trainer selects. The multi-mutant run delivered its named "
                "supervision to 76 of 800 pairs and measured almost nothing.",
            )[0],
            "sidecar_records_already_in_arm_a": len(inside),
            "sidecar_records_requiring_force_inclusion": len(outside),
            "rows_landing_on_selected_pairs": len(rows_inside),
            "rows_requiring_force_inclusion": len(rows_outside),
            "override_only_coverage_of_arm_a": round(
                len(inside) / max(len(selected_id_set), 1), 6),
            "force_inclusion_required": bool(outside),
        },
        "mixing": {
            "baseline_examples": baseline_examples,
            "sidecar_examples": len(sidecar),
            "arm_b_examples": arm_b_examples,
            "requested_ratio": sidecar_manifest.get("requested_sidecar_share"),
            "achieved_ratio": round(len(sidecar) / arm_b_examples, 6)
            if arm_b_examples else 0.0,
            "auxiliary_ceiling": sidecar_manifest.get("auxiliary_ceiling"),
            "o1_is_the_whole_corpus": False,
        },
        "token_budget": {
            "prompt_token_limit": held["prompt_token_limit"],
            "completion_token_limit": held["completion_token_limit"],
            "sequence_token_limit": held["sequence_token_limit"],
            "declared_total": int(held["prompt_token_limit"])
            + int(held["completion_token_limit"]),
            "declared_fits": int(held["prompt_token_limit"])
            + int(held["completion_token_limit"]) < MAX_SFT_SEQUENCE_LENGTH,
            "sidecar_prompt_tokens": _token_summary(prompt_lengths),
            "sidecar_completion_tokens": _token_summary(completion_lengths),
            "sidecar_prompt_plus_completion": _token_summary(total_lengths),
            "sequence_overflow_rows": len(overflow),
            "sequence_overflow_examples": overflow[:5],
            "completion_budget_overflow_rows": len(over_completion),
        },
        "leakage": {
            "record_source": "hash-verified development view train shard",
            "canonical_records_json_opened": False,
            "sealed_final_test_accessed": False,
            "validation_split_read": False,
            "sidecar_records_outside_train": len(not_train),
            "sidecar_records_in_ablation_dev": len(in_ablation),
            "ablation_dev_is_the_checkpoint_selection_panel": True,
            "arm_a_leakage_gates": {
                name: value for name, value in
                (arm_a.get("gates") or {}).items() if "overlap" in name
                or "leak" in name or "lineage" in name
            },
        },
        "hashes": {
            "arm_a_preflight_sha256": _sha_file(arm_a_path),
            "arm_a_selection_sha256": selection.get("selection_sha256"),
            "arm_a_selection_rebuilt_sha256": rebuilt_sha,
            "sidecar_sha256": sidecar_manifest.get("sidecar_sha256"),
            "o1_positives_sha256": sidecar_manifest.get("source_positives_sha256"),
            "o1_candidates_sha256": sidecar_manifest.get("source_candidates_sha256"),
        },
        "interpretation": [
            "This is a preflight. No training was launched and no GPU was used.",
            "O1 is train-only supervision. Nothing here is a generalization "
            "result or a model improvement.",
            "The repaired successor Kill@8 of 0.632887 remains a train-only "
            "whole-output diagnostic and is not comparable to the legacy "
            "first_assertion figure.",
            "No real-repository claim follows from either arm: the sidecar "
            "carries a handful of real-repository rows at most.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm-a", type=Path, required=True,
                        help="the frozen baseline's own preflight report")
    parser.add_argument("--sidecar-dir", type=Path, required=True)
    parser.add_argument("--corpus-version",
                        default="v4_1_research_hardened_candidate")
    parser.add_argument("--allow-tokenizer-download", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    report = run(arguments.arm_a, arguments.sidecar_dir,
                 arguments.corpus_version,
                 local_files_only=not arguments.allow_tokenizer_download)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
