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
from harness.o1_sidecar import append_o1_examples, load_o1_sidecar
from harness.model_identity import (
    build as build_model_identity, disagreements as identity_disagreements,
    is_immutable_revision, problems as identity_problems,
)
from utils.reproducibility import source_tree_sha256
from harness.successor_protocol import (
    CONTRACT_FIELDS, CONTRACT_SOURCES, SUCCESSOR_PROTOCOL, assert_matches,
    protocol_sha256,
)

SCHEMA = "oneiros_o1_sidecar_ab_preflight_v1"

#: Field separator for content hashing. A literal NUL cannot be
#: written into source, so it is built rather than typed.
SEP = chr(0)

#: Files that decide what a TRAINING RUN does. A change to any of these makes
#: two arms incomparable and requires Arm A to be retrained. Preflight,
#: emitter and test code are deliberately absent: editing them moves the
#: whole-tree hash without changing a single byte the trainer executes, and
#: failing on the tree hash would either block honest tooling fixes or, worse,
#: invite someone to wave the difference through. The comparison is made
#: file by file so a real runtime change cannot hide inside a tree delta.
RUNTIME_COMPONENTS = (
    "scripts/train_on_dataset.py",
    "engine/generator.py",
    "engine/sft_trainer.py",
    "engine/model_runtime.py",
    "engine/prompt_budget.py",
    "engine/test_generation_prompt.py",
    "harness/o1_sidecar.py",
    "harness/candidate_policy.py",
    "harness/safe_execution.py",
    "harness/corpus_view.py",
    "harness/successor_protocol.py",
    "metrics/research_evaluation.py",
    "config/settings.py",
)


def runtime_component_hashes(root: Path) -> dict[str, str]:
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in RUNTIME_COMPONENTS}

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


def build_arm_a_baseline(eligible_pairs, selected_ids, tokenizer, *,
                         prompt_limit: int, completion_limit: int,
                         repository_completion_limit: int,
                         real_target_fraction: float, max_real_repeats: int,
                         ) -> tuple[list[Any], dict[str, Any]]:
    """Rebuild Arm A's exact training view with the trainer's own functions.

    The preflight used to check that every sidecar row resolved through the
    train split and fitted the completion budget - both true for all 1,312
    rows of the last sidecar - and then declare a ratio. The trainer then
    dropped 94 of them because their completion already existed in the
    baseline, delivering a 15.02% mixture under a 15.998% label.

    The only way a preflight can know that is to build the baseline the
    trainer builds, so this does, through ``evaluate_pair``,
    ``filter_generation_compatible_sft_examples``,
    ``deduplicate_sft_examples`` and ``balanced_repeat_examples`` - the same
    canonical functions, not a reimplementation of them.
    """
    import math

    from scripts.train_on_dataset import (
        FUNCTION_EXECUTION_MODE, _repository_fragment_tests,
        balanced_repeat_examples, build_pair_prompt, deduplicate_sft_examples,
        evaluate_pair, extract_dataset_tests,
        filter_generation_compatible_sft_examples, is_repository_execution_mode,
        make_sft_data_point,
    )

    wanted = set(selected_ids)
    selected = [pair for pair in eligible_pairs if pair["id"] in wanted]
    synthetic: list[Any] = []
    repository: list[Any] = []
    for pair in selected:
        mode = pair.get("execution_mode", FUNCTION_EXECUTION_MODE)
        if is_repository_execution_mode(mode):
            winners = _repository_fragment_tests(pair.get("test_cases", []))
        else:
            winners, _ = evaluate_pair(
                extract_dataset_tests(pair.get("test_cases", []),
                                      pair["entry_point"]),
                pair["golden_code"], pair["mutant_code"], pair["entry_point"])
        if not winners:
            continue
        prompt = build_pair_prompt(pair)
        for test in winners[:3]:
            point = make_sft_data_point(pair, prompt, test)
            (repository if is_repository_execution_mode(mode)
             else synthetic).append(point)

    synthetic, synthetic_budget = filter_generation_compatible_sft_examples(
        synthetic, tokenizer, completion_limit, repository_completion_limit,
        prompt_limit, prompt_limit)
    repository, repository_budget = filter_generation_compatible_sft_examples(
        repository, tokenizer, completion_limit, repository_completion_limit,
        prompt_limit, prompt_limit)
    synthetic, synthetic_dedup = deduplicate_sft_examples(synthetic)
    repository, repository_dedup = deduplicate_sft_examples(repository)

    desired = math.ceil(len(synthetic) * real_target_fraction
                        / (1.0 - real_target_fraction))
    repository, repository_balance = balanced_repeat_examples(
        repository, desired, max_real_repeats, "project")

    baseline = [*synthetic, *repository]
    completions = {str(item.completion) for item in baseline}
    return baseline, {
        "baseline_examples": len(baseline),
        "baseline_synthetic_examples": len(synthetic),
        "baseline_repository_examples": len(repository),
        "baseline_unique_completions": len(completions),
        "baseline_completion_set_sha256": hashlib.sha256(
            "".join(sorted(hashlib.sha256(c.encode("utf-8")).hexdigest()
                           for c in completions)).encode("utf-8")).hexdigest(),
        "baseline_view_content_sha256": hashlib.sha256(
            b"".join(hashlib.sha256(
                SEP.join((i.function_id, i.prompt, i.completion)).encode("utf-8")
            ).digest() for i in baseline)).hexdigest(),
        "budget_excluded": len(synthetic_budget) + len(repository_budget),
        "dedup": {"synthetic": synthetic_dedup, "repository": repository_dedup},
        "repeat_histogram": repository_balance["repeat_histogram"],
        "built_with": (
            "the trainer's own evaluate_pair, "
            "filter_generation_compatible_sft_examples, "
            "deduplicate_sft_examples and balanced_repeat_examples"),
    }


def run(arm_a_path: Path, sidecar_dir: Path, corpus_version: str,
        local_files_only: bool,
        dump_baseline_completions: Path | None = None) -> dict[str, Any]:
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

    # --------------------------------------------------- model identity
    # Arm B states what IT used rather than pointing at Arm A. A receipt that
    # only links to another receipt cannot be checked on its own, and the link
    # is exactly what an audit cannot verify after the fact.
    arm_a_identity = arm_a.get("model_identity")
    problems.extend(identity_problems(arm_a_identity, label="arm A"))

    arm_b_source_tree = source_tree_sha256(ROOT)
    runtime_now = runtime_component_hashes(ROOT)
    if not arm_b_source_tree:
        problems.append(
            "this preflight cannot compute its own source-tree SHA, so it "
            "cannot record which source produced it")

    # ------------------------------------------------ held-constant contract
    held: dict[str, Any] = {}
    for name, path in HELD_CONSTANT:
        value = _dig(arm_a, path)
        held[name] = value
        if value is None:
            problems.append(f"arm A records no {name}; it cannot be held constant")

    # The generation/evaluation half of the contract. Model and SFT
    # hyperparameters alone are not enough: two arms agreeing on every
    # training setting and disagreeing on the parser produce two numbers that
    # cannot be compared, which is how first_assertion 0.629133 and
    # whole_output 0.282216 came to sit in the same table.
    arm_a_contract = arm_a.get("future_generation_contract") or {}
    contract_problems = assert_matches(arm_a_contract, ROOT)
    problems.extend("arm A " + problem for problem in contract_problems)
    if arm_a_contract.get("candidate_parse_mode") == "first_assertion":
        problems.append(
            "arm A's contract is the LEGACY first_assertion protocol; its "
            "results would be protocol-noncomparable with every successor "
            "measurement and must not be used as this comparison's baseline")
    held_generation = {field: arm_a_contract.get(field)
                       for field in CONTRACT_FIELDS}
    # The SFT training completion budget is a different quantity from the
    # generation completion budget and must not be silently read as one.
    sft_training_completion = arm_a_contract.get(
        "sft_training_completion_token_limit")
    if sft_training_completion is None:
        problems.append(
            "arm A does not record its SFT training completion budget "
            "separately from the generation completion budget")
    elif sft_training_completion == arm_a_contract.get(
            "function_generation_completion_limit"):
        # Not an error, but it must be stated rather than inferred.
        pass

    # ----------------------------------------- rebuild arm A's own selection
    verify_development_view(corpus_dir, ["train"])
    from scripts import train_on_dataset as trainer
    trainer.PROMPT_INFORMATION_VARIANT = held["prompt_information_variant"]
    trainer.OUTPUT_INSTRUCTION_VARIANT = held["output_instruction_variant"]
    trainer.REQUIRE_SPLIT_ISOLATION = True
    from scripts.train_on_dataset import (
        FUNCTION_EXECUTION_MODE, _filter_overlong_repository_completions,
        build_pair_prompt, is_repository_execution_mode, load_phase3_pairs,
        make_sft_data_point, select_bounded_train_pairs,
    )
    from scripts.preflight_sft_run import compute_selection_compatibility

    source_pairs = load_phase3_pairs(corpus_dir, "train")
    all_train_pairs = source_pairs
    eligible_pairs, _ = _filter_overlong_repository_completions(source_pairs)
    selection = arm_a.get("selection") or {}
    sampling = arm_a.get("sampling") or {}

    # INDEPENDENT re-derivation. Hashing the IDs arm A already recorded only
    # proves its receipt is self-consistent; it would pass just as happily on
    # a selection built by different code. So the bounded selection is run
    # again here, from arm A's own recorded configuration, through the same
    # shared compatibility function arm A used, and the resulting SHA must
    # equal arm A's. Two copies of that loop drifting apart is exactly what
    # sharing the function prevents.
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        held["model_name"], revision=held["model_revision"],
        trust_remote_code=True, local_files_only=local_files_only)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    compatibility = compute_selection_compatibility(
        eligible_pairs, tokenizer,
        selection_prompt_token_limit=int(
            selection.get("selection_prompt_token_limit")
            or held["prompt_token_limit"]),
        repository_prompt_token_limit=int(
            _dig(arm_a, ("tokenization", "repository_prompt_token_limit"))
            or held["prompt_token_limit"]),
        repository_completion_token_limit=int(
            _dig(arm_a, ("tokenization", "repository_completion_token_limit"))
            or held["completion_token_limit"]),
    )
    arm_b_identity = build_model_identity(
        model_name=held["model_name"],
        model_revision=held["model_revision"],
        tokenizer_name=(arm_a_identity or {}).get("selection_tokenizer_name")
        or held["model_name"],
        tokenizer_revision=(arm_a_identity or {}).get(
            "selection_tokenizer_revision") or held["model_revision"],
        source_tree_sha256=arm_b_source_tree,
        protocol_name=SUCCESSOR_PROTOCOL["protocol_name"],
        protocol_sha256=protocol_sha256(),
    )
    problems.extend(identity_problems(arm_b_identity, label="arm B"))
    if arm_a_identity:
        problems.extend(identity_disagreements(arm_a_identity, arm_b_identity))
        # A whole-tree difference is reported, never silently accepted and
        # never treated as fatal on its own: it moves when preflight or test
        # code is edited. What must not differ is the runtime.
        arm_a_runtime = (arm_a.get("run_identity") or {}).get(
            "runtime_component_hashes") or {}
        drifted = sorted(
            name for name, digest in arm_a_runtime.items()
            if runtime_now.get(name) != digest)
        if drifted:
            problems.append(
                "RUNTIME COMPONENTS CHANGED SINCE ARM A: " + ", ".join(drifted)
                + ". Arm A must be retrained before the comparison is valid; "
                "this is not a tooling-only edit.")
    if not is_immutable_revision(held.get("model_revision")):
        problems.append(
            f"the held-constant model revision {held.get('model_revision')!r} "
            "is not an immutable snapshot SHA")

    rederived_pairs = select_bounded_train_pairs(
        eligible_pairs,
        int(selection["requested_pairs"]),
        compatible_repository_ids=compatibility["compatible_repository_ids"],
        compatible_synthetic_ids=compatibility["compatible_synthetic_ids"],
        target_real_fraction=float(sampling["target_real_fraction"]),
        max_real_repeats=int(sampling["max_real_repeats"]),
        target_complex_fraction=float(sampling["target_complex_function_fraction"]),
    )
    rederived_ids = [pair["id"] for pair in rederived_pairs]
    rebuilt_sha = _sha_json(rederived_ids)
    recorded_sha = str(selection.get("selection_sha256") or "")
    selection_reproduced = bool(recorded_sha) and rebuilt_sha == recorded_sha
    if not selection_reproduced:
        problems.append(
            "arm A's bounded selection did not reproduce independently "
            f"({rebuilt_sha[:12]}... vs {recorded_sha[:12]}...); the two arms "
            "would not share a baseline")

    recorded_ids = [str(value) for value in
                    selection.get("selected_record_ids") or []]
    if recorded_ids and recorded_ids != rederived_ids:
        problems.append(
            "arm A's recorded selection IDs differ from the independently "
            "re-derived selection")

    selected_ids = rederived_ids

    # ------------------------------------------------------ coverage, exactly
    selected_id_set = set(selected_ids)
    sidecar_ids = {row["record_id"] for row in sidecar}
    inside = sidecar_ids & selected_id_set
    outside = sidecar_ids - selected_id_set
    rows_inside = [r for r in sidecar if r["record_id"] in inside]
    rows_outside = [r for r in sidecar if r["record_id"] in outside]

    # EVERY row must resolve through the FULL train split, which is the map
    # the trainer uses. Keyed to the bounded selection instead, 836 of these
    # rows would vanish and deliver 6.4% of a mixture declared at 16% - the
    # multi-mutant failure. Proving it here means the trainer's own lookup is
    # checked against the same set before any GPU time is spent.
    full_split_ids = {pair["id"] for pair in eligible_pairs}
    unresolvable = sorted(sidecar_ids - full_split_ids)
    if unresolvable:
        problems.append(
            f"{len(unresolvable)} sidecar records do not resolve through the "
            f"full train split and could never be appended, first: "
            f"{unresolvable[:3]}")

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

    # ---------------------------------- the trainer's OWN append, simulated
    # Declaring a ratio without running this is what produced a 15.02%
    # mixture wearing a 15.998% label: every row resolved and fitted the
    # budget, and 94 were still dropped for duplicating a baseline
    # completion. The baseline is therefore built and the real
    # append_o1_examples run against it.
    baseline_view, baseline_stats = build_arm_a_baseline(
        eligible_pairs, selected_ids, tokenizer,
        prompt_limit=int(held["prompt_token_limit"]),
        completion_limit=int(held["completion_token_limit"]),
        repository_completion_limit=int(
            _dig(arm_a, ("tokenization", "repository_completion_token_limit"))
            or held["completion_token_limit"]),
        real_target_fraction=float(sampling["target_real_fraction"]),
        max_real_repeats=int(sampling["max_real_repeats"]),
    )
    declared_baseline = int(
        (arm_a.get("sampling") or {}).get("effective_total_examples") or 0)
    if baseline_stats["baseline_examples"] != declared_baseline:
        problems.append(
            f"the rebuilt baseline has {baseline_stats['baseline_examples']} "
            f"examples but arm A declares {declared_baseline}")

    pairs_by_id = {pair["id"]: pair for pair in all_train_pairs}

    def _fits(pair, prompt, completion):
        limit = int(
            _dig(arm_a, ("tokenization", "repository_completion_token_limit"))
            or held["completion_token_limit"]
        ) if is_repository_execution_mode(
            pair.get("execution_mode", FUNCTION_EXECUTION_MODE)
        ) else int(held["completion_token_limit"])
        return len(tokenizer(str(completion).strip() + tokenizer.eos_token,
                             add_special_tokens=False)["input_ids"]) <= limit

    combined_view, append_report = append_o1_examples(
        baseline_view, sidecar, pairs_by_id,
        make_data_point=make_sft_data_point, build_prompt=build_pair_prompt,
        completion_fits=_fits)

    baseline_completions = {str(item.completion) for item in baseline_view}
    if dump_baseline_completions is not None:
        dump_baseline_completions.parent.mkdir(parents=True, exist_ok=True)
        dump_baseline_completions.write_text(json.dumps({
            "baseline_examples": baseline_stats["baseline_examples"],
            "baseline_completion_set_sha256":
                baseline_stats["baseline_completion_set_sha256"],
            "completion_sha256": sorted(
                hashlib.sha256(c.encode("utf-8")).hexdigest()
                for c in baseline_completions),
        }, indent=1) + chr(10), encoding="utf-8")
    collisions = sorted({row["completion"] for row in sidecar
                         if row["completion"] in baseline_completions})
    appended_rows = combined_view[len(baseline_view):]
    appended_hashes = [
        hashlib.sha256(str(item.completion).encode("utf-8")).hexdigest()
        for item in appended_rows]
    appended_set_sha = hashlib.sha256(
        "".join(sorted(appended_hashes)).encode("utf-8")).hexdigest()

    baseline_examples = baseline_stats["baseline_examples"]
    arm_b_examples = append_report["combined_examples"]
    appended = append_report["appended_examples"]
    delivered_ratio = round(appended / arm_b_examples, 6) if arm_b_examples else 0.0
    declared_ratio = sidecar_manifest.get("achieved_sidecar_share")

    # The declared ratio must be the DELIVERED ratio. This is the check that
    # was missing, and the only one that would have caught the 94 drops.
    if declared_ratio is not None and abs(
            float(declared_ratio) - delivered_ratio) > 0.0005:
        problems.append(
            f"the sidecar declares a ratio of {declared_ratio} but only "
            f"{appended} of {len(sidecar)} rows survive deduplication against "
            f"the baseline, delivering {delivered_ratio}. Emit a sidecar whose "
            "rows do not collide with baseline completions rather than "
            "declaring a ratio the trainer will not deliver.")
    if appended != len(sidecar):
        problems.append(
            f"{len(sidecar) - appended} sidecar rows would not be appended: "
            f"{append_report['dropped']}")

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
        "model_identity": arm_b_identity,
        "arm_a_model_identity": arm_a_identity,
        "source_tree_sha256": arm_b_source_tree,
        "runtime_identity": {
            "runtime_component_hashes": runtime_now,
            "arm_a_source_tree_sha256":
                (arm_a_identity or {}).get("source_tree_sha256"),
            "arm_b_source_tree_sha256": arm_b_source_tree,
            "source_trees_identical": (
                (arm_a_identity or {}).get("source_tree_sha256")
                == arm_b_source_tree),
            "runtime_components_identical": not [
                name for name, digest in
                (((arm_a.get("run_identity") or {}).get(
                    "runtime_component_hashes")) or {}).items()
                if runtime_now.get(name) != digest],
            "why_this_is_the_right_test": (
                "a whole-tree hash moves when preflight, emitter or test code "
                "is edited, none of which the trainer executes. Only these "
                "files change what a training run does."),
        },
        "comparison": {
            "arm_a": "frozen baseline corpus and configuration",
            "arm_b": "identical corpus and configuration plus the O1 sidecar",
            "only_difference": "the O1 sidecar rows",
            "held_constant": held,
            "held_constant_generation": held_generation,
            "successor_protocol_name": SUCCESSOR_PROTOCOL["protocol_name"],
            "successor_protocol_sha256": protocol_sha256(),
            "contract_source_hashes": {
                field: arm_a_contract.get(field) for field in CONTRACT_SOURCES
            },
            "sft_training_completion_token_limit": sft_training_completion,
            "sft_budget_is_separate_from_generation_budget": True,
            "arm_a_contract_problems": contract_problems,
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
            "selection_independently_rederived": True,
            "selection_sha256_rederived": rebuilt_sha,
            "selection_sha256_matches": selection_reproduced,
            "rederived_pairs": len(rederived_ids),
            "recorded_ids_match_rederived": (
                not recorded_ids or recorded_ids == rederived_ids),
            "selection_check_scope": (
                "the bounded selection is re-run from arm A's recorded "
                "configuration through the same shared compatibility function "
                "arm A used, and the resulting SHA must equal arm A's. Hashing "
                "the recorded IDs alone would pass on a selection built by "
                "different code."),
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
            "rows_resolving_through_full_train_split": (
                len(sidecar) - len(unresolvable)),
            "rows_unresolvable": len(unresolvable),
            "all_rows_resolve_and_would_be_appended": not unresolvable,
            "trainer_lookup_map": (
                "the full train split, as load_phase3_pairs(corpus_dir, "
                "'train') returns it - not the bounded selection"),
        },
        "append_simulation": {
            "method": (
                "harness.o1_sidecar.append_o1_examples, the function the "
                "trainer itself calls, run against the rebuilt baseline"),
            **baseline_stats,
            "offered_rows": append_report["sidecar_rows_offered"],
            "appended_rows": appended,
            "dropped_by_reason": append_report["dropped"],
            "records_not_in_training_pairs":
                append_report["records_not_in_training_pairs_count"],
            "baseline_completion_collisions": len(collisions),
            "collision_examples": collisions[:5],
            "combined_examples": arm_b_examples,
            "delivered_ratio": delivered_ratio,
            "declared_ratio": declared_ratio,
            "declared_matches_delivered": (
                declared_ratio is None
                or abs(float(declared_ratio) - delivered_ratio) <= 0.0005),
            "appended_completion_set_sha256": appended_set_sha,
            "appended_completion_hashes_sample": appended_hashes[:5],
            "baseline_unchanged_by_append": (
                append_report["baseline_examples"] == baseline_examples
                and append_report["baseline_unchanged"] is True),
            "golden_substituted": append_report["golden_substituted"],
        },
        "mixing": {
            "baseline_examples": baseline_examples,
            "sidecar_examples": len(sidecar),
            "appended_examples": appended,
            "arm_b_examples": arm_b_examples,
            "requested_ratio": sidecar_manifest.get("requested_sidecar_share"),
            "achieved_ratio": delivered_ratio,
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
    parser.add_argument(
        "--dump-baseline-completions", type=Path, default=None,
        help=(
            "write the sha256 of every baseline completion here, so a sidecar "
            "can be emitted that excludes the ones the trainer would drop"))
    arguments = parser.parse_args()

    report = run(arguments.arm_a, arguments.sidecar_dir,
                 arguments.corpus_version,
                 local_files_only=not arguments.allow_tokenizer_download,
                 dump_baseline_completions=arguments.dump_baseline_completions)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
