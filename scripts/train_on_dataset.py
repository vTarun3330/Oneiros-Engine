"""Oneiros Phase 3 training: verified SFT followed by mutation-aware DPO.

The pipeline is intentionally fail-closed: DPO will not start unless an SFT
adapter and its immutable reference snapshot have both been saved.
"""
import argparse
import gc
import hashlib
import json
import math
import random
import re
import shutil
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import CANONICAL_CORPUS_VERSION, model_config, training_config
from harness.candidate_policy import validate_function_assertion
from harness.corpus import sha256_file, valid_corpus_version, verify_corpus
from harness.corpus_view import (
    load_complexity_index,
    load_development_split,
    verify_development_view,
)
from harness.function_complexity import analyze_function_complexity
from harness.safe_execution import classify_assertions
from harness.training_data import extract_dataset_assertions
from metrics.research_evaluation import (
    DEFAULT_K_VALUES,
    evaluate_candidate_slots,
    evaluation_profile_sha256,
    function_result as build_function_result,
    prioritise_diverse_slots,
    sanitise_family_name,
    summarise_function_results,
)
from engine.execution_feedback import build_feedback_prompt, collect_execution_feedback
from engine.prompt_budget import (
    PROMPT_COMPACTION_STRATEGY,
    PromptBudgetError,
    compact_unified_user_prompt,
)
from engine.test_generation_prompt import (
    OUTPUT_INSTRUCTION_VARIANTS,
    PROMPT_INFORMATION_VARIANTS,
    PROMPT_SCHEMA_VERSION,
    build_unified_user_prompt,
    format_chat_prompt,
    normalize_target_symbols,
    task_mode_for_execution_mode,
    test_format_for_execution_mode,
)
from utils.reproducibility import build_reproducibility_manifest, functional_identity
from utils.dataset_identity import DATASET_IDENTITY_POLICY, dataset_name_for_pair, dataset_name_from_source
from utils.sampling_audit import summarize_sampling_weights

DATA_DIR = Path(__file__).parent.parent / "data"
RESULTS_DIR = Path(__file__).parent.parent / "results"
ADAPTER_DIR = Path(__file__).parent.parent / "checkpoints" / "v4_1_research_hardened_sft"
CORPUS_VERSION = CANONICAL_CORPUS_VERSION
EXECUTION_MODE_FILTER = None
TRAINING_PHASE = "sft"
RESTART_DPO = False
CONFIRM_FINAL_TEST = False
EVAL_FEEDBACK_ROUNDS = 0
EVAL_DIVERSITY_MODE = "none"
HOLDOUT_BUG_FAMILY = None
EVALUATION_SPLIT = "val"
PROMPT_INFORMATION_VARIANT = "full"
OUTPUT_INSTRUCTION_VARIANT = "self_contained"

SEED = 42
TESTS_PER_PAIR = 8
BATCH_GEN_SIZE = 2
DPO_BUFFER_SIZE = 8
DPO_BATCH_SIZE = 1
MAX_LOSERS_PER_WINNER = 3
MAX_TRAIN_PAIRS = None
MAX_VALIDATION_PAIRS = None
# DPO is allowed to make model-selection decisions only on ``val``.  The
# group-disjoint ``test`` split is intentionally reserved for one final,
# post-selection measurement.
DPO_VALIDATION_SPLIT = "val"
# Evaluate after this many *trained DPO preference pairs* (not merely source
# records read from the corpus).
DPO_VALIDATION_INTERVAL_PAIRS = 500
MIN_SFT_FUNCTION_KILL_RATE_FOR_DPO = 0.58
SFT_EPOCHS_OVERRIDE = None
SFT_LEARNING_RATE_OVERRIDE = None
SFT_BATCH_SIZE_OVERRIDE = None
SFT_LR_SCHEDULER_TYPE_OVERRIDE = None
SFT_REPOSITORY_COMPLETION_TOKEN_LIMIT_OVERRIDE = None
# A base-model/backend swap is itself a declared ablation, never a silent
# default change: these stay None (canonical Phi-3/eager) unless a run
# explicitly opts in, and the resolved values are recorded in that run's
# reproducibility manifest so it can never be confused with the canonical model.
BASE_MODEL_NAME_OVERRIDE = None
BASE_MODEL_REVISION_OVERRIDE = None
BASE_MODEL_ATTENTION_IMPLEMENTATION_OVERRIDE = None
# Preflight accepts --checkpoint-steps, so production must accept the same
# override or the two can plan different monitor schedules for one run.
SFT_CHECKPOINT_STEPS_OVERRIDE = None
# Which tokenizer decides supervision eligibility. It defaults to the run's own
# base model, because eligibility means "does this actually fit the budget for
# the model being trained". A controlled base-model comparison pins every arm to
# one common tokenizer instead, so the arms train on identical records and only
# the model varies; that pinning must be declared, never inherited by accident.
SFT_SELECTION_TOKENIZER_NAME_OVERRIDE = None


def resolved_base_model_identity() -> Tuple[str, str]:
    """Return the (name, revision) actually used to load and tokenize."""
    name = BASE_MODEL_NAME_OVERRIDE or model_config.model_name
    if BASE_MODEL_REVISION_OVERRIDE is not None:
        revision = BASE_MODEL_REVISION_OVERRIDE
    elif name == model_config.model_name:
        revision = model_config.model_revision
    else:
        revision = "main"
    return name, revision


def resolved_selection_tokenizer_identity() -> Tuple[str, str]:
    """Return the (name, revision) that decides supervision eligibility."""
    if SFT_SELECTION_TOKENIZER_NAME_OVERRIDE:
        name = SFT_SELECTION_TOKENIZER_NAME_OVERRIDE
        revision = (
            model_config.model_revision
            if name == model_config.model_name else "main"
        )
        return name, revision
    return resolved_base_model_identity()
# Real repository records are rare in V2/V3.  Bound their deterministic
# repetition during SFT so verified real behaviour is not drowned out by the
# synthetic corpus, without letting a tiny real subset dominate the model.
SFT_REAL_TARGET_FRACTION = 0.20
SFT_MAX_REAL_REPEATS = 8
SFT_REAL_TARGET_FRACTION_OVERRIDE = None
SFT_MAX_REAL_REPEATS_OVERRIDE = None
SFT_CHECKPOINT_MONITOR_ENABLED = True
SFT_MONITOR_VALIDATION_FUNCTIONS = 500
SFT_MONITOR_PATIENCE = 5
SFT_MONITOR_MIN_FUNCTION_KILL_RATE_OVERRIDE = None
SFT_MIN_MONITOR_CHECKPOINTS_OVERRIDE = None
# New SFT runs use deterministic project-balanced repetition and exact
# supervision deduplication. Existing named runs retain their frozen run
# configuration and cannot silently resume with this changed distribution.
SFT_BALANCED_SAMPLING_ENABLED = True
SFT_SYNTHETIC_BALANCE_FRACTION = 0.0
SFT_SYNTHETIC_BALANCE_MODE = "none"
SFT_MAX_SYNTHETIC_REPEATS = 2
# A bounded run must not accidentally collapse to short, low-branching
# functions. This fraction is a minimum pair-level representation target based
# solely on buggy-side AST metrics. It is separately ablated on ablation_dev.
SFT_COMPLEX_TARGET_FRACTION = 0.60
SFT_COMPLEX_TARGET_FRACTION_OVERRIDE = None
# Prompt-budget ablations must select from one common eligibility universe.
# When unset, bounded selection uses the active training prompt budget. Group J
# explicitly pins this to the smallest admissible budget (1,024) for both arms.
SFT_SELECTION_PROMPT_TOKEN_LIMIT_OVERRIDE = None
REQUIRE_SPLIT_ISOLATION = False
MAX_NEW_TOKENS_OVERRIDE = 128
PROMPT_TOKEN_LIMIT = training_config.sft_prompt_token_limit
REPOSITORY_PROMPT_TOKEN_LIMIT = training_config.sft_repository_prompt_token_limit
MAX_SFT_COMPLETION_TOKENS = 2048
MAX_SFT_GENERATION_COMPATIBLE_TOKENS = MAX_NEW_TOKENS_OVERRIDE
MAX_SFT_REPOSITORY_GENERATION_COMPATIBLE_TOKENS = (
    training_config.sft_repository_completion_token_limit
)
MAX_DPO_COMPLETION_TOKENS = 1024
VALIDATION_ACCOUNTING_SCHEMA_VERSION = 3
FUNCTION_EXECUTION_MODE = "function_assertion"
REPOSITORY_EXECUTION_MODE = "repository_pytest_fragment"
REPOSITORY_UNITTEST_EXECUTION_MODE = "repository_unittest_fragment"
REPOSITORY_EXECUTION_MODES = {
    REPOSITORY_EXECUTION_MODE,
    REPOSITORY_UNITTEST_EXECUTION_MODE,
}
REPOSITORY_EVALUATION_STATUS = "not_implemented_requires_native_project_environment"


def _evaluation_profile_slug() -> str:
    parts = []
    if EVALUATION_SPLIT != "val":
        parts.append(EVALUATION_SPLIT.replace("_", "-"))
    if MAX_VALIDATION_PAIRS:
        parts.append(f"smoke{MAX_VALIDATION_PAIRS}")
    if EVAL_FEEDBACK_ROUNDS:
        parts.append(f"feedback{EVAL_FEEDBACK_ROUNDS}")
    if EVAL_DIVERSITY_MODE != "none":
        parts.append(f"diversity-{EVAL_DIVERSITY_MODE}")
    if HOLDOUT_BUG_FAMILY:
        family = re.sub(r"[^a-z0-9]+", "-", HOLDOUT_BUG_FAMILY.lower()).strip("-")
        parts.append(f"holdout-{family}")
    if PROMPT_INFORMATION_VARIANT != "full":
        parts.append(f"prompt-{PROMPT_INFORMATION_VARIANT.replace('_', '-')}")
    if OUTPUT_INSTRUCTION_VARIANT != "self_contained":
        parts.append(f"instruction-{OUTPUT_INSTRUCTION_VARIANT.replace('_', '-')}")
    return "_".join(parts) or "standard"


def evaluation_results_filename(model_label: str, seed: int) -> str:
    """Keep model/profile/seed evaluations separate and immutable."""
    return f"{model_label}_validation_{_evaluation_profile_slug()}_seed_{seed}.json"


def sft_validation_results_filename(seed: int) -> str:
    """Keep ordered research metrics separate from legacy aggregate results."""
    return evaluation_results_filename("sft", seed)


def normalized_sft_run_hyperparameters(hyperparameters: Dict) -> Dict:
    """Add defaults for opt-in fields introduced after an older run began."""
    normalized = dict(hyperparameters)
    normalized.setdefault("balanced_sampling_enabled", False)
    normalized.setdefault("synthetic_balance_fraction", 0.0)
    normalized.setdefault("synthetic_balance_mode", "none")
    normalized.setdefault("max_synthetic_repeats", 2)
    normalized.setdefault("complex_target_fraction", 0.0)
    normalized.setdefault(
        "selection_prompt_token_limit",
        normalized.get("prompt_token_limit", PROMPT_TOKEN_LIMIT),
    )
    # Runs created before this field existed always used cosine.
    normalized.setdefault("lr_scheduler_type", "cosine")
    # Preserve the identity of historical V3/V4 runs that predate the field.
    # New V4.1 runs always persist the explicit locked 0.58 value.
    normalized.setdefault("min_function_kill_rate", 0.50)
    return normalized


def sft_training_scope(
    max_train_pairs: Optional[int], run_sft: bool, execution_mode: Optional[str],
) -> str:
    """Describe only the supervision scope that produced the frozen SFT adapter.

    ``--max-pairs`` also bounds a DPO smoke selection.  That DPO-only bound must
    not rewrite the SFT identity or a smoke can never load the full-run frozen
    adapter and its locked validation baseline.  The exact bounded DPO records
    are independently fingerprinted by ``_dpo_training_scope_sha256``.
    """
    bounded_sft_pairs = max_train_pairs if run_sft else None
    scope = (
        f"first_{bounded_sft_pairs}_train_records"
        if bounded_sft_pairs else "full_train_split"
    )
    if bounded_sft_pairs:
        scope += ":bounded_selection=stratified_generation_compatible_complex_v5"
    scope += f":execution_mode={execution_mode or 'all'}"
    scope += f":repository_completion_limit={MAX_SFT_COMPLETION_TOKENS}"
    if HOLDOUT_BUG_FAMILY:
        scope += f":holdout_bug_family={HOLDOUT_BUG_FAMILY}"
    scope += f":prompt_token_limit={PROMPT_TOKEN_LIMIT}"
    if bounded_sft_pairs:
        selection_prompt_limit = (
            SFT_SELECTION_PROMPT_TOKEN_LIMIT_OVERRIDE
            if SFT_SELECTION_PROMPT_TOKEN_LIMIT_OVERRIDE is not None
            else PROMPT_TOKEN_LIMIT
        )
        scope += f":selection_prompt_token_limit={selection_prompt_limit}"
    # Only recorded when eligibility is deliberately pinned away from the run's
    # own base model, so fingerprints of existing runs stay stable.
    selection_tokenizer, _ = resolved_selection_tokenizer_identity()
    base_model, _ = resolved_base_model_identity()
    if selection_tokenizer != base_model:
        scope += f":selection_tokenizer={selection_tokenizer}"
    scope += f":repository_prompt_token_limit={REPOSITORY_PROMPT_TOKEN_LIMIT}"
    scope += f":prompt_compaction={PROMPT_COMPACTION_STRATEGY}"
    scope += f":prompt_schema={PROMPT_SCHEMA_VERSION}"
    scope += f":prompt_information={PROMPT_INFORMATION_VARIANT}"
    scope += f":output_instruction={OUTPUT_INSTRUCTION_VARIANT}"
    scope += f":dataset_identity={DATASET_IDENTITY_POLICY}"
    complex_fraction = (
        SFT_COMPLEX_TARGET_FRACTION_OVERRIDE
        if SFT_COMPLEX_TARGET_FRACTION_OVERRIDE is not None
        else SFT_COMPLEX_TARGET_FRACTION
    )
    scope += f":complex_target_fraction={complex_fraction}"
    scope += (
        ":generation_completion_limit="
        f"{MAX_SFT_GENERATION_COMPATIBLE_TOKENS}"
    )
    repository_generation_limit = (
        SFT_REPOSITORY_COMPLETION_TOKEN_LIMIT_OVERRIDE
        if SFT_REPOSITORY_COMPLETION_TOKEN_LIMIT_OVERRIDE is not None
        else MAX_SFT_REPOSITORY_GENERATION_COMPATIBLE_TOKENS
    )
    scope += f":repository_generation_completion_limit={repository_generation_limit}"
    return scope


def resolve_sft_dataset_fingerprint(
    manifest: Dict,
    training_scope: str,
    dataset_version_file: Path,
    run_sft: bool,
) -> str:
    """Use the adapter's frozen SFT scope for evaluation/DPO-only launches."""
    corpus_identity = (
        f"{manifest['corpus_id']}:{manifest['schema_version']}:"
        f"{manifest['files']['records.json']['sha256']}:"
        f"{manifest['files']['splits.json']['sha256']}:"
    )
    computed = f"{corpus_identity}sft_scope={training_scope}"
    if run_sft or not dataset_version_file.exists():
        return computed
    frozen = dataset_version_file.read_text(encoding="utf-8").strip()
    if not frozen.startswith(corpus_identity):
        raise RuntimeError(
            "Existing SFT adapter fingerprint belongs to a different canonical corpus."
        )
    return frozen


def is_repository_execution_mode(execution_mode: str) -> bool:
    return execution_mode in REPOSITORY_EXECUTION_MODES


def extract_dataset_tests(test_cases: List[str], entry_point: str) -> List[str]:
    """Compatibility wrapper for AST-safe dataset assertion extraction."""
    return extract_dataset_assertions(test_cases, entry_point)


def build_prompt(
    code_under_test: str, entry_point: str, specification: str = "",
    execution_mode: str = FUNCTION_EXECUTION_MODE,
    support_context: str = "", target_symbols: Optional[List[str]] = None,
) -> str:
    """Build the one dataset-agnostic, reference-free Oneiros prompt.

    The fixed implementation, mutation diff, dataset identity, oracle result,
    and expected completion are deliberately absent from this API so callers
    cannot accidentally leak them into the model-visible prompt.
    """
    return build_unified_user_prompt(
        code_under_test=code_under_test,
        execution_mode=execution_mode,
        specification=specification,
        support_context=support_context,
        target_symbols=target_symbols,
        entry_point=entry_point,
        information_variant=PROMPT_INFORMATION_VARIANT,
        output_instruction_variant=OUTPUT_INSTRUCTION_VARIANT,
    )


def build_pair_prompt(pair: Dict) -> str:
    """Build a unified prompt from an adapted canonical record."""
    return build_prompt(
        pair.get("prompt_code_under_test") or pair["mutant_code"],
        pair.get("entry_point", ""),
        pair.get("specification", ""),
        pair.get("execution_mode", FUNCTION_EXECUTION_MODE),
        pair.get("support_context", ""),
        pair.get("target_symbols", []),
    )


def _record_to_pair(
    record: Dict, complexity: Optional[Dict[str, Any]] = None,
) -> Dict:
    """Adapt a canonical record to the internal oracle evaluation shape."""
    provenance = record.get("provenance", {})
    source = record.get("source", {})
    execution_mode = record.get("quality", {}).get("execution_mode", FUNCTION_EXECUTION_MODE)
    target_symbols = normalize_target_symbols(
        record.get("target_symbols"), record.get("entry_point", "")
    )
    if execution_mode == FUNCTION_EXECUTION_MODE and complexity is None:
        complexity = analyze_function_complexity(
            record.get("prompt_code_under_test") or record["code_under_test"],
            record["entry_point"],
        ).to_dict()
    return {
        "id": record["id"],
        "task_type": record["task_type"],
        "source": record["source"],
        "entry_point": record["entry_point"],
        "execution_mode": execution_mode,
        "task_mode": record.get("task_mode", task_mode_for_execution_mode(execution_mode)),
        "test_format": record.get("test_format", test_format_for_execution_mode(execution_mode)),
        "target_symbols": target_symbols,
        "support_context": record.get("support_context", ""),
        "prompt_code_under_test": record.get("prompt_code_under_test", record["code_under_test"]),
        "specification": record["specification"],
        "mutant_code": record["code_under_test"],
        "golden_code": record["reference_code"],
        "test_cases": [test["code"] for test in record["tests"]],
        "source_name": source.get("name", "unknown") if isinstance(source, dict) else str(source),
        "dataset_name": dataset_name_from_source(source),
        "dataset_identity_policy": DATASET_IDENTITY_POLICY,
        "project": (
            provenance.get("project")
            or provenance.get("repository")
            or "synthetic"
        ),
        "group_id": record.get("group_id", record["id"]),
        "complexity_tier": (
            str(complexity.get("tier", "unknown")) if complexity else "repository"
        ),
        "function_complexity": dict(complexity or {}),
        "bug_family": (
            provenance.get("mutation_type")
            or provenance.get("category")
            or record["task_type"]
        ),
    }


def make_sft_data_point(pair: Dict, prompt: str, completion: str):
    """One metadata adapter shared by training and exact CPU preflight."""
    from engine.sft_trainer import SFTDataPoint

    dataset = dataset_name_for_pair(pair)
    family = pair.get("bug_family", "unknown") or "unknown"
    return SFTDataPoint(
        prompt=prompt, completion=completion, function_id=pair["id"],
        project=pair.get("project", "synthetic"), bug_family=family,
        semantic_group=pair.get("group_id", pair["id"]),
        execution_mode=pair.get("execution_mode", FUNCTION_EXECUTION_MODE),
        dataset=dataset, dataset_family=f"{dataset}::{family}",
    )


def load_phase3_pairs(corpus_dir: Path, split: str) -> List[Dict]:
    """Load one already-verified canonical split without legacy fallbacks."""
    complexity_index: Dict[str, Dict[str, Any]] = {}
    if REQUIRE_SPLIT_ISOLATION:
        records = load_development_split(corpus_dir, split)
        if split in {"train", "ablation_dev"}:
            complexity_index = load_complexity_index(corpus_dir)
        return [
            _record_to_pair(record, complexity_index.get(record["id"]))
            for record in records
        ]

    records = json.loads((corpus_dir / "records.json").read_text(encoding="utf-8"))
    split_ids = json.loads((corpus_dir / "splits.json").read_text(encoding="utf-8"))[split]
    if split in {"train", "ablation_dev"}:
        exclusions_path = corpus_dir / "training_exclusions.json"
        excluded_ids = {
            item["record_id"]
            for item in (
                json.loads(exclusions_path.read_text(encoding="utf-8"))
                if exclusions_path.exists() else []
            )
        }
        split_ids = [record_id for record_id in split_ids if record_id not in excluded_ids]
    by_id = {record["id"]: record for record in records}
    return [_record_to_pair(by_id[record_id]) for record_id in split_ids]


def _evenly_spaced(items: List[Dict], count: int) -> List[Dict]:
    """Select a deterministic, coverage-oriented subset from an ordered list."""
    if count <= 0:
        return []
    if count >= len(items):
        return list(items)
    if count == 1:
        return [items[len(items) // 2]]
    indices = [round(index * (len(items) - 1) / (count - 1)) for index in range(count)]
    return [items[index] for index in indices]


def _stratified_subset(
    items: List[Dict], count: int, attributes: Tuple[str, ...]
) -> List[Dict]:
    """Select deterministic group-diverse records without changing source order."""
    if count <= 0:
        return []
    if count >= len(items):
        return list(items)
    groups: Dict[Tuple[str, ...], List[Dict]] = {}
    for item in items:
        key = tuple(str(item.get(attribute, "unknown") or "unknown") for attribute in attributes)
        groups.setdefault(key, []).append(item)

    selected: List[Dict] = []
    selected_ids = set()
    group_keys = sorted(groups)
    initial_keys = _evenly_spaced(
        [{"key": key} for key in group_keys], min(count, len(group_keys))
    )
    for descriptor in initial_keys:
        group_items = groups[descriptor["key"]]
        item = group_items[len(group_items) // 2]
        selected.append(item)
        selected_ids.add(item["id"])

    remaining = [item for item in items if item["id"] not in selected_ids]
    selected.extend(_evenly_spaced(remaining, count - len(selected)))
    return selected


def select_bounded_train_pairs(
    pairs: List[Dict], limit: int,
    compatible_repository_ids: Optional[set] = None,
    compatible_synthetic_ids: Optional[set] = None,
    target_real_fraction: float = SFT_REAL_TARGET_FRACTION,
    max_real_repeats: int = SFT_MAX_REAL_REPEATS,
    target_complex_fraction: float = SFT_COMPLEX_TARGET_FRACTION,
) -> List[Dict]:
    """Build a representative smoke subset containing synthetic and real data.

    Canonical split ordering places synthetic records first, so a raw prefix
    silently omits repository supervision. Keep a small real slice here and
    let the normal bounded 8x sampler exercise the intended weighting path.
    """
    if not 0.0 <= target_real_fraction < 1.0:
        raise ValueError("target_real_fraction must be in [0, 1)")
    if max_real_repeats < 1:
        raise ValueError("max_real_repeats must be at least one")
    if not 0.0 <= target_complex_fraction <= 1.0:
        raise ValueError("target_complex_fraction must be in [0, 1]")
    if limit <= 0:
        return list(pairs)
    repository = [
        pair for pair in pairs
        if is_repository_execution_mode(
            pair.get("execution_mode", FUNCTION_EXECUTION_MODE)
        )
    ]
    if compatible_repository_ids is not None:
        repository = [
            pair for pair in repository if pair["id"] in compatible_repository_ids
        ]
    synthetic = [
        pair for pair in pairs
        if not is_repository_execution_mode(
            pair.get("execution_mode", FUNCTION_EXECUTION_MODE)
        )
    ]
    if compatible_synthetic_ids is not None:
        synthetic = [
            pair for pair in synthetic if pair["id"] in compatible_synthetic_ids
        ]
    eligible_ids = {pair["id"] for pair in [*synthetic, *repository]}
    eligible_in_original_order = [
        pair for pair in pairs if pair["id"] in eligible_ids
    ]
    if limit >= len(eligible_in_original_order):
        return eligible_in_original_order
    if not synthetic or limit < 2:
        return eligible_in_original_order[:limit]

    # A synthetic record can contribute three retained winners while a
    # generation-compatible repository record is guaranteed only one. Reserve
    # enough repository records that the bounded 8x sampler can still reach
    # the requested real share in that conservative 3:1 supervision case.
    max_winners_per_synthetic_pair = 3
    real_pair_denominator = (
        max_real_repeats * (1.0 - target_real_fraction)
        + max_winners_per_synthetic_pair * target_real_fraction
    )
    target_repository_count = (
        math.ceil(
            max_winners_per_synthetic_pair
            * target_real_fraction
            * limit
            / real_pair_denominator
        )
        if target_real_fraction
        else 0
    )
    repository_count = (
        min(
            len(repository),
            max(2 if limit >= 32 else 1, target_repository_count),
        )
        if repository
        else 0
    )
    repository_count = min(repository_count, limit - 1)
    synthetic_count = min(len(synthetic), limit - repository_count)
    repository_count = min(len(repository), limit - synthetic_count)
    selected_synthetic = _stratified_subset(
        synthetic, synthetic_count, ("bug_family", "source_name")
    )
    selected_synthetic_ids = {pair["id"] for pair in selected_synthetic}
    current_complex_count = sum(
        pair.get("complexity_tier") == "complex" for pair in selected_synthetic
    )
    desired_complex_count = min(
        sum(pair.get("complexity_tier") == "complex" for pair in synthetic),
        math.ceil(synthetic_count * target_complex_fraction),
    )
    replacements_needed = max(0, desired_complex_count - current_complex_count)
    if replacements_needed:
        replacement_candidates = [
            pair for pair in synthetic
            if pair.get("complexity_tier") == "complex"
            and pair["id"] not in selected_synthetic_ids
        ]
        replacements = _stratified_subset(
            replacement_candidates,
            replacements_needed,
            ("bug_family", "source_name"),
        )
        replacement_ids = {pair["id"] for pair in replacements}
        removable_ids = [
            pair["id"] for pair in reversed(selected_synthetic)
            if pair.get("complexity_tier") != "complex"
        ]
        removable_ids = set(removable_ids[:len(replacements)])
        selected_synthetic = [
            pair for pair in selected_synthetic if pair["id"] not in removable_ids
        ]
        selected_synthetic.extend(
            pair for pair in replacements if pair["id"] in replacement_ids
        )
    selected = [
        *selected_synthetic,
        *_stratified_subset(
            repository, repository_count, ("project", "bug_family")
        ),
    ]
    original_order = {pair["id"]: index for index, pair in enumerate(pairs)}
    return sorted(selected, key=lambda pair: original_order[pair["id"]])


def summarize_train_pair_selection(pairs: List[Dict]) -> Dict:
    """Persist the source/project/family mix used by a bounded run."""
    def counts(attribute: str) -> Dict[str, int]:
        result: Dict[str, int] = {}
        for pair in pairs:
            value = str(pair.get(attribute, "unknown") or "unknown")
            result[value] = result.get(value, 0) + 1
        return dict(sorted(result.items()))

    repository = [
        pair for pair in pairs
        if is_repository_execution_mode(
            pair.get("execution_mode", FUNCTION_EXECUTION_MODE)
        )
    ]
    return {
        "total_pairs": len(pairs),
        "synthetic_pairs": len(pairs) - len(repository),
        "repository_pairs": len(repository),
        "bug_family_counts": counts("bug_family"),
        "dataset_counts": dict(sorted(Counter(
            dataset_name_for_pair(pair) for pair in pairs
        ).items())),
        "ingestion_source_counts": counts("source_name"),
        "dataset_identity_policy": DATASET_IDENTITY_POLICY,
        "complexity_tier_counts": counts("complexity_tier"),
        "complex_function_pairs": sum(
            pair.get("complexity_tier") == "complex" for pair in pairs
        ),
        "complex_function_fraction": round(
            sum(pair.get("complexity_tier") == "complex" for pair in pairs)
            / max(1, len(pairs) - len(repository)),
            6,
        ),
        "repository_project_counts": dict(sorted({
            project: sum(1 for pair in repository if str(pair.get("project", "unknown") or "unknown") == project)
            for project in {str(pair.get("project", "unknown") or "unknown") for pair in repository}
        }.items())),
    }


def balanced_repeat_examples(
    examples: List,
    target_total: int,
    max_repeats: int,
    group_attribute: str,
) -> Tuple[List, Dict]:
    """Repeat examples deterministically while reducing group dominance.

    Every input example is retained exactly once.  Additional copies are
    assigned to the currently smallest eligible group, then to the
    least-repeated example in that group.  The per-example cap prevents a tiny
    project or mutation family from being memorized merely to make aggregate
    percentages look balanced.
    """
    if max_repeats < 1:
        raise ValueError("max_repeats must be at least one")
    if not examples:
        return [], {
            "raw_examples": 0,
            "target_examples": 0,
            "effective_examples": 0,
            "max_repeats": max_repeats,
            "group_attribute": group_attribute,
            "raw_group_counts": {},
            "effective_group_counts": {},
            "repeat_histogram": {},
            "achievable_max_examples": 0,
            "target_reached": target_total <= 0,
        }

    bounded_target = min(
        max(len(examples), int(target_total)),
        len(examples) * max_repeats,
    )
    group_indices: Dict[str, List[int]] = {}
    for index, example in enumerate(examples):
        group = str(getattr(example, group_attribute, "") or "unknown")
        group_indices.setdefault(group, []).append(index)

    repeat_counts = [1] * len(examples)
    group_totals = {group: len(indices) for group, indices in group_indices.items()}
    additional_indices: List[int] = []
    while len(examples) + len(additional_indices) < bounded_target:
        eligible_groups = [
            group for group, indices in group_indices.items()
            if any(repeat_counts[index] < max_repeats for index in indices)
        ]
        if not eligible_groups:
            break
        group = min(eligible_groups, key=lambda item: (group_totals[item], item))
        eligible_indices = [
            index for index in group_indices[group]
            if repeat_counts[index] < max_repeats
        ]
        index = min(
            eligible_indices,
            key=lambda item: (
                repeat_counts[item],
                str(getattr(examples[item], "function_id", "")),
                str(getattr(examples[item], "completion", "")),
                item,
            ),
        )
        repeat_counts[index] += 1
        group_totals[group] += 1
        additional_indices.append(index)

    raw_group_counts = {
        group: len(indices) for group, indices in sorted(group_indices.items())
    }
    repeat_histogram: Dict[str, int] = {}
    for count in repeat_counts:
        repeat_histogram[str(count)] = repeat_histogram.get(str(count), 0) + 1
    return [*examples, *(examples[index] for index in additional_indices)], {
        "raw_examples": len(examples),
        "target_examples": int(target_total),
        "effective_examples": len(examples) + len(additional_indices),
        "max_repeats": max_repeats,
        "group_attribute": group_attribute,
        "raw_group_counts": raw_group_counts,
        "effective_group_counts": dict(sorted(group_totals.items())),
        "repeat_histogram": dict(sorted(repeat_histogram.items(), key=lambda item: int(item[0]))),
        "achievable_max_examples": len(examples) * max_repeats,
        "target_reached": len(examples) + len(additional_indices) >= int(target_total),
    }


def deduplicate_sft_examples(examples: List) -> Tuple[List, Dict]:
    """Remove exact prompt/completion duplicates without semantic guessing."""
    retained = []
    seen = set()
    for example in examples:
        key = (str(example.prompt), str(example.completion).strip())
        if key in seen:
            continue
        seen.add(key)
        retained.append(example)
    return retained, {
        "input_examples": len(examples),
        "retained_examples": len(retained),
        "exact_duplicates_excluded": len(examples) - len(retained),
    }


def filter_generation_compatible_sft_examples(
    examples: List, tokenizer, max_completion_tokens: int,
    max_repository_completion_tokens: Optional[int] = None,
    max_prompt_tokens: Optional[int] = None,
    max_repository_prompt_tokens: Optional[int] = None,
) -> Tuple[List, List[Dict]]:
    """Exclude phase-incompatible targets without altering canonical records.

    SFT previously accepted completions up to the 1,536-token training context
    even though live generation is capped at 128 new tokens.  Optimizing those
    targets can lower teacher-forced loss while teaching output the deployed
    policy cannot emit.  Keep every source record in the corpus and make the
    phase-specific exclusion explicit and reproducible instead of truncating a
    behaviorally verified test.
    """
    repository_completion_limit = (
        max_completion_tokens
        if max_repository_completion_tokens is None
        else max_repository_completion_tokens
    )
    if max_completion_tokens <= 0 or repository_completion_limit <= 0:
        raise ValueError("SFT live-generation completion limit must be positive")
    if not tokenizer.eos_token:
        raise RuntimeError("SFT generation preflight requires a tokenizer EOS token")

    retained = []
    excluded = []
    from engine.sft_trainer import (
        MAX_SFT_SEQUENCE_LENGTH,
        sft_completion_limit_for_execution_mode,
        sft_prompt_limit_for_execution_mode,
    )
    for example in examples:
        completion_tokens = len(tokenizer(
            example.completion.strip() + tokenizer.eos_token,
            add_special_tokens=False,
        )["input_ids"])
        completion_limit = sft_completion_limit_for_execution_mode(
            example.execution_mode,
            max_completion_tokens,
            repository_completion_limit,
        )
        evidence = {
            "record_id": str(example.function_id),
            "execution_mode": str(example.execution_mode),
            "project": str(example.project),
            "bug_family": str(example.bug_family),
            "completion_tokens": completion_tokens,
        }
        if completion_tokens > completion_limit:
            excluded.append({
                **evidence,
                "limit_tokens": completion_limit,
                "reason": "completion_exceeds_live_generation_limit",
            })
            continue

        # A completion is not actually generation-compatible when its prompt
        # cannot preserve the complete target and required semantic sections.
        # The optional arguments retain backwards compatibility for callers
        # performing only a completion audit; every production SFT path passes
        # both prompt budgets.
        if max_prompt_tokens is not None:
            repository_prompt_limit = (
                max_prompt_tokens
                if max_repository_prompt_tokens is None
                else max_repository_prompt_tokens
            )
            prompt_limit = sft_prompt_limit_for_execution_mode(
                example.execution_mode,
                max_prompt_tokens,
                repository_prompt_limit,
            )
            allowed_prompt_tokens = min(
                prompt_limit,
                MAX_SFT_SEQUENCE_LENGTH - completion_tokens,
            )
            try:
                compaction = compact_unified_user_prompt(
                    tokenizer,
                    example.prompt,
                    allowed_prompt_tokens,
                    format_chat_prompt,
                )
            except (PromptBudgetError, ValueError) as exc:
                excluded.append({
                    **evidence,
                    "prompt_limit_tokens": max(0, allowed_prompt_tokens),
                    "reason": "required_prompt_sections_exceed_mode_budget",
                    "detail": str(exc),
                })
                continue
            if compaction.final_token_count + completion_tokens > MAX_SFT_SEQUENCE_LENGTH:
                excluded.append({
                    **evidence,
                    "prompt_tokens": compaction.final_token_count,
                    "sequence_limit_tokens": MAX_SFT_SEQUENCE_LENGTH,
                    "reason": "prompt_and_completion_exceed_sequence_budget",
                })
                continue

        retained.append(example)
    return retained, excluded


def supervision_exclusion_summary(record_ids: List[str]) -> Dict:
    """Create a deterministic audit record for selected pairs with no SFT target."""
    ordered_ids = list(dict.fromkeys(str(record_id) for record_id in record_ids))
    return {
        "reason": "no_policy_and_reference_valid_mutation_killing_completion",
        "count": len(ordered_ids),
        "record_ids": ordered_ids,
        "record_ids_sha256": hashlib.sha256(
            json.dumps(ordered_ids, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "canonical_records_modified": False,
    }


def format_generation_prompt(tokenizer, prompt: str) -> str:
    """Render the unified system/user prefix shared with SFT, DPO, and inference."""
    return format_chat_prompt(tokenizer, prompt)


def evaluate_pair(
    tests: List[str], golden_code: str, mutant_code: str, entry_point: str,
):
    """Return policy-valid, reference-valid killing and non-killing assertions."""
    winners, losers = [], []
    policy_valid = []
    for test in tests:
        policy = validate_function_assertion(test, entry_point)
        if policy.valid:
            policy_valid.append(test)
    for outcome in classify_assertions(policy_valid, golden_code, mutant_code):
        if not outcome["valid"]:
            continue
        if outcome["killed"]:
            winners.append(outcome["test"])
        else:
            losers.append(outcome["test"])
    return winners, losers


def generate_tests_mock(golden_code: str, entry_point: str, num: int = 4) -> List[str]:
    """Generate deterministic non-model candidates for bootstrap negatives only."""
    from baseline.benchmark_runner import StaticBaseline

    return StaticBaseline().generate_tests(golden_code, entry_point, [], num_tests=num)


def bootstrap_losers(pair: Dict) -> List[str]:
    """Supply evaluated rejected completions when dataset tests are all winners."""
    candidates = generate_tests_mock(
        pair["golden_code"], pair["entry_point"], num=TESTS_PER_PAIR
    )
    _, losers = evaluate_pair(
        candidates, pair["golden_code"], pair["mutant_code"], pair["entry_point"]
    )
    # A valid no-op assertion is a last-resort rejected completion. It is only
    # used if no evaluated non-killing function call can be generated.
    return losers or ["assert True"]


def generate_tests_ai_batched(
    generator, pairs_chunk: List[Dict], num: int = 4,
    return_accounting: bool = False,
    prompt_additions: Optional[Dict[int, str]] = None,
    rank_offset: int = 0,
):
    """Generate multiple samples per pair using the same chat prompt as training."""
    # Keep corpus/audit imports lightweight.  CUDA/PyTorch is required only
    # when this live model-generation path is actually invoked.
    import torch

    if any(pair.get("execution_mode", FUNCTION_EXECUTION_MODE) != FUNCTION_EXECUTION_MODE for pair in pairs_chunk):
        raise ValueError("Live assertion generation only supports function_assertion records")
    if not generator.is_loaded:
        generator.load_model()

    tokenizer = generator.tokenizer
    prompt_additions = prompt_additions or {}
    compacted_prompt_ids = []
    # Section-aware compaction is fail-closed: it refuses to slice a target
    # function in half.  A record whose required sections cannot fit the mode
    # budget must therefore be recorded as an unusable evaluation prompt, not
    # allowed to abort the whole batch.  Its candidate slots stay in the
    # accounting so requested-candidate counts and Kill@k denominators remain
    # exact.
    generable_indexes: List[int] = []
    prompt_budget_failures: Dict[int, str] = {}
    for index, pair in enumerate(pairs_chunk):
        prompt = build_pair_prompt(pair)
        addition = prompt_additions.get(index, "").strip()
        if addition:
            prompt = f"{prompt}\n\n{addition}"
        try:
            compaction = compact_unified_user_prompt(
                tokenizer,
                prompt,
                PROMPT_TOKEN_LIMIT,
                format_chat_prompt,
            )
        except (PromptBudgetError, ValueError) as exc:
            prompt_budget_failures[index] = str(exc)
            continue
        compacted_prompt_ids.append(compaction.token_ids)
        generable_indexes.append(index)

    outputs: List[Any] = []
    input_length = 0
    if compacted_prompt_ids:
        original_padding_side = tokenizer.padding_side
        tokenizer.padding_side = "left"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        try:
            inputs = tokenizer.pad(
                [
                    {
                        "input_ids": token_ids,
                        "attention_mask": [1] * len(token_ids),
                    }
                    for token_ids in compacted_prompt_ids
                ],
                padding=True,
                return_tensors="pt",
            ).to(generator.model.device)
            input_length = inputs.input_ids.shape[1]
            generator.model.eval()
            with torch.inference_mode():
                outputs = generator.model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS_OVERRIDE,
                    temperature=generator.temperature,
                    top_p=generator.top_p,
                    do_sample=True,
                    num_return_sequences=num,
                    pad_token_id=tokenizer.pad_token_id,
                    use_cache=True,
                )
        finally:
            tokenizer.padding_side = original_padding_side

    results: Dict[int, List[str]] = {}
    accounting = {
        index: {
            "requested_candidates": num,
            "raw_generated_sequences": 0,
            "parsed_candidates": 0,
            "generation_invalid_candidates": num,
            "candidate_slots": [
                {
                    "rank": rank_offset + rank + 1,
                    "parse_valid": False,
                    "code": None,
                    "raw_output_sha256": None,
                }
                for rank in range(num)
            ],
            "prompt_budget_failure": index in prompt_budget_failures,
            "prompt_budget_failure_reason": prompt_budget_failures.get(index),
        }
        for index in range(len(pairs_chunk))
    }
    for sequence_index, output in enumerate(outputs):
        generable_position = sequence_index // num
        if generable_position >= len(generable_indexes):
            break
        pair_index = generable_indexes[generable_position]
        accounting[pair_index]["raw_generated_sequences"] += 1
        text = tokenizer.decode(output[input_length:], skip_special_tokens=True)
        local_rank = sequence_index % num
        slot = accounting[pair_index]["candidate_slots"][local_rank]
        slot["raw_output_sha256"] = hashlib.sha256(text.encode("utf-8")).hexdigest()
        parsed = generator._parse_output(text, pairs_chunk[pair_index]["entry_point"])
        if parsed.is_valid:
            results.setdefault(pair_index, []).append(parsed.input_code)
            accounting[pair_index]["parsed_candidates"] += 1
            slot["parse_valid"] = True
            slot["code"] = parsed.input_code
    for item in accounting.values():
        item["generation_invalid_candidates"] = max(
            0, item["requested_candidates"] - item["parsed_candidates"]
        )
    return (results, accounting) if return_accounting else results


def _append_preferences(buffer, dpo_trainer, pair: Dict, winners: List[str], losers: List[str]) -> int:
    """Append bounded winner/loser preferences with the DPO chat-formatted prompt."""
    from engine.dpo_trainer import DPODataPoint

    if not winners or not losers:
        return 0
    prompt = dpo_trainer.format_prompt(
        build_pair_prompt(pair)
    )
    added = 0
    for winner in winners[:2]:
        for loser in random.sample(losers, min(len(losers), MAX_LOSERS_PER_WINNER)):
            buffer.append(DPODataPoint(
                prompt=prompt,
                chosen=winner,
                rejected=loser,
                function_id=pair["id"],
            ))
            added += 1
    return added


def _repository_fragment_tests(test_cases: List[str]) -> List[str]:
    """Keep only syntactically complete official pytest fragments.

    Their fixed-pass/buggy-fail result was proved by the repository ingestion
    gate.  Re-executing them against concatenated source excerpts would be a
    different, invalid oracle, so this intentionally performs syntax-only
    validation here.
    """
    valid = []
    for test in test_cases:
        try:
            compile(test, "<official-pytest-fragment>", "exec")
        except SyntaxError:
            continue
        valid.append(test)
    return valid


def _filter_overlong_repository_completions(
    pairs: List[Dict],
    max_completion_tokens: int = MAX_SFT_COMPLETION_TOKENS,
    exclusion_reason: str = "completion_exceeds_sft_context",
) -> Tuple[List[Dict], List[Dict]]:
    """Keep whole repository completions that fit an explicit token budget.

    A repository fragment must remain whole: token truncation could remove the
    assertion that the official fixed-pass/buggy-fail evidence certified.  Any
    overlong completion is therefore retained in the canonical corpus and
    explicitly excluded from the affected training phase.
    """
    if max_completion_tokens <= 0:
        raise ValueError("Repository completion token budget must be positive")
    repository_pairs = [
        pair for pair in pairs
        if is_repository_execution_mode(pair.get("execution_mode", FUNCTION_EXECUTION_MODE))
    ]
    if not repository_pairs:
        return pairs, []

    from transformers import AutoTokenizer

    selection_model, selection_revision = resolved_selection_tokenizer_identity()
    tokenizer = AutoTokenizer.from_pretrained(
        selection_model,
        revision=selection_revision,
        trust_remote_code=True,
    )
    if not tokenizer.eos_token:
        raise RuntimeError("Repository SFT preflight requires a tokenizer EOS token")

    filtered: List[Dict] = []
    excluded: List[Dict] = []
    for pair in pairs:
        mode = pair.get("execution_mode", FUNCTION_EXECUTION_MODE)
        if not is_repository_execution_mode(mode):
            filtered.append(pair)
            continue
        retained_tests = []
        for completion in _repository_fragment_tests(pair.get("test_cases", [])):
            token_count = len(tokenizer(
                completion.strip() + tokenizer.eos_token, add_special_tokens=False
            )["input_ids"])
            if token_count < max_completion_tokens:
                retained_tests.append(completion)
            else:
                excluded.append({
                    "record_id": pair["id"],
                    "execution_mode": mode,
                    "completion_tokens": token_count,
                    "limit_tokens": max_completion_tokens,
                    "reason": exclusion_reason,
                })
        if retained_tests:
            filtered_pair = dict(pair)
            filtered_pair["test_cases"] = retained_tests
            filtered.append(filtered_pair)

    return filtered, excluded


def _append_repository_preferences(buffer, dpo_trainer, pair: Dict, winners: List[str]) -> int:
    """Add official pytest winners against an explicit non-discriminating loser."""
    from engine.dpo_trainer import DPODataPoint

    if not winners:
        return 0
    prompt = dpo_trainer.format_prompt(
        build_pair_prompt(pair)
    )
    for winner in winners[:2]:
        rejected = _length_matched_repository_loser(
            winner, dpo_trainer.tokenizer, MAX_DPO_COMPLETION_TOKENS
        )
        buffer.append(DPODataPoint(
            prompt=prompt, chosen=winner, rejected=rejected, function_id=pair["id"]
        ))
    return min(len(winners), 2)


def _length_matched_repository_loser(
    winner: str, tokenizer, max_completion_tokens: int = MAX_DPO_COMPLETION_TOKENS,
) -> str:
    """Create a valid non-discriminating pytest completion of similar length.

    DPO works on sequence log-probabilities. Pairing a multi-hundred-token
    official fragment with ``assert True`` creates an avoidable length-driven
    preference and can destabilize half-precision gradients.  The filler is
    deliberately inert: it neither imports nor calls repository code.
    """
    def token_count(text: str) -> int:
        return len(tokenizer(text.strip(), add_special_tokens=False)["input_ids"])

    winner_tokens = token_count(winner)
    if winner_tokens >= max_completion_tokens:
        raise ValueError(
            "Repository DPO winner exceeds the explicit completion context gate: "
            f"{winner_tokens} >= {max_completion_tokens} tokens"
        )

    prefix = [
        "def test_oneiros_non_discriminating():",
        '    """Intentionally does not exercise repository behaviour."""',
    ]

    def candidate(padding_words: int) -> str:
        lines = list(prefix)
        if padding_words:
            padding = " ".join(["oneiros"] * padding_words)
            lines.append(f'    ignored_context = "{padding}"')
        lines.append("    assert True")
        return "\n".join(lines)

    # Find the closest inert completion at or below the winner's token count.
    # This avoids the old character-length approximation, which could expand
    # into more than 1,024 tokens even when the verified winner itself fit.
    target_tokens = max(winner_tokens, token_count(candidate(0)))
    low = 0
    high = 1
    while (
        token_count(candidate(high)) <= target_tokens
        and token_count(candidate(high)) < max_completion_tokens
    ):
        low = high
        high *= 2
    while low + 1 < high:
        middle = (low + high) // 2
        count = token_count(candidate(middle))
        if count <= target_tokens and count < max_completion_tokens:
            low = middle
        else:
            high = middle
    rejected = candidate(low)
    rejected_tokens = token_count(rejected)
    if rejected_tokens >= max_completion_tokens:
        raise RuntimeError("Generated repository DPO loser exceeded its context gate")
    compile(rejected, "<non-discriminating-repository-test>", "exec")
    return rejected


def _dpo_training_scope_sha256(pairs: List[Dict]) -> str:
    """Fingerprint the exact ordered records and completions visible to DPO."""
    payload = [
        {
            "record_id": pair["id"],
            "execution_mode": pair.get("execution_mode", FUNCTION_EXECUTION_MODE),
            "test_case_sha256": [
                hashlib.sha256(test.encode("utf-8")).hexdigest()
                for test in pair.get("test_cases", [])
            ],
        }
        for pair in pairs
    ]
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _release_sft_trainer(sft_trainer) -> None:
    """Release all references after either SFT success or failure."""
    import torch

    if sft_trainer is not None:
        if getattr(sft_trainer, "model", None) is not None:
            sft_trainer.model = None
        if getattr(sft_trainer, "tokenizer", None) is not None:
            sft_trainer.tokenizer = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _copy_adapter_snapshot(source_dir: Path, destination_dir: Path) -> str:
    """Copy a persisted adapter snapshot without removing any existing files."""
    source_adapter = source_dir / "adapter_model.safetensors"
    if not source_adapter.exists():
        raise RuntimeError(f"Best validation adapter is missing: {source_adapter}")
    destination_dir.mkdir(parents=True, exist_ok=True)
    for source_file in sorted(source_dir.iterdir()):
        if source_file.is_file() and source_file.name != "validation_metrics.json":
            shutil.copy2(source_file, destination_dir / source_file.name)
    destination_adapter = destination_dir / "adapter_model.safetensors"
    if sha256_file(destination_adapter) != sha256_file(source_adapter):
        raise RuntimeError("Copied validation adapter checksum does not match its source")
    return sha256_file(destination_adapter)


def _evaluation_scope_pairs(corpus_dir: Path, evaluation_split: str) -> List[Dict]:
    """Load the exact deterministic records visible to an adapter evaluation."""
    pairs = load_phase3_pairs(corpus_dir, evaluation_split)
    if EXECUTION_MODE_FILTER:
        pairs = [
            pair for pair in pairs
            if pair.get("execution_mode", FUNCTION_EXECUTION_MODE) == EXECUTION_MODE_FILTER
        ]
    if HOLDOUT_BUG_FAMILY:
        pairs = [
            pair for pair in pairs
            if str(pair.get("bug_family", "unknown")).strip().lower()
            == HOLDOUT_BUG_FAMILY
        ]
    if MAX_VALIDATION_PAIRS:
        pairs = [
            pair for pair in pairs
            if pair.get("execution_mode", FUNCTION_EXECUTION_MODE)
            == FUNCTION_EXECUTION_MODE
        ][:MAX_VALIDATION_PAIRS]
    return pairs


def _candidate_round_sizes(total: int, feedback_rounds: int) -> List[int]:
    """Allocate one fixed generation budget across initial and repair rounds."""
    if total <= 0 or feedback_rounds < 0:
        raise ValueError("Candidate budget must be positive and feedback rounds non-negative")
    rounds = min(total, feedback_rounds + 1)
    quotient, remainder = divmod(total, rounds)
    return [quotient + int(index < remainder) for index in range(rounds)]


def _generate_evaluation_candidates(generator, pairs_chunk: List[Dict]):
    """Generate a fixed-budget batch with optional reference-free feedback."""
    round_sizes = _candidate_round_sizes(TESTS_PER_PAIR, EVAL_FEEDBACK_ROUNDS)
    combined: Dict[int, Dict[str, Any]] = {
        index: {
            "requested_candidates": 0,
            "raw_generated_sequences": 0,
            "parsed_candidates": 0,
            "generation_invalid_candidates": 0,
            "candidate_slots": [],
            "feedback_rounds_completed": 0,
            "prompt_budget_failure": False,
            "prompt_budget_failure_reason": None,
        }
        for index in range(len(pairs_chunk))
    }
    additions: Dict[int, str] = {}
    prior_codes: Dict[int, List[str]] = {index: [] for index in combined}
    rank_offset = 0
    for round_index, round_size in enumerate(round_sizes):
        if EVAL_DIVERSITY_MODE != "none" and round_index == 0:
            additions = {
                index: (
                    "Prefer an unusual boundary or input structure and avoid a generic happy-path test."
                )
                for index in combined
            }
        generated, accounting = generate_tests_ai_batched(
            generator,
            pairs_chunk,
            round_size,
            return_accounting=True,
            prompt_additions=additions,
            rank_offset=rank_offset,
        )
        for index, pair in enumerate(pairs_chunk):
            item = combined[index]
            current = accounting[index]
            for field in (
                "requested_candidates",
                "raw_generated_sequences",
                "parsed_candidates",
                "generation_invalid_candidates",
            ):
                item[field] += int(current[field])
            item["candidate_slots"].extend(current["candidate_slots"])
            prior_codes[index].extend(generated.get(index, []))
            item["feedback_rounds_completed"] = round_index
            if current.get("prompt_budget_failure"):
                item["prompt_budget_failure"] = True
                item["prompt_budget_failure_reason"] = current.get(
                    "prompt_budget_failure_reason"
                )

        rank_offset += round_size
        if round_index + 1 >= len(round_sizes):
            continue
        additions = {}
        for index, pair in enumerate(pairs_chunk):
            feedback = collect_execution_feedback(
                prior_codes[index], pair["mutant_code"]
            )
            additions[index] = build_feedback_prompt(
                feedback,
                require_novel_shape=EVAL_DIVERSITY_MODE != "none",
            )

    generated_results: Dict[int, List[str]] = {}
    for index, pair in enumerate(pairs_chunk):
        slots = prioritise_diverse_slots(
            combined[index]["candidate_slots"],
            pair["entry_point"],
            EVAL_DIVERSITY_MODE,
        )
        combined[index]["candidate_slots"] = slots
        generated_results[index] = [
            str(slot["code"])
            for slot in slots
            if slot.get("parse_valid") and slot.get("code")
        ]
    return generated_results, combined


def _adapter_evaluation_context(
    dataset_fingerprint: str,
    adapter_label: str,
    adapter_sha256: str,
    evaluation_split: str,
    checkpoint_pairs: Optional[int],
    evaluation_scope_sha256: str,
    function_count: int,
) -> Dict:
    """Identity fields that make validation progress safe to resume."""
    resolved_base_model_name = BASE_MODEL_NAME_OVERRIDE or model_config.model_name
    resolved_base_model_revision = (
        BASE_MODEL_REVISION_OVERRIDE
        if BASE_MODEL_REVISION_OVERRIDE is not None
        else (
            model_config.model_revision
            if resolved_base_model_name == model_config.model_name
            else "main"
        )
    )
    reproducibility = build_reproducibility_manifest(
        Path(__file__).parent.parent,
        resolved_base_model_name,
        resolved_base_model_revision,
    )
    profile = {
        "feedback_rounds": EVAL_FEEDBACK_ROUNDS,
        "diversity_mode": EVAL_DIVERSITY_MODE,
        "holdout_bug_family": HOLDOUT_BUG_FAMILY,
        "candidate_budget": TESTS_PER_PAIR,
        "max_validation_functions": MAX_VALIDATION_PAIRS,
        "k_values": list(DEFAULT_K_VALUES),
        "dataset_identity_policy": DATASET_IDENTITY_POLICY,
    }
    return {
        "format_version": 3,
        "validation_accounting_schema_version": VALIDATION_ACCOUNTING_SCHEMA_VERSION,
        "dataset_fingerprint": dataset_fingerprint,
        "adapter": adapter_label,
        "adapter_sha256": adapter_sha256,
        "model_artifact_sha256": adapter_sha256,
        "evaluation_split": evaluation_split,
        "final_test_measurement": evaluation_split == "test",
        "checkpoint_pairs": checkpoint_pairs,
        "seed": SEED,
        "tests_per_function": TESTS_PER_PAIR,
        "batch_size": BATCH_GEN_SIZE,
        "evaluation_scope_sha256": evaluation_scope_sha256,
        "function_validation_records": function_count,
        "evaluation_profile": profile,
        "evaluation_profile_sha256": evaluation_profile_sha256(profile),
        "reproducibility": reproducibility,
    }


def _save_adapter_evaluation_progress(
    results_stem: str,
    context: Dict,
    completed_functions: int,
    killed_functions: int,
    generated_candidates: int,
    mutation_killing_candidates: int,
    elapsed_wall_time: float,
    requested_candidates: int = 0,
    parsed_candidates: int = 0,
    generation_invalid_candidates: int = 0,
    execution_invalid_candidates: int = 0,
    function_results: Optional[List[Dict]] = None,
) -> Path:
    """Append one immutable validation checkpoint, including exact RNG state."""
    import torch

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    progress_path = (
        RESULTS_DIR / f"{results_stem}.progress.{completed_functions:06d}.pt"
    )
    if progress_path.exists():
        existing = torch.load(progress_path, map_location="cpu", weights_only=False)
        if existing.get("context") != context:
            raise RuntimeError(
                f"Existing validation checkpoint has incompatible identity: {progress_path}"
            )
        return progress_path
    payload = {
        "context": context,
        "completed_functions": completed_functions,
        "function_validation_killed": killed_functions,
        "generated_candidates": generated_candidates,
        "mutation_killing_candidates": mutation_killing_candidates,
        "requested_candidates": requested_candidates,
        "parsed_candidates": parsed_candidates,
        "generation_invalid_candidates": generation_invalid_candidates,
        "execution_invalid_candidates": execution_invalid_candidates,
        "function_results": list(function_results or []),
        "elapsed_wall_time": elapsed_wall_time,
        "python_rng_state": random.getstate(),
        "torch_cpu_rng_state": torch.get_rng_state(),
        "torch_cuda_rng_states": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
        ),
    }
    torch.save(payload, progress_path)
    return progress_path


def _load_adapter_evaluation_progress(
    results_stem: str, context: Dict,
) -> Optional[Dict]:
    """Load the newest complete compatible checkpoint without removing any file."""
    import torch

    if not RESULTS_DIR.exists():
        return None
    candidates = sorted(
        RESULTS_DIR.glob(f"{results_stem}.progress.*.pt"), reverse=True
    )
    for candidate in candidates:
        try:
            payload = torch.load(candidate, map_location="cpu", weights_only=False)
        except Exception as exc:
            print(
                f"[VALIDATION RESUME] Ignoring unreadable checkpoint {candidate.name}: {exc}",
                flush=True,
            )
            continue
        if payload.get("context") != context:
            continue
        completed = payload.get("completed_functions")
        if (
            not isinstance(completed, int)
            or completed < 0
            or completed > context["function_validation_records"]
            or (
                completed != context["function_validation_records"]
                and completed % context["batch_size"] != 0
            )
        ):
            continue
        payload["checkpoint_path"] = str(candidate)
        return payload
    return None


def _restore_adapter_evaluation_rng(progress: Dict) -> None:
    """Restore the exact generation RNG position saved after a completed batch."""
    import torch

    random.setstate(progress["python_rng_state"])
    torch.set_rng_state(progress["torch_cpu_rng_state"])
    cuda_states = progress.get("torch_cuda_rng_states", [])
    if cuda_states:
        if not torch.cuda.is_available():
            raise RuntimeError("Validation checkpoint requires CUDA RNG state but CUDA is unavailable")
        if len(cuda_states) != torch.cuda.device_count():
            raise RuntimeError("Validation checkpoint CUDA topology changed; refusing unsafe resume")
        torch.cuda.set_rng_state_all(cuda_states)


def _evaluate_adapter_kill_rate(
    corpus_dir: Path,
    dataset_fingerprint: str,
    adapter_dir: Optional[Path],
    adapter_label: str,
    results_filename: str,
    evaluation_split: str = "val",
    checkpoint_pairs: Optional[int] = None,
) -> Dict:
    """Measure one frozen adapter on a group-disjoint evaluation split.

    This is inference-only: it never creates a DPO trainer, changes adapter
    weights, or consults reference implementations when prompting the model.
    Repository records are reported separately because their official test
    fragments require their original repository environments to execute.
    """
    import torch
    from engine.generator import Phi3Generator

    if evaluation_split not in {"ablation_dev", "val", "test"}:
        raise ValueError("Evaluation split must be 'ablation_dev', 'val', or 'test'")
    adapter_file = adapter_dir / "adapter_model.safetensors" if adapter_dir else None
    if adapter_file is not None and not adapter_file.exists():
        raise RuntimeError(f"{adapter_label} validation requires its frozen adapter")
    resolved_base_model_name = BASE_MODEL_NAME_OVERRIDE or model_config.model_name
    resolved_base_model_revision = (
        BASE_MODEL_REVISION_OVERRIDE
        if BASE_MODEL_REVISION_OVERRIDE is not None
        else (
            model_config.model_revision
            if resolved_base_model_name == model_config.model_name
            else "main"
        )
    )
    adapter_sha256 = (
        sha256_file(adapter_file)
        if adapter_file is not None
        else hashlib.sha256(
            f"{resolved_base_model_name}@{resolved_base_model_revision}".encode("utf-8")
        ).hexdigest()
    )

    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    started = time.time()
    generator = Phi3Generator(
        model_name=BASE_MODEL_NAME_OVERRIDE,
        model_revision=BASE_MODEL_REVISION_OVERRIDE,
        attention_implementation=BASE_MODEL_ATTENTION_IMPLEMENTATION_OVERRIDE,
    )
    try:
        generator.load_model()
        if adapter_dir is not None:
            generator.load_lora_adapter(adapter_dir)
        generator.max_new_tokens = MAX_NEW_TOKENS_OVERRIDE

        all_eval_pairs = _evaluation_scope_pairs(corpus_dir, evaluation_split)

        repository_validation_records = sum(
            is_repository_execution_mode(pair.get("execution_mode", FUNCTION_EXECUTION_MODE))
            for pair in all_eval_pairs
        )
        function_pairs = [
            pair for pair in all_eval_pairs
            if pair.get("execution_mode", FUNCTION_EXECUTION_MODE) == FUNCTION_EXECUTION_MODE
        ]
        if not function_pairs:
            raise RuntimeError("Evaluation split has no function-assertion records to evaluate")
        evaluation_scope_sha256 = hashlib.sha256(
            json.dumps(
                [pair["id"] for pair in function_pairs], separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest()

        context = _adapter_evaluation_context(
            dataset_fingerprint,
            adapter_label,
            adapter_sha256,
            evaluation_split,
            checkpoint_pairs,
            evaluation_scope_sha256,
            len(function_pairs),
        )
        context["model_runtime_profile"] = dict(generator.runtime_profile)
        results_stem = Path(results_filename).stem
        final_result_path = RESULTS_DIR / results_filename
        if final_result_path.exists():
            existing_result = json.loads(final_result_path.read_text(encoding="utf-8"))
            if all(existing_result.get(key) == value for key, value in context.items()):
                print(
                    f"[VALIDATION COMPLETE] Reusing verified {results_filename}", flush=True
                )
                return existing_result
            raise RuntimeError(
                f"Existing {results_filename} does not match the requested immutable evaluation"
            )

        progress = _load_adapter_evaluation_progress(results_stem, context)
        if progress:
            start_index = progress["completed_functions"]
            killed_functions = progress["function_validation_killed"]
            generated_candidates = progress["generated_candidates"]
            mutation_killing_candidates = progress["mutation_killing_candidates"]
            requested_candidates = int(progress.get("requested_candidates", 0))
            parsed_candidates = int(progress.get("parsed_candidates", 0))
            generation_invalid_candidates = int(
                progress.get("generation_invalid_candidates", 0)
            )
            execution_invalid_candidates = int(
                progress.get("execution_invalid_candidates", 0)
            )
            function_results = list(progress.get("function_results", []))
            prior_wall_time = float(progress.get("elapsed_wall_time", 0.0))
            _restore_adapter_evaluation_rng(progress)
            print(
                f"[VALIDATION RESUME] Continuing at function {start_index + 1}/"
                f"{len(function_pairs)} from {Path(progress['checkpoint_path']).name}",
                flush=True,
            )
        else:
            start_index = 0
            killed_functions = 0
            generated_candidates = 0
            mutation_killing_candidates = 0
            requested_candidates = 0
            parsed_candidates = 0
            generation_invalid_candidates = 0
            execution_invalid_candidates = 0
            function_results = []
            prior_wall_time = 0.0

        for start in range(start_index, len(function_pairs), BATCH_GEN_SIZE):
            chunk = function_pairs[start:start + BATCH_GEN_SIZE]
            _, generation_accounting = _generate_evaluation_candidates(
                generator, chunk
            )
            for index, pair in enumerate(chunk):
                accounting = generation_accounting[index]
                candidate_outcomes = evaluate_candidate_slots(
                    accounting["candidate_slots"],
                    pair["golden_code"],
                    pair["mutant_code"],
                    pair["entry_point"],
                )
                item = build_function_result(
                    pair["id"],
                    str(pair.get("bug_family", "unknown") or "unknown"),
                    pair["entry_point"],
                    candidate_outcomes,
                    source_name=str(pair.get("source_name", "unknown")),
                    dataset_name=dataset_name_for_pair(pair),
                    project=str(pair.get("project", "unknown")),
                    prompt_budget_failure=bool(
                        accounting.get("prompt_budget_failure")
                    ),
                    prompt_budget_failure_reason=accounting.get(
                        "prompt_budget_failure_reason"
                    ),
                )
                function_results.append(item)
                killed_functions += int(item["killed"])
                mutation_killing_candidates += item["killing_candidates"]
                requested_candidates += item["requested_candidates"]
                parsed_candidates += item["parsed_candidates"]
                generation_invalid_candidates += item["generation_invalid_candidates"]
                execution_invalid_candidates += item["execution_invalid_candidates"]
                generated_candidates += item["valid_candidates"]
            completed = min(start + len(chunk), len(function_pairs))
            checkpoint_path = _save_adapter_evaluation_progress(
                results_stem,
                context,
                completed,
                killed_functions,
                generated_candidates,
                mutation_killing_candidates,
                prior_wall_time + time.time() - started,
                requested_candidates=requested_candidates,
                parsed_candidates=parsed_candidates,
                generation_invalid_candidates=generation_invalid_candidates,
                execution_invalid_candidates=execution_invalid_candidates,
                function_results=function_results,
            )
            print(
                f"{adapter_label} {evaluation_split} evaluation progress="
                f"{completed}/{len(function_pairs)} killed={killed_functions} "
                f"checkpoint={checkpoint_path.name}",
                flush=True,
            )

        research_summary = summarise_function_results(function_results)
        result = {
            "mode": f"{adapter_label}_validation_only",
            **context,
            "evaluation_split_records": len(all_eval_pairs),
            **research_summary,
            "function_results": function_results,
            "repository_validation_records_held": repository_validation_records,
            "repository_evaluation": {
                "status": REPOSITORY_EVALUATION_STATUS,
                "evaluated": 0,
                "held_records": repository_validation_records,
                "included_in_function_kill_rate": False,
            },
            "wall_time": round(prior_wall_time + time.time() - started, 1),
            "resumed_from_completed_functions": start_index,
            "validation_checkpointing": "every_generation_batch",
        }
        if adapter_file is not None and sha256_file(adapter_file) != adapter_sha256:
            raise RuntimeError("Immutable adapter changed during validation; result was not saved")
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        with open(RESULTS_DIR / results_filename, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)
        print(json.dumps(result, indent=2))
        return result
    finally:
        generator.model = None
        generator.tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _locked_sft_monitor_pairs(
    corpus_dir: Path, dataset_fingerprint: str, limit: int
) -> Tuple[List[Dict], str]:
    """Create or verify the fixed validation panel used for SFT early stopping."""
    validation_pairs = [
        pair for pair in load_phase3_pairs(corpus_dir, EVALUATION_SPLIT)
        if pair.get("execution_mode", FUNCTION_EXECUTION_MODE) == FUNCTION_EXECUTION_MODE
        and (
            not HOLDOUT_BUG_FAMILY
            or str(pair.get("bug_family", "unknown")).strip().lower()
            != HOLDOUT_BUG_FAMILY
        )
    ]
    selected = _evenly_spaced(validation_pairs, min(limit, len(validation_pairs)))
    selected_ids = [pair["id"] for pair in selected]
    selection_sha256 = hashlib.sha256(
        json.dumps(selected_ids, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    payload = {
        "dataset_fingerprint": dataset_fingerprint,
        "evaluation_split": EVALUATION_SPLIT,
        "final_test_measurement": False,
        "selection_method": "evenly_spaced_canonical_validation_functions",
        "training_holdout_bug_family": HOLDOUT_BUG_FAMILY,
        "function_count": len(selected),
        "record_ids": selected_ids,
        "selection_sha256": selection_sha256,
    }
    selection_path = RESULTS_DIR / "sft_monitor_selection.json"
    if selection_path.exists():
        existing = json.loads(selection_path.read_text(encoding="utf-8"))
        if existing != payload:
            raise RuntimeError("SFT monitor selection changed for an existing run")
    else:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        selection_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return selected, selection_sha256


def _evaluate_loaded_sft_monitor(
    model,
    tokenizer,
    function_pairs: List[Dict],
    dataset_fingerprint: str,
    selection_sha256: str,
    checkpoint_step: int,
) -> Dict:
    """Evaluate the in-memory SFT policy without changing its training RNG state."""
    import torch
    from engine.generator import Phi3Generator
    from engine.model_runtime import resolve_compute_dtype, runtime_profile

    cpu_rng_state = torch.get_rng_state()
    cuda_rng_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    python_rng_state = random.getstate()
    was_training = bool(model.training)
    original_use_cache = getattr(model.config, "use_cache", None)
    started = time.time()
    generator = Phi3Generator()
    generator.model = model
    generator.tokenizer = tokenizer
    generator.is_loaded = True
    generator.max_new_tokens = MAX_NEW_TOKENS_OVERRIDE
    _, dtype_name = resolve_compute_dtype(torch)
    generator.runtime_profile = runtime_profile(dtype_name)
    try:
        random.seed(SEED)
        torch.manual_seed(SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(SEED)
        if original_use_cache is not None:
            model.config.use_cache = True

        function_results = []
        for start in range(0, len(function_pairs), BATCH_GEN_SIZE):
            chunk = function_pairs[start:start + BATCH_GEN_SIZE]
            _, generation_accounting = generate_tests_ai_batched(
                generator, chunk, TESTS_PER_PAIR, return_accounting=True
            )
            for index, pair in enumerate(chunk):
                accounting = generation_accounting[index]
                outcomes = evaluate_candidate_slots(
                    accounting["candidate_slots"],
                    pair["golden_code"],
                    pair["mutant_code"],
                    pair["entry_point"],
                )
                function_results.append(build_function_result(
                    pair["id"],
                    str(pair.get("bug_family", "unknown") or "unknown"),
                    pair["entry_point"],
                    outcomes,
                    source_name=str(pair.get("source_name", "unknown")),
                    dataset_name=dataset_name_for_pair(pair),
                    project=str(pair.get("project", "unknown")),
                    prompt_budget_failure=bool(
                        accounting.get("prompt_budget_failure")
                    ),
                    prompt_budget_failure_reason=accounting.get(
                        "prompt_budget_failure_reason"
                    ),
                ))
            completed = min(start + len(chunk), len(function_pairs))
            if completed % 50 == 0 or completed == len(function_pairs):
                killed_so_far = sum(item["killed"] for item in function_results)
                print(
                    f"SFT monitor step={checkpoint_step} progress={completed}/"
                    f"{len(function_pairs)} killed={killed_so_far}",
                    flush=True,
                )
        research_metrics = summarise_function_results(
            function_results, DEFAULT_K_VALUES
        )
        function_outcomes_sha256 = hashlib.sha256(
            json.dumps(
                function_results, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest()
        result = {
            "mode": "sft_checkpoint_validation",
            "validation_accounting_schema_version": VALIDATION_ACCOUNTING_SCHEMA_VERSION,
            "dataset_fingerprint": dataset_fingerprint,
            "evaluation_split": EVALUATION_SPLIT,
            "final_test_measurement": False,
            "selection_sha256": selection_sha256,
            "checkpoint_step": checkpoint_step,
            "seed": SEED,
            "tests_per_function": TESTS_PER_PAIR,
            **research_metrics,
            "model_runtime_profile": dict(generator.runtime_profile),
            "function_outcomes_sha256": function_outcomes_sha256,
            "function_results": function_results,
            "wall_time": round(time.time() - started, 1),
        }
        filename = (
            "sft_monitor_baseline.json" if checkpoint_step == 0
            else f"sft_monitor_checkpoint_{checkpoint_step}.json"
        )
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        RESULTS_DIR.joinpath(filename).write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8"
        )
        return result
    finally:
        if original_use_cache is not None:
            model.config.use_cache = original_use_cache
        if was_training:
            model.train()
        torch.set_rng_state(cpu_rng_state)
        if cuda_rng_states is not None:
            torch.cuda.set_rng_state_all(cuda_rng_states)
        random.setstate(python_rng_state)


def _wilson_interval(successes: int, total: int, z: float = 1.959964) -> List[float]:
    """Return a two-sided Wilson score interval without a SciPy dependency."""
    if total <= 0:
        return [0.0, 0.0]
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    margin = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / total
            + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return [round(max(0.0, center - margin), 6), round(min(1.0, center + margin), 6)]


def _paired_function_diagnostics(reference: Dict, evaluation: Dict) -> Dict:
    """Compare the exact same locked functions across two checkpoints."""
    reference_results = {
        item["record_id"]: bool(item["killed"])
        for item in reference.get("function_results", [])
    }
    evaluation_results = {
        item["record_id"]: bool(item["killed"])
        for item in evaluation.get("function_results", [])
    }
    if not reference_results or reference_results.keys() != evaluation_results.keys():
        return {"available": False}
    improved_ids = sorted(
        record_id for record_id in reference_results
        if not reference_results[record_id] and evaluation_results[record_id]
    )
    regressed_ids = sorted(
        record_id for record_id in reference_results
        if reference_results[record_id] and not evaluation_results[record_id]
    )
    discordant = len(improved_ids) + len(regressed_ids)
    if discordant:
        tail = min(len(improved_ids), len(regressed_ids))
        exact_p = min(
            1.0,
            2.0 * sum(
                math.comb(discordant, value) for value in range(tail + 1)
            ) / (2 ** discordant),
        )
    else:
        exact_p = 1.0
    return {
        "available": True,
        "improved_functions": len(improved_ids),
        "regressed_functions": len(regressed_ids),
        "unchanged_functions": len(reference_results) - discordant,
        "net_function_gain": len(improved_ids) - len(regressed_ids),
        "mcnemar_exact_p_value": round(exact_p, 6),
        "statistically_significant_05": exact_p < 0.05,
        "improved_record_ids": improved_ids,
        "regressed_record_ids": regressed_ids,
    }


def _sft_monitor_gate_decision(
    baseline: Dict, trend: List[Dict], evaluation: Dict, patience: int
) -> Dict:
    """Apply best-so-far early stopping with consecutive-checkpoint patience."""
    previous = trend[-1] if trend else baseline
    eligible_best = [
        baseline, *(item for item in trend if item.get("improved", False))
    ]
    best_before = max(
        eligible_best,
        key=lambda item: int(item["function_validation_killed"]),
    )
    previous_killed = int(previous["function_validation_killed"])
    best_killed = int(best_before["function_validation_killed"])
    panel_size = int(evaluation.get("function_validation_records", 0))
    paired = _paired_function_diagnostics(best_before, evaluation)
    minimum_practical_gain = (
        max(2, math.ceil(panel_size * 0.01))
        if paired.get("available") and panel_size else 1
    )
    function_gain = int(evaluation["function_validation_killed"]) - best_killed
    best_candidate_rate = float(
        best_before.get(
            "end_to_end_candidate_kill_rate",
            best_before.get("candidate_kill_rate", 0.0),
        )
    )
    candidate_rate = float(
        evaluation.get(
            "end_to_end_candidate_kill_rate",
            evaluation.get("candidate_kill_rate", 0.0),
        )
    )
    best_parse_rate = float(best_before.get("parse_success_rate", 1.0))
    parse_rate = float(evaluation.get("parse_success_rate", 1.0))
    candidate_health_passed = candidate_rate >= best_candidate_rate - 0.01
    parse_health_passed = parse_rate >= best_parse_rate - 0.01
    improved = (
        function_gain >= minimum_practical_gain
        and candidate_health_passed
        and parse_health_passed
    )
    consecutive_non_improving = (
        0 if improved
        else int(previous.get("consecutive_non_improving_checkpoints", 0)) + 1
    )
    should_stop = consecutive_non_improving >= patience
    return {
        **evaluation,
        "previous_checkpoint_step": int(previous.get("checkpoint_step", 0)),
        "previous_function_kill_rate": float(previous["function_kill_rate"]),
        "previous_function_validation_killed": previous_killed,
        "best_prior_checkpoint_step": int(best_before.get("checkpoint_step", 0)),
        "best_prior_function_kill_rate": float(best_before["function_kill_rate"]),
        "best_prior_function_validation_killed": best_killed,
        "function_gain_over_best": function_gain,
        "minimum_practical_function_gain": minimum_practical_gain,
        "candidate_health_passed": candidate_health_passed,
        "parse_health_passed": parse_health_passed,
        "paired_function_diagnostics": paired,
        "improved": improved,
        "consecutive_non_improving_checkpoints": consecutive_non_improving,
        "patience": patience,
        "should_stop": should_stop,
        "decision": "early_stop" if should_stop else "continue",
    }


def sft_monitor_acceptance_passed(
    monitor_enabled: bool,
    stopped_early: bool,
    best_adapter_path: Optional[str],
    best_metrics: Optional[Dict],
    minimum_function_kill_rate: float,
) -> bool:
    """Require both a preserved improvement and the run's explicit bar."""
    if not 0.0 <= minimum_function_kill_rate <= 1.0:
        raise ValueError("SFT monitor acceptance rate must be in [0, 1]")
    if not monitor_enabled:
        return not stopped_early
    return bool(best_adapter_path) and float(
        (best_metrics or {}).get("function_kill_rate", 0.0)
    ) >= minimum_function_kill_rate


def _build_sft_checkpoint_monitor(
    corpus_dir: Path,
    dataset_fingerprint: str,
    model,
    tokenizer,
):
    """Return a callback that stops when validation kill rate ceases increasing."""
    function_pairs, selection_sha256 = _locked_sft_monitor_pairs(
        corpus_dir, dataset_fingerprint, SFT_MONITOR_VALIDATION_FUNCTIONS
    )
    baseline_path = RESULTS_DIR / "sft_monitor_baseline.json"
    trend_path = RESULTS_DIR / "sft_monitor_trend.json"
    if baseline_path.exists():
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        if baseline.get("selection_sha256") != selection_sha256:
            raise RuntimeError("SFT monitor baseline uses a different validation panel")
    else:
        baseline = _evaluate_loaded_sft_monitor(
            model, tokenizer, function_pairs, dataset_fingerprint,
            selection_sha256, checkpoint_step=0,
        )

    trend = []
    if trend_path.exists():
        trend = json.loads(trend_path.read_text(encoding="utf-8"))
        if not isinstance(trend, list):
            raise RuntimeError("SFT monitor trend is malformed")
    completed_checkpoint_metrics = {
        int(item["checkpoint_step"]): dict(item)
        for item in trend
        if "checkpoint_step" in item
    }

    initial_best_adapter_path = None
    initial_best_metrics = None
    improved_history = [item for item in trend if item.get("improved", False)]
    if improved_history:
        initial_best_metrics = max(
            improved_history,
            key=lambda item: int(item["function_validation_killed"]),
        )
        initial_best_step = int(initial_best_metrics["checkpoint_step"])
        rolling_checkpoint = ADAPTER_DIR / "sft_tmp" / f"checkpoint-{initial_best_step}"
        preserved_checkpoint = (
            ADAPTER_DIR / "sft_validation_best" / f"checkpoint-{initial_best_step}"
        )
        if not preserved_checkpoint.joinpath("adapter_model.safetensors").exists():
            _copy_adapter_snapshot(rolling_checkpoint, preserved_checkpoint)
            preserved_checkpoint.joinpath("validation_metrics.json").write_text(
                json.dumps(initial_best_metrics, indent=2) + "\n", encoding="utf-8"
            )
        initial_best_adapter_path = str(preserved_checkpoint)

    def monitor(checkpoint_step: int, current_model, current_tokenizer) -> Dict:
        nonlocal trend
        existing = next(
            (item for item in trend if item.get("checkpoint_step") == checkpoint_step),
            None,
        )
        if existing is not None:
            return existing
        evaluation = _evaluate_loaded_sft_monitor(
            current_model, current_tokenizer, function_pairs,
            dataset_fingerprint, selection_sha256, checkpoint_step,
        )
        checkpoint = _sft_monitor_gate_decision(
            baseline, trend, evaluation, SFT_MONITOR_PATIENCE
        )
        trend = [item for item in trend if item.get("checkpoint_step") != checkpoint_step]
        trend.append(checkpoint)
        trend.sort(key=lambda item: item["checkpoint_step"])
        completed_checkpoint_metrics[checkpoint_step] = dict(checkpoint)
        trend_path.write_text(json.dumps(trend, indent=2) + "\n", encoding="utf-8")
        print(
            f"[SFT MONITOR] step={checkpoint_step} "
            f"kill_rate={checkpoint['function_kill_rate']:.2%} "
            f"previous={checkpoint['previous_function_kill_rate']:.2%} "
            f"decision={checkpoint['decision']}",
            flush=True,
        )
        return checkpoint

    monitor.initial_best_adapter_path = initial_best_adapter_path
    monitor.initial_best_metrics = initial_best_metrics
    monitor.completed_checkpoint_metrics = completed_checkpoint_metrics

    return monitor, baseline


def _write_resume_state(
    path: Path, last_index: int, totals: Dict[str, int], adapter_dir: Path,
    dpo_training_scope_sha256: str,
) -> None:
    """Persist progress only after saving the matching policy adapter.

    The checkpoint index and SHA-256 make it impossible to skip records after
    an interrupted DPO update.  A stale or hand-edited state is rejected on
    resume instead of silently mixing a different policy with later counters.
    """
    adapter_file = adapter_dir / "adapter_model.safetensors"
    if not adapter_file.exists():
        raise RuntimeError("Cannot save DPO resume state without a policy adapter checkpoint")
    state = {
        "last_processed_index": last_index,
        "adapter_checkpoint_last_processed_index": last_index,
        "adapter_model_sha256": sha256_file(adapter_file),
        "dpo_training_scope_sha256": dpo_training_scope_sha256,
        **totals,
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(state, handle, indent=2)


def _latest_sft_trainer_checkpoint(adapter_dir: Path) -> Optional[Path]:
    """Locate a complete SFT Trainer checkpoint for a safe interrupted-run resume."""
    candidates = []
    for candidate in adapter_dir.joinpath("sft_tmp").glob("checkpoint-*"):
        try:
            step = int(candidate.name.rsplit("-", 1)[1])
        except ValueError:
            continue
        if candidate.joinpath("trainer_state.json").exists():
            candidates.append((step, candidate))
    return max(candidates, default=(None, None))[1]


def _load_sft_validation_baseline(
    results_dir: Path, dataset_fingerprint: str,
    expected_function_ids: Optional[List[str]] = None,
    expected_repository_records: Optional[int] = None,
) -> Dict:
    """Load a comparable locked-validation result before DPO may start."""
    baseline_path = results_dir / sft_validation_results_filename(SEED)
    if not baseline_path.exists():
        raise RuntimeError(
            "DPO requires a completed SFT validation benchmark before it can start. "
            "Run --phase sft_eval first."
        )
    try:
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError("SFT validation result is not valid JSON; rerun --phase sft_eval.") from exc
    if baseline.get("dataset_fingerprint") != dataset_fingerprint:
        raise RuntimeError("SFT validation result does not match this corpus and SFT scope.")
    if baseline.get("adapter") != "sft_adapter":
        raise RuntimeError("SFT validation result does not identify the immutable SFT adapter.")
    if baseline.get("evaluation_split") != DPO_VALIDATION_SPLIT:
        raise RuntimeError("SFT validation must use the locked validation split, not the test split.")
    if baseline.get("final_test_measurement") is not False:
        raise RuntimeError("SFT validation baseline must explicitly confirm the test split was not used.")
    if baseline.get("seed") != SEED or baseline.get("tests_per_function") != TESTS_PER_PAIR:
        raise RuntimeError("SFT validation baseline does not use the locked seed/candidate count.")
    try:
        rate = float(baseline["function_kill_rate"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("SFT validation result has no usable function kill rate.") from exc
    if not 0.0 <= rate <= 1.0:
        raise RuntimeError("SFT validation function kill rate is outside [0, 1].")
    for metric_name in ("function_kill_rate", "candidate_kill_rate"):
        try:
            metric = float(baseline[metric_name])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(f"SFT validation has no usable {metric_name}.") from exc
        if not math.isfinite(metric) or not 0.0 <= metric <= 1.0:
            raise RuntimeError(f"SFT validation {metric_name} is not a finite rate in [0, 1].")
    if expected_function_ids is not None:
        expected_count = len(expected_function_ids)
        if baseline.get("function_validation_records") != expected_count:
            raise RuntimeError(
                "SFT validation baseline does not cover the complete locked function scope: "
                f"expected {expected_count}, got {baseline.get('function_validation_records')}."
            )
        expected_scope_sha256 = hashlib.sha256(
            json.dumps(expected_function_ids, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        recorded_scope = baseline.get("evaluation_scope_sha256")
        if recorded_scope is not None and recorded_scope != expected_scope_sha256:
            raise RuntimeError("SFT validation function selection hash does not match DPO validation.")
    if (
        expected_repository_records is not None
        and baseline.get("repository_validation_records_held") != expected_repository_records
    ):
        raise RuntimeError("SFT validation held-repository count does not match the locked split.")
    return baseline


def _read_dpo_validation_trend(results_dir: Path) -> List[Dict]:
    trend_path = results_dir / "dpo_validation_trend.json"
    if not trend_path.exists():
        return []
    try:
        trend = json.loads(trend_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError("DPO validation trend is corrupt; refusing an ambiguous resume.") from exc
    if not isinstance(trend, list):
        raise RuntimeError("DPO validation trend must be a list; refusing an ambiguous resume.")
    return trend


def _append_dpo_validation_checkpoint(
    corpus_dir: Path,
    dataset_fingerprint: str,
    adapter_dir: Path,
    results_dir: Path,
    checkpoint_pairs: int,
    trained_dpo_pairs: int,
    processed_records: int,
    dpo_updates: int,
    sft_baseline: Dict,
) -> Dict:
    """Evaluate one saved DPO policy and append an auditable gate decision."""
    result_filename = f"dpo_validation_checkpoint_{checkpoint_pairs}.json"
    evaluation = _evaluate_adapter_kill_rate(
        corpus_dir, dataset_fingerprint, adapter_dir, "dpo_adapter",
        result_filename, evaluation_split=DPO_VALIDATION_SPLIT,
        checkpoint_pairs=checkpoint_pairs,
    )
    baseline_rate = float(sft_baseline["function_kill_rate"])
    evaluation_rate = float(evaluation["function_kill_rate"])
    improved = evaluation_rate > baseline_rate
    checkpoint = {
        "checkpoint_pairs": checkpoint_pairs,
        "trained_dpo_pairs": trained_dpo_pairs,
        "processed_records": processed_records,
        "dpo_updates": dpo_updates,
        "evaluation_split": DPO_VALIDATION_SPLIT,
        "function_kill_rate": evaluation_rate,
        "function_validation_killed": evaluation["function_validation_killed"],
        "function_validation_records": evaluation["function_validation_records"],
        "candidate_kill_rate": evaluation["candidate_kill_rate"],
        "sft_baseline_function_kill_rate": baseline_rate,
        "improved_over_sft": improved,
        "decision": "continue" if improved else "early_stop_and_pivot_to_sft",
    }
    trend = _read_dpo_validation_trend(results_dir)
    trend = [item for item in trend if item.get("checkpoint_pairs") != checkpoint_pairs]
    trend.append(checkpoint)
    trend.sort(key=lambda item: item["checkpoint_pairs"])
    (results_dir / "dpo_validation_trend.json").write_text(
        json.dumps(trend, indent=2) + "\n", encoding="utf-8"
    )
    return checkpoint


def reset_dpo_from_frozen_sft(adapter_dir: Path, results_dir: Path) -> Dict:
    """Reset only mutable DPO state while preserving the completed SFT snapshot.

    The root adapter is DPO's trainable policy; ``sft_adapter`` is the
    immutable reference policy. A restart must restore the root from that
    reference instead of resuming from an interrupted DPO checkpoint.
    """
    adapter_dir = Path(adapter_dir)
    results_dir = Path(results_dir)
    sft_adapter_dir = adapter_dir / "sft_adapter"
    source_adapter_file = sft_adapter_dir / "adapter_model.safetensors"
    target_adapter_file = adapter_dir / "adapter_model.safetensors"
    if not source_adapter_file.exists():
        raise RuntimeError("Cannot restart DPO: the immutable SFT adapter is missing.")

    source_checksum = sha256_file(source_adapter_file)
    prior_root_checksum = sha256_file(target_adapter_file) if target_adapter_file.exists() else None
    prior_resume_state = None
    state_file = adapter_dir / "resume_state.json"
    if state_file.exists():
        try:
            prior_resume_state = json.loads(state_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            prior_resume_state = {"unreadable": True}

    restored_files = []
    for source_file in sorted(sft_adapter_dir.iterdir()):
        if source_file.is_file():
            shutil.copy2(source_file, adapter_dir / source_file.name)
            restored_files.append(source_file.name)

    if sha256_file(source_adapter_file) != source_checksum:
        raise RuntimeError("Immutable SFT adapter changed during DPO reset; aborting.")
    if sha256_file(target_adapter_file) != source_checksum:
        raise RuntimeError("DPO reset did not restore the exact frozen SFT adapter.")

    removed_artifacts = []
    for artifact_name in (
        "resume_state.json", "dpo_complete.marker", "dpo_metadata.json", "training_stats.json",
        "dpo_early_stop.json",
    ):
        artifact = adapter_dir / artifact_name
        if artifact.exists():
            artifact.unlink()
            removed_artifacts.append(artifact_name)

    archived_result = None
    prior_result = results_dir / "dpo_training_results.json"
    if prior_result.exists():
        archive_index = 1
        while True:
            candidate = results_dir / f"dpo_training_results.pre_restart_{archive_index}.json"
            if not candidate.exists():
                prior_result.replace(candidate)
                archived_result = candidate.name
                break

    # Validation checkpoints are tied to a particular DPO policy trajectory.
    # A restart begins from immutable SFT, so carrying them forward would make
    # the 500-pair gate report a result from a different policy.
    removed_validation_artifacts = []
    for artifact in [
        results_dir / "dpo_validation_trend.json",
        *results_dir.glob("dpo_validation_checkpoint_*.json"),
    ]:
        if artifact.exists():
            artifact.unlink()
            removed_validation_artifacts.append(artifact.name)

    audit = {
        "event": "dpo_restart_from_frozen_sft",
        "sft_adapter_sha256": source_checksum,
        "prior_root_adapter_sha256": prior_root_checksum,
        "restored_files": restored_files,
        "removed_dpo_artifacts": removed_artifacts,
        "archived_dpo_result": archived_result,
        "removed_validation_artifacts": removed_validation_artifacts,
        "prior_resume_state": prior_resume_state,
    }
    (adapter_dir / "dpo_restart_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    print("[DPO RESTART] Restored the exact immutable SFT adapter; DPO resumes at pair 1.")
    return audit


def run_training(use_mock: bool = False, fresh: bool = False) -> Dict:
    random.seed(SEED)

    if TRAINING_PHASE not in {"base_eval", "sft", "sft_eval", "dpo", "dpo_eval", "sft_then_dpo"}:
        raise RuntimeError(f"Invalid training phase: {TRAINING_PHASE!r}")
    if TRAINING_PHASE == "dpo_eval" and not CONFIRM_FINAL_TEST:
        raise RuntimeError(
            "The final test split is sealed. Re-run with --confirm-final-test only "
            "after model selection and validation are frozen."
        )
    if RESTART_DPO and (TRAINING_PHASE != "dpo" or fresh):
        raise RuntimeError("--restart-dpo is valid only for a non-fresh DPO-only run.")
    if TRAINING_PHASE == "sft_then_dpo":
        raise RuntimeError(
            "Combined SFT→DPO is disabled by the quality gate. Run SFT, validate it on val, "
            "then start DPO only after the SFT kill rate reaches 58%."
        )
    run_sft = TRAINING_PHASE in {"sft", "sft_then_dpo"}
    run_dpo = TRAINING_PHASE in {"dpo", "sft_then_dpo"}

    if not valid_corpus_version(CORPUS_VERSION):
        raise RuntimeError(f"Invalid corpus version: {CORPUS_VERSION!r}")
    corpus_dir = DATA_DIR / "corpus" / CORPUS_VERSION
    if REQUIRE_SPLIT_ISOLATION:
        required_splits = ["train", EVALUATION_SPLIT]
        if TRAINING_PHASE == "dpo_eval":
            raise RuntimeError(
                "The development-only local corpus view deliberately excludes the "
                "sealed test. Final measurement requires the separately authorized "
                "post-selection path."
            )
        verify_development_view(corpus_dir, required_splits)
        manifest = json.loads(
            (corpus_dir / "manifest.json").read_text(encoding="utf-8")
        )
        print(
            "Sealed-test-free development corpus view verified for: "
            + ", ".join(required_splits)
        )
    else:
        manifest = verify_corpus(corpus_dir)
    print(
        "Canonical corpus verified: "
        f"{manifest['training_records']:,} behaviorally checked records; "
        "splits are group-disjoint and external evaluation is locked."
    )

    if fresh:
        print(f"[FRESH] Removing checkpoints in {ADAPTER_DIR}")
        if ADAPTER_DIR.exists():
            shutil.rmtree(ADAPTER_DIR)
    ADAPTER_DIR.mkdir(parents=True, exist_ok=True)

    train_pairs = load_phase3_pairs(corpus_dir, "train")
    if EXECUTION_MODE_FILTER:
        if EXECUTION_MODE_FILTER not in {FUNCTION_EXECUTION_MODE, *REPOSITORY_EXECUTION_MODES}:
            raise RuntimeError(f"Invalid execution-mode filter: {EXECUTION_MODE_FILTER!r}")
        train_pairs = [
            pair for pair in train_pairs
            if pair.get("execution_mode", FUNCTION_EXECUTION_MODE) == EXECUTION_MODE_FILTER
        ]
    if HOLDOUT_BUG_FAMILY:
        before_holdout = len(train_pairs)
        train_pairs = [
            pair for pair in train_pairs
            if str(pair.get("bug_family", "unknown")).strip().lower()
            != HOLDOUT_BUG_FAMILY
        ]
        print(
            "[FAMILY HOLDOUT] Excluded "
            f"{before_holdout - len(train_pairs)} training record(s) from family "
            f"{HOLDOUT_BUG_FAMILY!r}."
        )
    train_pairs, overlong_repository_completions = _filter_overlong_repository_completions(
        train_pairs
    )
    if overlong_repository_completions:
        print(
            "[CONTEXT GATE] Excluded "
            f"{len(overlong_repository_completions)} overlong repository completion(s) "
            f"from SFT and DPO while retaining their canonical corpus records: "
            + ", ".join(item["record_id"] for item in overlong_repository_completions)
        )
    dpo_overlong_repository_completions = []
    if run_dpo:
        train_pairs, dpo_overlong_repository_completions = (
            _filter_overlong_repository_completions(
                train_pairs,
                max_completion_tokens=MAX_DPO_COMPLETION_TOKENS,
                exclusion_reason="completion_exceeds_dpo_context",
            )
        )
        if dpo_overlong_repository_completions:
            print(
                "[DPO CONTEXT GATE] Excluded "
                f"{len(dpo_overlong_repository_completions)} additional repository "
                "completion(s) from DPO only; all remain in the canonical corpus and "
                "were still eligible for SFT: "
                + ", ".join(
                    item["record_id"] for item in dpo_overlong_repository_completions
                )
            )
    available_train_pairs = len(train_pairs)
    sft_preflight_tokenizer = None
    compatible_repository_ids = None
    compatible_synthetic_ids = None
    if MAX_TRAIN_PAIRS and run_sft and not use_mock:
        from transformers import AutoTokenizer
        _selection_model, _selection_revision = (
            resolved_selection_tokenizer_identity()
        )
        sft_preflight_tokenizer = AutoTokenizer.from_pretrained(
            _selection_model,
            revision=_selection_revision,
            trust_remote_code=True,
        )
        compatible_repository_ids = set()
        compatible_synthetic_ids = set()
        selection_prompt_limit = (
            SFT_SELECTION_PROMPT_TOKEN_LIMIT_OVERRIDE
            if SFT_SELECTION_PROMPT_TOKEN_LIMIT_OVERRIDE is not None
            else PROMPT_TOKEN_LIMIT
        )
        repository_generation_limit = (
            SFT_REPOSITORY_COMPLETION_TOKEN_LIMIT_OVERRIDE
            if SFT_REPOSITORY_COMPLETION_TOKEN_LIMIT_OVERRIDE is not None
            else MAX_SFT_REPOSITORY_GENERATION_COMPATIBLE_TOKENS
        )
        for pair in train_pairs:
            mode = pair.get("execution_mode", FUNCTION_EXECUTION_MODE)
            pair_prompt = build_pair_prompt(pair)
            if not is_repository_execution_mode(mode):
                try:
                    compact_unified_user_prompt(
                        sft_preflight_tokenizer,
                        pair_prompt,
                        selection_prompt_limit,
                        format_chat_prompt,
                    )
                except (PromptBudgetError, ValueError):
                    continue
                compatible_synthetic_ids.add(pair["id"])
                continue
            for completion in _repository_fragment_tests(
                pair.get("test_cases", [])
            )[:3]:
                completion_tokens = len(sft_preflight_tokenizer(
                    completion.strip() + sft_preflight_tokenizer.eos_token,
                    add_special_tokens=False,
                )["input_ids"])
                if completion_tokens > repository_generation_limit:
                    continue
                from engine.sft_trainer import MAX_SFT_SEQUENCE_LENGTH
                allowed_prompt_tokens = min(
                    REPOSITORY_PROMPT_TOKEN_LIMIT,
                    MAX_SFT_SEQUENCE_LENGTH - completion_tokens,
                )
                try:
                    compact_unified_user_prompt(
                        sft_preflight_tokenizer,
                        pair_prompt,
                        allowed_prompt_tokens,
                        format_chat_prompt,
                    )
                except (PromptBudgetError, ValueError):
                    continue
                compatible_repository_ids.add(pair["id"])
                break
    if MAX_TRAIN_PAIRS:
        train_pairs = select_bounded_train_pairs(
            train_pairs,
            MAX_TRAIN_PAIRS,
            compatible_repository_ids=compatible_repository_ids,
            compatible_synthetic_ids=compatible_synthetic_ids,
            target_real_fraction=(
                SFT_REAL_TARGET_FRACTION_OVERRIDE
                if SFT_REAL_TARGET_FRACTION_OVERRIDE is not None
                else SFT_REAL_TARGET_FRACTION
            ),
            max_real_repeats=(
                SFT_MAX_REAL_REPEATS_OVERRIDE
                if SFT_MAX_REAL_REPEATS_OVERRIDE is not None
                else SFT_MAX_REAL_REPEATS
            ),
            target_complex_fraction=(
                SFT_COMPLEX_TARGET_FRACTION_OVERRIDE
                if SFT_COMPLEX_TARGET_FRACTION_OVERRIDE is not None
                else SFT_COMPLEX_TARGET_FRACTION
            ),
        )
    dpo_training_scope_sha256 = _dpo_training_scope_sha256(train_pairs) if run_dpo else None
    selected_repository_pairs = sum(
        is_repository_execution_mode(
            pair.get("execution_mode", FUNCTION_EXECUTION_MODE)
        )
        for pair in train_pairs
    )
    selected_synthetic_pairs = len(train_pairs) - selected_repository_pairs
    bounded_selection_stats = summarize_train_pair_selection(train_pairs)
    print(
        f"Loaded {len(train_pairs):,} training records"
        f"{f' (mode={EXECUTION_MODE_FILTER})' if EXECUTION_MODE_FILTER else ''}"
    )
    if MAX_TRAIN_PAIRS:
        print(
            "Bounded smoke selection: "
            f"{selected_synthetic_pairs} synthetic + "
            f"{selected_repository_pairs} verified repository records "
            f"from {available_train_pairs:,} context-eligible records"
        )

    sft_marker = ADAPTER_DIR / "sft_complete.marker"
    sft_adapter_dir = ADAPTER_DIR / "sft_adapter"
    sft_adapter_file = sft_adapter_dir / "adapter_model.safetensors"
    root_adapter_file = ADAPTER_DIR / "adapter_model.safetensors"
    dataset_version_file = ADAPTER_DIR / "dataset_manifest.sha256"
    training_scope = sft_training_scope(
        MAX_TRAIN_PAIRS, run_sft, EXECUTION_MODE_FILTER,
    )
    dataset_fingerprint = resolve_sft_dataset_fingerprint(
        manifest, training_scope, dataset_version_file, run_sft
    )
    verified_sft_examples = 0
    requested_sft_examples = 0
    optimizer_sft_examples = 0
    optimizer_padding_examples = 0
    repository_sft_examples = 0
    effective_repository_sft_examples = 0
    real_sampling_repeats = 1
    sft_sampling_stats = {}
    sft_optimizer_plan = {}
    sft_generation_incompatible_completions = []
    sft_records_without_verified_winners = []
    sft_records_without_generation_compatible_winners = []
    sft_prompt_truncated_examples = 0
    sft_loss = None
    sft_monitor_stopped_early = False
    sft_monitor_baseline = None
    sft_monitor_history = []
    sft_monitor_best_adapter = None
    sft_monitor_best_metrics = None
    sft_monitor_gate_passed = not SFT_CHECKPOINT_MONITOR_ENABLED
    started = time.time()
    sft_metadata_file = ADAPTER_DIR / "sft_metadata.json"
    sft_run_config_file = ADAPTER_DIR / "sft_run_config.json"
    dpo_metadata_file = ADAPTER_DIR / "dpo_metadata.json"
    dpo_complete_marker = ADAPTER_DIR / "dpo_complete.marker"
    from config import training_config
    sft_hyperparameters = {
        "epochs": (
            SFT_EPOCHS_OVERRIDE if SFT_EPOCHS_OVERRIDE is not None else training_config.sft_epochs
        ),
        "learning_rate": (
            SFT_LEARNING_RATE_OVERRIDE
            if SFT_LEARNING_RATE_OVERRIDE is not None else training_config.sft_learning_rate
        ),
        "lr_scheduler_type": (
            SFT_LR_SCHEDULER_TYPE_OVERRIDE
            if SFT_LR_SCHEDULER_TYPE_OVERRIDE is not None
            else training_config.sft_lr_scheduler_type
        ),
        "batch_size": (
            SFT_BATCH_SIZE_OVERRIDE
            if SFT_BATCH_SIZE_OVERRIDE is not None else training_config.sft_batch_size
        ),
        "seed": SEED,
        "max_sequence_length": MAX_SFT_COMPLETION_TOKENS,
        "prompt_token_limit": PROMPT_TOKEN_LIMIT,
        "selection_prompt_token_limit": (
            SFT_SELECTION_PROMPT_TOKEN_LIMIT_OVERRIDE
            if SFT_SELECTION_PROMPT_TOKEN_LIMIT_OVERRIDE is not None
            else PROMPT_TOKEN_LIMIT
        ),
        "repository_prompt_token_limit": REPOSITORY_PROMPT_TOKEN_LIMIT,
        "prompt_compaction_strategy": PROMPT_COMPACTION_STRATEGY,
        "prompt_schema_version": PROMPT_SCHEMA_VERSION,
        "prompt_information_variant": PROMPT_INFORMATION_VARIANT,
        "output_instruction_variant": OUTPUT_INSTRUCTION_VARIANT,
        "dataset_identity_policy": DATASET_IDENTITY_POLICY,
        "generation_completion_token_limit": MAX_SFT_GENERATION_COMPATIBLE_TOKENS,
        "repository_generation_completion_token_limit": (
            SFT_REPOSITORY_COMPLETION_TOKEN_LIMIT_OVERRIDE
            if SFT_REPOSITORY_COMPLETION_TOKEN_LIMIT_OVERRIDE is not None
            else MAX_SFT_REPOSITORY_GENERATION_COMPATIBLE_TOKENS
        ),
        "warmup_steps": training_config.sft_warmup_steps,
        "checkpoint_save_steps": (
            SFT_CHECKPOINT_STEPS_OVERRIDE
            if SFT_CHECKPOINT_STEPS_OVERRIDE is not None
            else training_config.sft_checkpoint_steps
        ),
        "minimum_monitor_checkpoints": (
            SFT_MIN_MONITOR_CHECKPOINTS_OVERRIDE
            if SFT_MIN_MONITOR_CHECKPOINTS_OVERRIDE is not None
            else training_config.sft_min_monitor_checkpoints
        ),
        "real_target_fraction": (
            SFT_REAL_TARGET_FRACTION_OVERRIDE
            if SFT_REAL_TARGET_FRACTION_OVERRIDE is not None else SFT_REAL_TARGET_FRACTION
        ),
        "max_real_repeats": (
            SFT_MAX_REAL_REPEATS_OVERRIDE
            if SFT_MAX_REAL_REPEATS_OVERRIDE is not None else SFT_MAX_REAL_REPEATS
        ),
        "checkpoint_monitor_enabled": SFT_CHECKPOINT_MONITOR_ENABLED,
        "monitor_validation_functions": SFT_MONITOR_VALIDATION_FUNCTIONS,
        "monitor_patience": SFT_MONITOR_PATIENCE,
        "min_function_kill_rate": (
            SFT_MONITOR_MIN_FUNCTION_KILL_RATE_OVERRIDE
            if SFT_MONITOR_MIN_FUNCTION_KILL_RATE_OVERRIDE is not None
            else training_config.sft_min_function_kill_rate
        ),
        "monitor_interval_optimizer_steps": (
            SFT_CHECKPOINT_STEPS_OVERRIDE
            if SFT_CHECKPOINT_STEPS_OVERRIDE is not None
            else training_config.sft_checkpoint_steps
        ),
        "balanced_sampling_enabled": SFT_BALANCED_SAMPLING_ENABLED,
        "synthetic_balance_fraction": SFT_SYNTHETIC_BALANCE_FRACTION,
        "synthetic_balance_mode": SFT_SYNTHETIC_BALANCE_MODE,
        "max_synthetic_repeats": SFT_MAX_SYNTHETIC_REPEATS,
        "complex_target_fraction": (
            SFT_COMPLEX_TARGET_FRACTION_OVERRIDE
            if SFT_COMPLEX_TARGET_FRACTION_OVERRIDE is not None
            else SFT_COMPLEX_TARGET_FRACTION
        ),
    }
    resolved_base_model_name = BASE_MODEL_NAME_OVERRIDE or model_config.model_name
    resolved_base_model_revision = (
        BASE_MODEL_REVISION_OVERRIDE
        if BASE_MODEL_REVISION_OVERRIDE is not None
        else (
            model_config.model_revision
            if resolved_base_model_name == model_config.model_name
            else "main"
        )
    )
    resolved_attention_implementation = (
        BASE_MODEL_ATTENTION_IMPLEMENTATION_OVERRIDE
        or model_config.attention_implementation
    )
    reproducibility = build_reproducibility_manifest(
        Path(__file__).parent.parent,
        resolved_base_model_name,
        resolved_base_model_revision,
    )
    if (
        sft_hyperparameters["epochs"] <= 0
        or sft_hyperparameters["learning_rate"] <= 0
        or sft_hyperparameters["batch_size"] <= 0
        or not 0.0 <= sft_hyperparameters["real_target_fraction"] < 1.0
        or sft_hyperparameters["max_real_repeats"] < 1
        or sft_hyperparameters["synthetic_balance_fraction"] < 0.0
        or sft_hyperparameters["synthetic_balance_mode"]
        not in {"none", "dataset", "dataset_family"}
        or sft_hyperparameters["max_synthetic_repeats"] < 1
        or not 0.0 <= sft_hyperparameters["complex_target_fraction"] <= 1.0
        or sft_hyperparameters["prompt_token_limit"] <= 0
        or sft_hyperparameters["repository_prompt_token_limit"] <= 0
        or sft_hyperparameters["generation_completion_token_limit"] <= 0
        or sft_hyperparameters[
            "repository_generation_completion_token_limit"
        ] <= 0
        or sft_hyperparameters[
            "repository_generation_completion_token_limit"
        ] >= sft_hyperparameters["max_sequence_length"]
        or sft_hyperparameters["warmup_steps"] < 0
        or sft_hyperparameters["lr_scheduler_type"]
        not in {"cosine", "constant_with_warmup"}
        or sft_hyperparameters["checkpoint_save_steps"] <= 0
        or sft_hyperparameters["minimum_monitor_checkpoints"] < 1
        or not 0.0 <= sft_hyperparameters["min_function_kill_rate"] <= 1.0
    ):
        raise RuntimeError(
            "SFT hyperparameters are invalid; epochs/LR/batch/limits/checkpoint steps "
            "must be positive and sampling fractions/repeats must be in range."
        )

    if TRAINING_PHASE == "base_eval":
        if use_mock:
            raise RuntimeError("Base-model validation requires real model inference")
        return _evaluate_adapter_kill_rate(
            corpus_dir,
            dataset_fingerprint,
            None,
            "base_model",
            evaluation_results_filename("base", SEED),
            evaluation_split=EVALUATION_SPLIT,
        )

    if not use_mock:
        has_training_artifacts = sft_marker.exists() or root_adapter_file.exists()
        if has_training_artifacts:
            previous_fingerprint = (
                dataset_version_file.read_text(encoding="utf-8").strip()
                if dataset_version_file.exists() else ""
            )
            if sft_marker.exists() and previous_fingerprint != dataset_fingerprint:
                raise RuntimeError(
                    "Existing adapter was trained on a different corpus or SFT training scope. "
                    "Re-run with --fresh; resuming would mix incompatible training runs."
                )
        if sft_marker.exists() and not sft_adapter_file.exists():
            raise RuntimeError("SFT marker exists without its reference adapter. Re-run with --fresh.")
        if not sft_marker.exists() and root_adapter_file.exists() and not _latest_sft_trainer_checkpoint(ADAPTER_DIR):
            raise RuntimeError(
                "An adapter without an SFT marker or a complete SFT Trainer checkpoint was found. "
                "Re-run with --fresh so DPO cannot start from an unknown adapter."
            )

        if sft_marker.exists():
            print("[SKIP] Verified SFT adapter and marker found")
            if TRAINING_PHASE in {"dpo", "sft_eval", "dpo_eval"}:
                if not sft_metadata_file.exists():
                    raise RuntimeError(
                        "Verified SFT marker exists without SFT metadata; refusing DPO from an "
                        "unaccounted reference adapter. Re-run SFT with --fresh."
                    )
                sft_metadata = json.loads(sft_metadata_file.read_text(encoding="utf-8"))
                if sft_metadata.get("dataset_fingerprint") != dataset_fingerprint:
                    raise RuntimeError("SFT metadata does not match the requested DPO training scope")
                if TRAINING_PHASE in {"dpo", "dpo_eval"}:
                    # Compare what actually determines the artifact, which is
                    # exactly what this check's message claims to require.
                    # Comparing the whole manifest also compared git_commit, so
                    # committing the SFT run's own results - required by section
                    # 47 - invalidated the adapter that produced them. Source
                    # identity is the stronger test regardless, since it also
                    # catches uncommitted edits a commit SHA would miss.
                    recorded = functional_identity(
                        sft_metadata.get("reproducibility") or {}
                    )
                    current = functional_identity(reproducibility)
                    if recorded != current:
                        differing = sorted(
                            field for field in current
                            if recorded.get(field) != current.get(field)
                        )
                        raise RuntimeError(
                            "DPO requires an SFT adapter produced by this exact source, "
                            "dependency specification, Python runtime, and pinned base "
                            f"model. Differing fields: {differing}. "
                            "The existing adapter is legacy or mismatched; start a new SFT run."
                        )
                requested_sft_examples = sft_metadata["requested_sft_examples"]
                verified_sft_examples = sft_metadata["verified_sft_examples"]
                optimizer_sft_examples = sft_metadata.get(
                    "optimizer_sft_examples", verified_sft_examples
                )
                optimizer_padding_examples = sft_metadata.get(
                    "optimizer_padding_examples", 0
                )
                repository_sft_examples = sft_metadata["repository_sft_examples"]
                sft_loss = sft_metadata["sft_loss"]
            if RESTART_DPO:
                reset_dpo_from_frozen_sft(ADAPTER_DIR, RESULTS_DIR)
        elif run_sft:
            print("\n" + "=" * 60)
            print("PHASE 0: SFT (verified mutation-killing assertions)")
            print("=" * 60)
            sft_trainer = None
            try:
                from engine.sft_trainer import (
                    OneirosSFTTrainer,
                    plan_sft_optimizer_schedule,
                )

                if sft_run_config_file.exists():
                    try:
                        existing_run_config = json.loads(
                            sft_run_config_file.read_text(encoding="utf-8")
                        )
                    except json.JSONDecodeError as exc:
                        raise RuntimeError("SFT run configuration is corrupt; use --fresh for this run name.") from exc
                    if (
                        existing_run_config.get("dataset_fingerprint") != dataset_fingerprint
                        or existing_run_config.get("reproducibility") != reproducibility
                        or normalized_sft_run_hyperparameters(
                            existing_run_config.get("hyperparameters", {})
                        ) != normalized_sft_run_hyperparameters(sft_hyperparameters)
                    ):
                        raise RuntimeError(
                            "Interrupted SFT run configuration does not match this launch. "
                            "Use the original hyperparameters or a new run name."
                        )
                elif _latest_sft_trainer_checkpoint(ADAPTER_DIR):
                    raise RuntimeError(
                        "Found an interrupted SFT checkpoint without its run configuration. "
                        "Use a new run name; this checkpoint cannot be resumed reproducibly."
                    )
                else:
                    sft_run_config_file.write_text(json.dumps({
                        "status": "in_progress",
                        "dataset_fingerprint": dataset_fingerprint,
                        "reproducibility": reproducibility,
                        "hyperparameters": sft_hyperparameters,
                    }, indent=2) + "\n", encoding="utf-8")

                synthetic_sft_data = []
                repository_sft_data = []
                for pair in train_pairs:
                    if is_repository_execution_mode(pair.get("execution_mode", FUNCTION_EXECUTION_MODE)):
                        verified_winners = _repository_fragment_tests(pair.get("test_cases", []))
                        repository_sft_examples += len(verified_winners[:3])
                    else:
                        source_tests = extract_dataset_tests(pair.get("test_cases", []), pair["entry_point"])
                        verified_winners, _ = evaluate_pair(
                            source_tests, pair["golden_code"], pair["mutant_code"],
                            pair["entry_point"],
                        )
                    if not verified_winners:
                        sft_records_without_verified_winners.append(pair["id"])
                        continue
                    prompt = build_pair_prompt(pair)
                    for test in verified_winners[:3]:
                        data_point = make_sft_data_point(pair, prompt, test)
                        if is_repository_execution_mode(
                            pair.get("execution_mode", FUNCTION_EXECUTION_MODE)
                        ):
                            repository_sft_data.append(data_point)
                        else:
                            synthetic_sft_data.append(data_point)

                verified_supervision_exclusions = supervision_exclusion_summary(
                    sft_records_without_verified_winners
                )
                sft_sampling_stats["verified_supervision_exclusions"] = (
                    verified_supervision_exclusions
                )
                if sft_records_without_verified_winners:
                    print(
                        "[VERIFIED SUPERVISION GATE] Explicitly excluded "
                        f"{len(sft_records_without_verified_winners):,} selected "
                        "record(s) with no policy- and reference-valid mutation-killing "
                        "completion; canonical corpus records remain unchanged."
                    )

                # Apply the live 128-token generation gate to the exact
                # behaviorally verified winners selected above.  Loading only
                # the tokenizer keeps this fail-fast check cheap; no model or
                # GPU training starts when the supervision is incompatible.
                if sft_preflight_tokenizer is None:
                    from transformers import AutoTokenizer
                    _sel_model, _sel_revision = (
                        resolved_selection_tokenizer_identity()
                    )
                    sft_preflight_tokenizer = AutoTokenizer.from_pretrained(
                        _sel_model,
                        revision=_sel_revision,
                        trust_remote_code=True,
                    )
                synthetic_sft_data, synthetic_generation_exclusions = (
                    filter_generation_compatible_sft_examples(
                        synthetic_sft_data,
                        sft_preflight_tokenizer,
                        sft_hyperparameters["generation_completion_token_limit"],
                        sft_hyperparameters[
                            "repository_generation_completion_token_limit"
                        ],
                        sft_hyperparameters["prompt_token_limit"],
                        sft_hyperparameters["repository_prompt_token_limit"],
                    )
                )
                repository_sft_data, repository_generation_exclusions = (
                    filter_generation_compatible_sft_examples(
                        repository_sft_data,
                        sft_preflight_tokenizer,
                        sft_hyperparameters["generation_completion_token_limit"],
                        sft_hyperparameters[
                            "repository_generation_completion_token_limit"
                        ],
                        sft_hyperparameters["prompt_token_limit"],
                        sft_hyperparameters["repository_prompt_token_limit"],
                    )
                )
                sft_generation_incompatible_completions = [
                    *synthetic_generation_exclusions,
                    *repository_generation_exclusions,
                ]
                generation_compatible_record_ids = {
                    example.function_id
                    for example in [*synthetic_sft_data, *repository_sft_data]
                }
                no_verified_winner_ids = set(sft_records_without_verified_winners)
                raw_verified_record_ids = {
                    pair["id"] for pair in train_pairs
                    if pair["id"] not in no_verified_winner_ids
                }
                sft_records_without_generation_compatible_winners = [
                    pair["id"] for pair in train_pairs
                    if pair["id"] in raw_verified_record_ids
                    and pair["id"] not in generation_compatible_record_ids
                ]
                sft_sampling_stats["live_generation_compatibility"] = {
                    "completion_token_limit": sft_hyperparameters[
                        "generation_completion_token_limit"
                    ],
                    "repository_completion_token_limit": sft_hyperparameters[
                        "repository_generation_completion_token_limit"
                    ],
                    "synthetic_excluded": len(synthetic_generation_exclusions),
                    "repository_excluded": len(repository_generation_exclusions),
                    "total_excluded": len(sft_generation_incompatible_completions),
                    "records_without_compatible_completions": (
                        sft_records_without_generation_compatible_winners
                    ),
                    "canonical_records_modified": False,
                }
                if sft_generation_incompatible_completions:
                    print(
                        "[LIVE GENERATION GATE] Explicitly excluded "
                        f"{len(sft_generation_incompatible_completions):,} verified "
                        "completion(s) that cannot fit the deployed output budget; "
                        "canonical corpus records remain unchanged."
                    )
                if sft_hyperparameters["balanced_sampling_enabled"]:
                    synthetic_sft_data, synthetic_dedup_stats = deduplicate_sft_examples(
                        synthetic_sft_data
                    )
                    repository_sft_data, repository_dedup_stats = deduplicate_sft_examples(
                        repository_sft_data
                    )
                    sft_sampling_stats["exact_supervision_deduplication"] = {
                        "synthetic": synthetic_dedup_stats,
                        "repository": repository_dedup_stats,
                    }
                unresampled_sft_data = [*synthetic_sft_data, *repository_sft_data]
                repository_sft_examples = len(repository_sft_data)
                effective_repository_sft_examples = len(repository_sft_data)
                effective_synthetic_sft_data = list(synthetic_sft_data)
                if (
                    sft_hyperparameters["balanced_sampling_enabled"]
                    and synthetic_sft_data
                    and sft_hyperparameters["synthetic_balance_fraction"]
                    and sft_hyperparameters["synthetic_balance_mode"] != "none"
                ):
                    synthetic_target = math.ceil(
                        len(synthetic_sft_data)
                        * (1.0 + sft_hyperparameters["synthetic_balance_fraction"])
                    )
                    effective_synthetic_sft_data, synthetic_sampling_stats = (
                        balanced_repeat_examples(
                            synthetic_sft_data,
                            synthetic_target,
                            sft_hyperparameters["max_synthetic_repeats"],
                            sft_hyperparameters["synthetic_balance_mode"],
                        )
                    )
                    sft_sampling_stats["synthetic_semantic_group_balance"] = (
                        synthetic_sampling_stats
                    )

                if repository_sft_data and sft_hyperparameters["real_target_fraction"]:
                    desired_real_examples = math.ceil(
                        len(effective_synthetic_sft_data)
                        * sft_hyperparameters["real_target_fraction"]
                        / (1.0 - sft_hyperparameters["real_target_fraction"])
                    )
                    if sft_hyperparameters["balanced_sampling_enabled"]:
                        repository_sft_data, repository_sampling_stats = (
                            balanced_repeat_examples(
                                repository_sft_data,
                                desired_real_examples,
                                sft_hyperparameters["max_real_repeats"],
                                "project",
                            )
                        )
                        sft_sampling_stats["repository_project_balance"] = (
                            repository_sampling_stats
                        )
                        effective_repository_sft_examples = len(repository_sft_data)
                        real_sampling_repeats = max(
                            (int(value) for value in repository_sampling_stats["repeat_histogram"]),
                            default=1,
                        )
                    else:
                        real_sampling_repeats = min(
                            sft_hyperparameters["max_real_repeats"],
                            max(1, math.ceil(desired_real_examples / len(repository_sft_data))),
                        )
                        effective_repository_sft_examples = (
                            len(repository_sft_data) * real_sampling_repeats
                        )
                sft_data = [
                    *effective_synthetic_sft_data,
                    *(
                        repository_sft_data
                        if sft_hyperparameters["balanced_sampling_enabled"]
                        else repository_sft_data * real_sampling_repeats
                    ),
                ]
                sft_sampling_stats["example_weights"] = summarize_sampling_weights(
                    unresampled_sft_data, sft_data,
                )
                actual_real_fraction = (
                    effective_repository_sft_examples / len(sft_data)
                    if sft_data else 0.0
                )
                sft_sampling_stats["effective_mix"] = {
                    "target_real_fraction": sft_hyperparameters["real_target_fraction"],
                    "actual_real_fraction": round(actual_real_fraction, 6),
                    "synthetic_examples": len(effective_synthetic_sft_data),
                    "repository_examples": effective_repository_sft_examples,
                    "target_reached": (
                        actual_real_fraction + 1e-12
                        >= sft_hyperparameters["real_target_fraction"]
                    ),
                }
                if not sft_sampling_stats["effective_mix"]["target_reached"]:
                    print(
                        "[SAMPLING GATE] Safe repetition cap limits compatible real "
                        f"supervision to {actual_real_fraction:.2%}, below the "
                        f"{sft_hyperparameters['real_target_fraction']:.2%} target. "
                        "The run will report the true share and will not inflate repeats."
                    )
                random.shuffle(sft_data)
                requested_sft_examples = len(sft_data)
                if not requested_sft_examples:
                    raise RuntimeError("No verified mutation-killing assertions were available for SFT")

                sft_optimizer_plan = plan_sft_optimizer_schedule(
                    requested_sft_examples,
                    sft_hyperparameters["epochs"],
                    sft_hyperparameters["batch_size"],
                    sft_hyperparameters["warmup_steps"],
                    sft_hyperparameters["checkpoint_save_steps"],
                )
                minimum_monitored_steps = (
                    sft_optimizer_plan["effective_checkpoint_steps"]
                    * sft_hyperparameters["minimum_monitor_checkpoints"]
                )
                sft_optimizer_plan["minimum_monitored_optimizer_steps"] = (
                    minimum_monitored_steps
                )
                sft_optimizer_plan["minimum_monitor_checkpoints"] = (
                    sft_hyperparameters["minimum_monitor_checkpoints"]
                )
                sft_sampling_stats["optimizer_preflight"] = dict(sft_optimizer_plan)
                if (
                    SFT_CHECKPOINT_MONITOR_ENABLED
                    and sft_optimizer_plan["planned_optimizer_steps"]
                    < minimum_monitored_steps
                ):
                    minimum_examples = math.ceil(
                        minimum_monitored_steps
                        * sft_optimizer_plan["samples_per_optimizer_step"]
                        / sft_hyperparameters["epochs"]
                    )
                    raise RuntimeError(
                        "SFT monitor preflight rejected an underpowered run before GPU model "
                        f"setup: planned_optimizer_steps={sft_optimizer_plan['planned_optimizer_steps']}, "
                        f"required={minimum_monitored_steps}. Retain at least "
                        f"{minimum_examples} optimizer examples or disable the checkpoint monitor "
                        "for a format-only smoke."
                    )

                print(
                    f"SFT examples after verification: {requested_sft_examples:,} "
                    f"({repository_sft_examples:,} raw / {effective_repository_sft_examples:,} "
                    "effective official repository fragments)"
                )
                sft_trainer = OneirosSFTTrainer(
                    output_dir=ADAPTER_DIR,
                    learning_rate=sft_hyperparameters["learning_rate"],
                    max_prompt_tokens=sft_hyperparameters["prompt_token_limit"],
                    max_repository_prompt_tokens=sft_hyperparameters[
                        "repository_prompt_token_limit"
                    ],
                    max_completion_tokens=sft_hyperparameters[
                        "generation_completion_token_limit"
                    ],
                    max_repository_completion_tokens=sft_hyperparameters[
                        "repository_generation_completion_token_limit"
                    ],
                    warmup_steps=sft_hyperparameters["warmup_steps"],
                    checkpoint_steps=sft_hyperparameters["checkpoint_save_steps"],
                    lr_scheduler_type=sft_hyperparameters["lr_scheduler_type"],
                    model_name=BASE_MODEL_NAME_OVERRIDE,
                    model_revision=BASE_MODEL_REVISION_OVERRIDE,
                    attention_implementation=BASE_MODEL_ATTENTION_IMPLEMENTATION_OVERRIDE,
                )
                sft_trainer.setup_model()
                checkpoint_monitor = None
                if SFT_CHECKPOINT_MONITOR_ENABLED:
                    checkpoint_monitor, sft_monitor_baseline = _build_sft_checkpoint_monitor(
                        corpus_dir,
                        dataset_fingerprint,
                        sft_trainer.model,
                        sft_trainer.tokenizer,
                    )
                    print(
                        "[SFT MONITOR] baseline kill rate="
                        f"{sft_monitor_baseline['function_kill_rate']:.2%} on "
                        f"{sft_monitor_baseline['function_validation_records']} locked "
                        f"validation functions; patience={SFT_MONITOR_PATIENCE}"
                        f"; acceptance={sft_hyperparameters['min_function_kill_rate']:.2%}"
                    )
                result = sft_trainer.train(
                    sft_data,
                    num_epochs=sft_hyperparameters["epochs"],
                    batch_size=sft_hyperparameters["batch_size"],
                    checkpoint_monitor=checkpoint_monitor,
                )
                verified_sft_examples = result["retained_examples"]
                optimizer_sft_examples = result["optimizer_examples"]
                optimizer_padding_examples = result["optimizer_padding_examples"]
                sft_prompt_truncated_examples = result["prompt_truncated_examples"]
                sft_monitor_stopped_early = result["monitor_stopped_early"]
                sft_monitor_history = result["monitor_history"]
                sft_monitor_best_adapter = result.get("best_validation_adapter_path")
                sft_monitor_best_metrics = result.get("best_validation_metrics")
                sft_monitor_gate_passed = sft_monitor_acceptance_passed(
                    SFT_CHECKPOINT_MONITOR_ENABLED,
                    sft_monitor_stopped_early,
                    sft_monitor_best_adapter,
                    sft_monitor_best_metrics,
                    sft_hyperparameters["min_function_kill_rate"],
                )
                if result["dropped_overlong_examples"]:
                    raise RuntimeError(
                        "SFT retained fewer examples than the verified corpus supplied; "
                        "compact the repository fragment before training."
                    )
                if SFT_CHECKPOINT_MONITOR_ENABLED:
                    # Preserve the terminal policy for diagnosis.  If any
                    # checkpoint beat the baseline, promote that exact
                    # snapshot into the root and immutable SFT reference.
                    sft_trainer.save_adapter(ADAPTER_DIR / "sft_terminal_adapter")
                    if sft_monitor_best_adapter:
                        best_adapter_dir = Path(sft_monitor_best_adapter)
                        root_sft_checksum = _copy_adapter_snapshot(
                            best_adapter_dir, ADAPTER_DIR
                        )
                    else:
                        sft_trainer.save_adapter(ADAPTER_DIR)
                        root_sft_checksum = sha256_file(root_adapter_file)
                else:
                    sft_trainer.save_adapter(ADAPTER_DIR)
                    root_sft_checksum = sha256_file(root_adapter_file)
                reference_sft_checksum = None
                if sft_monitor_gate_passed:
                    if SFT_CHECKPOINT_MONITOR_ENABLED:
                        reference_sft_checksum = _copy_adapter_snapshot(
                            Path(sft_monitor_best_adapter), sft_adapter_dir
                        )
                    else:
                        sft_trainer.save_adapter(sft_adapter_dir)
                        reference_sft_checksum = sha256_file(sft_adapter_file)
                    if root_sft_checksum != reference_sft_checksum:
                        raise RuntimeError(
                            "SFT root adapter and immutable reference differ; refusing to mark SFT complete."
                        )
                dataset_version_file.write_text(dataset_fingerprint + "\n", encoding="utf-8")
                sft_loss = result["loss"]
                sft_metadata_file.write_text(json.dumps({
                    "status": "complete" if sft_monitor_gate_passed else "validation_gate_failed",
                    "dataset_fingerprint": dataset_fingerprint,
                    "reproducibility": reproducibility,
                    "hyperparameters": sft_hyperparameters,
                    "requested_sft_examples": requested_sft_examples,
                    "verified_sft_examples": verified_sft_examples,
                    "optimizer_sft_examples": optimizer_sft_examples,
                    "optimizer_padding_examples": optimizer_padding_examples,
                    "prompt_truncated_examples": sft_prompt_truncated_examples,
                    "generation_incompatible_completions_excluded": (
                        sft_generation_incompatible_completions
                    ),
                    "records_without_verified_winners": (
                        sft_records_without_verified_winners
                    ),
                    "records_without_generation_compatible_winners": (
                        sft_records_without_generation_compatible_winners
                    ),
                    "repository_sft_examples": repository_sft_examples,
                    "effective_repository_sft_examples": effective_repository_sft_examples,
                    "real_sampling_repeats": real_sampling_repeats,
                    "sampling_stats": sft_sampling_stats,
                    "bounded_selection_stats": bounded_selection_stats,
                    "optimizer_plan": sft_optimizer_plan,
                    "sft_loss": sft_loss,
                    "resumed_from_checkpoint": result["resumed_from_checkpoint"],
                    "trainer_checkpoint": result["trainer_checkpoint"],
                    "checkpoint_save_steps": result["checkpoint_save_steps"],
                    "warmup_steps": result["warmup_steps"],
                    "lr_scheduler_type": result["lr_scheduler_type"],
                    "model_runtime_profile": result["model_runtime_profile"],
                    "max_prompt_tokens": result["max_prompt_tokens"],
                    "max_repository_prompt_tokens": result[
                        "max_repository_prompt_tokens"
                    ],
                    "max_completion_tokens": result["max_completion_tokens"],
                    "planned_optimizer_steps": result["planned_optimizer_steps"],
                    "completed_optimizer_steps": result["completed_optimizer_steps"],
                    "completed_epochs": result["completed_epochs"],
                    "monitor_stopped_early": sft_monitor_stopped_early,
                    "monitor_baseline": sft_monitor_baseline,
                    "monitor_history": sft_monitor_history,
                    "monitor_best_adapter": sft_monitor_best_adapter,
                    "monitor_best_metrics": sft_monitor_best_metrics,
                    "monitor_gate_passed": sft_monitor_gate_passed,
                    "root_adapter_sha256": root_sft_checksum,
                    "sft_adapter_sha256": reference_sft_checksum,
                }, indent=2) + "\n", encoding="utf-8")
                sft_run_config_file.write_text(json.dumps({
                    "status": "complete" if sft_monitor_gate_passed else "validation_gate_failed",
                    "dataset_fingerprint": dataset_fingerprint,
                    "reproducibility": reproducibility,
                    "hyperparameters": sft_hyperparameters,
                    "optimizer_plan": sft_optimizer_plan,
                    "sft_loss": sft_loss,
                    "resumed_from_checkpoint": result["resumed_from_checkpoint"],
                }, indent=2) + "\n", encoding="utf-8")
                if not sft_monitor_gate_passed:
                    ADAPTER_DIR.joinpath("sft_early_stop.json").write_text(
                        json.dumps(
                            sft_monitor_history[-1]
                            if sft_monitor_history else sft_monitor_baseline,
                            indent=2,
                        ) + "\n",
                        encoding="utf-8",
                    )
                    print(
                        "SFT validation gate was not met at optimizer step "
                        f"{result['completed_optimizer_steps']}; best="
                        f"{float((sft_monitor_best_metrics or {}).get('function_kill_rate', 0.0)):.2%}, "
                        f"required={sft_hyperparameters['min_function_kill_rate']:.2%}. "
                        "DPO remains blocked."
                    )
                else:
                    sft_marker.write_text("complete\n", encoding="utf-8")
                    print(
                        f"SFT complete: loss={result['loss']:.4f}; "
                        f"reference saved to {sft_adapter_dir}"
                    )
            except Exception as exc:
                print(f"SFT failed; DPO will not start: {exc}")
                raise RuntimeError("Phase 3 stopped because SFT did not complete") from exc
            finally:
                _release_sft_trainer(sft_trainer)
        elif run_dpo:
            raise RuntimeError("DPO requires a verified SFT adapter and marker")
        elif TRAINING_PHASE in {"sft_eval", "dpo_eval"}:
            raise RuntimeError("Adapter validation requires a verified SFT adapter and marker")

    if TRAINING_PHASE == "sft_eval":
        if use_mock:
            raise RuntimeError("SFT validation must use the real immutable SFT adapter")
        if HOLDOUT_BUG_FAMILY and (
            f":holdout_bug_family={HOLDOUT_BUG_FAMILY}" not in dataset_fingerprint
        ):
            raise RuntimeError(
                "Leave-one-family-out evaluation requires an adapter trained with the same "
                "--holdout-bug-family setting."
            )
        return _evaluate_adapter_kill_rate(
            corpus_dir, dataset_fingerprint, ADAPTER_DIR / "sft_adapter", "sft_adapter",
            sft_validation_results_filename(SEED), evaluation_split=EVALUATION_SPLIT,
        )

    if TRAINING_PHASE == "dpo_eval":
        if use_mock:
            raise RuntimeError("DPO validation must use the real DPO adapter")
        if not dpo_complete_marker.exists() or not dpo_metadata_file.exists():
            raise RuntimeError("DPO validation requires a completed and accounted DPO adapter")
        dpo_metadata = json.loads(dpo_metadata_file.read_text(encoding="utf-8"))
        if dpo_metadata.get("dataset_fingerprint") != dataset_fingerprint:
            raise RuntimeError("DPO adapter metadata does not match the requested corpus and scope")
        return _evaluate_adapter_kill_rate(
            corpus_dir, dataset_fingerprint, ADAPTER_DIR, "dpo_adapter",
            "dpo_test_results.json", evaluation_split="test",
        )

    if run_sft and not run_dpo:
        results = {
            "mode": (
                "sft_validation_gate_failed"
                if not sft_monitor_gate_passed
                else "sft_best_checkpoint_selected"
                if SFT_CHECKPOINT_MONITOR_ENABLED
                else "sft_only"
            ),
            "train_pairs": len(train_pairs),
            "available_train_pairs_after_context_gate": available_train_pairs,
            "selected_synthetic_pairs": selected_synthetic_pairs,
            "selected_repository_pairs": selected_repository_pairs,
            "bounded_selection_stats": bounded_selection_stats,
            "verified_sft_examples": verified_sft_examples,
            "requested_sft_examples": requested_sft_examples,
            "optimizer_sft_examples": optimizer_sft_examples,
            "optimizer_padding_examples": optimizer_padding_examples,
            "prompt_truncated_examples": sft_prompt_truncated_examples,
            "sft_dropped_overlong_examples": requested_sft_examples - verified_sft_examples,
            "repository_overlong_completions_excluded": overlong_repository_completions,
            "generation_incompatible_completions_excluded": (
                sft_generation_incompatible_completions
            ),
            "records_without_verified_winners": (
                sft_records_without_verified_winners
            ),
            "records_without_generation_compatible_winners": (
                sft_records_without_generation_compatible_winners
            ),
            "repository_sft_examples": repository_sft_examples,
            "effective_repository_sft_examples": effective_repository_sft_examples,
            "real_sampling_repeats": real_sampling_repeats,
            "sft_sampling_stats": sft_sampling_stats,
            "sft_optimizer_plan": sft_optimizer_plan,
            "dataset_fingerprint": dataset_fingerprint,
            "reproducibility": reproducibility,
            "sft_loss": sft_loss,
            "sft_hyperparameters": sft_hyperparameters,
            "sft_monitor_stopped_early": sft_monitor_stopped_early,
            "sft_monitor_baseline": sft_monitor_baseline,
            "sft_monitor_history": sft_monitor_history,
            "sft_monitor_best_adapter": sft_monitor_best_adapter,
            "sft_monitor_best_metrics": sft_monitor_best_metrics,
            "sft_monitor_gate_passed": sft_monitor_gate_passed,
            "dpo_pairs": 0,
            "dpo_updates": 0,
            "repository_system_evaluation_status": REPOSITORY_EVALUATION_STATUS,
            "wall_time": round(time.time() - started, 1),
        }
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        with open(RESULTS_DIR / "training_results.json", "w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2)
        print(json.dumps(results, indent=2))
        return results

    sft_validation_baseline = None
    if run_dpo and not use_mock:
        # DPO selection is gated by the locked validation split.  A run that
        # has not yet met this SFT bar should spend compute improving SFT, not
        # attempting preference optimization on an underperforming policy.
        locked_validation_pairs = _evaluation_scope_pairs(
            corpus_dir, DPO_VALIDATION_SPLIT
        )
        expected_function_ids = [
            pair["id"] for pair in locked_validation_pairs
            if pair.get("execution_mode", FUNCTION_EXECUTION_MODE)
            == FUNCTION_EXECUTION_MODE
        ]
        expected_repository_records = sum(
            is_repository_execution_mode(
                pair.get("execution_mode", FUNCTION_EXECUTION_MODE)
            )
            for pair in locked_validation_pairs
        )
        sft_validation_baseline = _load_sft_validation_baseline(
            RESULTS_DIR,
            dataset_fingerprint,
            expected_function_ids=expected_function_ids,
            expected_repository_records=expected_repository_records,
        )
        sft_validation_rate = float(sft_validation_baseline["function_kill_rate"])
        if sft_validation_rate < MIN_SFT_FUNCTION_KILL_RATE_FOR_DPO:
            raise RuntimeError(
                "DPO is blocked by the SFT quality gate: locked validation function kill "
                f"rate is {sft_validation_rate:.2%}, below the required "
                f"{MIN_SFT_FUNCTION_KILL_RATE_FOR_DPO:.0%}. Improve and revalidate SFT first."
            )
        print(
            "[DPO GATE] SFT validation function kill rate "
            f"{sft_validation_rate:.2%} meets the "
            f"{MIN_SFT_FUNCTION_KILL_RATE_FOR_DPO:.0%} threshold."
        )

    dpo_trainer = None
    generator = None
    if use_mock:
        print("Running explicit mock mode; no SFT or DPO updates will occur")
    else:
        from engine.dpo_trainer import DPOTrainer
        from engine.generator import Phi3Generator

        try:
            dpo_trainer = DPOTrainer(
                output_dir=ADAPTER_DIR,
                model_name=BASE_MODEL_NAME_OVERRIDE,
                model_revision=BASE_MODEL_REVISION_OVERRIDE,
                attention_implementation=BASE_MODEL_ATTENTION_IMPLEMENTATION_OVERRIDE,
            )
            dpo_trainer.setup_model()
            if not dpo_trainer.has_reference_adapter:
                raise RuntimeError("DPO setup did not load the SFT reference adapter")
            generator = Phi3Generator()
            generator.model = dpo_trainer.model
            generator.tokenizer = dpo_trainer.tokenizer
            generator.is_loaded = True
            generator.max_new_tokens = MAX_NEW_TOKENS_OVERRIDE
            print("DPO model and generator share one GPU model instance")
        except Exception as exc:
            raise RuntimeError("DPO setup failed; training is stopped") from exc

    totals = {
        "total_winners": 0,
        "total_losers": 0,
        "total_dataset_winners": 0,
        "total_ai_winners": 0,
        "repository_dataset_winners": 0,
        "repository_dpo_pairs": 0,
        "dpo_pairs": 0,
        "dpo_pairs_trained": 0,
        "dpo_updates": 0,
    }
    state_file = ADAPTER_DIR / "resume_state.json"
    start_index = 0
    if not use_mock and state_file.exists():
        with open(state_file, "r", encoding="utf-8") as handle:
            state = json.load(handle)
        checkpoint_index = state.get("adapter_checkpoint_last_processed_index")
        expected_checksum = state.get("adapter_model_sha256")
        if not isinstance(checkpoint_index, int) or not expected_checksum:
            raise RuntimeError(
                "DPO resume state predates atomic checkpoints and cannot be resumed safely. "
                "Use --restart-dpo to restore the immutable SFT policy."
            )
        if state.get("dpo_training_scope_sha256") != dpo_training_scope_sha256:
            raise RuntimeError(
                "DPO resume state does not match the exact ordered, context-eligible training "
                "selection. Use --restart-dpo to restore the immutable SFT policy."
            )
        if state.get("last_processed_index") != checkpoint_index:
            raise RuntimeError(
                "DPO resume state is ahead of its adapter checkpoint; refusing to skip records. "
                "Use --restart-dpo to restore the immutable SFT policy."
            )
        adapter_file = ADAPTER_DIR / "adapter_model.safetensors"
        if not adapter_file.exists() or sha256_file(adapter_file) != expected_checksum:
            raise RuntimeError(
                "DPO policy adapter does not match its resume state; refusing a mixed resume. "
                "Use --restart-dpo to restore the immutable SFT policy."
            )
        start_index = checkpoint_index + 1
        for key in totals:
            totals[key] = state.get(key, 0)
        if start_index:
            print(f"[RESUME] Continuing at pair {start_index + 1}")

    def stop_dpo_for_validation(checkpoint: Dict) -> Dict:
        """Persist an explicit non-success result without marking DPO complete."""
        result = {
            "mode": "dpo_early_stopped",
            "reason": "locked_validation_did_not_improve_over_sft",
            "checkpoint_pairs": checkpoint["checkpoint_pairs"],
            "train_pairs": len(train_pairs),
            "dpo_pairs": totals["dpo_pairs"],
            "dpo_pairs_trained": totals["dpo_pairs_trained"],
            "dpo_updates": totals["dpo_updates"],
            "dpo_training_scope_sha256": dpo_training_scope_sha256,
            "repository_dpo_overlong_completions_excluded": (
                dpo_overlong_repository_completions
            ),
            "sft_baseline_function_kill_rate": checkpoint["sft_baseline_function_kill_rate"],
            "dpo_validation_function_kill_rate": checkpoint["function_kill_rate"],
            "validation_split": DPO_VALIDATION_SPLIT,
            "wall_time": round(time.time() - started, 1),
        }
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        (RESULTS_DIR / "dpo_training_results.json").write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8"
        )
        (ADAPTER_DIR / "dpo_early_stop.json").write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8"
        )
        print("[DPO EARLY STOP] Locked validation did not beat the SFT baseline; pivot to SFT.")
        print(json.dumps(result, indent=2))
        return result

    def pending_validation_thresholds() -> List[int]:
        if not dpo_trainer or not sft_validation_baseline:
            return []
        completed = {
            item.get("checkpoint_pairs") for item in _read_dpo_validation_trend(RESULTS_DIR)
        }
        return [
            threshold
            for threshold in range(
                DPO_VALIDATION_INTERVAL_PAIRS,
                totals["dpo_pairs_trained"] + 1,
                DPO_VALIDATION_INTERVAL_PAIRS,
            )
            if threshold not in completed
        ]

    def evaluate_dpo_checkpoint_if_needed(
        checkpoint_pairs: int, processed_records: int,
    ) -> Optional[Dict]:
        if not dpo_trainer or not sft_validation_baseline:
            return None
        trend = _read_dpo_validation_trend(RESULTS_DIR)
        existing = next(
            (item for item in trend if item.get("checkpoint_pairs") == checkpoint_pairs), None
        )
        checkpoint = existing or _append_dpo_validation_checkpoint(
            corpus_dir, dataset_fingerprint, ADAPTER_DIR, RESULTS_DIR,
            checkpoint_pairs, totals["dpo_pairs_trained"], processed_records,
            totals["dpo_updates"], sft_validation_baseline,
        )
        print(
            "[DPO VALIDATION] checkpoint="
            f"{checkpoint_pairs} function_kill_rate={checkpoint['function_kill_rate']:.2%} "
            f"vs_sft={checkpoint['sft_baseline_function_kill_rate']:.2%} "
            f"decision={checkpoint['decision']}"
        )
        return stop_dpo_for_validation(checkpoint) if not checkpoint["improved_over_sft"] else None

    dpo_buffer = []
    ai_kill_rates = []
    # A container can fail while evaluating a saved checkpoint.  On resume,
    # evaluate that checkpoint before training another record so it cannot be
    # silently bypassed.
    if dpo_trainer and start_index:
        for threshold in pending_validation_thresholds():
            early_stop_result = evaluate_dpo_checkpoint_if_needed(threshold, start_index)
            if early_stop_result is not None:
                return early_stop_result
    index = start_index
    while index < len(train_pairs):
        chunk_end = min(index + BATCH_GEN_SIZE, len(train_pairs))
        chunk = train_pairs[index:chunk_end]
        print(f"\nBatch [{index + 1}-{chunk_end}] / {len(train_pairs)}", flush=True)

        for pair in chunk:
            if is_repository_execution_mode(pair.get("execution_mode", FUNCTION_EXECUTION_MODE)):
                repository_winners = _repository_fragment_tests(pair.get("test_cases", []))
                totals["repository_dataset_winners"] += len(repository_winners)
                if dpo_trainer:
                    added = _append_repository_preferences(dpo_buffer, dpo_trainer, pair, repository_winners)
                    totals["dpo_pairs"] += added
                    totals["repository_dpo_pairs"] += added
                continue
            dataset_tests = extract_dataset_tests(pair.get("test_cases", []), pair["entry_point"])
            dataset_winners, dataset_losers = evaluate_pair(
                dataset_tests, pair["golden_code"], pair["mutant_code"],
                pair["entry_point"],
            )
            totals["total_dataset_winners"] += len(dataset_winners)
            if dpo_trainer and dataset_winners:
                totals["dpo_pairs"] += _append_preferences(
                    dpo_buffer,
                    dpo_trainer,
                    pair,
                    dataset_winners,
                    dataset_losers or bootstrap_losers(pair),
                )

        function_pairs = [
            (item_index, pair) for item_index, pair in enumerate(chunk)
            if pair.get("execution_mode", FUNCTION_EXECUTION_MODE) == FUNCTION_EXECUTION_MODE
        ]
        if use_mock:
            ai_test_map = {
                item_index: generate_tests_mock(pair["golden_code"], pair["entry_point"], TESTS_PER_PAIR)
                for item_index, pair in function_pairs
            }
        elif function_pairs:
            generated = generate_tests_ai_batched(
                generator, [pair for _, pair in function_pairs], TESTS_PER_PAIR
            )
            ai_test_map = {
                item_index: generated.get(generated_index, [])
                for generated_index, (item_index, _) in enumerate(function_pairs)
            }
        else:
            ai_test_map = {}

        for item_index, pair in enumerate(chunk):
            if pair.get("execution_mode", FUNCTION_EXECUTION_MODE) != FUNCTION_EXECUTION_MODE:
                continue
            ai_tests = ai_test_map.get(item_index, [])
            ai_winners, ai_losers = evaluate_pair(
                ai_tests, pair["golden_code"], pair["mutant_code"], pair["entry_point"]
            )
            totals["total_ai_winners"] += len(ai_winners)
            totals["total_winners"] += len(ai_winners)
            totals["total_losers"] += len(ai_losers)
            if ai_winners or ai_losers:
                ai_kill_rates.append(len(ai_winners) / (len(ai_winners) + len(ai_losers)))
                ai_kill_rates = ai_kill_rates[-200:]
            if dpo_trainer:
                totals["dpo_pairs"] += _append_preferences(
                    dpo_buffer, dpo_trainer, pair, ai_winners, ai_losers
                )

        while dpo_trainer and len(dpo_buffer) >= DPO_BUFFER_SIZE:
            batch = dpo_buffer[:DPO_BUFFER_SIZE]
            try:
                result = dpo_trainer.train(batch, num_epochs=1, batch_size=DPO_BATCH_SIZE)
            except Exception as exc:
                raise RuntimeError(
                    "DPO update failed; resume state was not advanced beyond the last verified "
                    "adapter checkpoint."
                ) from exc
            dpo_buffer = dpo_buffer[DPO_BUFFER_SIZE:]
            totals["dpo_pairs_trained"] += len(batch)
            totals["dpo_updates"] += 1
            print(f"DPO update {totals['dpo_updates']}: loss={result['loss']:.4f}", flush=True)

        current_ai_rate = sum(ai_kill_rates) / len(ai_kill_rates) if ai_kill_rates else 0.0
        elapsed = time.time() - started
        print(
            f"progress={chunk_end}/{len(train_pairs)} "
            f"datasetW={totals['total_dataset_winners']} aiW={totals['total_ai_winners']} "
            f"ai_kill={current_ai_rate:.1%} elapsed={elapsed / 60:.1f}m",
            flush=True,
        )

        validation_thresholds = pending_validation_thresholds()
        if dpo_trainer and (chunk_end % 50 == 0 or validation_thresholds):
            dpo_trainer.save_adapter(ADAPTER_DIR)
            _write_resume_state(
                state_file, chunk_end - 1, totals, ADAPTER_DIR,
                dpo_training_scope_sha256,
            )
            print(f"[CHECKPOINT] Saved at pair {chunk_end}")
            for threshold in validation_thresholds:
                early_stop_result = evaluate_dpo_checkpoint_if_needed(threshold, chunk_end)
                if early_stop_result is not None:
                    return early_stop_result
        index = chunk_end

    if dpo_trainer and dpo_buffer:
        result = dpo_trainer.train(dpo_buffer, num_epochs=1, batch_size=DPO_BATCH_SIZE)
        totals["dpo_pairs_trained"] += len(dpo_buffer)
        totals["dpo_updates"] += 1
        print(f"Final DPO update: loss={result['loss']:.4f}")

    if dpo_trainer:
        dpo_trainer.save_adapter(ADAPTER_DIR)
        _write_resume_state(
            state_file, len(train_pairs) - 1, totals, ADAPTER_DIR,
            dpo_training_scope_sha256,
        )
        for threshold in pending_validation_thresholds():
            early_stop_result = evaluate_dpo_checkpoint_if_needed(threshold, len(train_pairs))
            if early_stop_result is not None:
                return early_stop_result

    val_rate = None
    if not use_mock and TRAINING_PHASE == "sft_then_dpo":
        all_val_pairs = load_phase3_pairs(corpus_dir, "val")
        if EXECUTION_MODE_FILTER:
            all_val_pairs = [
                pair for pair in all_val_pairs
                if pair.get("execution_mode", FUNCTION_EXECUTION_MODE) == EXECUTION_MODE_FILTER
            ]
        # A smoke run must constrain validation as well as training; otherwise a
        # tiny --max-pairs run silently launches the full held-out evaluation.
        if MAX_TRAIN_PAIRS:
            all_val_pairs = all_val_pairs[:MAX_TRAIN_PAIRS]
        repository_validation_records = sum(
            is_repository_execution_mode(pair.get("execution_mode", FUNCTION_EXECUTION_MODE))
            for pair in all_val_pairs
        )
        val_pairs = [
            pair for pair in all_val_pairs
            if pair.get("execution_mode", FUNCTION_EXECUTION_MODE) == FUNCTION_EXECUTION_MODE
        ]
        killed = 0
        for pair in val_pairs:
            tests = generate_tests_ai_batched(generator, [pair], TESTS_PER_PAIR).get(0, [])
            winners, _ = evaluate_pair(
                tests, pair["golden_code"], pair["mutant_code"], pair["entry_point"]
            )
            killed += int(bool(winners))
        val_rate = killed / max(len(val_pairs), 1) if val_pairs else None
        print(
            f"Validation function kill rate: {(val_rate or 0.0):.1%} ({killed}/{len(val_pairs)}); "
            f"repository fragments held for repository-level evaluation: {repository_validation_records}"
        )

    ai_kill_rate = totals["total_winners"] / max(
        totals["total_winners"] + totals["total_losers"], 1
    )
    results = {
        "mode": "mock" if use_mock else TRAINING_PHASE,
        "train_pairs": len(train_pairs),
        "available_train_pairs_after_context_gate": available_train_pairs,
        "selected_synthetic_pairs": selected_synthetic_pairs,
        "selected_repository_pairs": selected_repository_pairs,
        "bounded_selection_stats": bounded_selection_stats,
        "verified_sft_examples": verified_sft_examples,
        "requested_sft_examples": requested_sft_examples,
        "optimizer_sft_examples": optimizer_sft_examples,
        "optimizer_padding_examples": optimizer_padding_examples,
        "prompt_truncated_examples": sft_prompt_truncated_examples,
        "sft_dropped_overlong_examples": requested_sft_examples - verified_sft_examples,
        "repository_overlong_completions_excluded": overlong_repository_completions,
        "generation_incompatible_completions_excluded": (
            sft_generation_incompatible_completions
        ),
        "repository_dpo_overlong_completions_excluded": (
            dpo_overlong_repository_completions
        ),
        "repository_sft_examples": repository_sft_examples,
        "effective_repository_sft_examples": effective_repository_sft_examples,
        "real_sampling_repeats": real_sampling_repeats,
        "sft_optimizer_plan": sft_optimizer_plan,
        "dataset_fingerprint": dataset_fingerprint,
        "reproducibility": reproducibility,
        "sft_hyperparameters": sft_hyperparameters,
        "sft_loss": sft_loss,
        "dpo_pairs": totals["dpo_pairs"],
        "dpo_pairs_trained": totals["dpo_pairs_trained"],
        "dpo_updates": totals["dpo_updates"],
        "dpo_training_scope_sha256": dpo_training_scope_sha256,
        "dpo_validation_checkpoints": (
            _read_dpo_validation_trend(RESULTS_DIR) if run_dpo and not use_mock else []
        ),
        "dataset_winners": totals["total_dataset_winners"],
        "repository_dataset_winners": totals["repository_dataset_winners"],
        "repository_dpo_pairs": totals["repository_dpo_pairs"],
        "ai_winners": totals["total_ai_winners"],
        "ai_kill_rate": round(ai_kill_rate, 4),
        "val_kill_rate": round(val_rate, 4) if val_rate is not None else None,
        "repository_system_evaluation_status": REPOSITORY_EVALUATION_STATUS,
        "wall_time": round(time.time() - started, 1),
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_filename = "dpo_training_results.json" if TRAINING_PHASE == "dpo" else "training_results.json"
    with open(RESULTS_DIR / results_filename, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    if run_dpo and not use_mock:
        dpo_metadata_file.write_text(json.dumps({
            "dataset_fingerprint": dataset_fingerprint,
            "reproducibility": reproducibility,
            "seed": SEED,
            "sft_reference_adapter": "sft_adapter",
            "dpo_pairs": totals["dpo_pairs"],
            "dpo_pairs_trained": totals["dpo_pairs_trained"],
            "dpo_updates": totals["dpo_updates"],
            "dpo_training_scope_sha256": dpo_training_scope_sha256,
            "repository_dpo_overlong_completions_excluded": (
                dpo_overlong_repository_completions
            ),
            "validation_split": DPO_VALIDATION_SPLIT,
            "sft_validation_function_kill_rate": sft_validation_baseline["function_kill_rate"],
        }, indent=2) + "\n", encoding="utf-8")
        dpo_complete_marker.write_text("complete\n", encoding="utf-8")
    print(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Oneiros verified SFT followed by DPO")
    parser.add_argument("--mock", action="store_true", help="Run static generation only; skip model training")
    parser.add_argument("--fresh", action="store_true", help="Delete Phase 3 checkpoints before SFT")
    parser.add_argument("--max-pairs", type=int, default=None, help="Limit training pairs for a smoke test")
    parser.add_argument(
        "--max-validation-functions", type=int, default=None,
        help="Bound validation for an explicitly labelled smoke run; never changes training scope",
    )
    parser.add_argument("--seed", type=int, default=SEED, help="Non-negative training and generation seed")
    parser.add_argument(
        "--eval-feedback-rounds", type=int, default=0,
        help="Reference-free execution-repair rounds within the fixed eight-candidate budget",
    )
    parser.add_argument(
        "--eval-diversity-mode", choices=["none", "ast", "input_shape"], default="none",
        help="Equal-budget candidate prioritisation ablation",
    )
    parser.add_argument(
        "--evaluation-split", choices=["ablation_dev", "val"], default="val",
        help="Use training-only ablation_dev for design experiments; val remains locked model selection",
    )
    parser.add_argument(
        "--holdout-bug-family", default="",
        help="Exclude one mutation family from training and evaluate only that family",
    )
    parser.add_argument(
        "--prompt-information-variant",
        choices=PROMPT_INFORMATION_VARIANTS,
        default="full",
        help="Ablation A: code only, code plus specification, or the full legitimate context",
    )
    parser.add_argument(
        "--output-instruction-variant",
        choices=OUTPUT_INSTRUCTION_VARIANTS,
        default="self_contained",
        help="Ablation C: legacy exactly-one wording or the V4.1 self-contained-test wording",
    )
    parser.add_argument(
        "--dpo-validation-interval-pairs", type=int, default=DPO_VALIDATION_INTERVAL_PAIRS,
        help="Evaluate locked validation after this many trained DPO preference pairs (default: 500)",
    )
    parser.add_argument("--sft-epochs", type=int, default=None, help="Override SFT epochs for this named run")
    parser.add_argument(
        "--sft-learning-rate", type=float, default=None,
        help="Override SFT learning rate for this named run",
    )
    parser.add_argument(
        "--sft-lr-scheduler-type",
        choices=["cosine", "constant_with_warmup"],
        default=None,
        help="Override the explicit SFT LR scheduler for this named run",
    )
    parser.add_argument(
        "--sft-batch-size", type=int, default=None,
        help="Override SFT micro-batch size for this named run",
    )
    parser.add_argument(
        "--sft-prompt-token-limit",
        type=int,
        default=None,
        help=(
            "Function-mode prompt budget for training and generation. It is part "
            "of the evaluation scope hash, so changing it declares a different "
            "protocol and must be run as a recorded ablation, never mixed into "
            "an existing comparison."
        ),
    )
    parser.add_argument(
        "--sft-selection-prompt-token-limit",
        type=int,
        default=None,
        help=(
            "Prompt budget used only to define bounded-selection eligibility. "
            "Pin this to a common admissible floor when comparing prompt budgets."
        ),
    )
    parser.add_argument(
        "--sft-repository-completion-token-limit",
        type=int,
        default=None,
        help=(
            "Mode-specific SFT completion budget for verified repository fragments "
            "(function assertions remain capped at 128)"
        ),
    )
    parser.add_argument(
        "--sft-real-target-fraction", type=float, default=None,
        help="Target effective share of verified repository examples in SFT (default: 0.20)",
    )
    parser.add_argument(
        "--sft-max-real-repeats", type=int, default=None,
        help="Maximum deterministic repeats per verified repository SFT example (default: 8)",
    )
    parser.add_argument(
        "--sft-balanced-sampling", action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Enable or disable exact deduplication and deterministic project/group-balanced "
            "repetition (enabled by default for new runs)"
        ),
    )
    parser.add_argument(
        "--sft-synthetic-balance-fraction", type=float,
        default=SFT_SYNTHETIC_BALANCE_FRACTION,
        help="Optional additional rare-family SFT fraction for the balanced sampler",
    )
    parser.add_argument(
        "--sft-synthetic-balance-mode",
        choices=["none", "dataset", "dataset_family"],
        default=SFT_SYNTHETIC_BALANCE_MODE,
        help="Ablation G grouping used for deterministic synthetic balancing",
    )
    parser.add_argument(
        "--sft-max-synthetic-repeats", type=int,
        default=SFT_MAX_SYNTHETIC_REPEATS,
        help="Maximum copies of any synthetic example under balanced sampling",
    )
    parser.add_argument(
        "--sft-complex-target-fraction",
        type=float,
        default=SFT_COMPLEX_TARGET_FRACTION,
        help=(
            "Minimum complex-function share in bounded synthetic SFT selection, "
            "using only buggy-side AST metrics (default: 0.60)"
        ),
    )
    parser.add_argument(
        "--sft-selection-tokenizer-name",
        default=None,
        help=(
            "Tokenizer that decides supervision eligibility. Defaults to the "
            "run's own base model. Pin every arm of a base-model comparison to "
            "one common tokenizer so the arms train on identical records."
        ),
    )
    parser.add_argument(
        "--sft-checkpoint-steps",
        type=int,
        default=None,
        help=(
            "Optimizer-step interval between monitored checkpoints. Must match "
            "the --checkpoint-steps value given to preflight, or the planned and "
            "runtime monitor schedules will disagree."
        ),
    )
    parser.add_argument(
        "--sft-monitor-kill-rate", action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable the locked 50-step validation monitor (enabled by default)",
    )
    parser.add_argument(
        "--sft-monitor-validation-functions", type=int,
        default=SFT_MONITOR_VALIDATION_FUNCTIONS,
        help="Number of locked validation functions used by the SFT checkpoint monitor",
    )
    parser.add_argument(
        "--sft-monitor-patience", type=int, default=SFT_MONITOR_PATIENCE,
        help="Consecutive checkpoints without a new best kill rate before stopping",
    )
    parser.add_argument(
        "--sft-monitor-min-function-kill-rate",
        type=float,
        default=None,
        help="Required best locked-panel function kill rate before SFT is promoted",
    )
    parser.add_argument(
        "--sft-min-monitor-checkpoints",
        type=int,
        default=None,
        help=(
            "Minimum checkpoint evaluations required by preflight; use one only "
            "for a declared terminal-monitor integration run"
        ),
    )
    parser.add_argument(
        "--corpus-version",
        default=CANONICAL_CORPUS_VERSION,
        help="Canonical corpus version",
    )
    parser.add_argument(
        "--base-model-name", default=None,
        help=(
            "Override the canonical Phi-3 base model for a declared base-model "
            "ablation (e.g. Qwen/Qwen2.5-Coder-1.5B-Instruct). Leave unset for "
            "every canonical/production run; the choice is recorded in the "
            "run's reproducibility manifest so it can never be mistaken for "
            "the canonical model."
        ),
    )
    parser.add_argument(
        "--base-model-revision", default=None,
        help="Pin an exact snapshot for --base-model-name (defaults to 'main' if unset)",
    )
    parser.add_argument(
        "--attention-implementation", default=None, choices=[None, "eager", "sdpa", "flash_attention_2"],
        help="Override the canonical eager attention backend for this run",
    )
    parser.add_argument(
        "--phase", default="sft", choices=["base_eval", "sft", "sft_eval", "dpo", "dpo_eval", "sft_then_dpo"],
        help="Run SFT, locked adapter validation, DPO from a verified SFT adapter, or the combined legacy flow",
    )
    parser.add_argument(
        "--execution-mode", default="",
        choices=["", FUNCTION_EXECUTION_MODE, *sorted(REPOSITORY_EXECUTION_MODES)],
        help="Optionally train only one canonical execution mode (useful for targeted smoke tests)",
    )
    parser.add_argument(
        "--restart-dpo", action="store_true",
        help="Restore the immutable SFT adapter and restart DPO from pair 1",
    )
    parser.add_argument(
        "--confirm-final-test", action="store_true",
        help="Explicit one-time authorization for the sealed dpo_eval test split",
    )
    parser.add_argument(
        "--run-name", required=True,
        help="Local run identity; isolates checkpoints and results without Modal",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print local paths/options without opening the corpus or loading a model",
    )
    args = parser.parse_args()
    from utils.local_run import local_run_paths

    try:
        ADAPTER_DIR, RESULTS_DIR = local_run_paths(
            Path(__file__).resolve().parent.parent, args.run_name, fresh=args.fresh,
        )
    except ValueError as exc:
        parser.error(str(exc))
    if args.dry_run:
        print(json.dumps({
            "backend": "local_cuda",
            "adapter_dir": str(ADAPTER_DIR),
            "results_dir": str(RESULTS_DIR),
            "options": vars(args),
            "training_launched": False,
            "corpus_opened": False,
            "readiness_checked": False,
            "warning": "Execution requires research preflight and sealed-data access isolation",
        }, indent=2))
        sys.exit(0)
    if args.seed < 0:
        raise ValueError("--seed must be non-negative")
    SEED = args.seed
    if args.eval_feedback_rounds < 0 or args.eval_feedback_rounds >= TESTS_PER_PAIR:
        raise ValueError(f"--eval-feedback-rounds must be between 0 and {TESTS_PER_PAIR - 1}")
    EVAL_FEEDBACK_ROUNDS = args.eval_feedback_rounds
    EVAL_DIVERSITY_MODE = args.eval_diversity_mode
    EVALUATION_SPLIT = args.evaluation_split
    HOLDOUT_BUG_FAMILY = sanitise_family_name(args.holdout_bug_family)
    PROMPT_INFORMATION_VARIANT = args.prompt_information_variant
    OUTPUT_INSTRUCTION_VARIANT = args.output_instruction_variant
    if args.max_pairs:
        MAX_TRAIN_PAIRS = args.max_pairs
    if args.max_validation_functions:
        if args.max_validation_functions <= 0:
            raise ValueError("--max-validation-functions must be positive")
        MAX_VALIDATION_PAIRS = args.max_validation_functions
    if args.dpo_validation_interval_pairs <= 0:
        raise ValueError("--dpo-validation-interval-pairs must be positive")
    if args.sft_epochs is not None and args.sft_epochs <= 0:
        raise ValueError("--sft-epochs must be positive")
    if args.sft_learning_rate is not None and args.sft_learning_rate <= 0:
        raise ValueError("--sft-learning-rate must be positive")
    if args.sft_batch_size is not None and args.sft_batch_size <= 0:
        raise ValueError("--sft-batch-size must be positive")
    if (
        args.sft_repository_completion_token_limit is not None
        and not 0 < args.sft_repository_completion_token_limit < MAX_SFT_COMPLETION_TOKENS
    ):
        raise ValueError(
            "--sft-repository-completion-token-limit must be between 1 and 2047"
        )
    if args.sft_real_target_fraction is not None and not 0.0 <= args.sft_real_target_fraction < 1.0:
        raise ValueError("--sft-real-target-fraction must be in [0, 1)")
    if args.sft_max_real_repeats is not None and args.sft_max_real_repeats < 1:
        raise ValueError("--sft-max-real-repeats must be at least one")
    if args.sft_synthetic_balance_fraction < 0.0:
        raise ValueError("--sft-synthetic-balance-fraction must be non-negative")
    if args.sft_synthetic_balance_mode == "none" and args.sft_synthetic_balance_fraction:
        raise ValueError(
            "--sft-synthetic-balance-fraction requires dataset or dataset_family mode"
        )
    if args.sft_max_synthetic_repeats < 1:
        raise ValueError("--sft-max-synthetic-repeats must be at least one")
    if not 0.0 <= args.sft_complex_target_fraction <= 1.0:
        raise ValueError("--sft-complex-target-fraction must be in [0, 1]")
    if args.sft_monitor_validation_functions <= 0:
        raise ValueError("--sft-monitor-validation-functions must be positive")
    if args.sft_monitor_patience <= 0:
        raise ValueError("--sft-monitor-patience must be positive")
    if (
        args.sft_monitor_min_function_kill_rate is not None
        and not 0.0 <= args.sft_monitor_min_function_kill_rate <= 1.0
    ):
        raise ValueError("--sft-monitor-min-function-kill-rate must be in [0, 1]")
    if (
        args.sft_min_monitor_checkpoints is not None
        and args.sft_min_monitor_checkpoints < 1
    ):
        raise ValueError("--sft-min-monitor-checkpoints must be at least one")
    DPO_VALIDATION_INTERVAL_PAIRS = args.dpo_validation_interval_pairs
    SFT_EPOCHS_OVERRIDE = args.sft_epochs
    SFT_LEARNING_RATE_OVERRIDE = args.sft_learning_rate
    SFT_LR_SCHEDULER_TYPE_OVERRIDE = args.sft_lr_scheduler_type
    SFT_BATCH_SIZE_OVERRIDE = args.sft_batch_size
    if args.sft_prompt_token_limit is not None:
        from engine.sft_trainer import MAX_SFT_SEQUENCE_LENGTH

        if not 0 < args.sft_prompt_token_limit < MAX_SFT_SEQUENCE_LENGTH:
            raise ValueError(
                "--sft-prompt-token-limit must be between 1 and the sequence limit"
            )
        PROMPT_TOKEN_LIMIT = args.sft_prompt_token_limit
    if args.sft_selection_prompt_token_limit is not None:
        from engine.sft_trainer import MAX_SFT_SEQUENCE_LENGTH

        if not 0 < args.sft_selection_prompt_token_limit < MAX_SFT_SEQUENCE_LENGTH:
            raise ValueError(
                "--sft-selection-prompt-token-limit must be between 1 and the sequence limit"
            )
        SFT_SELECTION_PROMPT_TOKEN_LIMIT_OVERRIDE = (
            args.sft_selection_prompt_token_limit
        )
    SFT_REPOSITORY_COMPLETION_TOKEN_LIMIT_OVERRIDE = (
        args.sft_repository_completion_token_limit
    )
    SFT_REAL_TARGET_FRACTION_OVERRIDE = args.sft_real_target_fraction
    SFT_MAX_REAL_REPEATS_OVERRIDE = args.sft_max_real_repeats
    if args.sft_balanced_sampling is not None:
        SFT_BALANCED_SAMPLING_ENABLED = args.sft_balanced_sampling
    SFT_SYNTHETIC_BALANCE_FRACTION = args.sft_synthetic_balance_fraction
    SFT_SYNTHETIC_BALANCE_MODE = args.sft_synthetic_balance_mode
    SFT_MAX_SYNTHETIC_REPEATS = args.sft_max_synthetic_repeats
    SFT_COMPLEX_TARGET_FRACTION_OVERRIDE = args.sft_complex_target_fraction
    REQUIRE_SPLIT_ISOLATION = True
    if args.sft_monitor_kill_rate is not None:
        SFT_CHECKPOINT_MONITOR_ENABLED = args.sft_monitor_kill_rate
    SFT_MONITOR_VALIDATION_FUNCTIONS = args.sft_monitor_validation_functions
    SFT_MONITOR_PATIENCE = args.sft_monitor_patience
    SFT_MONITOR_MIN_FUNCTION_KILL_RATE_OVERRIDE = (
        args.sft_monitor_min_function_kill_rate
    )
    SFT_MIN_MONITOR_CHECKPOINTS_OVERRIDE = args.sft_min_monitor_checkpoints
    CORPUS_VERSION = args.corpus_version
    BASE_MODEL_NAME_OVERRIDE = args.base_model_name
    BASE_MODEL_REVISION_OVERRIDE = args.base_model_revision
    BASE_MODEL_ATTENTION_IMPLEMENTATION_OVERRIDE = args.attention_implementation
    if args.sft_checkpoint_steps is not None:
        if args.sft_checkpoint_steps <= 0:
            raise ValueError("--sft-checkpoint-steps must be positive")
        SFT_CHECKPOINT_STEPS_OVERRIDE = args.sft_checkpoint_steps
    SFT_SELECTION_TOKENIZER_NAME_OVERRIDE = args.sft_selection_tokenizer_name
    EXECUTION_MODE_FILTER = args.execution_mode or None
    TRAINING_PHASE = args.phase
    if TRAINING_PHASE in {"dpo", "dpo_eval", "sft_then_dpo"} and EVALUATION_SPLIT != "val":
        raise ValueError("DPO gating and final comparison require the locked val split")
    RESTART_DPO = args.restart_dpo
    CONFIRM_FINAL_TEST = args.confirm_final_test
    if not args.mock:
        import torch

        if not torch.cuda.is_available():
            parser.error("Local execution requires a CUDA GPU; CPU fallback is disabled")
        print(f"Local GPU: {torch.cuda.get_device_name(0)}", flush=True)
    run_training(use_mock=args.mock, fresh=args.fresh)
