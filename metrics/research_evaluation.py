"""Research-grade metrics and diagnostics for Oneiros candidate evaluation.

The training loop deliberately keeps the reference implementation out of the
model prompt.  This module receives generated candidates only after inference
and records their ordered oracle outcomes.  Keeping the raw generation rank is
what makes prefix metrics such as Kill@1/2/4/8 reproducible.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from copy import deepcopy
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from harness.candidate_policy import validate_function_assertion
from harness.safe_execution import classify_assertions


RESEARCH_METRICS_SCHEMA_VERSION = 2
DEFAULT_K_VALUES: Tuple[int, ...] = (1, 2, 4, 8)


def wilson_interval(successes: int, total: int, z: float = 1.959964) -> List[float]:
    """Return a two-sided Wilson score interval for a binomial proportion."""
    if total <= 0:
        return [0.0, 0.0]
    proportion = successes / total
    denominator = 1.0 + (z * z / total)
    centre = (proportion + z * z / (2.0 * total)) / denominator
    margin = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / total
            + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return [round(max(0.0, centre - margin), 6), round(min(1.0, centre + margin), 6)]


def _normalise_code(code: str) -> str:
    return " ".join(str(code).strip().split())


class _ConstantShapeNormaliser(ast.NodeTransformer):
    """Replace literal values while retaining their type and container shape."""

    def visit_Constant(self, node: ast.Constant) -> ast.AST:  # noqa: N802
        value = node.value
        if value is None or isinstance(value, bool):
            marker = repr(value)
        else:
            marker = f"<{type(value).__name__}>"
        return ast.copy_location(ast.Constant(value=marker), node)


def ast_shape_key(code: str) -> str:
    """Return a value-insensitive AST key, falling back to normalised text."""
    try:
        tree = ast.parse(code)
        tree = _ConstantShapeNormaliser().visit(tree)
        ast.fix_missing_locations(tree)
        return ast.dump(tree, annotate_fields=True, include_attributes=False)
    except (SyntaxError, TypeError, ValueError):
        return f"invalid:{_normalise_code(code)}"


def _literal_shape(node: ast.AST) -> str:
    if isinstance(node, ast.Constant):
        value = node.value
        if isinstance(value, (str, bytes)):
            size = len(value)
            bucket = "empty" if size == 0 else "one" if size == 1 else "many"
            return f"{type(value).__name__}:{bucket}"
        return type(value).__name__
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        size = len(node.elts)
        bucket = "empty" if size == 0 else "one" if size == 1 else "many"
        return f"{type(node).__name__.lower()}:{bucket}"
    if isinstance(node, ast.Dict):
        size = len(node.keys)
        bucket = "empty" if size == 0 else "one" if size == 1 else "many"
        return f"dict:{bucket}"
    if isinstance(node, ast.UnaryOp):
        return f"unary:{type(node.op).__name__}:{_literal_shape(node.operand)}"
    return type(node).__name__


def input_shape_key(code: str, entry_point: str) -> str:
    """Describe the argument shape used to invoke the target entry point."""
    try:
        tree = ast.parse(code)
    except (SyntaxError, TypeError, ValueError):
        return "invalid"
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = None
        if isinstance(node.func, ast.Name):
            name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            name = node.func.attr
        if name != entry_point:
            continue
        positional = ",".join(_literal_shape(arg) for arg in node.args)
        keywords = ",".join(
            f"{keyword.arg or '**'}:{_literal_shape(keyword.value)}"
            for keyword in node.keywords
        )
        return f"args[{positional}]|kwargs[{keywords}]"
    return "no_target_call"


def prioritise_diverse_slots(
    slots: Sequence[Mapping[str, Any]], entry_point: str, mode: str = "none"
) -> List[Dict[str, Any]]:
    """Prioritise structurally novel candidates without increasing generation budget.

    This is candidate prioritisation, not free oversampling: the same raw model
    sequences are retained and only their prefix order changes.  Invalid slots
    remain at the end in their original order.
    """
    if mode not in {"none", "ast", "input_shape"}:
        raise ValueError(f"Unsupported diversity mode: {mode!r}")
    copied = [deepcopy(dict(slot)) for slot in slots]
    if mode == "none":
        return copied

    valid = [slot for slot in copied if slot.get("parse_valid") and slot.get("code")]
    invalid = [slot for slot in copied if not (slot.get("parse_valid") and slot.get("code"))]
    key_fn = ast_shape_key if mode == "ast" else lambda code: input_shape_key(code, entry_point)
    buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    key_order: List[str] = []
    for slot in valid:
        key = key_fn(str(slot["code"]))
        if key not in buckets:
            key_order.append(key)
        buckets[key].append(slot)

    ordered: List[Dict[str, Any]] = []
    while any(buckets.values()):
        for key in key_order:
            if buckets[key]:
                ordered.append(buckets[key].pop(0))
    ordered.extend(invalid)
    for rank, slot in enumerate(ordered, 1):
        slot["original_rank"] = int(slot.get("rank", rank))
        slot["rank"] = rank
    return ordered


def _outcome_mode(outcome: Mapping[str, Any]) -> str:
    if not outcome.get("parse_valid"):
        return "generation_invalid"
    if not outcome.get("policy_valid"):
        return "policy_invalid"
    if not outcome.get("reference_valid"):
        return f"reference_{outcome.get('reference_status', 'invalid')}"
    if outcome.get("killed"):
        return f"killed_{outcome.get('mutant_status', 'failure')}"
    return "survived"


def evaluate_candidate_slots(
    slots: Sequence[Mapping[str, Any]],
    golden_code: str,
    mutant_code: str,
    entry_point: str,
) -> List[Dict[str, Any]]:
    """Classify ordered generation slots without losing invalid candidates."""
    outcomes = [deepcopy(dict(slot)) for slot in slots]
    executable_codes: List[str] = []
    executable_indexes: List[int] = []
    for index, outcome in enumerate(outcomes):
        code = outcome.get("code")
        outcome.setdefault("parse_valid", bool(code))
        outcome["policy_valid"] = False
        outcome["execution_valid"] = False
        outcome["reference_valid"] = False
        outcome["killed"] = False
        if not outcome["parse_valid"] or not isinstance(code, str):
            continue
        policy = validate_function_assertion(code, entry_point)
        outcome["policy_valid"] = bool(policy.valid)
        if not policy.valid:
            outcome["policy_error"] = policy.reason
            continue
        executable_indexes.append(index)
        executable_codes.append(code)

    rows = classify_assertions(executable_codes, golden_code, mutant_code)
    if len(rows) != len(executable_indexes):
        raise RuntimeError("Execution harness returned an incomplete candidate classification")
    for index, row in zip(executable_indexes, rows):
        outcome = outcomes[index]
        golden = row.get("golden") or {}
        mutant = row.get("mutant") or {}
        outcome["execution_valid"] = str(golden.get("status", "unknown")) in {
            "pass", "assertion_error"
        }
        outcome["reference_valid"] = bool(row.get("valid"))
        outcome["killed"] = bool(row.get("killed"))
        outcome["reference_status"] = str(golden.get("status", "unknown"))
        outcome["mutant_status"] = str(mutant.get("status", "unknown"))
        if not outcome["reference_valid"]:
            outcome["reference_error"] = str(golden.get("error", ""))[:240]
    for outcome in outcomes:
        outcome["failure_mode"] = _outcome_mode(outcome)
    return outcomes


def candidate_diversity(
    outcomes: Sequence[Mapping[str, Any]], entry_point: str
) -> Dict[str, Any]:
    """Measure exact, structural, input-shape, and outcome diversity."""
    parsed = [item for item in outcomes if item.get("parse_valid") and item.get("code")]
    valid = [item for item in parsed if item.get("reference_valid")]

    def unique_count(items: Sequence[Mapping[str, Any]], key_fn) -> int:
        return len({key_fn(str(item["code"])) for item in items})

    exact = unique_count(parsed, _normalise_code)
    ast_unique = unique_count(parsed, ast_shape_key)
    shapes = unique_count(parsed, lambda code: input_shape_key(code, entry_point))
    outcome_modes = len({str(item.get("failure_mode", "unknown")) for item in outcomes})
    denominator = max(len(parsed), 1)
    return {
        "parsed_candidates": len(parsed),
        "reference_valid_candidates": len(valid),
        "exact_unique": exact,
        "exact_unique_ratio": round(exact / denominator, 6),
        "ast_shape_unique": ast_unique,
        "ast_shape_unique_ratio": round(ast_unique / denominator, 6),
        "input_shape_unique": shapes,
        "input_shape_unique_ratio": round(shapes / denominator, 6),
        "outcome_mode_unique": outcome_modes,
    }


def function_result(
    record_id: str,
    bug_family: str,
    entry_point: str,
    outcomes: Sequence[Mapping[str, Any]],
    source_name: str = "unknown",
    project: str = "unknown",
    prompt_budget_failure: bool = False,
    prompt_budget_failure_reason: Optional[str] = None,
) -> Dict[str, Any]:
    """Build the canonical per-function research result.

    ``prompt_budget_failure`` marks a record the generator could not prompt at
    all because its required sections exceed the declared mode budget.  Such a
    record still occupies its denominator slot; recording it separately keeps a
    budget defect distinguishable from a model that generated nothing useful.
    """
    ordered = sorted((deepcopy(dict(item)) for item in outcomes), key=lambda item: item["rank"])
    parsed = sum(bool(item.get("parse_valid")) for item in ordered)
    execution_valid = sum(bool(item.get("execution_valid")) for item in ordered)
    valid = sum(bool(item.get("reference_valid")) for item in ordered)
    killed_candidates = sum(bool(item.get("killed")) for item in ordered)
    generation_invalid = sum(not bool(item.get("parse_valid")) for item in ordered)
    execution_invalid = sum(
        bool(item.get("parse_valid")) and not bool(item.get("execution_valid"))
        for item in ordered
    )
    return {
        "record_id": record_id,
        "bug_family": bug_family or "unknown",
        "source_name": source_name or "unknown",
        "project": project or "unknown",
        "entry_point": entry_point,
        "requested_candidates": len(ordered),
        "parsed_candidates": parsed,
        "execution_valid_candidates": execution_valid,
        "generation_invalid_candidates": generation_invalid,
        "execution_invalid_candidates": execution_invalid,
        "invalid_candidates": len(ordered) - valid,
        "valid_candidates": valid,
        "killing_candidates": killed_candidates,
        "killed": bool(killed_candidates),
        "prompt_budget_failure": bool(prompt_budget_failure),
        "prompt_budget_failure_reason": prompt_budget_failure_reason,
        "candidate_outcomes": ordered,
        "diversity": candidate_diversity(ordered, entry_point),
    }


def _prefix_success(outcomes: Sequence[Mapping[str, Any]], k: int, field: str) -> bool:
    return any(bool(item.get(field)) for item in outcomes if int(item.get("rank", 0)) <= k)


def summarise_function_results(
    function_results: Sequence[Mapping[str, Any]],
    k_values: Sequence[int] = DEFAULT_K_VALUES,
) -> Dict[str, Any]:
    """Aggregate ordered per-function results into headline research metrics."""
    if not function_results:
        raise ValueError("At least one function result is required")
    ks = sorted({int(k) for k in k_values if int(k) > 0})
    total_functions = len(function_results)
    requested = sum(int(item.get("requested_candidates", 0)) for item in function_results)
    parsed = sum(int(item.get("parsed_candidates", 0)) for item in function_results)
    execution_valid = sum(
        int(item.get("execution_valid_candidates", 0)) for item in function_results
    )
    valid = sum(int(item.get("valid_candidates", 0)) for item in function_results)
    killing = sum(int(item.get("killing_candidates", 0)) for item in function_results)
    generation_invalid = sum(
        int(item.get("generation_invalid_candidates", 0)) for item in function_results
    )
    execution_invalid = sum(
        int(item.get("execution_invalid_candidates", 0)) for item in function_results
    )
    killed_functions = sum(bool(item.get("killed")) for item in function_results)
    prompt_budget_failures = sum(
        bool(item.get("prompt_budget_failure")) for item in function_results
    )

    kill_at_k: Dict[str, Any] = {}
    pass_at_k: Dict[str, Any] = {}
    for k in ks:
        killed = sum(
            _prefix_success(item.get("candidate_outcomes", []), k, "killed")
            for item in function_results
        )
        passed = sum(
            _prefix_success(item.get("candidate_outcomes", []), k, "reference_valid")
            for item in function_results
        )
        kill_at_k[str(k)] = {
            "functions": killed,
            "rate": round(killed / total_functions, 6),
            "wilson_95": wilson_interval(killed, total_functions),
        }
        pass_at_k[str(k)] = {
            "functions": passed,
            "rate": round(passed / total_functions, 6),
            "wilson_95": wilson_interval(passed, total_functions),
            "definition": "at least one reference-valid generated test in the first k raw slots",
        }

    family_buckets: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    source_buckets: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for item in function_results:
        family_buckets[str(item.get("bug_family", "unknown"))].append(item)
        source_buckets[str(item.get("source_name", "unknown"))].append(item)

    def compact_group(items: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        count = len(items)
        killed = sum(bool(item.get("killed")) for item in items)
        group_requested = sum(int(item.get("requested_candidates", 0)) for item in items)
        group_parsed = sum(int(item.get("parsed_candidates", 0)) for item in items)
        group_executed = sum(int(item.get("execution_valid_candidates", 0)) for item in items)
        group_valid = sum(int(item.get("valid_candidates", 0)) for item in items)
        group_killing = sum(int(item.get("killing_candidates", 0)) for item in items)
        group_kill_at_k = {}
        for k in ks:
            successes = sum(
                _prefix_success(item.get("candidate_outcomes", []), k, "killed")
                for item in items
            )
            group_kill_at_k[str(k)] = {
                "functions": successes,
                "rate": round(successes / max(count, 1), 6),
                "wilson_95": wilson_interval(successes, count),
            }
        return {
            "functions": count,
            "killed_functions": killed,
            "function_kill_rate": round(killed / max(count, 1), 6),
            "function_kill_rate_wilson_95": wilson_interval(killed, count),
            "requested_candidates": group_requested,
            "parsed_candidates": group_parsed,
            "execution_valid_candidates": group_executed,
            "reference_valid_candidates": group_valid,
            "killing_candidates": group_killing,
            "parse_valid_rate": round(group_parsed / max(group_requested, 1), 6),
            "execution_valid_rate": round(group_executed / max(group_requested, 1), 6),
            "reference_valid_rate": round(group_valid / max(group_requested, 1), 6),
            "bug_killing_candidate_rate": round(group_killing / max(group_requested, 1), 6),
            "kill_at_k": group_kill_at_k,
        }

    diversity_fields = (
        "exact_unique_ratio",
        "ast_shape_unique_ratio",
        "input_shape_unique_ratio",
        "outcome_mode_unique",
    )
    diversity_mean = {
        field: round(
            statistics.fmean(float(item.get("diversity", {}).get(field, 0.0)) for item in function_results),
            6,
        )
        for field in diversity_fields
    }
    redundant_killing_candidates = max(0, killing - killed_functions)
    source_metrics = {
        key: compact_group(items) for key, items in sorted(source_buckets.items())
    }
    macro_sources = [value for value in source_metrics.values() if value["functions"]]
    equal_weight_source_macro = {
        "source_count": len(macro_sources),
        "function_kill_rate": round(statistics.fmean(
            item["function_kill_rate"] for item in macro_sources
        ), 6) if macro_sources else 0.0,
        "reference_valid_rate": round(statistics.fmean(
            item["reference_valid_rate"] for item in macro_sources
        ), 6) if macro_sources else 0.0,
        "execution_valid_rate": round(statistics.fmean(
            item["execution_valid_rate"] for item in macro_sources
        ), 6) if macro_sources else 0.0,
        "kill_at_k": {
            str(k): round(statistics.fmean(
                item["kill_at_k"][str(k)]["rate"] for item in macro_sources
            ), 6) if macro_sources else 0.0
            for k in ks
        },
        "definition": "equal-weight mean across function benchmark sources",
    }
    return {
        "research_metrics_schema_version": RESEARCH_METRICS_SCHEMA_VERSION,
        "function_validation_records": total_functions,
        "function_validation_killed": killed_functions,
        "prompt_budget_failed_functions": prompt_budget_failures,
        "prompt_budget_failure_rate": round(
            prompt_budget_failures / total_functions, 6
        ),
        "function_kill_rate": round(killed_functions / total_functions, 6),
        "function_kill_rate_wilson_95": wilson_interval(killed_functions, total_functions),
        "requested_candidates": requested,
        "parsed_candidates": parsed,
        "execution_valid_candidates": execution_valid,
        "reference_valid_candidates": valid,
        "generated_candidates": valid,
        "mutation_killing_candidates": killing,
        "generation_invalid_candidates": generation_invalid,
        "execution_invalid_candidates": execution_invalid,
        "invalid_candidates": requested - valid,
        "candidate_kill_rate": round(killing / max(valid, 1), 6),
        "end_to_end_candidate_kill_rate": round(killing / max(requested, 1), 6),
        "parse_success_rate": round(parsed / max(requested, 1), 6),
        "execution_valid_rate": round(execution_valid / max(requested, 1), 6),
        "reference_valid_rate": round(valid / max(requested, 1), 6),
        "kill_at_k": kill_at_k,
        "pass_at_k": pass_at_k,
        "candidate_redundancy": {
            "redundant_killing_candidates": redundant_killing_candidates,
            "redundancy_ratio": round(redundant_killing_candidates / max(killing, 1), 6),
            "definition": "killing candidates beyond the first kill for each killed function",
        },
        "diversity_mean": diversity_mean,
        "bug_family_metrics": {
            key: compact_group(items) for key, items in sorted(family_buckets.items())
        },
        "source_metrics": source_metrics,
        "equal_weight_source_macro": equal_weight_source_macro,
    }


def _t_critical_95(degrees_of_freedom: int) -> float:
    # Two-sided 95% Student-t critical values. Normal approximation above 30.
    values = {
        1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
        6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
        11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
        16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
        21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060,
        26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
    }
    return values.get(max(1, degrees_of_freedom), 1.96)


def aggregate_seed_results(results: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Aggregate complete, scope-compatible seed results with mean/std/CI."""
    if len(results) < 2:
        raise ValueError("At least two complete seeds are required for seed aggregation")
    identity_fields = (
        "evaluation_split",
        "evaluation_scope_sha256",
        "tests_per_function",
        "model_artifact_sha256",
        "evaluation_profile_sha256",
    )
    reference = results[0]
    for result in results[1:]:
        for field in identity_fields:
            if reference.get(field) != result.get(field):
                raise ValueError(f"Seed results are not comparable: {field} differs")
    rates = [float(result["function_kill_rate"]) for result in results]
    mean = statistics.fmean(rates)
    standard_deviation = statistics.stdev(rates)
    standard_error = standard_deviation / math.sqrt(len(rates))
    margin = _t_critical_95(len(rates) - 1) * standard_error
    return {
        "seed_count": len(results),
        "seeds": [int(result["seed"]) for result in results],
        "function_kill_rate_mean": round(mean, 6),
        "function_kill_rate_standard_deviation": round(standard_deviation, 6),
        "function_kill_rate_standard_error": round(standard_error, 6),
        "function_kill_rate_seed_mean_t_95": [
            round(max(0.0, mean - margin), 6),
            round(min(1.0, mean + margin), 6),
        ],
        "function_kill_rate_min": round(min(rates), 6),
        "function_kill_rate_max": round(max(rates), 6),
        "identity": {field: reference.get(field) for field in identity_fields},
    }


def compare_policy_results(
    reference: Mapping[str, Any], evaluation: Mapping[str, Any]
) -> Dict[str, Any]:
    """Explain unique coverage, regressions, redundancy, and diversity changes."""
    reference_by_id = {
        str(item["record_id"]): item for item in reference.get("function_results", [])
    }
    evaluation_by_id = {
        str(item["record_id"]): item for item in evaluation.get("function_results", [])
    }
    if set(reference_by_id) != set(evaluation_by_id):
        raise ValueError("Policy comparison requires the exact same function IDs")
    ids = sorted(reference_by_id)
    newly_killed = [
        item_id for item_id in ids
        if not reference_by_id[item_id].get("killed") and evaluation_by_id[item_id].get("killed")
    ]
    regressions = [
        item_id for item_id in ids
        if reference_by_id[item_id].get("killed") and not evaluation_by_id[item_id].get("killed")
    ]
    retained = [
        item_id for item_id in ids
        if reference_by_id[item_id].get("killed") and evaluation_by_id[item_id].get("killed")
    ]

    def policy_stats(result: Mapping[str, Any]) -> Dict[str, float]:
        items = result.get("function_results", [])
        killing = sum(int(item.get("killing_candidates", 0)) for item in items)
        killed = sum(bool(item.get("killed")) for item in items)
        return {
            "unique_functions_killed": killed,
            "killing_candidates": killing,
            "redundant_killing_candidates": max(0, killing - killed),
            "redundancy_ratio": round(max(0, killing - killed) / max(killing, 1), 6),
            "mean_ast_diversity": round(statistics.fmean(
                float(item.get("diversity", {}).get("ast_shape_unique_ratio", 0.0))
                for item in items
            ), 6) if items else 0.0,
            "mean_input_shape_diversity": round(statistics.fmean(
                float(item.get("diversity", {}).get("input_shape_unique_ratio", 0.0))
                for item in items
            ), 6) if items else 0.0,
        }

    ref_stats = policy_stats(reference)
    eval_stats = policy_stats(evaluation)
    return {
        "functions_compared": len(ids),
        "newly_killed": newly_killed,
        "regressions": regressions,
        "retained_kills": retained,
        "net_unique_function_gain": len(newly_killed) - len(regressions),
        "reference": ref_stats,
        "evaluation": eval_stats,
        "deltas": {
            key: round(eval_stats[key] - ref_stats[key], 6)
            for key in (
                "unique_functions_killed",
                "killing_candidates",
                "redundant_killing_candidates",
                "redundancy_ratio",
                "mean_ast_diversity",
                "mean_input_shape_diversity",
            )
        },
    }


def evaluation_profile_sha256(profile: Mapping[str, Any]) -> str:
    """Return a stable identity for an ablation/evaluation configuration."""
    payload = json.dumps(dict(profile), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


FAILURE_TAXONOMY_SCHEMA_VERSION = 1

#: Every category a candidate outcome can be assigned.  ``killed`` is the only
#: success label; it is kept in the same vocabulary so per-slice totals always
#: sum to the requested-candidate count.
FAILURE_TAXONOMY_CATEGORIES: Tuple[str, ...] = (
    "killed",
    "syntax_invalid",
    "wrong_target_api",
    "undefined_symbol",
    "environment_failure",
    "fixture_missing",
    "incorrect_exception_expectation",
    "wrong_expected_value",
    "reference_invalid",
    "timeout",
    "repository_context_hallucination",
    "boundary_miss",
    "off_by_one_miss",
    "logical_condition_miss",
    "indexing_miss",
    "passes_both",
    "not_generated",
)

_SURVIVED_FAMILY_CATEGORY = {
    "boundary": "boundary_miss",
    "off_by_one": "off_by_one_miss",
    "logical": "logical_condition_miss",
    "boolean": "logical_condition_miss",
    "comparison": "logical_condition_miss",
    "index": "indexing_miss",
    "indexing": "indexing_miss",
}


def classify_candidate_failure(
    outcome: Mapping[str, Any],
    bug_family: str = "unknown",
    execution_mode: str = "function_assertion",
) -> str:
    """Assign one ordered-candidate outcome to a single failure category.

    Classification uses only post-hoc oracle evidence that already exists in the
    recorded outcome.  It never re-executes anything and never consults the
    reference implementation directly, so it is safe to run over any legitimately
    produced evaluation artifact.
    """
    if outcome.get("prompt_budget_failure"):
        return "not_generated"
    if not outcome.get("parse_valid"):
        return "syntax_invalid"
    if not outcome.get("policy_valid"):
        return "wrong_target_api"

    reference_status = str(outcome.get("reference_status", "")).strip()
    if not reference_status:
        return "not_generated"

    error = str(outcome.get("reference_error", "") or "")
    exception = error.split(":", 1)[0].strip()
    repository = str(execution_mode or "").startswith("repository")

    if reference_status == "timeout":
        return "timeout"
    if reference_status in {"system_exit", "keyboard_interrupt", "worker_error"}:
        return "environment_failure"
    if reference_status == "source_policy_error":
        return "environment_failure"

    if reference_status == "error":
        if exception in {"ImportError", "ModuleNotFoundError"}:
            return (
                "repository_context_hallucination" if repository
                else "environment_failure"
            )
        if exception == "NameError":
            return (
                "repository_context_hallucination" if repository
                else "undefined_symbol"
            )
        if exception in {"AttributeError", "TypeError"}:
            return "wrong_target_api"
        if "fixture" in error.lower():
            return "fixture_missing"
        if exception in {"Failed", "DidNotRaise"} or "raises" in error.lower():
            return "incorrect_exception_expectation"
        return "reference_invalid"

    if reference_status == "assertion_error":
        # The candidate is executable but asserts behaviour the reference does
        # not exhibit, so its expected value is wrong rather than its syntax.
        return "wrong_expected_value"

    if not outcome.get("reference_valid"):
        return "reference_invalid"
    if outcome.get("killed"):
        return "killed"

    family = sanitise_family_name(bug_family) or "unknown"
    return _SURVIVED_FAMILY_CATEGORY.get(family, "passes_both")


def summarise_failure_taxonomy(
    function_results: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Aggregate §43 failure categories overall and per dataset/mutation family.

    Records that carry no ``candidate_outcomes`` (older artifact schemas) are
    counted separately instead of being silently treated as zero failures.
    """
    overall: Counter[str] = Counter()
    by_source: Dict[str, Counter[str]] = defaultdict(Counter)
    by_family: Dict[str, Counter[str]] = defaultdict(Counter)
    classified_functions = 0
    unclassifiable_functions = 0
    classified_candidates = 0

    for item in function_results:
        outcomes = item.get("candidate_outcomes")
        if not outcomes:
            unclassifiable_functions += 1
            continue
        classified_functions += 1
        family = str(item.get("bug_family", "unknown") or "unknown")
        source = str(item.get("source_name", "unknown") or "unknown")
        execution_mode = str(item.get("execution_mode", "function_assertion"))
        for outcome in outcomes:
            merged = dict(outcome)
            merged.setdefault(
                "prompt_budget_failure", bool(item.get("prompt_budget_failure"))
            )
            category = classify_candidate_failure(merged, family, execution_mode)
            overall[category] += 1
            by_source[source][category] += 1
            by_family[family][category] += 1
            classified_candidates += 1

    def _table(counter: Counter[str]) -> Dict[str, Any]:
        total = sum(counter.values())
        return {
            "candidates": total,
            "counts": {
                category: counter.get(category, 0)
                for category in FAILURE_TAXONOMY_CATEGORIES
                if counter.get(category, 0)
            },
            "rates": {
                category: round(counter[category] / total, 6)
                for category in FAILURE_TAXONOMY_CATEGORIES
                if counter.get(category, 0)
            } if total else {},
        }

    return {
        "failure_taxonomy_schema_version": FAILURE_TAXONOMY_SCHEMA_VERSION,
        "classified_functions": classified_functions,
        "unclassifiable_functions": unclassifiable_functions,
        "classified_candidates": classified_candidates,
        "overall": _table(overall),
        "by_source": {key: _table(value) for key, value in sorted(by_source.items())},
        "by_mutation_family": {
            key: _table(value) for key, value in sorted(by_family.items())
        },
    }


def sanitise_family_name(value: Optional[str]) -> Optional[str]:
    if value is None or not value.strip():
        return None
    family = re.sub(r"[\s-]+", "_", value.strip().lower()).strip("_")
    if not re.fullmatch(r"[a-z0-9_+]+", family):
        raise ValueError(f"Invalid mutation-family name: {value!r}")
    return family
