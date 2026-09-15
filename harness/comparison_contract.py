"""Comparison identity for future arms. Applies to nothing already measured.

Locked validation failed a frozen integrity criterion for a reason that had
nothing to do with the data. The criterion required both arms to share a
``run_contract_sha256``. A later change put ``locked_validation_binding`` -
which carries ``adapter_sha256`` - inside the run contract, and an arm that
loads an adapter can never share a contract hash with one that does not. The
two contracts differed in that single field and in nothing else.

The mistake was not the hash. It was putting two different kinds of fact in one
bucket and then demanding they match:

*shared* facts
    What was measured and how: corpus, split, panel scope, protocol, parse
    mode, candidate count, sampling, budgets, evaluator, candidate policy,
    timeout policy, prompt builder. If two arms differ here, their numbers are
    not comparable and the comparison is void.

*per-arm* facts
    Which weights answered: model name and revision, adapter path, adapter
    hash, provenance, training run. These are *supposed* to differ - that
    difference is the experiment. Requiring them to match is requiring the
    arms to be the same arm.

So this contract hashes the two groups separately. ``shared_sha256`` must be
identical across arms; each arm's ``arm_sha256`` is expected to differ, and a
comparison where two arms share an ``arm_sha256`` is the suspicious one,
because it means the same weights were measured twice under two names.

This module is version 2 and deliberately separate. It does not read, re-score,
re-hash or reinterpret any locked-validation or development artifact. Those
were produced under version 1 and stay exactly as they are; the defect they
record is history, and history that gets edited once it becomes inconvenient is
not evidence.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Iterable, List

#: Separately versioned. v1 is the implicit whole-run-contract comparison used
#: by the development and locked-validation measurements; it is not re-run.
COMPARISON_CONTRACT_VERSION = "oneiros_comparison_contract_v2"

#: Facts that decide what was measured and how it was scored. Two arms that
#: differ in any of these are not comparable at all.
SHARED_FIELDS = (
    "corpus_version",
    "corpus_records_sha256",
    "evaluation_split",
    "evaluation_scope_sha256",
    "protocol_name",
    "protocol_sha256",
    "candidate_parse_mode",
    "retain_raw_output",
    "candidates_per_function",
    "temperature",
    "top_p",
    "generation_seed",
    "generation_completion_token_limit",
    "prompt_token_limit",
    "max_sequence_tokens",
    "prompt_schema_version",
    "prompt_information_variant",
    "output_instruction_variant",
    "evaluator_source_canonical_sha256",
    "candidate_policy_source_canonical_sha256",
    "timeout_policy_source_canonical_sha256",
    "prompt_builder_source_canonical_sha256",
    "feedback_rounds",
    "diversity_mode",
)

#: Facts about which weights answered. Expected to differ between arms.
#:
#: ``arm_name`` is deliberately absent. It is a label, not a weight, and
#: including it made the arm hash unique per name - so the same checkpoint
#: entered twice under two names produced two different identities and the
#: duplicate-weights check could never fire. The hash answers "which weights",
#: and a label is not an answer to that.
ARM_FIELDS = (
    "base_model_name",
    "base_model_revision",
    "adapter_provenance",
    "adapter_path",
    "adapter_sha256",
    "adapter_source_run",
    "adapter_checkpoint_step",
)

#: A base arm loads no adapter. The evaluator already has a convention for
#: this - sha256("<model>@<revision>") - and reusing it keeps "which weights"
#: answerable for every arm rather than null for one of them.
BASE_MODEL_PROVENANCE = "immutable_base_model_no_adapter"
ADAPTER_PROVENANCES = frozenset({
    BASE_MODEL_PROVENANCE,
    "trained_in_this_run",
    "external_evaluation_adapter",
})


def base_model_adapter_identity(model_name: str, model_revision: str) -> str:
    """The digest a base arm uses in place of an adapter hash."""
    return hashlib.sha256(
        f"{model_name}@{model_revision}".encode("utf-8")).hexdigest()


def _digest(payload: Dict[str, Any], fields: Iterable[str]) -> str:
    return hashlib.sha256(json.dumps(
        {name: payload.get(name) for name in sorted(fields)},
        sort_keys=True, separators=(",", ":"), default=str,
    ).encode("utf-8")).hexdigest()


def shared_sha256(shared: Dict[str, Any]) -> str:
    """Hash of what was measured and how. Must match across arms."""
    return _digest(shared, SHARED_FIELDS)


def arm_sha256(arm: Dict[str, Any]) -> str:
    """Hash of which weights answered. Expected to differ across arms."""
    return _digest(arm, ARM_FIELDS)


def missing_fields(shared: Dict[str, Any], arms: Iterable[Dict[str, Any]]) -> List[str]:
    """Absent or empty required fields, named so a refusal is actionable."""
    found: List[str] = []
    for name in SHARED_FIELDS:
        if shared.get(name) is None:
            found.append(f"shared.{name}")
    for arm in arms:
        label = arm.get("arm_name") or "<unnamed arm>"
        for name in ARM_FIELDS:
            if arm.get(name) is None and name not in {
                "adapter_source_run", "adapter_checkpoint_step", "adapter_path",
            }:
                found.append(f"{label}.{name}")
    return found


def comparison_problems(
    shared: Dict[str, Any], arms: List[Dict[str, Any]],
) -> List[str]:
    """Everything that would make this comparison invalid.

    Deliberately NOT in this list: "the arms' adapter hashes differ". That is
    the experiment, not a defect, and demanding otherwise is what made a frozen
    integrity criterion unsatisfiable once the adapter hash moved into the
    shared bucket.
    """
    problems: List[str] = []
    problems.extend(missing_fields(shared, arms))

    if len(arms) < 2:
        problems.append("a comparison needs at least two arms")
        return problems

    names = [arm.get("arm_name") for arm in arms]
    if len(set(names)) != len(names):
        problems.append(f"arm names are not unique: {names}")

    # Every arm must agree about what was measured.
    reference = shared_sha256(shared)
    for arm in arms:
        recorded = arm.get("shared_sha256")
        if recorded is not None and recorded != reference:
            problems.append(
                f"{arm.get('arm_name')} was measured under a different shared "
                f"contract: {recorded} vs {reference}")

    # Two arms with the same arm hash are the same weights under two names.
    by_hash: Dict[str, List[str]] = {}
    for arm in arms:
        by_hash.setdefault(arm_sha256(arm), []).append(arm.get("arm_name"))
    for digest, group in by_hash.items():
        if len(group) > 1:
            problems.append(
                f"arms {group} share an arm identity {digest[:16]}...; they are "
                "the same weights measured twice, not a comparison")

    for arm in arms:
        provenance = arm.get("adapter_provenance")
        if provenance not in ADAPTER_PROVENANCES:
            problems.append(
                f"{arm.get('arm_name')} has unknown adapter_provenance "
                f"{provenance!r}; expected one of {sorted(ADAPTER_PROVENANCES)}")
        if provenance == BASE_MODEL_PROVENANCE:
            expected = base_model_adapter_identity(
                arm.get("base_model_name") or "", arm.get("base_model_revision") or "")
            if arm.get("adapter_sha256") != expected:
                problems.append(
                    f"{arm.get('arm_name')} is a base arm but its adapter_sha256 is "
                    "not the base-model identity digest")
    return problems


def build(shared: Dict[str, Any], arms: List[Dict[str, Any]]) -> Dict[str, Any]:
    """The comparison contract for a future measurement.

    Records both hash families explicitly, states which one must match, and
    says in the artifact itself that a differing arm hash is expected - so the
    next person reading it does not have to rediscover why.
    """
    reference = shared_sha256(shared)
    built_arms = []
    for arm in arms:
        built_arms.append(dict(
            arm,
            arm_sha256=arm_sha256(arm),
            shared_sha256=reference,
        ))
    problems = comparison_problems(shared, built_arms)
    return {
        "comparison_contract_version": COMPARISON_CONTRACT_VERSION,
        "applies_to": "future measurements only",
        "does_not_re_score": [
            "results/v4_2_frozen_development_evaluation_receipt.json",
            "results/v4_2_development_selection_receipt.json",
            "results/v4_2_locked_validation_preflight.json",
            "results/v4_2_locked_validation_result_receipt.json",
        ],
        "shared_sha256": reference,
        "shared": {name: shared.get(name) for name in SHARED_FIELDS},
        "arms": built_arms,
        "arm_identities_are_expected_to_differ": True,
        "rule": (
            "shared_sha256 must be identical across arms; arm_sha256 must differ. "
            "An arm that loads an adapter can never share an arm identity with a "
            "base arm, and requiring it to is what made a frozen locked-validation "
            "integrity criterion unsatisfiable. That criterion's failure is "
            "preserved as recorded history and is not re-scored here."
        ),
        "problems": problems,
        "valid": not problems,
    }
