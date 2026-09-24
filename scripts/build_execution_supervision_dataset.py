"""Build the train-only execution-supervision corpus and matched A/B arms.

The builder reads only the hash-verified train development view plus the
already-audited O1 candidate artifact.  It freezes semantic lineages before
runtime execution; later failures reduce a split and are never backfilled from
pilot or confirmation.  The confirmation partition is emitted as identifiers
only and remains unopened until the pilot decision rule permits it.

The first causal pilot deliberately does not train on long traces.  At 128
declared positions Arm A and Arm B have byte-identical verified assertions;
only the prompt changes from canonical test generation to focused
call-to-oracle prediction.  Full trace targets are nevertheless built and
token-audited as a separately reported future intervention.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict, deque
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import immutable_revision_for
from engine.test_generation_prompt import (
    build_unified_user_prompt,
    format_chat_prompt,
    sanitize_behavioral_specification,
)
from harness.candidate_policy import validate_function_assertion
from harness.corpus import sha256_file, write_json
from harness.corpus_view import load_development_split
from harness.execution_supervision import (
    ExecutionSupervisionRejected,
    build_execution_supervision_completion,
    build_output_prediction_completion,
    build_output_prediction_prompt,
    collect_execution_evidence,
    execution_supervision_call_policy_error,
    format_output_prediction_chat_prompt,
)
from harness.execution_supervision_sidecar import (
    ARM_SIZE,
    REPLACEMENT_COUNT,
    SCHEMA,
    assemble_controlled_arms,
    freeze_lineage_split,
    sha256_text,
    stable_rank,
)
from harness.parallel_execution import progress_map
from harness.safe_execution import classify_assertions
from harness.training_data import extract_dataset_assertions
from scripts.measure_prompt_giveaway_effect import _states_a_killing_value
from utils.dataset_identity import dataset_name_from_source


MODEL_NAME = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
MODEL_REVISION = "2e1fd397ee46e1388853d2af2c993145b0f1098a"
PROMPT_TOKEN_LIMIT = 1024
OUTPUT_COMPLETION_TOKEN_LIMIT = 128
TRACE_COMPLETION_TOKEN_LIMIT = 1024
MAX_SEQUENCE_TOKENS = 3072
TIMEOUT_SECONDS = 1.0


def _json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8")


def _source_dataset(record: dict[str, Any]) -> str:
    return dataset_name_from_source(record.get("source") or {})


def _execution_mode(record: dict[str, Any]) -> str:
    return str((record.get("quality") or {}).get(
        "execution_mode", "function_assertion"
    ))


def _bug_family(record: dict[str, Any], fallback: str = "unknown") -> str:
    provenance = record.get("provenance") or {}
    return str(provenance.get("mutation_type") or provenance.get("category")
               or fallback or "unknown")


def _complexity_tier(record: dict[str, Any], fallback: str = "unknown") -> str:
    quality = record.get("quality") or {}
    if quality.get("complexity_tier"):
        return str(quality["complexity_tier"])
    lines = [line for line in str(
        record.get("prompt_code_under_test") or record.get("code_under_test") or ""
    ).splitlines() if line.strip()]
    return "simple" if len(lines) <= 6 else "moderate" if len(lines) <= 14 else "complex"


def _canonical_prompt(record: dict[str, Any]) -> str:
    mode = _execution_mode(record)
    return build_unified_user_prompt(
        code_under_test=str(record.get("prompt_code_under_test")
                            or record.get("code_under_test") or ""),
        execution_mode=mode,
        specification=str(record.get("specification") or ""),
        support_context=str(record.get("support_context") or ""),
        target_symbols=record.get("target_symbols") or [],
        entry_point=str(record.get("entry_point") or ""),
        information_variant="full",
        output_instruction_variant="self_contained",
    )


def _combined_source(record: dict[str, Any], field: str) -> str:
    support = str(record.get("support_context") or "").strip()
    code = str(record.get(field) or "").strip()
    return (support + "\n" + code).strip() if support else code


def _giveaway_job(job: dict[str, Any]) -> dict[str, Any]:
    verdict = _states_a_killing_value(job["record"], TIMEOUT_SECONDS)
    return {"record_id": job["record_id"], "verdict": verdict}


def _load_verified_inputs(
    candidates_path: Path, oracle_manifest_path: Path, corpus_dir: Path,
    workers: int,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, Any]]:
    oracle_manifest = json.loads(oracle_manifest_path.read_text(encoding="utf-8"))
    if oracle_manifest.get("evaluation_split") != "train":
        raise ValueError("oracle candidate artifact is not train-only")
    if oracle_manifest.get("sealed_final_test_accessed") is not False:
        raise ValueError("oracle manifest does not deny sealed-test access")
    if sha256_file(candidates_path) != oracle_manifest.get("candidates_sha256"):
        raise ValueError("oracle candidates do not match their manifest")
    records = {
        str(record["id"]): record
        for record in load_development_split(corpus_dir, "train", include_excluded=True)
    }
    candidates = json.loads(candidates_path.read_text(encoding="utf-8"))

    literal_safe: list[dict[str, Any]] = []
    for row in candidates:
        replay = row.get("call_replay") or {}
        call = str(replay.get("call") or "")
        record = records.get(str(row.get("record_id") or ""))
        if record is None or not call:
            continue
        entry = str(record.get("entry_point") or "")
        if execution_supervision_call_policy_error(call, entry):
            continue
        if str(row.get("source_dataset")) not in {
            "mbpp", "humaneval", "manual_curated_examples",
        }:
            continue
        literal_safe.append(row)

    # A HumanEval record is excluded when any worked example in its own prompt
    # already states a killing value.  Cache per record; mutation siblings are
    # never repeatedly audited.
    humaneval_ids = sorted({
        str(row["record_id"]) for row in literal_safe
        if row.get("source_dataset") == "humaneval"
    })
    giveaway_results = progress_map(
        _giveaway_job,
        [{"record_id": record_id, "record": records[record_id]}
         for record_id in humaneval_ids],
        "HumanEval giveaway audit",
        every=100,
        workers=workers,
    )
    giveaway = {
        result["record_id"] for result in giveaway_results
        if result.get("verdict") is True
    }
    unverifiable = {
        result["record_id"] for result in giveaway_results
        if result.get("verdict") is None
    }
    eligible = [
        row for row in literal_safe
        if str(row["record_id"]) not in giveaway
        and str(row["record_id"]) not in unverifiable
    ]
    eligible_lineages = {str(row["function_lineage"]) for row in eligible}
    if len(eligible_lineages) != 585:
        raise ValueError(
            f"post-giveaway eligible universe drifted to {len(eligible_lineages)} "
            "lineages; expected the audited 585"
        )
    audit = {
        "candidate_rows": len(candidates),
        "literal_safe_call_rows": len(literal_safe),
        "humaneval_records_audited": len(humaneval_ids),
        "humaneval_giveaway_records_excluded": len(giveaway),
        "humaneval_unverifiable_records_excluded": len(unverifiable),
        "eligible_rows": len(eligible),
        "eligible_lineages": len(eligible_lineages),
    }
    return eligible, records, audit


def _intended_value(evidence) -> Any:
    return ast.literal_eval(str(evidence.actual["literal"]))


def _materialize_row(
    row: dict[str, Any], record: dict[str, Any], tokenizer,
) -> tuple[dict[str, Any] | None, str | None]:
    replay = row.get("call_replay") or {}
    call = str(replay.get("call") or "")
    entry = str(record.get("entry_point") or "")
    specification = sanitize_behavioral_specification(
        str(record.get("specification") or "")
    ).strip()
    if not specification:
        return None, "empty_sanitized_specification"
    shown_source = _combined_source(record, "code_under_test")
    intended_source = _combined_source(record, "reference_code")
    try:
        focused_prompt = build_output_prediction_prompt(
            specification=specification,
            shown_code=shown_source,
            entry_point=entry,
            call_expression=call,
        )
        shown = collect_execution_evidence(
            shown_code=shown_source,
            entry_point=entry,
            call_expression=call,
            timeout_seconds=TIMEOUT_SECONDS,
        )
        intended = collect_execution_evidence(
            shown_code=intended_source,
            entry_point=entry,
            call_expression=call,
            timeout_seconds=TIMEOUT_SECONDS,
        )
        completion, evidence = build_output_prediction_completion(
            call_expression=call,
            shown_evidence=shown,
            intended_evidence=intended,
        )
        trace_completion = build_execution_supervision_completion(
            shown, intended_output=_intended_value(intended)
        )
    except ExecutionSupervisionRejected as exc:
        return None, exc.reason

    policy = validate_function_assertion(completion, entry)
    if not policy.valid:
        return None, "completion_policy_invalid"
    scored = classify_assertions(
        [completion], intended_source, shown_source, TIMEOUT_SECONDS
    )
    outcome = scored[0] if scored else {}
    if not outcome.get("valid"):
        return None, "completion_reference_invalid"
    differs = bool(evidence["differs"])
    if differs != bool(outcome.get("killed")):
        return None, "execution_evidence_kill_mismatch"

    rendered_focused = format_output_prediction_chat_prompt(tokenizer, focused_prompt)
    prompt_tokens = len(tokenizer(
        rendered_focused, add_special_tokens=False
    )["input_ids"])
    output_tokens = len(tokenizer(
        completion + tokenizer.eos_token, add_special_tokens=False
    )["input_ids"])
    trace_tokens = len(tokenizer(
        trace_completion + tokenizer.eos_token, add_special_tokens=False
    )["input_ids"])
    if prompt_tokens > PROMPT_TOKEN_LIMIT:
        return None, "focused_prompt_token_budget"
    if output_tokens > OUTPUT_COMPLETION_TOKEN_LIMIT:
        return None, "output_completion_token_budget"
    # Trace overflow is reported but does not reject the output-prediction row;
    # trace supervision is a separate future arm.
    trace_eligible = trace_tokens <= TRACE_COMPLETION_TOKEN_LIMIT
    result = {
        "record_id": str(row["record_id"]),
        "function_lineage": str(row["function_lineage"]),
        "source_dataset": str(row.get("source_dataset") or "unknown"),
        "bug_family": str(row.get("bug_family") or _bug_family(record)),
        "complexity_tier": str(row.get("complexity_tier") or _complexity_tier(record)),
        "call_expression": call,
        "canonical_prompt": _canonical_prompt(record),
        "focused_prompt": focused_prompt,
        "completion": completion,
        "trace_completion": trace_completion,
        "execution_evidence": evidence,
        "trace_event_count": len(shown.trace),
        "focused_prompt_tokens": prompt_tokens,
        "completion_tokens_with_eos": output_tokens,
        "trace_completion_tokens_with_eos": trace_tokens,
        "trace_training_eligible": trace_eligible,
        "candidate_identity": {
            "rank": row.get("rank"),
            "candidate_position": row.get("candidate_position"),
            "candidate_code_sha256": row.get("candidate_code_sha256"),
        },
        "verified": True,
        "evaluation_split": "train",
    }
    result["prompt_sha256"] = sha256_text(focused_prompt)
    result["completion_sha256"] = sha256_text(completion)
    result["trace_completion_sha256"] = sha256_text(trace_completion)
    return result, None


def _select_replacements(
    rows: list[dict[str, Any]], records: dict[str, dict[str, Any]],
    train_lineages: set[str], tokenizer,
) -> tuple[list[dict[str, Any]], Counter]:
    candidates = [
        row for row in rows
        if str(row["function_lineage"]) in train_lineages
        and row.get("label") == "wrong_oracle"
        and (row.get("call_replay") or {}).get("distinguishing") is True
        and (row.get("verified_correction") or {}).get("verified") is True
    ]
    candidates.sort(key=lambda row: stable_rank(
        "replacement", row.get("source_dataset"), row.get("function_lineage"),
        row.get("record_id"), row.get("rank"), row.get("candidate_position"),
    ))
    selected: list[dict[str, Any]] = []
    rejections: Counter = Counter()
    per_lineage: Counter = Counter()
    per_source: Counter = Counter()
    per_family: Counter = Counter()
    seen_completion: set[str] = set()
    for row in candidates:
        source = str(row.get("source_dataset") or "unknown")
        lineage = str(row["function_lineage"])
        family = str(row.get("bug_family") or "unknown")
        if per_lineage[lineage] >= 2:
            continue
        # With the independent per-family 35% gate retained, 46 unique
        # non-MBPP rows survive every execution/token/duplicate filter. Freeze
        # the strongest feasible no-repeat mix (64.1/35.9%), still stricter
        # than the original 0.70 MBPP ceiling.
        if source == "mbpp" and per_source[source] >= 82:
            continue
        if source != "mbpp" and sum(
            count for key, count in per_source.items() if key != "mbpp"
        ) >= 46:
            continue
        if per_family[family] >= 44:
            continue
        materialized, reason = _materialize_row(
            row, records[str(row["record_id"])], tokenizer
        )
        if materialized is None:
            rejections[str(reason)] += 1
            continue
        if materialized["completion_sha256"] in seen_completion:
            rejections["duplicate_completion"] += 1
            continue
        selected.append(materialized)
        seen_completion.add(materialized["completion_sha256"])
        per_lineage[lineage] += 1
        per_source[source] += 1
        per_family[family] += 1
        if len(selected) == REPLACEMENT_COUNT:
            break
    if len(selected) != REPLACEMENT_COUNT:
        raise ValueError(
            f"only {len(selected)} of {REPLACEMENT_COUNT} replacement rows survived; "
            f"sources={dict(per_source)}, rejections={dict(rejections)}"
        )
    if per_source["mbpp"] > 82 or sum(per_source.values()) != REPLACEMENT_COUNT:
        raise AssertionError("replacement source cap failed")
    return selected, rejections


def _canonical_candidates(
    records: dict[str, dict[str, Any]], forbidden_lineages: set[str],
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for record_id, record in records.items():
        lineage = str(record.get("group_id") or record_id)
        if lineage in forbidden_lineages:
            continue
        mode = _execution_mode(record)
        test_blocks = [
            str(test.get("code") or "") for test in record.get("tests") or []
            if isinstance(test, dict)
        ]
        if mode == "function_assertion":
            completions = extract_dataset_assertions(
                test_blocks, str(record.get("entry_point") or "")
            )
        else:
            completions = []
            for block in test_blocks:
                try:
                    compile(block, "<repository-supervision>", "exec")
                except SyntaxError:
                    continue
                completions.append(block.strip())
        if not completions:
            continue
        source = _source_dataset(record)
        prompt = _canonical_prompt(record)
        for position, completion in enumerate(completions[:3]):
            if not completion or completion != completion.strip():
                continue
            candidates.append({
                "record_id": record_id,
                "function_lineage": lineage,
                "source_dataset": source,
                "bug_family": _bug_family(record),
                "complexity_tier": _complexity_tier(record),
                "canonical_prompt": prompt,
                "completion": completion,
                "execution_mode": mode,
                "position": position,
            })
    candidates.sort(key=lambda row: stable_rank(
        "shared", row["source_dataset"], row["function_lineage"],
        row["record_id"], row["position"], sha256_text(row["completion"]),
    ))
    return candidates


def _select_shared(
    pool: list[dict[str, Any]], replacement_rows: list[dict[str, Any]], tokenizer,
) -> list[dict[str, Any]]:
    target = ARM_SIZE - REPLACEMENT_COUNT
    queues: dict[str, deque] = defaultdict(deque)
    replacement_hashes = {row["completion_sha256"] for row in replacement_rows}
    seen: set[str] = set(replacement_hashes)
    for row in pool:
        digest = sha256_text(str(row["completion"]))
        if digest in seen:
            continue
        completion_tokens = len(tokenizer(
            str(row["completion"]) + tokenizer.eos_token,
            add_special_tokens=False,
        )["input_ids"])
        limit = (
            TRACE_COMPLETION_TOKEN_LIMIT
            if str(row.get("execution_mode")).startswith("repository_")
            else OUTPUT_COMPLETION_TOKEN_LIMIT
        )
        if completion_tokens > limit:
            continue
        row = dict(row)
        row["completion_tokens_with_eos"] = completion_tokens
        seen.add(digest)
        queues[str(row["source_dataset"])].append(row)
    selected: list[dict[str, Any]] = []
    source_counts = Counter(str(row["source_dataset"]) for row in replacement_rows)
    lineage_counts: Counter = Counter(str(row["function_lineage"])
                                      for row in replacement_rows)
    family_counts: Counter = Counter(str(row["bug_family"])
                                     for row in replacement_rows)
    # Four real/synthetic sources remain represented. The empirical clean
    # supply cannot meet 0.35 without duplication; 0.40 is the tightest cap
    # that fills 1,024 unique examples under the frozen holdout boundaries.
    max_source = int(ARM_SIZE * 0.40)
    max_family = int(ARM_SIZE * 0.35)
    while len(selected) < target:
        possible = [source for source, queue in queues.items() if queue
                    and source_counts[source] < max_source]
        if not possible:
            raise ValueError(
                f"canonical pool exhausted at {len(selected)}/{target}; "
                f"sources={dict(source_counts)}"
            )
        source = min(possible, key=lambda key: (source_counts[key], key))
        accepted = False
        while queues[source]:
            row = queues[source].popleft()
            lineage = str(row["function_lineage"])
            family = str(row["bug_family"])
            # Project-grouped repository rows need more than four records; the
            # source cap remains the controlling diversity bound there.
            lineage_cap = 64 if source in {"BugsInPy", "SWE-bench Verified"} else 4
            if lineage_counts[lineage] >= lineage_cap:
                continue
            if family_counts[family] >= max_family:
                continue
            selected.append(row)
            source_counts[source] += 1
            lineage_counts[lineage] += 1
            family_counts[family] += 1
            accepted = True
            break
        if not accepted and not queues[source]:
            continue
    return selected


def _materialize_pilot(
    eligible: list[dict[str, Any]], records: dict[str, dict[str, Any]],
    pilot_lineages: set[str], tokenizer,
) -> tuple[list[dict[str, Any]], Counter]:
    by_lineage: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in eligible:
        lineage = str(row["function_lineage"])
        if lineage in pilot_lineages:
            by_lineage[lineage].append(row)
    output: list[dict[str, Any]] = []
    rejections: Counter = Counter()
    for lineage in sorted(pilot_lineages):
        candidates = sorted(by_lineage.get(lineage, []), key=lambda row: stable_rank(
            "pilot", lineage, row.get("record_id"), row.get("rank"),
            row.get("candidate_position"),
        ))
        for row in candidates:
            materialized, reason = _materialize_row(
                row, records[str(row["record_id"])], tokenizer
            )
            if materialized is not None:
                output.append(materialized)
                break
            rejections[str(reason)] += 1
        # No cross-lineage backfill: a failed lineage remains absent.
    return output, rejections


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, default=ROOT / "results"
                        / "v4_2_oracle_dataset_full" / "candidates.json")
    parser.add_argument("--oracle-manifest", type=Path, default=ROOT / "results"
                        / "v4_2_oracle_dataset_full" / "manifest.json")
    parser.add_argument("--corpus", type=Path, default=ROOT / "data" / "corpus"
                        / "v4_1_research_hardened_candidate")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--manifest-output", type=Path,
        default=ROOT / "results" / "v4_3_execution_supervision_dataset_manifest.json",
        help="small tracked handoff manifest; large arm artifacts stay local",
    )
    parser.add_argument("--workers", type=int, default=8)
    arguments = parser.parse_args()

    if immutable_revision_for(MODEL_NAME) != MODEL_REVISION:
        raise ValueError("configured immutable Qwen revision drifted")
    eligible, records, input_audit = _load_verified_inputs(
        arguments.candidates, arguments.oracle_manifest, arguments.corpus,
        arguments.workers,
    )
    split = freeze_lineage_split(eligible)
    train_lineages = set(split["train_lineages"])
    pilot_lineages = set(split["pilot_development_lineages"])
    confirm_lineages = set(split["unopened_confirmation_lineages"])

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, revision=MODEL_REVISION, trust_remote_code=True
    )
    if tokenizer.eos_token is None:
        raise ValueError("tokenizer has no EOS token")

    replacements, replacement_rejections = _select_replacements(
        eligible, records, train_lineages, tokenizer
    )
    forbidden = pilot_lineages | confirm_lineages
    shared_pool = _canonical_candidates(records, forbidden)
    shared = _select_shared(shared_pool, replacements, tokenizer)
    arm_a, arm_b, arm_report = assemble_controlled_arms(shared, replacements)
    pilot, pilot_rejections = _materialize_pilot(
        eligible, records, pilot_lineages, tokenizer
    )

    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    split_path = arguments.output_dir / "lineage_split.json"
    sidecar_path = arguments.output_dir / "train.execution.json"
    pilot_path = arguments.output_dir / "pilot_development.execution.json"
    confirm_path = arguments.output_dir / "unopened_confirmation.ids.json"
    arm_a_path = arguments.output_dir / "arm_a.control.json"
    arm_b_path = arguments.output_dir / "arm_b.treatment.json"
    _json_dump(split_path, split)
    _json_dump(sidecar_path, replacements)
    _json_dump(pilot_path, pilot)
    # Confirmation labels/traces are deliberately not generated yet.
    _json_dump(confirm_path, {
        "schema_version": SCHEMA,
        "status": "unopened_identifiers_only",
        "function_lineages": sorted(confirm_lineages),
        "labels_or_execution_evidence_materialized": False,
    })
    _json_dump(arm_a_path, arm_a)
    _json_dump(arm_b_path, arm_b)

    source_files = [
        ROOT / "harness" / "execution_supervision.py",
        ROOT / "harness" / "execution_supervision_sidecar.py",
        ROOT / "engine" / "sft_trainer.py",
        ROOT / "engine" / "test_generation_prompt.py",
        Path(__file__).resolve(),
    ]
    manifest = {
        "schema_version": SCHEMA,
        "evaluation_split": "train",
        "sealed_final_test_accessed": False,
        "ablation_dev_accessed": False,
        "validation_accessed": False,
        "test_accessed": False,
        "canonical_records_json_opened": False,
        "record_source": "hash-verified development_view/train.records.json only",
        "model_name": MODEL_NAME,
        "model_revision": MODEL_REVISION,
        "token_budgets": {
            "function_prompt": PROMPT_TOKEN_LIMIT,
            "repository_prompt": 2048,
            "function_completion": OUTPUT_COMPLETION_TOKEN_LIMIT,
            "repository_completion": TRACE_COMPLETION_TOKEN_LIMIT,
            "future_trace_completion": TRACE_COMPLETION_TOKEN_LIMIT,
            "sequence": MAX_SEQUENCE_TOKENS,
            "truncation_allowed": False,
        },
        "input_audit": input_audit,
        "lineage_split_sha256": sha256_file(split_path),
        "sidecar_sha256": sha256_file(sidecar_path),
        "pilot_development_sha256": sha256_file(pilot_path),
        "unopened_confirmation_ids_sha256": sha256_file(confirm_path),
        "arm_a_sha256": sha256_file(arm_a_path),
        "arm_b_sha256": sha256_file(arm_b_path),
        "arm_report": arm_report,
        "replacement_rejections": dict(replacement_rejections.most_common()),
        "pilot_runtime_rejections": dict(pilot_rejections.most_common()),
        "pilot_lineages_requested": len(pilot_lineages),
        "pilot_rows_materialized": len(pilot),
        "confirmation_opened": False,
        "replacement_source_counts": dict(Counter(
            row["source_dataset"] for row in replacements
        ).most_common()),
        "arm_a_source_counts": dict(Counter(
            row["source_dataset"] for row in arm_a
        ).most_common()),
        "arm_b_source_counts": dict(Counter(
            row["source_dataset"] for row in arm_b
        ).most_common()),
        "trace_training_eligible_replacements": sum(
            bool(row["trace_training_eligible"]) for row in replacements
        ),
        "source_files_sha256": {
            path.relative_to(ROOT).as_posix(): sha256_file(path)
            for path in source_files
        },
        "source_candidates_sha256": sha256_file(arguments.candidates),
        "source_oracle_manifest_sha256": sha256_file(arguments.oracle_manifest),
        "corpus_development_view_manifest_sha256": sha256_file(
            arguments.corpus / "development_view" / "manifest.json"
        ),
        "assertions": {
            "arms_exactly_1024": len(arm_a) == len(arm_b) == ARM_SIZE,
            "completion_bytes_equal_at_every_position": all(
                left["completion"] == right["completion"]
                for left, right in zip(arm_a, arm_b)
            ),
            "replacement_count_exactly_128": len(replacements) == REPLACEMENT_COUNT,
            "mbpp_replacement_share_at_most_0_70": (
                sum(row["source_dataset"] == "mbpp" for row in replacements)
                / len(replacements) <= 0.70
            ),
            "no_holdout_lineage_in_training": not (
                {row["function_lineage"] for row in arm_a} & forbidden
            ),
            "unopened_confirmation_has_no_labels": True,
        },
    }
    if not all(manifest["assertions"].values()):
        raise ValueError(f"manifest assertion failed: {manifest['assertions']}")
    manifest_path = arguments.output_dir / "manifest.json"
    _json_dump(manifest_path, manifest)
    _json_dump(arguments.manifest_output, manifest)
    print(json.dumps({
        "output_dir": str(arguments.output_dir),
        "input_audit": input_audit,
        "replacement_sources": manifest["replacement_source_counts"],
        "arm_sources": manifest["arm_a_source_counts"],
        "pilot_rows": len(pilot),
        "trace_training_eligible_replacements": manifest[
            "trace_training_eligible_replacements"
        ],
        "manifest_sha256": sha256_file(manifest_path),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
