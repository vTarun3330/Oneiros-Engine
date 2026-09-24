"""Build matched unordered-event and ordered-trace SFT arms from train evidence.

Both arms start from the exact 1,024-example frozen control.  The same 122
positions and record identities are replaced.  Both arms predict the same
shown/intended values and the same verified execution-event multiset.  Only
the trace arm retains temporal order; the control sorts the identical events.
This nearly matches target token mass while isolating sequential information.
No confirmation labels or non-train split is opened.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness.execution_supervision import format_output_prediction_chat_prompt
from harness.execution_supervision_sidecar import sha256_file
from scripts.preflight_execution_supervision_ab import MODEL_NAME, MODEL_REVISION

SCHEMA = "oneiros_execution_trace_ab_v1"


def _atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _sections(focused: str, call_expression: str) -> tuple[str, str, str]:
    spec_marker = "### Intended behaviour\n\n"
    function_marker = "\n\n### Function (may be defective)\n\n"
    call_marker = "\n\n### Call\n\n"
    assertion_marker = "\n\n### Assertion"
    if not all(marker in focused for marker in (
        spec_marker, function_marker, call_marker, assertion_marker
    )):
        raise ValueError("focused prompt is missing a frozen section")
    tail = focused.split(spec_marker, 1)[1]
    specification, tail = tail.split(function_marker, 1)
    function, tail = tail.split(call_marker, 1)
    call = tail.split(assertion_marker, 1)[0].strip()
    if call != call_expression.strip():
        raise ValueError("focused prompt call mismatch")
    return specification.strip(), function.strip(), call


def build_pair_prompt(row: dict[str, Any], *, ordered: bool) -> str:
    specification, function, call = _sections(
        str(row["focused_prompt"]), str(row["call_expression"])
    )
    order_instruction = (
        "Preserve the temporal order in which line events occur."
        if ordered else
        "Return the verified execution events sorted by line number and source "
        "text; temporal order is intentionally removed."
    )
    schema = (
        '{"events":[{"line":<int>,"source":<string>}],"actual":{"type":<string>,'
        '"literal":<string>},"intended":{"type":<string>,"literal":<string>},'
        '"differs":<bool>}'
    )
    return (
        "### VERIFIED EXECUTION-SUPERVISION TASK\n\n"
        "Given the intended public behaviour, a possibly defective Python "
        "function, and one literal-input call, predict the verified executed "
        "line events, the actual shown-code value, the intended correct value, "
        f"and whether the values differ. {order_instruction}\n\n"
        "Return exactly one compact JSON object and no prose or Markdown. Use "
        f"this exact key order and schema:\n{schema}\n\n"
        f"### Intended behaviour\n\n{specification}\n\n"
        f"### Function (may be defective)\n\n{function}\n\n"
        f"### Call\n\n{call}\n\n### JSON\n"
    )


def event_completion(row: dict[str, Any], *, ordered: bool) -> str:
    text = str(row["trace_completion"])
    parsed = json.loads(text)
    if list(parsed) != ["trace", "actual", "intended", "differs"]:
        raise ValueError("trace completion key order/schema mismatch")
    if parsed["actual"] != row["execution_evidence"]["actual"]:
        raise ValueError("trace actual value mismatch")
    if parsed["intended"] != row["execution_evidence"]["intended"]:
        raise ValueError("trace intended value mismatch")
    events = list(parsed["trace"])
    if not ordered:
        events.sort(key=lambda event: (int(event["line"]), str(event["source"])))
    payload = {
        "events": events,
        "actual": parsed["actual"],
        "intended": parsed["intended"],
        "differs": bool(parsed["differs"]),
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _arm_hash(rows: list[dict[str, Any]]) -> str:
    return hashlib.sha256(json.dumps(
        rows, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=ROOT / "results"
                        / "v4_3_execution_supervision_v1")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results"
                        / "v4_3_execution_trace_v1")
    args = parser.parse_args()
    source_manifest_path = args.source_dir / "manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    for key in ("sealed_final_test_accessed", "ablation_dev_accessed",
                "validation_accessed", "test_accessed", "canonical_records_json_opened"):
        if source_manifest.get(key) is not False:
            raise SystemExit(f"REFUSED: source isolation field {key} is not false")
    if source_manifest.get("evaluation_split") != "train":
        raise SystemExit("REFUSED: source dataset is not train-only")

    control_path = args.source_dir / "arm_a.control.json"
    prompt_only_path = args.source_dir / "arm_b.treatment.json"
    if sha256_file(control_path) != source_manifest["arm_a_sha256"]:
        raise SystemExit("REFUSED: source control hash mismatch")
    if sha256_file(prompt_only_path) != source_manifest["arm_b_sha256"]:
        raise SystemExit("REFUSED: source treatment hash mismatch")
    control = json.loads(control_path.read_text(encoding="utf-8"))
    prompt_only = json.loads(prompt_only_path.read_text(encoding="utf-8"))
    if not (len(control) == len(prompt_only) == 1024):
        raise SystemExit("REFUSED: source arm size mismatch")

    sidecar_rows = json.loads(
        (args.source_dir / "train.execution.json").read_text(encoding="utf-8")
    )
    by_identity: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in sidecar_rows:
        by_identity[(str(row["record_id"]), str(row["completion"]))].append(row)

    replacements: list[tuple[int, dict[str, Any]]] = []
    rejected_positions: list[int] = []
    for position, row in enumerate(prompt_only):
        if row.get("task_kind") != "execution_output_prediction":
            continue
        matches = [candidate for candidate in by_identity.get(
            (str(row["record_id"]), str(row["completion"])), []
        ) if candidate.get("trace_training_eligible") is True]
        if not matches:
            rejected_positions.append(position)
            continue
        # The upstream builder already selected one candidate identity for the
        # prompt-only row. Require that exact candidate when it is available.
        identity = row.get("candidate_identity")
        exact = [candidate for candidate in matches
                 if not identity or candidate.get("candidate_identity") == identity]
        selected = exact[0] if exact else matches[0]
        replacements.append((position, selected))
    if len(replacements) != 122 or len(rejected_positions) != 6:
        raise SystemExit(
            f"REFUSED: expected 122 trace-eligible and 6 rejected positions; "
            f"got {len(replacements)} and {len(rejected_positions)}"
        )

    unordered_arm = [dict(row) for row in control]
    ordered_arm = [dict(row) for row in control]
    for position, evidence in replacements:
        base = control[position]
        if base["record_id"] != evidence["record_id"]:
            raise SystemExit(f"REFUSED: record identity mismatch at position {position}")
        common = {
            **base,
            "task_kind": "execution_output_prediction",
            "execution_supervision_candidate_identity": evidence["candidate_identity"],
        }
        unordered_text = event_completion(evidence, ordered=False)
        ordered_text = event_completion(evidence, ordered=True)
        unordered_prompt = build_pair_prompt(evidence, ordered=False)
        ordered_prompt = build_pair_prompt(evidence, ordered=True)
        unordered_arm[position] = {
            **common,
            "supervision_kind": "unordered_execution_events_and_value_pair",
            "prompt": unordered_prompt,
            "completion": unordered_text,
            "prompt_sha256": hashlib.sha256(unordered_prompt.encode("utf-8")).hexdigest(),
            "completion_sha256": hashlib.sha256(unordered_text.encode("utf-8")).hexdigest(),
        }
        ordered_arm[position] = {
            **common,
            "supervision_kind": "ordered_execution_trace_and_value_pair",
            "prompt": ordered_prompt,
            "completion": ordered_text,
            "prompt_sha256": hashlib.sha256(ordered_prompt.encode("utf-8")).hexdigest(),
            "completion_sha256": hashlib.sha256(ordered_text.encode("utf-8")).hexdigest(),
        }

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, revision=MODEL_REVISION, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    token_report: dict[str, Any] = {}
    for name, arm in (("unordered_events", unordered_arm), ("ordered_trace", ordered_arm)):
        prompt_tokens, completion_tokens = [], []
        for position, _ in replacements:
            row = arm[position]
            rendered = format_output_prediction_chat_prompt(tokenizer, row["prompt"])
            p = len(tokenizer(rendered, add_special_tokens=False)["input_ids"])
            c = len(tokenizer(
                row["completion"] + tokenizer.eos_token, add_special_tokens=False
            )["input_ids"])
            limit = 1024 if row["execution_mode"] == "function_assertion" else 2048
            # Trace/value auxiliary targets use their own 1,024-token training
            # ceiling.  This does not change the canonical assertion bytes;
            # generation evaluation remains capped at 128 tokens.
            completion_limit = 1024
            if p > limit or c > completion_limit or p + c > 3072:
                raise SystemExit(
                    f"REFUSED: {name} position {position} exceeds token budget "
                    f"(prompt={p}/{limit}, completion={c}/{completion_limit}, total={p+c}/3072)"
                )
            prompt_tokens.append(p)
            completion_tokens.append(c)
        token_report[name] = {
            "replacement_prompt_tokens_total": sum(prompt_tokens),
            "replacement_completion_tokens_total": sum(completion_tokens),
            "prompt_tokens_min": min(prompt_tokens),
            "prompt_tokens_max": max(prompt_tokens),
            "completion_tokens_min": min(completion_tokens),
            "completion_tokens_max": max(completion_tokens),
        }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    unordered_path = args.output_dir / "arm_unordered_events.json"
    ordered_path = args.output_dir / "arm_ordered_trace.json"
    _atomic(unordered_path, unordered_arm)
    _atomic(ordered_path, ordered_arm)
    manifest = {
        "schema_version": SCHEMA,
        "evaluation_split": "train",
        "examples_per_arm": 1024,
        "shared_canonical_examples": 902,
        "replacement_examples": 122,
        "replacement_positions": [position for position, _ in replacements],
        "unmodified_ineligible_positions": rejected_positions,
        "record_identity_equal_at_every_position": all(
            left["record_id"] == right["record_id"]
            for left, right in zip(unordered_arm, ordered_arm)
        ),
        "replacement_identity_equal": all(
            unordered_arm[position]["execution_supervision_candidate_identity"]
            == ordered_arm[position]["execution_supervision_candidate_identity"]
            for position, _ in replacements
        ),
        "optimizer_examples_equal": True,
        "optimizer_steps_per_arm": 64,
        "completion_token_mass_exactly_equal": False,
        "token_mass_control": (
            "both arms contain the identical event multiset and values; only event order "
            "and the corresponding prompt instruction differ"
        ),
        "token_report": token_report,
        "source_counts": dict(sorted(Counter(
            row["source_dataset"] for row in unordered_arm
        ).items())),
        "replacement_source_counts": dict(sorted(Counter(
            evidence["source_dataset"] for _, evidence in replacements
        ).items())),
        "source_dataset_manifest_sha256": sha256_file(source_manifest_path),
        "source_control_sha256": sha256_file(control_path),
        "source_prompt_only_sha256": sha256_file(prompt_only_path),
        "unordered_arm_sha256": sha256_file(unordered_path),
        "ordered_arm_sha256": sha256_file(ordered_path),
        "unordered_arm_content_sha256": _arm_hash(unordered_arm),
        "ordered_arm_content_sha256": _arm_hash(ordered_arm),
        "pilot_development_sha256": source_manifest["pilot_development_sha256"],
        "unopened_confirmation_ids_sha256": source_manifest[
            "unopened_confirmation_ids_sha256"
        ],
        "model": MODEL_NAME,
        "model_revision": MODEL_REVISION,
        "sealed_final_test_accessed": False,
        "ablation_dev_accessed": False,
        "validation_accessed": False,
        "test_accessed": False,
        "confirmation_opened": False,
        "source_files_sha256": {
            Path(__file__).relative_to(ROOT).as_posix(): sha256_file(Path(__file__)),
            "engine/sft_trainer.py": sha256_file(ROOT / "engine" / "sft_trainer.py"),
            "harness/execution_supervision.py": sha256_file(
                ROOT / "harness" / "execution_supervision.py"
            ),
        },
    }
    _atomic(args.output_dir / "manifest.json", manifest)
    print(json.dumps({
        "output_dir": str(args.output_dir),
        "replacement_examples": len(replacements),
        "unmodified_ineligible_positions": len(rejected_positions),
        "token_report": token_report,
        "unordered_arm_sha256": manifest["unordered_arm_sha256"],
        "ordered_arm_sha256": manifest["ordered_arm_sha256"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
