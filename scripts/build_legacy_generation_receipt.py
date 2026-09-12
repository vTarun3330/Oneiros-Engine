"""Label a legacy-protocol generation run for exactly what it can be used for.

The full train-split generation launched before the successor protocol was
selected is a useful artifact and a dangerous one. Useful, because it is the
first train-split generation that covers mbpp at all - the previous train panel
was 893 humaneval and 7 curated records with zero mbpp, which silently made
every "generalisation gap" in this project a comparison across two different
benchmark mixtures. Dangerous, because it looks like every other generation
artifact and is not interchangeable with one.

Two properties disqualify it as a source of oracle-training labels:

* ``candidate_parse_mode`` is ``first_assertion``. The stored ``code`` for each
  candidate is only the FIRST assert statement. This project measured the model
  emitting a mean of 3.65 assertions per output, so most of what the model
  actually wrote was discarded at parse time.
* ``retain_raw_output`` is false. Only ``raw_output_sha256`` survives, and a
  hash cannot be un-hashed, so the discarded text is unrecoverable.

Correcting an oracle requires seeing what the model actually produced. This
receipt states, in the artifact directory itself, that this run cannot supply
that - so the file cannot be picked up later by someone who only sees its name.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness.corpus import write_json
from harness.evaluation_protocol import LEGACY, protocol_of

RECEIPT_NAME = "LEGACY_PROTOCOL_RECEIPT.json"

PERMITTED_USE = "historical_train_panel_composition_analysis_only"
PROHIBITED_USES = (
    "oracle_structured_dataset_construction",
    "wrong_oracle_correction_labels",
    "any_supervision_derived_from_candidate_text",
    "comparison_against_successor_protocol_artifacts",
)


def _load(path: Path) -> tuple[dict[str, Any], bool]:
    """The finished artifact if present, else the newest progress checkpoint."""
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8")), True
    import torch

    checkpoints = sorted(path.parent.glob(path.stem + ".progress.*.pt"),
                         reverse=True)
    if not checkpoints:
        raise SystemExit(f"no artifact and no progress checkpoint for {path}")
    payload = torch.load(checkpoints[0], map_location="cpu", weights_only=False)
    context = dict(payload.get("context") or {})
    context["completed_functions"] = payload.get("completed_functions")
    return context, False


def build(artifact: Path) -> dict[str, Any]:
    payload, complete = _load(artifact)
    profile = payload.get("evaluation_profile") or {}
    protocol = protocol_of(payload)

    disqualifiers: list[str] = []
    if protocol == LEGACY:
        disqualifiers.append(
            "candidate_parse_mode is first_assertion: the stored candidate is "
            "only the first assert statement, and the model emits a mean of "
            "3.65 assertions per output")
    if not profile.get("retain_raw_output"):
        disqualifiers.append(
            "retain_raw_output is false: only raw_output_sha256 survives, and "
            "a hash cannot be un-hashed, so discarded output is unrecoverable")
    if not profile.get("allow_test_function_candidates"):
        disqualifiers.append(
            "allow_test_function_candidates is false: a candidate written as a "
            "test function was rejected rather than scored")

    return {
        "schema_version": "oneiros_legacy_generation_receipt_v1",
        "artifact": artifact.as_posix().split("results/", 1)[-1],
        "run_completed": complete,
        "completed_functions": payload.get("completed_functions"),
        "function_validation_records": payload.get("function_validation_records"),
        "evaluation_split": payload.get("evaluation_split"),
        "final_test_measurement": payload.get("final_test_measurement"),
        "sealed_final_test_accessed": False,
        "evaluation_protocol": protocol,
        "protocol_declaration": "legacy first-assertion protocol",
        "raw_outputs_available": bool(profile.get("retain_raw_output")),
        "raw_outputs_note": (
            "raw model outputs are NOT available; only raw_output_sha256 was "
            "retained"),
        "suitable_for_oracle_structured_dataset": False,
        "disqualifiers": disqualifiers,
        "permitted_use": PERMITTED_USE,
        "prohibited_uses": list(PROHIBITED_USES),
        "seed": payload.get("seed"),
        "adapter": payload.get("adapter"),
        "model_runtime_profile": payload.get("model_runtime_profile"),
        "evaluation_profile": profile,
        "evaluation_profile_sha256": payload.get("evaluation_profile_sha256"),
        "evaluation_scope_sha256": payload.get("evaluation_scope_sha256"),
        "dataset_fingerprint": payload.get("dataset_fingerprint"),
        "reproducibility": payload.get("reproducibility"),
        "why_it_is_kept": (
            "it is the first train-split generation covering mbpp. The previous "
            "train panel was 893 humaneval and 7 curated records with zero "
            "mbpp, which made every reported train/val generalisation gap a "
            "comparison across two different benchmark mixtures. This run is "
            "the correct instrument for that correction precisely BECAUSE it "
            "is under the same legacy protocol as the committed arm table."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    arguments = parser.parse_args()

    receipt = build(arguments.artifact)
    output = arguments.output or arguments.artifact.parent / RECEIPT_NAME
    write_json(output, receipt)
    print(json.dumps({k: v for k, v in receipt.items() if k not in (
        "reproducibility", "evaluation_profile", "model_runtime_profile")},
        indent=2))
    print("\nreceipt written to " + str(output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
