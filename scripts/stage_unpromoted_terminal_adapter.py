"""Make an unpromoted arm's terminal adapter evaluable, and mark it as such.

The monitor gate decides PROMOTION. It also, as a side effect, decides
MEASURABILITY: an arm that fails the gate writes no sft_adapter and no
completion marker, so the evaluator refuses to score it and the run produces no
validation number at all.

That conflation cost a real experiment. The mild regularisation arm reached the
same monitor kill rate as the promoted full-density arm (0.73, +8 functions on
100 targets) and was rejected because its parse success rate was 0.98 against a
0.985 tolerance - four candidates out of eight hundred. Whether that model
generalises better is exactly the question the arm was run to answer, and the
gate prevented anyone from finding out.

Refusing to PROMOTE a model on candidate health is right. Refusing to MEASURE
it is not: the project's own plan says to retain unsuccessful checkpoints and
rounds for comparison, which requires being able to evaluate them.

This stages the terminal adapter so the standard evaluator can read it, and
stamps the run so nothing downstream can mistake it for a promoted one:

* sft_metadata.json carries monitor_promoted: false and the rejection reason;
* an unpromoted_terminal_adapter.marker file sits beside the SFT marker;
* the run name is the caller's, so no reported artifact is touched.

It is a separate script rather than a flag on the trainer so it can never fire
as a side effect of a normal run.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness.corpus import write_json


def stage(run_name: str, reason: str, source: str = "sft_terminal_adapter",
          destination_run: str | None = None) -> dict[str, Any]:
    adapter_dir = ROOT / "checkpoints" / run_name
    if not adapter_dir.is_dir():
        raise SystemExit(f"no checkpoint directory for run {run_name!r}")

    terminal = adapter_dir / source
    if not terminal.is_dir():
        raise SystemExit(f"{run_name} has no {source} to stage")
    if not (terminal / "adapter_model.safetensors").exists():
        raise SystemExit(f"{terminal} is not a LoRA adapter directory")

    # An intermediate checkpoint is staged into its OWN run directory, so the
    # arm it came from keeps its promoted adapter and its reported numbers
    # untouched. Overwriting a measured run to inspect one of its checkpoints
    # would destroy the result the checkpoint is being compared against.
    if destination_run:
        target_dir = ROOT / "checkpoints" / destination_run
        if target_dir.exists():
            raise SystemExit(f"{destination_run} already exists; choose a new name")
        target_dir.mkdir(parents=True)
        # dataset_manifest.sha256 is what the trainer compares against to
        # refuse an adapter from a different corpus or training scope. Omitting
        # it makes the staged run look like it was trained on nothing, and
        # every evaluation is refused before it starts.
        for name in ("sft_metadata.json", "sft_run_config.json",
                     "dataset_manifest.sha256"):
            if (adapter_dir / name).exists():
                shutil.copy2(adapter_dir / name, target_dir / name)
        adapter_dir = target_dir

    marker = adapter_dir / "sft_complete.marker"
    promoted = adapter_dir / "sft_adapter"
    if marker.exists() or promoted.exists():
        raise SystemExit(
            f"{run_name} already has a promoted adapter; this script exists "
            "only for arms the monitor gate rejected, and must never overwrite "
            "a genuinely promoted one"
        )

    results = ROOT / "results" / run_name / "training_results.json"
    training = json.loads(results.read_text(encoding="utf-8")) if results.exists() else {}
    if training.get("sft_monitor_gate_passed") and source == "sft_terminal_adapter":
        raise SystemExit(
            f"{run_name} passed the monitor gate; it should have promoted "
            "normally and does not need staging"
        )

    shutil.copytree(terminal, promoted)

    metadata_path = adapter_dir / "sft_metadata.json"
    metadata = (
        json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata_path.exists() else {}
    )
    metadata["monitor_promoted"] = False
    metadata["monitor_rejection_reason"] = reason
    metadata["adapter_source"] = source
    metadata["staged_from_run"] = run_name
    metadata["measurement_only"] = (
        "staged so the arm can be evaluated. It failed the promotion gate and "
        "must never be reported as a promoted or selected adapter."
    )
    write_json(metadata_path, metadata)

    (adapter_dir / "unpromoted_terminal_adapter.marker").write_text(
        f"{reason}\n", encoding="utf-8"
    )
    marker.write_text("staged_unpromoted_terminal_adapter\n", encoding="utf-8")

    return {
        "run_name": run_name,
        "staged_from": terminal.as_posix().split("checkpoints/", 1)[-1],
        "staged_into": adapter_dir.as_posix().split("checkpoints/", 1)[-1],
        "monitor_promoted": False,
        "monitor_rejection_reason": reason,
        "marker_contents": "staged_unpromoted_terminal_adapter",
        "warning": (
            "any evaluation of this run measures an adapter the monitor "
            "rejected; report it as an unpromoted arm"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--source", default="sft_terminal_adapter",
                        help="subdirectory to stage, e.g. sft_tmp/checkpoint-100")
    parser.add_argument("--destination-run", default=None,
                        help="stage into a NEW run name, leaving the source run intact")
    parser.add_argument(
        "--reason", required=True,
        help="why the monitor rejected it, recorded in the metadata",
    )
    arguments = parser.parse_args()
    print(json.dumps(stage(arguments.run_name, arguments.reason,
                           arguments.source, arguments.destination_run), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
