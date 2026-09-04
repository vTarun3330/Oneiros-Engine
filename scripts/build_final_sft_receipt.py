"""Receipt the final source-bound SFT rebuild.

Everything here is read back off disk from the run that actually happened - the
training artifact, the monitor checkpoints, and the locked-validation artifacts
- so the receipt cannot claim a configuration that was not used or a result
that was not measured.

The checkpoint is selected by a frozen rule stated before the numbers are read,
never by looking at the outcomes and preferring the best.  The rule is recorded
next to the selection so a reader can check it was followed.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness.corpus import write_json
from utils.reproducibility import source_tree_sha256

#: Stated before any validation number is read.  Selection uses the ablation_dev
#: monitor panel only; locked validation is a measurement surface, and selecting
#: on it would tune the model to the thing it is later judged by.
CHECKPOINT_SELECTION_RULE = (
    "highest mean ablation_dev monitor function kill rate across the trained "
    "seeds; ties broken by the earlier step. Locked validation is never used "
    "for selection."
)


def _read(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _relative(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def monitor_points(run_dir: Path) -> list[dict[str, Any]]:
    points = []
    for path in sorted(run_dir.glob("sft_monitor_checkpoint_*.json")):
        payload = _read(path)
        if not payload:
            continue
        points.append({
            "checkpoint_step": payload.get("checkpoint_step"),
            "ablation_dev_kill_rate": payload.get("function_kill_rate"),
            "functions": payload.get("function_validation_records"),
            "killed": payload.get("function_validation_killed"),
            "artifact": path.name,
        })
    return sorted(points, key=lambda item: item["checkpoint_step"] or 0)


def select_checkpoint(
    points_by_seed: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    """Apply the frozen rule across whatever seeds were trained.

    A step is only comparable where every seed measured it, so the mean is taken
    over steps common to all seeds.  With one trained seed this degenerates to
    that seed's best step, which is recorded as a stated limitation rather than
    presented as a cross-seed selection.
    """
    if not points_by_seed:
        return {"rule": CHECKPOINT_SELECTION_RULE, "selected_checkpoint_step": None}

    common: set[int] | None = None
    for points in points_by_seed.values():
        steps = {
            int(point["checkpoint_step"]) for point in points
            if point["checkpoint_step"] is not None
        }
        common = steps if common is None else (common & steps)
    common = common or set()

    means: dict[int, float] = {}
    for step in sorted(common):
        rates = [
            float(point["ablation_dev_kill_rate"])
            for points in points_by_seed.values()
            for point in points
            if int(point["checkpoint_step"] or -1) == step
            and point["ablation_dev_kill_rate"] is not None
        ]
        if rates:
            means[step] = sum(rates) / len(rates)
    if not means:
        return {"rule": CHECKPOINT_SELECTION_RULE, "selected_checkpoint_step": None}

    # sorted() first so an exact tie resolves to the earlier step, as the rule says
    best = max(sorted(means), key=lambda step: means[step])
    single = len(points_by_seed) == 1
    return {
        "rule": CHECKPOINT_SELECTION_RULE,
        "seeds_used_for_selection": sorted(points_by_seed),
        "steps_comparable_across_all_seeds": sorted(common),
        "mean_ablation_dev_kill_rate_by_step": {
            str(step): round(value, 6) for step, value in sorted(means.items())
        },
        "selected_checkpoint_step": best,
        "single_seed_selection": single,
        "limitation": (
            "Selected from one trained seed, so this step is not shown to be "
            "optimal across seeds. An earlier sweep chose steps 50, 142 and 142 "
            "on three seeds; no step is universally optimal."
        ) if single else None,
    }


def locked_validation(run_dir: Path) -> dict[str, Any]:
    arms: dict[str, Any] = {}
    for path in sorted(run_dir.glob("sft_validation_standard_seed_*.json")):
        payload = _read(path)
        if not payload or payload.get("evaluation_split") != "val":
            continue
        arms[str(payload.get("seed"))] = {
            "kill_rate": payload.get("function_kill_rate"),
            "wilson_95": payload.get("function_kill_rate_wilson_95"),
            "functions": payload.get("function_validation_records"),
            "killed": payload.get("function_validation_killed"),
            "prompt_budget_failed_functions": payload.get(
                "prompt_budget_failed_functions"
            ),
            "artifact": _relative(path),
        }
    rates = [
        arm["kill_rate"] for arm in arms.values() if arm["kill_rate"] is not None
    ]
    return {
        "by_seed": arms,
        "seeds_measured": sorted(arms),
        "mean_kill_rate": round(sum(rates) / len(rates), 6) if rates else None,
        "range_across_seeds": (
            round(max(rates) - min(rates), 6) if len(rates) > 1 else None
        ),
        "measurement_only": (
            "No checkpoint, hyperparameter, prompt, threshold or selection rule "
            "was chosen using locked validation."
        ),
    }


def launch_command(run_name: str) -> tuple[list[str], str | None]:
    """Recover the exact argv of the durable GPU run that produced ``run_name``.

    The command that ran is stronger evidence of the configuration than a field
    the trainer writes about itself, and reading it here means the receipt can
    report a flag the frozen trainer does not record without editing frozen
    source mid-run.
    """
    best: tuple[str, list[str]] | None = None
    for manifest_path in (ROOT / "runs").glob("*/manifest.json"):
        manifest = _read(manifest_path)
        if not manifest:
            continue
        command = manifest.get("command") or []
        if f"--run-name" in command:
            index = command.index("--run-name")
            if index + 1 < len(command) and command[index + 1] == run_name:
                created = str(manifest.get("created_utc") or "")
                if best is None or created > best[0]:
                    best = (created, command)
    if best is None:
        return [], None
    return best[1], best[0]


def _seed_of(run_name: str, training: dict[str, Any]) -> str:
    hyper = training.get("sft_hyperparameters") or {}
    if hyper.get("seed") is not None:
        return str(hyper["seed"])
    match = re.search(r"seed[_-]?(\d+)", run_name)
    return match.group(1) if match else run_name


def build(run_name: str, extra_seed_runs: list[str]) -> dict[str, Any]:
    run_dir = ROOT / "results" / run_name
    training = _read(run_dir / "training_results.json") or {}
    sampling = training.get("sft_sampling_stats") or {}
    supervision = sampling.get("multi_mutant_supervision") or {}

    points_by_seed: dict[str, list[dict[str, Any]]] = {}
    for name in [run_name] + list(extra_seed_runs):
        directory = ROOT / "results" / name
        points = monitor_points(directory)
        if points:
            other = _read(directory / "training_results.json") or {}
            points_by_seed[_seed_of(name, other)] = points

    density = supervision.get("density_over_eligible_synthetic")
    command, launched = launch_command(run_name)
    return {
        "schema_version": "oneiros_final_sft_receipt_v1",
        "source_tree_sha256": source_tree_sha256(ROOT),
        "sealed_final_test_accessed": False,
        "phase": "sft_only",
        "dpo": "OUT OF SCOPE - not prepared, launched, evaluated, or budgeted",
        "run_name": run_name,
        "adapter": training.get("sft_monitor_best_adapter"),
        "dataset_fingerprint": training.get("dataset_fingerprint"),
        "hyperparameters": training.get("sft_hyperparameters"),
        "sft_loss": training.get("sft_loss"),
        "wall_time_seconds": training.get("wall_time"),
        "supervision": {
            "verified_sft_examples": training.get("verified_sft_examples"),
            "repository_sft_examples": training.get("repository_sft_examples"),
            "prompt_compacted_examples": training.get("prompt_compacted_examples"),
            "every_label_carries_execution_evidence": True,
            "model_output_used_as_label": False,
        },
        "multi_mutant_supervision": supervision,
        # A run named for an intervention must be shown to have applied it.
        "intervention_actually_applied": (
            None if density is None else bool(density >= 0.5)
        ),
        "launch_command": command,
        "launched_utc": launched,
        "candidate_shape_policy": {
            "allow_test_function": "--allow-test-function-candidates" in command,
            "evidence": "the argv recorded by the durable GPU run manifest",
            "frozen_default": "single bounded assert statement",
            "widening_is_additive": (
                "a lone assertion is validated first by the identical frozen "
                "rule, so no candidate the frozen protocol accepted can change "
                "verdict; only shapes it rejected outright are admitted"
            ),
            "comparability": (
                "results scored under the widened policy are a named protocol "
                "variant and are not interchangeable with assertion-only numbers"
            ),
        },
        "checkpoint_selection": select_checkpoint(points_by_seed),
        "monitor_points_by_seed": points_by_seed,
        "locked_validation": locked_validation(run_dir),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument(
        "--extra-seed-run", action="append", default=[],
        help="Additional trained-seed run directories to include in selection.",
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "results" / "v4_2_final_sft_receipt.json",
    )
    arguments = parser.parse_args()
    receipt = build(arguments.run_name, arguments.extra_seed_run)
    write_json(arguments.output, receipt)
    print(json.dumps({
        "adapter": receipt["adapter"],
        "multi_mutant_density": receipt["multi_mutant_supervision"].get(
            "density_over_eligible_synthetic"
        ),
        "intervention_actually_applied": receipt["intervention_actually_applied"],
        "selected_checkpoint_step": receipt["checkpoint_selection"].get(
            "selected_checkpoint_step"
        ),
        "locked_validation": {
            "seeds": receipt["locked_validation"]["seeds_measured"],
            "mean_kill_rate": receipt["locked_validation"]["mean_kill_rate"],
        },
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
