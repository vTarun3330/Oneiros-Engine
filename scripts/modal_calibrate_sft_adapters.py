"""Evaluate base and two immutable SFT adapters on one locked V3 val panel.

The run is inference-only and checkpoints every ten completed functions.  It
never trains, mutates adapters, evaluates the final test split, or removes a
local artifact.
"""

from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import modal

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

app = modal.App("oneiros-sft-calibration")
volume = modal.Volume.from_name("oneiros-data-volume", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "build-essential")
    .pip_install(
        "torch==2.5.1",
        "transformers==4.48.3",
        "peft==0.14.0",
        "trl==0.15.2",
        "datasets==3.2.0",
        "bitsandbytes==0.45.0",
        "accelerate==1.2.1",
        "sentence-transformers==3.3.1",
        "sentencepiece==0.2.0",
        "protobuf==5.29.3",
        "faiss-cpu==1.9.0.post1",
    )
    .add_local_dir("engine", remote_path="/root/oneiros/engine")
    .add_local_dir("baseline", remote_path="/root/oneiros/baseline")
    .add_local_dir("config", remote_path="/root/oneiros/config")
    .add_local_dir("scripts", remote_path="/root/oneiros/scripts")
    .add_local_dir("harness", remote_path="/root/oneiros/harness")
    .add_local_dir("metrics", remote_path="/root/oneiros/metrics")
    .add_local_dir("tests", remote_path="/root/oneiros/tests")
    .add_local_dir("utils", remote_path="/root/oneiros/utils")
    .add_local_file("requirements.txt", remote_path="/root/oneiros/requirements.txt")
    .add_local_file("pytest.ini", remote_path="/root/oneiros/pytest.ini")
)


INPUTS_NAME = "v3_sft_common_panel_calibration_inputs_20260820_1"
POLICIES = (
    {
        "label": "base_model",
        "checkpoint_step": 0,
        "adapter_relative_path": None,
    },
    {
        "label": "previous_v3_checkpoint100",
        "checkpoint_step": 100,
        "adapter_relative_path": "previous_checkpoint100",
    },
    {
        "label": "corrected_alignment_adapter",
        "checkpoint_step": 11,
        "adapter_relative_path": "corrected_alignment",
    },
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_once(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != payload:
            raise RuntimeError(f"Existing immutable result conflicts with {path.name}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _latest_progress(policy_dir: Path, context: dict[str, Any]) -> dict | None:
    import torch

    for candidate in sorted(policy_dir.glob("progress_*.pt"), reverse=True):
        try:
            payload = torch.load(candidate, map_location="cpu", weights_only=False)
        except Exception as exc:
            print(f"Ignoring unreadable calibration progress {candidate.name}: {exc}")
            continue
        if payload.get("context") != context:
            continue
        completed = payload.get("completed_functions")
        if not isinstance(completed, int) or completed < 0 or completed > context["function_count"]:
            continue
        payload["checkpoint_path"] = str(candidate)
        return payload
    return None


def _save_progress(
    policy_dir: Path,
    context: dict[str, Any],
    completed: int,
    function_results: list[dict[str, Any]],
    elapsed_wall_time: float,
) -> Path:
    import torch

    path = policy_dir / f"progress_{completed:04d}.pt"
    if path.exists():
        return path
    payload = {
        "context": context,
        "completed_functions": completed,
        "function_results": function_results,
        "elapsed_wall_time": elapsed_wall_time,
        "python_rng_state": random.getstate(),
        "torch_cpu_rng_state": torch.get_rng_state(),
        "torch_cuda_rng_states": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
        ),
    }
    torch.save(payload, path)
    return path


def _restore_rng(progress: dict) -> None:
    import torch

    random.setstate(progress["python_rng_state"])
    torch.set_rng_state(progress["torch_cpu_rng_state"])
    cuda_states = progress.get("torch_cuda_rng_states", [])
    if cuda_states:
        if not torch.cuda.is_available() or len(cuda_states) != torch.cuda.device_count():
            raise RuntimeError("Calibration CUDA topology changed; refusing unsafe resume")
        torch.cuda.set_rng_state_all(cuda_states)


def _family_metrics(function_results: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    totals: dict[str, dict[str, int]] = {}
    for item in function_results:
        family = item["bug_family"]
        family_totals = totals.setdefault(
            family,
            {
                "functions": 0,
                "killed_functions": 0,
                "requested_candidates": 0,
                "parsed_candidates": 0,
                "generation_invalid_candidates": 0,
                "execution_invalid_candidates": 0,
                "valid_candidates": 0,
                "killing_candidates": 0,
            },
        )
        family_totals["functions"] += 1
        family_totals["killed_functions"] += int(item["killed"])
        for key in (
            "requested_candidates",
            "parsed_candidates",
            "generation_invalid_candidates",
            "execution_invalid_candidates",
            "valid_candidates",
            "killing_candidates",
        ):
            family_totals[key] += int(item[key])

    result: dict[str, dict[str, Any]] = {}
    for family, item in sorted(totals.items()):
        result[family] = {
            **item,
            "invalid_candidates": (
                item["generation_invalid_candidates"]
                + item["execution_invalid_candidates"]
            ),
            "function_kill_rate": round(
                item["killed_functions"] / max(item["functions"], 1), 6
            ),
            "candidate_kill_rate": round(
                item["killing_candidates"] / max(item["valid_candidates"], 1), 6
            ),
            "end_to_end_candidate_kill_rate": round(
                item["killing_candidates"] / max(item["requested_candidates"], 1), 6
            ),
            "parse_success_rate": round(
                item["parsed_candidates"] / max(item["requested_candidates"], 1), 6
            ),
        }
    return result


def _evaluate_policy(
    policy: dict[str, Any],
    function_pairs: list[dict],
    selection_sha256: str,
    corpus_fingerprint: str,
    inputs_root: Path,
    results_root: Path,
) -> dict[str, Any]:
    import torch

    import scripts.train_on_dataset as trainer
    from engine.generator import Phi3Generator

    policy_dir = results_root / policy["label"]
    policy_dir.mkdir(parents=True, exist_ok=True)
    adapter_path = (
        inputs_root / policy["adapter_relative_path"]
        if policy["adapter_relative_path"]
        else None
    )
    adapter_hash = "base_model_no_adapter"
    if adapter_path is not None:
        adapter_file = adapter_path / "adapter_model.safetensors"
        adapter_config = adapter_path / "adapter_config.json"
        if not adapter_file.exists() or not adapter_config.exists():
            raise RuntimeError(f"Missing immutable calibration adapter: {adapter_path}")
        adapter_hash = _sha256_file(adapter_file)

    generator = Phi3Generator()
    generator.load_model()
    if adapter_path is not None:
        generator.load_lora_adapter(adapter_path)
    generator.max_new_tokens = trainer.MAX_NEW_TOKENS_OVERRIDE
    base_revision = str(
        getattr(generator.model.config, "_commit_hash", "")
        or getattr(generator.tokenizer, "init_kwargs", {}).get("_commit_hash", "")
        or "unreported"
    )
    context = {
        "mode": "common_panel_sft_calibration",
        "validation_accounting_schema_version": trainer.VALIDATION_ACCOUNTING_SCHEMA_VERSION,
        "corpus_fingerprint": corpus_fingerprint,
        "evaluation_split": "val",
        "final_test_measurement": False,
        "selection_sha256": selection_sha256,
        "function_count": len(function_pairs),
        "seed": trainer.SEED,
        "tests_per_function": trainer.TESTS_PER_PAIR,
        "policy_label": policy["label"],
        "checkpoint_step": policy["checkpoint_step"],
        "adapter_sha256": adapter_hash,
        "base_model": generator.model_name,
        "base_model_revision": base_revision,
        "model_runtime_profile": dict(generator.runtime_profile),
    }
    result_path = policy_dir / "result.json"
    if result_path.exists():
        existing = json.loads(result_path.read_text(encoding="utf-8"))
        if all(existing.get(key) == value for key, value in context.items()):
            print(f"Reusing completed calibration policy {policy['label']}", flush=True)
            generator.model = None
            generator.tokenizer = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return existing
        raise RuntimeError(f"Existing {policy['label']} result has a different identity")

    progress = _latest_progress(policy_dir, context)
    started = time.time()
    if progress:
        start_index = progress["completed_functions"]
        function_results = list(progress["function_results"])
        prior_wall_time = float(progress.get("elapsed_wall_time", 0.0))
        _restore_rng(progress)
        print(
            f"Resuming {policy['label']} at {start_index}/{len(function_pairs)} "
            f"from {Path(progress['checkpoint_path']).name}",
            flush=True,
        )
    else:
        start_index = 0
        function_results = []
        prior_wall_time = 0.0
        random.seed(trainer.SEED)
        torch.manual_seed(trainer.SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(trainer.SEED)

    original_use_cache = getattr(generator.model.config, "use_cache", None)
    if original_use_cache is not None:
        generator.model.config.use_cache = True
    generator.model.eval()
    try:
        for start in range(start_index, len(function_pairs), trainer.BATCH_GEN_SIZE):
            chunk = function_pairs[start : start + trainer.BATCH_GEN_SIZE]
            generated, accounting_by_index = trainer.generate_tests_ai_batched(
                generator,
                chunk,
                trainer.TESTS_PER_PAIR,
                return_accounting=True,
            )
            for index, pair in enumerate(chunk):
                tests = generated.get(index, [])
                accounting = accounting_by_index[index]
                winners, losers = trainer.evaluate_pair(
                    tests, pair["golden_code"], pair["mutant_code"],
                    pair["entry_point"],
                )
                valid_candidates = len(winners) + len(losers)
                execution_invalid = max(0, len(tests) - valid_candidates)
                function_results.append(
                    {
                        "record_id": pair["id"],
                        "bug_family": str(pair.get("bug_family", "unknown") or "unknown"),
                        "requested_candidates": accounting["requested_candidates"],
                        "parsed_candidates": accounting["parsed_candidates"],
                        "generation_invalid_candidates": accounting[
                            "generation_invalid_candidates"
                        ],
                        "execution_invalid_candidates": execution_invalid,
                        "invalid_candidates": (
                            accounting["generation_invalid_candidates"]
                            + execution_invalid
                        ),
                        "valid_candidates": valid_candidates,
                        "killing_candidates": len(winners),
                        "killed": bool(winners),
                    }
                )
            completed = min(start + len(chunk), len(function_pairs))
            if completed % 10 == 0 or completed == len(function_pairs):
                checkpoint = _save_progress(
                    policy_dir,
                    context,
                    completed,
                    function_results,
                    prior_wall_time + time.time() - started,
                )
                killed = sum(int(item["killed"]) for item in function_results)
                print(
                    f"{policy['label']} progress={completed}/{len(function_pairs)} "
                    f"killed={killed} checkpoint={checkpoint.name}",
                    flush=True,
                )

        if adapter_path is not None:
            final_adapter_hash = _sha256_file(
                adapter_path / "adapter_model.safetensors"
            )
            if final_adapter_hash != adapter_hash:
                raise RuntimeError("Immutable adapter changed during calibration")
        requested = sum(item["requested_candidates"] for item in function_results)
        parsed = sum(item["parsed_candidates"] for item in function_results)
        valid = sum(item["valid_candidates"] for item in function_results)
        killing = sum(item["killing_candidates"] for item in function_results)
        generation_invalid = sum(
            item["generation_invalid_candidates"] for item in function_results
        )
        execution_invalid = sum(
            item["execution_invalid_candidates"] for item in function_results
        )
        killed = sum(int(item["killed"]) for item in function_results)
        function_outcomes_sha256 = hashlib.sha256(
            json.dumps(
                function_results, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest()
        result = {
            **context,
            "function_validation_records": len(function_pairs),
            "function_validation_killed": killed,
            "function_kill_rate": round(killed / len(function_pairs), 6),
            "function_kill_rate_wilson_95": trainer._wilson_interval(
                killed, len(function_pairs)
            ),
            "requested_candidates": requested,
            "parsed_candidates": parsed,
            "generated_candidates": valid,
            "mutation_killing_candidates": killing,
            "generation_invalid_candidates": generation_invalid,
            "execution_invalid_candidates": execution_invalid,
            "invalid_candidates": generation_invalid + execution_invalid,
            "candidate_kill_rate": round(killing / max(valid, 1), 6),
            "end_to_end_candidate_kill_rate": round(killing / max(requested, 1), 6),
            "parse_success_rate": round(parsed / max(requested, 1), 6),
            "function_outcomes_sha256": function_outcomes_sha256,
            "function_results": function_results,
            "bug_family_metrics": _family_metrics(function_results),
            "wall_time": round(prior_wall_time + time.time() - started, 1),
            "resumed_from_completed_functions": start_index,
            "validation_checkpointing": "every_10_functions",
            "adapter_mutated": False,
        }
        _write_json_once(result_path, result)
        return result
    finally:
        if original_use_cache is not None:
            generator.model.config.use_cache = original_use_cache
        generator.model = None
        generator.tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


@app.function(
    image=image,
    gpu="A10G",
    volumes={"/root/oneiros/storage": volume},
    timeout=7200,
    memory=32768,
)
def calibrate(
    run_name: str,
    corpus_version: str,
    expected_corpus_fingerprint: str,
    expected_adapter_hashes: dict[str, str],
    panel_size: int = 100,
) -> dict[str, Any]:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", run_name):
        raise ValueError("Invalid calibration run name")
    if panel_size != 100:
        raise ValueError("This calibration is locked to exactly 100 validation functions")
    os.chdir("/root/oneiros")
    if "/root/oneiros" not in sys.path:
        sys.path.insert(0, "/root/oneiros")
    storage = Path("/root/oneiros/storage")
    for directory in ("data", "results"):
        local_path = Path("/root/oneiros") / directory
        storage_path = storage / directory
        storage_path.mkdir(parents=True, exist_ok=True)
        if local_path.exists() and not local_path.is_symlink():
            shutil.rmtree(local_path)
        if not local_path.exists():
            local_path.symlink_to(storage_path, target_is_directory=True)

    from harness.corpus import verify_corpus
    import scripts.train_on_dataset as trainer

    corpus_dir = Path("/root/oneiros/data/corpus") / corpus_version
    manifest = verify_corpus(corpus_dir)
    corpus_fingerprint = ":".join(
        manifest["files"][name]["sha256"]
        for name in ("records.json", "splits.json", "external_eval_index.json")
    )
    if corpus_fingerprint != expected_corpus_fingerprint:
        raise RuntimeError("Remote V3 corpus identity does not match local preflight")
    inputs_root = storage / "calibration_inputs" / INPUTS_NAME
    for policy in POLICIES:
        if not policy["adapter_relative_path"]:
            continue
        path = inputs_root / policy["adapter_relative_path"] / "adapter_model.safetensors"
        actual_hash = _sha256_file(path)
        if actual_hash != expected_adapter_hashes[policy["label"]]:
            raise RuntimeError(f"Remote adapter hash mismatch for {policy['label']}")

    function_pairs = [
        pair
        for pair in trainer.load_phase3_pairs(corpus_dir, "val")
        if pair.get("execution_mode", trainer.FUNCTION_EXECUTION_MODE)
        == trainer.FUNCTION_EXECUTION_MODE
    ]
    selected = trainer._evenly_spaced(function_pairs, panel_size)
    selected_ids = [pair["id"] for pair in selected]
    selection_sha256 = hashlib.sha256(
        json.dumps(selected_ids, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    results_root = Path("/root/oneiros/results") / run_name
    results_root.mkdir(parents=True, exist_ok=True)
    selection = {
        "corpus_fingerprint": corpus_fingerprint,
        "evaluation_split": "val",
        "final_test_measurement": False,
        "selection_method": "evenly_spaced_canonical_validation_functions",
        "function_count": len(selected),
        "record_ids": selected_ids,
        "selection_sha256": selection_sha256,
        "seed": trainer.SEED,
        "tests_per_function": trainer.TESTS_PER_PAIR,
    }
    _write_json_once(results_root / "selection.json", selection)

    results = []
    for policy in POLICIES:
        result = _evaluate_policy(
            policy,
            selected,
            selection_sha256,
            corpus_fingerprint,
            inputs_root,
            results_root,
        )
        results.append(result)
        volume.commit()

    comparisons = {}
    for reference, evaluation in (
        (results[0], results[1]),
        (results[0], results[2]),
        (results[1], results[2]),
    ):
        key = f"{reference['policy_label']}__to__{evaluation['policy_label']}"
        comparisons[key] = trainer._paired_function_diagnostics(
            reference, evaluation
        )
    summary = {
        "mode": "common_panel_sft_calibration_summary",
        "corpus_fingerprint": corpus_fingerprint,
        "evaluation_split": "val",
        "final_test_measurement": False,
        "selection_sha256": selection_sha256,
        "function_count": panel_size,
        "tests_per_function": trainer.TESTS_PER_PAIR,
        "policies": [
            {
                key: result[key]
                for key in (
                    "policy_label",
                    "checkpoint_step",
                    "adapter_sha256",
                    "function_validation_killed",
                    "function_kill_rate",
                    "candidate_kill_rate",
                    "end_to_end_candidate_kill_rate",
                    "parse_success_rate",
                    "invalid_candidates",
                    "wall_time",
                )
            }
            for result in results
        ],
        "paired_comparisons": comparisons,
        "all_metrics_finite": all(
            math.isfinite(float(result[key]))
            for result in results
            for key in (
                "function_kill_rate",
                "candidate_kill_rate",
                "end_to_end_candidate_kill_rate",
                "parse_success_rate",
            )
        ),
        "adapters_mutated": False,
    }
    _write_json_once(results_root / "calibration_summary.json", summary)
    volume.commit()
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def _modal_command(*arguments: str) -> list[str]:
    return [sys.executable, "-m", "modal", *arguments]


def _upload(local_path: Path, remote_path: str) -> None:
    environment = os.environ.copy()
    environment["PYTHONUTF8"] = "1"
    environment["PYTHONIOENCODING"] = "utf-8"
    result = subprocess.run(
        _modal_command(
            "volume",
            "put",
            "--force",
            "oneiros-data-volume",
            str(local_path),
            remote_path,
        ),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=environment,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip())


@app.local_entrypoint()
def main(
    run_name: str = "v3_sft_common_panel_calibration_20260820_1",
    corpus_version: str = "v3_final_candidate",
    panel_size: int = 100,
) -> None:
    from harness.corpus import verify_corpus

    manifest = verify_corpus(PROJECT_ROOT / "data" / "corpus" / corpus_version)
    corpus_fingerprint = ":".join(
        manifest["files"][name]["sha256"]
        for name in ("records.json", "splits.json", "external_eval_index.json")
    )
    inputs_root = PROJECT_ROOT / "checkpoints" / INPUTS_NAME
    adapter_sources = {
        "previous_v3_checkpoint100": (
            PROJECT_ROOT
            / "checkpoints/v3_full_sft_monitored_20260819_1/sft_validation_best/checkpoint-100"
        ),
        "corrected_alignment_adapter": (
            PROJECT_ROOT
            / "checkpoints/v3_sft_alignment_smoke_20260820_1/sft_validation_best/checkpoint-11"
        ),
    }
    expected_hashes = {}
    for label, source in adapter_sources.items():
        destination_name = (
            "previous_checkpoint100"
            if label == "previous_v3_checkpoint100"
            else "corrected_alignment"
        )
        destination = inputs_root / destination_name
        destination.mkdir(parents=True, exist_ok=True)
        for filename in ("adapter_config.json", "adapter_model.safetensors"):
            source_file = source / filename
            destination_file = destination / filename
            if not source_file.exists():
                raise RuntimeError(f"Missing immutable local adapter file: {source_file}")
            if not destination_file.exists():
                shutil.copy2(source_file, destination_file)
            elif _sha256_file(destination_file) != _sha256_file(source_file):
                raise RuntimeError(f"Existing calibration input conflicts: {destination_file}")
        expected_hashes[label] = _sha256_file(
            destination / "adapter_model.safetensors"
        )

    print("Uploading verified V3 corpus and immutable calibration adapters...", flush=True)
    _upload(PROJECT_ROOT / "data" / "corpus" / corpus_version, "data/corpus/")
    _upload(inputs_root, "calibration_inputs/")
    local_results = PROJECT_ROOT / "results" / run_name
    if local_results.exists():
        _upload(local_results, "results/")
    summary = calibrate.remote(
        run_name=run_name,
        corpus_version=corpus_version,
        expected_corpus_fingerprint=corpus_fingerprint,
        expected_adapter_hashes=expected_hashes,
        panel_size=panel_size,
    )
    print(json.dumps(summary, indent=2), flush=True)
    destination = PROJECT_ROOT / "results"
    destination.mkdir(parents=True, exist_ok=True)
    download = subprocess.run(
        _modal_command(
            "volume",
            "get",
            "oneiros-data-volume",
            f"results/{run_name}",
            str(destination),
            "--force",
        ),
        check=False,
    )
    if download.returncode:
        raise RuntimeError("Calibration completed but local result sync failed")


if __name__ == "__main__":
    run_name = "v3_sft_common_panel_calibration_20260820_1"
    for index, argument in enumerate(sys.argv):
        if argument == "--run-name" and index + 1 < len(sys.argv):
            run_name = sys.argv[index + 1]
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", run_name):
        raise ValueError("A safe explicit calibration run name is required")
    log_path = PROJECT_ROOT / f"pipeline_oneiros_calibration_{run_name}.log"
    command = [
        sys.executable,
        "-m",
        "modal",
        "run",
        "--detach",
        "scripts/modal_calibrate_sft_adapters.py",
        *sys.argv[1:],
    ]
    environment = os.environ.copy()
    environment["PYTHONUTF8"] = "1"
    environment["PYTHONIOENCODING"] = "utf-8"
    with log_path.open("a", encoding="utf-8", buffering=1) as handle:
        completed = subprocess.run(
            command,
            stdout=handle,
            stderr=subprocess.STDOUT,
            env=environment,
            check=False,
        )
    raise SystemExit(completed.returncode)
