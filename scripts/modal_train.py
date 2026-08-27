"""
Oneiros — DPO Training on Modal (Serverless GPU).

Usage:
    modal run scripts/modal_train.py                          # default A10G
    modal run scripts/modal_train.py --fresh                  # wipe checkpoints
    modal run scripts/modal_train.py --max-pairs 100          # smoke test

Or run natively (wraps modal run with clean logging):
    py scripts/modal_train.py
    py scripts/modal_train.py --fresh --max-pairs 200  # start with 10 for a smoke test
"""

import os
import sys
import logging
import json
import subprocess
import re
import shutil
import threading
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
# Modal imports this module from /root/modal_train.py while the complete
# project tree is baked into /root/oneiros.  Make that tree importable before
# resolving project modules such as config.
REMOTE_PROJECT_ROOT = Path("/root/oneiros")
if REMOTE_PROJECT_ROOT.is_dir() and str(REMOTE_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(REMOTE_PROJECT_ROOT))

import modal
from config import CANONICAL_CORPUS_VERSION

# ═══════════════════════════════════════════════════════════════════
# 1. Modal App, Volume, Image
# ═══════════════════════════════════════════════════════════════════

app = modal.App("oneiros-training")

# Persistent volume for checkpoints, data splits, and results
volume = modal.Volume.from_name("oneiros-data-volume", create_if_missing=True)

# Container image with all ML dependencies pre-installed
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
    # Bake project source code directly into the image
    .add_local_dir("engine",   remote_path="/root/oneiros/engine")
    .add_local_dir("baseline", remote_path="/root/oneiros/baseline")
    .add_local_dir("config",   remote_path="/root/oneiros/config")
    .add_local_dir("scripts",  remote_path="/root/oneiros/scripts")
    .add_local_dir("harness",  remote_path="/root/oneiros/harness")
    .add_local_dir("metrics",  remote_path="/root/oneiros/metrics")
    .add_local_dir("tests",    remote_path="/root/oneiros/tests")
    .add_local_dir("utils",    remote_path="/root/oneiros/utils")
    .add_local_file("requirements.txt", remote_path="/root/oneiros/requirements.txt")
    .add_local_file("pytest.ini", remote_path="/root/oneiros/pytest.ini")
)


# ═══════════════════════════════════════════════════════════════════
# 2. Sync Helpers (Local ↔ Volume)
# ═══════════════════════════════════════════════════════════════════

def sync_tree(src_dir: str, dst_dir: str, prefer_newer: bool = True):
    """Recursively copy files from src to dst, skipping unchanged files."""
    if not os.path.exists(src_dir):
        return
    os.makedirs(dst_dir, exist_ok=True)
    for item in os.listdir(src_dir):
        s = os.path.join(src_dir, item)
        d = os.path.join(dst_dir, item)
        if os.path.isdir(s):
            sync_tree(s, d, prefer_newer=prefer_newer)
            continue
        should_copy = not os.path.exists(d)
        if not should_copy:
            try:
                if os.path.getsize(s) != os.path.getsize(d):
                    should_copy = True
                elif prefer_newer and int(os.path.getmtime(s)) >= int(os.path.getmtime(d)):
                    should_copy = True
            except OSError:
                should_copy = True
        if should_copy:
            os.makedirs(os.path.dirname(d), exist_ok=True)
            shutil.copy2(s, d)


def safe_replace_with_symlink(local_path: str, target_path: str):
    """Replace a local path with a symlink to the volume path."""
    if os.path.islink(local_path):
        try:
            if os.readlink(local_path) == target_path:
                return
            os.unlink(local_path)
        except OSError:
            pass
    elif os.path.exists(local_path):
        if os.path.isdir(local_path):
            shutil.rmtree(local_path)
        else:
            os.remove(local_path)
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    os.symlink(target_path, local_path)


def setup_local_to_volume_symlink(local_path: str, remote_path: str):
    """Sync local data up to the volume, then replace local with a symlink."""
    if os.path.exists(local_path) and not os.path.islink(local_path):
        sync_tree(local_path, remote_path, prefer_newer=True)
    safe_replace_with_symlink(local_path, remote_path)


def sync_results_from_volume(
    run_name: str = "v3_hardened_phase3", results_filename: str = "training_results.json"
):
    """Pull checkpoints and results back down from the Modal Volume."""
    print("\n📦 Syncing trained checkpoints and results from Modal Volume...")
    modal_bin = [sys.executable, "-m", "modal"]

    # Non-default runs use a dedicated results directory. Pull it as a tree so
    # checkpoint validation trend files arrive with the terminal result.
    results_source = (
        f"results/{results_filename}" if run_name == "dataset_trained"
        else f"results/{run_name}"
    )
    results_destination = "./results/"
    sync_targets = [
        # `modal volume get` preserves the source directory name. Target its
        # parent so the adapter lands at checkpoints/dataset_trained, rather
        # than checkpoints/dataset_trained/dataset_trained.
        (f"checkpoints/{run_name}", "./checkpoints/"),
        (results_source, results_destination),
    ]

    try:
        os.makedirs("checkpoints", exist_ok=True)
        os.makedirs("results", exist_ok=True)

        synced_any = False
        for remote_path, local_dest in sync_targets:
            print(f"  ⬇️  Syncing {remote_path}...")
            os.makedirs(os.path.dirname(local_dest) if not local_dest.endswith("/") else local_dest, exist_ok=True)
            download = subprocess.run(
                modal_bin + ["volume", "get", "oneiros-data-volume", remote_path, local_dest, "--force"],
                check=False,
                capture_output=True,
            )
            if download.returncode == 0:
                synced_any = True
            else:
                print(f"  ⚠️  No persisted artifact available at {remote_path}")

        if synced_any:
            print("✅ Sync complete.")
        else:
            print("⚠️ No checkpoint or result artifacts were available to sync.")
    except Exception as e:
        print(f"⚠️ Sync failed: {e}")


# ═══════════════════════════════════════════════════════════════════
# 3. Modal Remote Execution Function (GPU Heavy Lifter)
# ═══════════════════════════════════════════════════════════════════

@app.function(
    image=image,
    gpu="A10G",
    volumes={"/root/oneiros/storage": volume},
    timeout=14400*6,      # 24 hours
    memory=32768,       # 32 GB RAM
)
def run_cloud_training(
    fresh: bool = False,
    max_pairs: int = None,
    max_validation_functions: int = 0,
    corpus_version: str = CANONICAL_CORPUS_VERSION,
    execution_mode: str = "", phase: str = "sft", run_name: str = "v3_hardened_phase3",
    seed: int = 42,
    eval_feedback_rounds: int = 0,
    eval_diversity_mode: str = "none",
    holdout_bug_family: str = "",
    confirm_final_test: bool = False,
    expected_corpus_fingerprint: str = "", restart_dpo: bool = False,
    dpo_validation_interval_pairs: int = 500,
    sft_epochs: int = 0, sft_learning_rate: float = 0.0,
    sft_lr_scheduler_type: str = "", sft_batch_size: int = 0,
    sft_repository_completion_token_limit: int = 0,
    sft_real_target_fraction: float = -1.0, sft_max_real_repeats: int = 0,
    sft_balanced_sampling: bool = True,
    sft_synthetic_balance_fraction: float = 0.0,
    sft_max_synthetic_repeats: int = 2,
    sft_monitor_kill_rate: bool = True, sft_monitor_validation_functions: int = 500,
    sft_monitor_patience: int = 5,
    sft_monitor_min_function_kill_rate: float = -1.0,
):
    """Run the full Oneiros DPO training loop on a Modal GPU."""
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    # Keep the pinned base-model download on the persistent Modal Volume.
    # Repeated validation seeds must not spend GPU time downloading the same
    # immutable Hugging Face shards into an ephemeral container cache.
    hf_home = "/root/oneiros/storage/huggingface"
    os.makedirs(hf_home, exist_ok=True)
    os.environ["HF_HOME"] = hf_home
    os.environ["HF_HUB_CACHE"] = os.path.join(hf_home, "hub")
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    import torch
    log.info("🚀 Container booted. Configuring remote workspace...")
    log.info(f"   CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        log.info(f"   GPU: {torch.cuda.get_device_name(0)}")
        try:
            vram = torch.cuda.get_device_properties(0).total_mem / 1e9
        except AttributeError:
            vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        log.info(f"   VRAM: {vram:.1f} GB")

    os.chdir("/root/oneiros")
    if "/root/oneiros" not in sys.path:
        sys.path.insert(0, "/root/oneiros")

    storage_root = "/root/oneiros/storage"
    project_root = "/root/oneiros"

    # Symlink data, checkpoints, results to the persistent volume
    for sd in ["data", "checkpoints", "results"]:
        remote_sd = os.path.join(storage_root, sd)
        local_sd = os.path.join(project_root, sd)
        os.makedirs(remote_sd, exist_ok=True)
        setup_local_to_volume_symlink(local_sd, remote_sd)

    # Training is allowed only from the versioned canonical corpus.
    from harness.corpus import valid_corpus_version
    if not valid_corpus_version(corpus_version):
        return {"error": f"Invalid corpus version: {corpus_version}"}
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", run_name):
        return {"error": f"Invalid run name: {run_name}"}
    if phase not in {"base_eval", "sft", "sft_eval", "dpo", "dpo_eval", "sft_then_dpo"}:
        return {"error": f"Invalid training phase: {phase}"}
    if phase == "dpo_eval" and not confirm_final_test:
        return {"error": "dpo_eval requires explicit --confirm-final-test authorization"}
    if restart_dpo and (phase != "dpo" or fresh):
        return {"error": "--restart-dpo is valid only for a non-fresh DPO-only run"}
    if dpo_validation_interval_pairs <= 0:
        return {"error": "DPO validation interval must be positive"}
    if seed < 0:
        return {"error": "Seed must be non-negative"}
    if eval_feedback_rounds < 0 or eval_feedback_rounds >= 8:
        return {"error": "Evaluation feedback rounds must be between 0 and 7"}
    if eval_diversity_mode not in {"none", "ast", "input_shape"}:
        return {"error": "Unsupported evaluation diversity mode"}
    if max_validation_functions < 0:
        return {"error": "Maximum validation functions must be non-negative"}
    if sft_epochs < 0 or sft_learning_rate < 0 or sft_batch_size < 0:
        return {"error": "SFT overrides must be positive when supplied"}
    if not 0 <= sft_repository_completion_token_limit < 1536:
        return {"error": "SFT repository completion limit must be between 1 and 1535 when supplied"}
    if sft_lr_scheduler_type not in {"", "cosine", "constant_with_warmup"}:
        return {"error": "Unsupported SFT LR scheduler"}
    if sft_real_target_fraction != -1.0 and not 0.0 <= sft_real_target_fraction < 1.0:
        return {"error": "SFT real target fraction must be in [0, 1) when supplied"}
    if sft_max_real_repeats < 0:
        return {"error": "SFT max real repeats must be positive when supplied"}
    if sft_synthetic_balance_fraction < 0.0 or sft_max_synthetic_repeats < 1:
        return {"error": "SFT synthetic balance settings are invalid"}
    if sft_monitor_validation_functions <= 0 or sft_monitor_patience <= 0:
        return {"error": "SFT monitor validation functions and patience must be positive"}
    if (
        sft_monitor_min_function_kill_rate != -1.0
        and not 0.0 <= sft_monitor_min_function_kill_rate <= 1.0
    ):
        return {"error": "SFT monitor minimum function kill rate must be in [0, 1]"}
    corpus_dir = os.path.join(project_root, "data", "corpus", corpus_version)
    if not os.path.exists(os.path.join(corpus_dir, "manifest.json")):
        log.error("❌ train_pairs.json not found in volume! Upload data first.")
        log.error("   Run: modal volume put oneiros-data-volume data/splits/ data/splits/")
        return {"error": "Missing training data"}

    # The Volume can retain an older valid corpus under the same version
    # directory.  Verify the exact immutable files uploaded by this launch so
    # no remote run can silently train on a stale snapshot.
    from harness.corpus import verify_corpus
    remote_manifest = verify_corpus(Path(corpus_dir))
    remote_fingerprint = ":".join(
        remote_manifest["files"][filename]["sha256"]
        for filename in ("records.json", "splits.json", "external_eval_index.json")
    )
    if expected_corpus_fingerprint and remote_fingerprint != expected_corpus_fingerprint:
        raise RuntimeError(
            "Remote canonical corpus does not match the locally verified upload; "
            "aborting before SFT."
        )
    log.info("Remote canonical corpus identity verified.")

    # Import and configure the training script
    import scripts.train_on_dataset as trainer

    trainer.DATA_DIR    = Path(project_root) / "data"
    trainer.RESULTS_DIR = (
        Path(project_root) / "results" if run_name == "dataset_trained"
        else Path(project_root) / "results" / run_name
    )
    trainer.ADAPTER_DIR = Path(project_root) / "checkpoints" / run_name
    trainer.CORPUS_VERSION = corpus_version
    trainer.EXECUTION_MODE_FILTER = execution_mode or None
    trainer.TRAINING_PHASE = phase
    trainer.SEED = seed
    trainer.EVAL_FEEDBACK_ROUNDS = eval_feedback_rounds
    trainer.EVAL_DIVERSITY_MODE = eval_diversity_mode
    trainer.HOLDOUT_BUG_FAMILY = trainer.sanitise_family_name(holdout_bug_family)
    trainer.MAX_VALIDATION_PAIRS = max_validation_functions or None
    trainer.CONFIRM_FINAL_TEST = confirm_final_test
    trainer.RESTART_DPO = restart_dpo
    trainer.DPO_VALIDATION_INTERVAL_PAIRS = dpo_validation_interval_pairs
    trainer.SFT_EPOCHS_OVERRIDE = sft_epochs or None
    trainer.SFT_LEARNING_RATE_OVERRIDE = sft_learning_rate or None
    trainer.SFT_LR_SCHEDULER_TYPE_OVERRIDE = sft_lr_scheduler_type or None
    trainer.SFT_BATCH_SIZE_OVERRIDE = sft_batch_size or None
    trainer.SFT_REPOSITORY_COMPLETION_TOKEN_LIMIT_OVERRIDE = (
        sft_repository_completion_token_limit or None
    )
    trainer.SFT_REAL_TARGET_FRACTION_OVERRIDE = (
        sft_real_target_fraction if sft_real_target_fraction >= 0.0 else None
    )
    trainer.SFT_MAX_REAL_REPEATS_OVERRIDE = sft_max_real_repeats or None
    trainer.SFT_BALANCED_SAMPLING_ENABLED = sft_balanced_sampling
    trainer.SFT_SYNTHETIC_BALANCE_FRACTION = sft_synthetic_balance_fraction
    trainer.SFT_MAX_SYNTHETIC_REPEATS = sft_max_synthetic_repeats
    trainer.SFT_CHECKPOINT_MONITOR_ENABLED = sft_monitor_kill_rate
    trainer.SFT_MONITOR_VALIDATION_FUNCTIONS = sft_monitor_validation_functions
    trainer.SFT_MONITOR_PATIENCE = sft_monitor_patience
    trainer.SFT_MONITOR_MIN_FUNCTION_KILL_RATE_OVERRIDE = (
        sft_monitor_min_function_kill_rate
        if sft_monitor_min_function_kill_rate >= 0.0 else None
    )

    if max_pairs:
        trainer.MAX_TRAIN_PAIRS = max_pairs

    log.info("=" * 60)
    log.info(f"  Fresh start:  {fresh}")
    log.info(f"  Max pairs:    {max_pairs or 'ALL (8,000)'}")
    log.info(f"  Corpus:       {corpus_version}")
    log.info(f"  Mode filter:  {execution_mode or 'all'}")
    log.info(f"  Training phase: {phase}")
    log.info(f"  Seed:          {seed}")
    log.info(
        "  Evaluation profile: feedback_rounds=%s diversity=%s holdout_family=%s max_functions=%s",
        eval_feedback_rounds,
        eval_diversity_mode,
        holdout_bug_family or "none",
        max_validation_functions or "all",
    )
    log.info(f"  Restart DPO:   {restart_dpo}")
    log.info(f"  DPO validation interval (trained pairs): {dpo_validation_interval_pairs}")
    if sft_epochs or sft_learning_rate or sft_lr_scheduler_type or sft_batch_size or sft_repository_completion_token_limit or sft_real_target_fraction >= 0.0 or sft_max_real_repeats:
        log.info(
            "  SFT overrides: epochs=%s learning_rate=%s lr_scheduler=%s batch_size=%s repository_completion_limit=%s real_target_fraction=%s max_real_repeats=%s",
            sft_epochs or "default", sft_learning_rate or "default",
            sft_lr_scheduler_type or "default", sft_batch_size or "default",
            sft_repository_completion_token_limit or "default",
            sft_real_target_fraction if sft_real_target_fraction >= 0.0 else "default",
            sft_max_real_repeats or "default",
        )
    if sft_monitor_kill_rate:
        log.info(
            "  SFT kill-rate monitor: %s validation functions, patience=%s, minimum=%s, interval=50 steps",
            sft_monitor_validation_functions,
            sft_monitor_patience,
            sft_monitor_min_function_kill_rate
            if sft_monitor_min_function_kill_rate >= 0.0 else "default",
        )
    if sft_balanced_sampling:
        log.info(
            "  SFT sampler: exact deduplication and project-balanced real repeats; "
            "synthetic_balance_fraction=%s max_synthetic_repeats=%s",
            sft_synthetic_balance_fraction,
            sft_max_synthetic_repeats,
        )
    log.info(f"  Run name:     {run_name}")
    log.info(f"  Adapter dir:  {trainer.ADAPTER_DIR}")
    log.info("=" * 60)

    # Always commit completed SFT artifacts and failure diagnostics. A failed
    # phase must be resumable, but the train script prevents DPO from running
    # without a verified SFT marker and reference adapter.
    commit_stop = threading.Event()

    def commit_volume_periodically():
        while not commit_stop.wait(60):
            try:
                volume.commit()
                log.info("Periodic Volume commit completed.")
            except Exception as error:
                log.warning("Periodic Volume commit failed: %s", error)

    commit_thread = threading.Thread(
        target=commit_volume_periodically,
        name="oneiros-volume-commit",
        daemon=True,
    )
    commit_thread.start()
    try:
        result = trainer.run_training(use_mock=False, fresh=fresh)
    finally:
        commit_stop.set()
        commit_thread.join(timeout=5)
        volume.commit()
        log.info("Volume committed. Checkpoints and results are persisted.")

    return result

# ═══════════════════════════════════════════════════════════════════
# 4. Local Entrypoint (Orchestrates sync + remote launch)
# ═══════════════════════════════════════════════════════════════════

@app.local_entrypoint()
def training_main(
    fresh: bool = False,
    max_pairs: int = 0,
    max_validation_functions: int = 0,
    corpus_version: str = CANONICAL_CORPUS_VERSION,
    execution_mode: str = "", phase: str = "sft", run_name: str = "v3_hardened_phase3",
    seed: int = 42,
    eval_feedback_rounds: int = 0,
    eval_diversity_mode: str = "none",
    holdout_bug_family: str = "",
    confirm_final_test: bool = False,
    restart_dpo: bool = False, dpo_validation_interval_pairs: int = 500,
    sft_epochs: int = 0, sft_learning_rate: float = 0.0,
    sft_lr_scheduler_type: str = "", sft_batch_size: int = 0,
    sft_repository_completion_token_limit: int = 0,
    sft_real_target_fraction: float = -1.0, sft_max_real_repeats: int = 0,
    sft_balanced_sampling: bool = True,
    sft_synthetic_balance_fraction: float = 0.0,
    sft_max_synthetic_repeats: int = 2,
    sft_monitor_kill_rate: bool = True, sft_monitor_validation_functions: int = 500,
    sft_monitor_patience: int = 5,
    sft_monitor_min_function_kill_rate: float = -1.0,
):
    """Entry point for `modal run scripts/modal_train.py`."""
    # Validate locally before uploading or reserving a GPU. The remote training
    # function repeats this check against the files mounted from the Volume.
    from harness.corpus import valid_corpus_version
    if not valid_corpus_version(corpus_version):
        raise ValueError(f"Invalid corpus version: {corpus_version!r}")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", run_name):
        raise ValueError(f"Invalid run name: {run_name!r}")
    if phase not in {"base_eval", "sft", "sft_eval", "dpo", "dpo_eval", "sft_then_dpo"}:
        raise ValueError(f"Invalid training phase: {phase!r}")
    if phase == "dpo_eval" and not confirm_final_test:
        raise ValueError("dpo_eval requires explicit --confirm-final-test authorization")
    if restart_dpo and (phase != "dpo" or fresh):
        raise ValueError("--restart-dpo is valid only for a non-fresh DPO-only run")
    if dpo_validation_interval_pairs <= 0:
        raise ValueError("DPO validation interval must be positive")
    if seed < 0:
        raise ValueError("Seed must be non-negative")
    if eval_feedback_rounds < 0 or eval_feedback_rounds >= 8:
        raise ValueError("Evaluation feedback rounds must be between 0 and 7")
    if eval_diversity_mode not in {"none", "ast", "input_shape"}:
        raise ValueError("Unsupported evaluation diversity mode")
    if max_validation_functions < 0:
        raise ValueError("Maximum validation functions must be non-negative")
    from metrics.research_evaluation import sanitise_family_name
    holdout_bug_family = sanitise_family_name(holdout_bug_family) or ""
    if sft_epochs < 0 or sft_learning_rate < 0 or sft_batch_size < 0:
        raise ValueError("SFT overrides must be positive when supplied")
    if not 0 <= sft_repository_completion_token_limit < 1536:
        raise ValueError(
            "SFT repository completion limit must be between 1 and 1535 when supplied"
        )
    if sft_lr_scheduler_type not in {"", "cosine", "constant_with_warmup"}:
        raise ValueError("Unsupported SFT LR scheduler")
    if sft_real_target_fraction != -1.0 and not 0.0 <= sft_real_target_fraction < 1.0:
        raise ValueError("SFT real target fraction must be in [0, 1) when supplied")
    if sft_max_real_repeats < 0:
        raise ValueError("SFT max real repeats must be positive when supplied")
    if sft_synthetic_balance_fraction < 0.0 or sft_max_synthetic_repeats < 1:
        raise ValueError("SFT synthetic balance settings are invalid")
    if sft_monitor_validation_functions <= 0 or sft_monitor_patience <= 0:
        raise ValueError("SFT monitor validation functions and patience must be positive")
    if (
        sft_monitor_min_function_kill_rate != -1.0
        and not 0.0 <= sft_monitor_min_function_kill_rate <= 1.0
    ):
        raise ValueError("SFT monitor minimum function kill rate must be in [0, 1]")
    if phase == "sft" and sft_monitor_kill_rate and max_pairs > 0:
        from config import training_config
        from scripts.modal_train_failover import validate_bounded_sft_monitor_capacity
        validate_bounded_sft_monitor_capacity(
            max_pairs,
            sft_epochs or training_config.sft_epochs,
            sft_batch_size or training_config.sft_batch_size,
            sft_max_real_repeats or 8,
            checkpoint_steps=training_config.sft_checkpoint_steps,
            minimum_checkpoints=training_config.sft_min_monitor_checkpoints,
        )
    from harness.corpus import verify_corpus
    manifest = verify_corpus(Path("data") / "corpus" / corpus_version)
    expected_corpus_fingerprint = ":".join(
        manifest["files"][filename]["sha256"]
        for filename in ("records.json", "splits.json", "external_eval_index.json")
    )
    print(
        "✓ Canonical corpus verified locally: "
        f"{manifest['training_records']:,} behaviorally validated records."
    )
    print(f"🚀 Pushing Oneiros {phase} training to Modal cloud (A10G GPU)...")
    print(f"   Fresh start: {fresh}")
    print(f"   Corpus: {corpus_version}")
    print(f"   Training phase: {phase}")
    print(
        "   Evaluation profile: "
        f"feedback_rounds={eval_feedback_rounds} diversity={eval_diversity_mode} "
        f"holdout_family={holdout_bug_family or 'none'} "
        f"max_functions={max_validation_functions or 'all'}"
    )
    print(f"   Restart DPO: {restart_dpo}")
    if phase == "dpo":
        print(f"   DPO validation interval: {dpo_validation_interval_pairs} trained preference pairs")
    if sft_epochs or sft_learning_rate or sft_lr_scheduler_type or sft_batch_size or sft_repository_completion_token_limit or sft_real_target_fraction >= 0.0 or sft_max_real_repeats:
        print(
            "   SFT overrides: "
            f"epochs={sft_epochs or 'default'} lr={sft_learning_rate or 'default'} "
            f"lr_scheduler={sft_lr_scheduler_type or 'default'} "
            f"batch={sft_batch_size or 'default'} "
            f"repository_completion_limit={sft_repository_completion_token_limit or 'default'} "
            f"real_target_fraction={sft_real_target_fraction if sft_real_target_fraction >= 0.0 else 'default'} "
            f"max_real_repeats={sft_max_real_repeats or 'default'}"
        )
    if sft_monitor_kill_rate:
        print(
            "   SFT kill-rate monitor: "
            f"{sft_monitor_validation_functions} validation functions, "
            f"patience={sft_monitor_patience}, "
            f"minimum={sft_monitor_min_function_kill_rate if sft_monitor_min_function_kill_rate >= 0.0 else 'default'}, "
            "interval=50 optimizer steps"
        )
    if sft_balanced_sampling:
        print(
            "   SFT sampler: exact deduplication and project-balanced real repeats; "
            f"synthetic_balance_fraction={sft_synthetic_balance_fraction} "
            f"max_synthetic_repeats={sft_max_synthetic_repeats}"
        )
    if execution_mode:
        print(f"   Execution mode: {execution_mode}")
    print(f"   Run name: {run_name}")
    if max_pairs > 0:
        print(f"   Max pairs: {max_pairs}")

    # ── Upward Data Sync ──────────────────────────────────────
    print("\n⏳ Syncing local data UP to the Modal Volume...")
    modal_bin = [sys.executable, "-m", "modal"]

    # A trailing slash makes the Modal CLI treat this as the parent directory,
    # ensuring the complete versioned corpus replaces data/corpus/<version>.
    upload_targets = [(f"data/corpus/{corpus_version}", "data/corpus/")]

    if restart_dpo:
        from scripts.train_on_dataset import reset_dpo_from_frozen_sft
        local_results_dir = Path("results") if run_name == "dataset_trained" else Path("results") / run_name
        reset_dpo_from_frozen_sft(Path("checkpoints") / run_name, local_results_dir)
        # Overwrite the interrupted policy on the Volume before the remote
        # process starts; the remote guard repeats this operation defensively.
        upload_targets.append((f"checkpoints/{run_name}", "checkpoints/"))

    # Also push the named run's checkpoint when resuming.  The previous
    # hard-coded dataset_trained path could leave a non-default run using a
    # stale policy on the Volume.
    checkpoint_dir = os.path.join("checkpoints", run_name)
    if not fresh and os.path.exists(checkpoint_dir):
        pth_files = [f for f in os.listdir(checkpoint_dir) if f.endswith(('.safetensors', '.json'))]
        if pth_files:
            upload_targets.append((checkpoint_dir, "checkpoints/"))
    local_results_dir = os.path.join("results", run_name)
    if not fresh and os.path.exists(local_results_dir):
        upload_targets.append((local_results_dir, "results/"))

    for local_path, remote_path in upload_targets:
        if os.path.exists(local_path):
            print(f"  ⬆️  Pushing {local_path} -> Volume:/{remote_path}")
            modal_env = os.environ.copy()
            modal_env["PYTHONUTF8"] = "1"
            modal_env["PYTHONIOENCODING"] = "utf-8"
            upload = subprocess.run(
                modal_bin + ["volume", "put", "--force", "oneiros-data-volume", local_path, remote_path],
                capture_output=True, text=True, encoding="utf-8", errors="replace",
                check=False, env=modal_env,
            )
            if upload.returncode:
                raise RuntimeError(
                    f"Failed to upload canonical corpus to Modal Volume: {upload.stderr.strip()}"
                )

    print("✅ Upward sync complete.\n")

    # ── Trigger Cloud Execution ───────────────────────────────
    from scripts import train_on_dataset as evaluation_names
    evaluation_names.EVAL_FEEDBACK_ROUNDS = eval_feedback_rounds
    evaluation_names.EVAL_DIVERSITY_MODE = eval_diversity_mode
    evaluation_names.HOLDOUT_BUG_FAMILY = holdout_bug_family or None
    evaluation_names.MAX_VALIDATION_PAIRS = max_validation_functions or None
    result_filename = {
        "base_eval": evaluation_names.evaluation_results_filename("base", seed),
        "sft_eval": (
            evaluation_names.sft_validation_results_filename(seed)
        ),
        "dpo": "dpo_training_results.json",
        "dpo_eval": "dpo_test_results.json",
    }.get(phase, "training_results.json")
    try:
        result = run_cloud_training.remote(
            fresh=fresh,
            max_pairs=max_pairs if max_pairs > 0 else None,
            max_validation_functions=max_validation_functions,
            corpus_version=corpus_version,
            execution_mode=execution_mode,
            phase=phase,
            run_name=run_name,
            seed=seed,
            eval_feedback_rounds=eval_feedback_rounds,
            eval_diversity_mode=eval_diversity_mode,
            holdout_bug_family=holdout_bug_family,
            confirm_final_test=confirm_final_test,
            expected_corpus_fingerprint=expected_corpus_fingerprint,
            restart_dpo=restart_dpo,
            dpo_validation_interval_pairs=dpo_validation_interval_pairs,
            sft_epochs=sft_epochs,
            sft_learning_rate=sft_learning_rate,
            sft_lr_scheduler_type=sft_lr_scheduler_type,
            sft_batch_size=sft_batch_size,
            sft_repository_completion_token_limit=(
                sft_repository_completion_token_limit
            ),
            sft_real_target_fraction=sft_real_target_fraction,
            sft_max_real_repeats=sft_max_real_repeats,
            sft_balanced_sampling=sft_balanced_sampling,
            sft_synthetic_balance_fraction=sft_synthetic_balance_fraction,
            sft_max_synthetic_repeats=sft_max_synthetic_repeats,
            sft_monitor_kill_rate=sft_monitor_kill_rate,
            sft_monitor_validation_functions=sft_monitor_validation_functions,
            sft_monitor_patience=sft_monitor_patience,
            sft_monitor_min_function_kill_rate=(
                sft_monitor_min_function_kill_rate
            ),
        )

        print("\n" + "=" * 60)
        print("TRAINING COMPLETE (Modal)")
        print("=" * 60)
        if isinstance(result, dict):
            for k, v in result.items():
                print(f"  {k}: {v}")
        else:
            print(f"  Result: {result}")
    finally:
        # A credit/budget stop can interrupt the remote call before normal
        # completion. Always recover the latest persisted checkpoint locally
        # so a different Modal profile can resume the same named run.
        sync_results_from_volume(run_name, result_filename)

    print("\n✅ All results synced to local directories.")


# ═══════════════════════════════════════════════════════════════════
# 5. Native Python Entry Point (wraps `modal run` with clean logs)
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Windows can expose a legacy cp1252 stdout even when the child Modal
    # process is configured for UTF-8.  Configure this launcher before its
    # first status message so Unicode progress labels cannot abort the run.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    run_name = "v3_hardened_phase3"
    for index, argument in enumerate(sys.argv):
        if argument == "--run-name" and index + 1 < len(sys.argv):
            run_name = sys.argv[index + 1]
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", run_name):
        raise ValueError(f"Invalid run name: {run_name!r}")
    log_filename = f"pipeline_oneiros_training_{run_name}.log"
    # Keep the remote training call alive if the local terminal, Codex session,
    # or network connection disappears. Normal completion and failure statuses
    # are still returned while this launcher remains connected.
    cmd = [sys.executable, "-m", "modal", "run", "--detach", "scripts/modal_train.py"]

    # Forward CLI args
    if len(sys.argv) > 1:
        cmd.extend(sys.argv[1:])

    ignore_patterns = [
        r"Creating objects", r"Creating mount", r"Uploaded", r"Finalizing index",
        r"Creating function", r"Created objects", r"Initializing\.\.\.",
        r"Running app", r"Worker assigned", r"Loading images",
        r"Running \(\d+/\d+ containers active\)",
        r"Created mount", r"Created function",
        r"Mounting .+", r"Connecting from Modal", r"keyboard interrupt",
        r"0/1 \[00:00", r"1/1 \[00:00", r"0/2 \[00:00", r"2/2 \[00:00",
        r"Batches:.*\b0/1\b", r"Batches:.*\b1/1\b",
        r"Batches:   0%\|", r"Batches: 100%\|",
    ]
    spinner_start = r"^[|/\\-]\s"

    fresh_flag = "--fresh" in sys.argv
    max_p = ""
    for i, a in enumerate(sys.argv):
        if a == "--max-pairs" and i + 1 < len(sys.argv):
            max_p = sys.argv[i + 1]

    print(f"🚀 Launching Oneiros DPO Training Pipeline on Modal...")
    print(f"   Fresh: {fresh_flag}  Max-Pairs: {max_p or 'ALL'}")
    print(f"📋 Logging clean output to {log_filename}\n")

    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"

    with open(log_filename, "a", encoding="utf-8") as f:
        profile_name = os.environ.get("MODAL_PROFILE", "active-profile")
        f.write(f"\n=== Modal launch using profile {profile_name} ===\n")
        f.flush()
        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", bufsize=1, env=env,
        )
        url_printed = False

        while True:
            raw_line = process.stdout.readline()
            if not raw_line:
                if process.poll() is not None:
                    break
                continue

            clean = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", raw_line)
            segments = clean.split("\r")
            line = ""
            for seg in reversed(segments):
                if seg.strip():
                    line = seg.strip()
                    break

            if not line:
                continue
            if "View app at" in line or "modal.com" in line:
                if not url_printed:
                    f.write(line + "\n")
                    f.flush()
                    print(f"🔗 {line}")
                    url_printed = True
                continue

            if any(re.search(p, line) for p in ignore_patterns):
                continue
            if re.match(spinner_start, line):
                continue

            f.write(line + "\n")
            f.flush()
            try:
                print(line)
            except UnicodeEncodeError:
                print(line.encode(sys.stdout.encoding, errors="replace").decode(sys.stdout.encoding))

    process.wait()
    if process.returncode == 0:
        print("\n✅ Oneiros Training Pipeline completed. Exit code: 0")
    else:
        print(f"\n❌ Oneiros Training Pipeline failed. Exit code: {process.returncode}")
        raise SystemExit(process.returncode)
