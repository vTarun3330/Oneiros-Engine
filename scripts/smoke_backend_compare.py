"""Diagnostic-only base-model/attention-backend speed-quality smoke test.

Compares candidate backends on the exact fixed 32-function ablation_dev
monitor panel (results/local_j1024_integration_32_seed42_v5/sft_monitor_selection.json)
using the identical seed, candidate count, and verification pipeline already
used for the Phi-3/eager reference run. This never trains an adapter and
never touches the locked ``val`` split or the sealed ``test`` split; it exists
only to decide, before spending an 800-pair GPU training budget, whether a
faster attention backend or a smaller base model is worth training.

Usage:
    python scripts/smoke_backend_compare.py --backend phi3_sdpa
    python scripts/smoke_backend_compare.py --backend qwen_sdpa
    python scripts/smoke_backend_compare.py --backend phi3_eager   # re-verify only

Results are written to results/smoke_backend_<backend>_seed<seed>.json.
Run scripts/smoke_backend_report.py afterwards to print the comparison table.
"""
from __future__ import annotations

import argparse
import gc
import glob
import hashlib
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

import scripts.train_on_dataset as tod  # noqa: E402  (reuse the exact production eval path)
from config import CANONICAL_CORPUS_VERSION  # noqa: E402
from engine.generator import Phi3Generator  # noqa: E402
from harness.corpus import verify_corpus  # noqa: E402
from harness.corpus_view import verify_development_view  # noqa: E402

RESULTS_DIR = Path(__file__).parent.parent / "results"
PANEL_FILE = RESULTS_DIR / "local_j1024_integration_32_seed42_v5" / "sft_monitor_selection.json"

BACKENDS = {
    "phi3_eager": dict(
        model_name=None,
        model_revision=None,
        attention_implementation="eager",
        label="Phi-3-mini-4k-instruct (eager)",
    ),
    "phi3_sdpa": dict(
        model_name=None,
        model_revision=None,
        attention_implementation="sdpa",
        label="Phi-3-mini-4k-instruct (SDPA)",
    ),
    "qwen_sdpa": dict(
        model_name="Qwen/Qwen2.5-Coder-1.5B-Instruct",
        model_revision=None,
        attention_implementation="sdpa",
        label="Qwen2.5-Coder-1.5B-Instruct (SDPA)",
    ),
}


def _resolve_cached_snapshot_hash(model_name: str) -> str:
    """Best-effort lookup of the actual downloaded snapshot commit hash."""
    cache_root = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    repo_dir = "models--" + model_name.replace("/", "--")
    snapshots = sorted(glob.glob(str(cache_root / repo_dir / "snapshots" / "*")))
    if not snapshots:
        return "unresolved"
    return Path(snapshots[-1]).name


def _load_fixed_panel(seed: int):
    """Load the exact 32-function ablation_dev panel used by the Phi-3 reference run."""
    if not PANEL_FILE.exists():
        raise RuntimeError(f"Fixed comparison panel not found: {PANEL_FILE}")
    panel = json.loads(PANEL_FILE.read_text(encoding="utf-8"))
    record_ids = panel["record_ids"]
    split = panel["evaluation_split"]
    if split not in {"ablation_dev", "train"}:
        raise RuntimeError(
            f"Refusing to reuse a fixed panel drawn from split={split!r}; "
            "this smoke test may only use training-partition data."
        )

    tod.REQUIRE_SPLIT_ISOLATION = True
    tod.CORPUS_VERSION = CANONICAL_CORPUS_VERSION
    tod.SEED = seed
    tod.HOLDOUT_BUG_FAMILY = None
    tod.EVAL_FEEDBACK_ROUNDS = 0
    tod.EVAL_DIVERSITY_MODE = "none"

    corpus_dir = tod.DATA_DIR / "corpus" / CANONICAL_CORPUS_VERSION
    verify_corpus(corpus_dir)
    verify_development_view(corpus_dir, ["train", split])

    all_pairs = {
        pair["id"]: pair
        for pair in tod.load_phase3_pairs(corpus_dir, split)
        if pair.get("execution_mode", tod.FUNCTION_EXECUTION_MODE) == tod.FUNCTION_EXECUTION_MODE
    }
    missing = [record_id for record_id in record_ids if record_id not in all_pairs]
    if missing:
        raise RuntimeError(
            f"Fixed panel references {len(missing)} record(s) no longer present in the "
            f"current {split} split (corpus may have been rebuilt): {missing[:5]}"
        )
    ordered_pairs = [all_pairs[record_id] for record_id in record_ids]
    return ordered_pairs, panel


def run_smoke_backend(backend: str, seed: int, prompt_token_limit: int) -> dict:
    if backend not in BACKENDS:
        raise ValueError(f"Unknown backend {backend!r}; choose from {sorted(BACKENDS)}")
    spec = BACKENDS[backend]

    function_pairs, panel = _load_fixed_panel(seed)
    tod.PROMPT_TOKEN_LIMIT = prompt_token_limit
    tod.TESTS_PER_PAIR = 8
    tod.BATCH_GEN_SIZE = 2

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    started = time.time()
    generator = Phi3Generator(
        model_name=spec["model_name"],
        model_revision=spec["model_revision"],
        attention_implementation=spec["attention_implementation"],
    )
    try:
        generator.load_model()
        generator.max_new_tokens = tod.MAX_NEW_TOKENS_OVERRIDE

        resolved_revision = generator.model_revision
        if resolved_revision == "main":
            resolved_revision = _resolve_cached_snapshot_hash(generator.model_name)

        function_results = []
        for start in range(0, len(function_pairs), tod.BATCH_GEN_SIZE):
            chunk = function_pairs[start:start + tod.BATCH_GEN_SIZE]
            _, generation_accounting = tod._generate_evaluation_candidates(generator, chunk)
            for index, pair in enumerate(chunk):
                accounting = generation_accounting[index]
                candidate_outcomes = tod.evaluate_candidate_slots(
                    accounting["candidate_slots"],
                    pair["golden_code"],
                    pair["mutant_code"],
                    pair["entry_point"],
                )
                item = tod.build_function_result(
                    pair["id"],
                    str(pair.get("bug_family", "unknown") or "unknown"),
                    pair["entry_point"],
                    candidate_outcomes,
                    source_name=str(pair.get("source_name", "unknown")),
                    project=str(pair.get("project", "unknown")),
                    prompt_budget_failure=bool(accounting.get("prompt_budget_failure")),
                    prompt_budget_failure_reason=accounting.get("prompt_budget_failure_reason"),
                )
                function_results.append(item)
            completed = min(start + len(chunk), len(function_pairs))
            print(f"[{backend}] progress={completed}/{len(function_pairs)}", flush=True)

        wall_time = round(time.time() - started, 1)
        summary = tod.summarise_function_results(function_results)
        result = {
            "mode": "smoke_backend_comparison",
            "final_test_measurement": False,
            "backend": backend,
            "backend_label": spec["label"],
            "model_name": generator.model_name,
            "requested_model_revision": spec["model_revision"],
            "resolved_model_revision": resolved_revision,
            "attention_implementation": generator.attention_implementation,
            "model_runtime_profile": dict(generator.runtime_profile),
            "panel_source": str(PANEL_FILE),
            "panel_dataset_fingerprint": panel["dataset_fingerprint"],
            "panel_selection_sha256": panel["selection_sha256"],
            "evaluation_split": panel["evaluation_split"],
            "seed": seed,
            "tests_per_function": tod.TESTS_PER_PAIR,
            "prompt_token_limit": prompt_token_limit,
            "function_validation_records": len(function_pairs),
            "wall_time_seconds": wall_time,
            "seconds_per_function": round(wall_time / max(1, len(function_pairs)), 2),
            **summary,
            "function_results": function_results,
        }
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        out_path = RESULTS_DIR / f"v4_1_smoke_backend_{backend}_seed{seed}.json"
        out_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"[{backend}] wrote {out_path}", flush=True)
        print(
            f"[{backend}] kill_rate={summary.get('function_kill_rate'):.4f} "
            f"wall_time={wall_time}s ({result['seconds_per_function']}s/fn)",
            flush=True,
        )
        return result
    finally:
        generator.model = None
        generator.tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", required=True, choices=sorted(BACKENDS))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt-token-limit", type=int, default=1024)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        parser.error("Local execution requires a CUDA GPU; CPU fallback is disabled")
    print(f"Local GPU: {torch.cuda.get_device_name(0)}", flush=True)
    run_smoke_backend(args.backend, args.seed, args.prompt_token_limit)
