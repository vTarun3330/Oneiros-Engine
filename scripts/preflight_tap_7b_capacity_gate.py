"""Freeze the 7B TAP capacity protocol before downloading weights or generating."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.analyse_tap_7b_capacity_gate import (  # noqa: E402
    MAX_ANSWER_RATE_REGRESSION_PP,
    MAX_CAP_HIT_RATE,
    MIN_CI95_LOW_PP,
    MIN_TAP_MUT_GAIN_PP,
)
from scripts.analyse_tap_capacity_gate import validate_artifact  # noqa: E402
from scripts.run_tap_7b_capacity_gate import (  # noqa: E402
    MODEL_NAME,
    MODEL_REVISION,
    TOKENIZER_MAX_LENGTH,
    source_bundle,
    validate_items,
)

BASELINE_MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
BASELINE_REVISION = "2e1fd397ee46e1388853d2af2c993145b0f1098a"
REQUIRED_ITEMS_SHA256 = "c15398bdee5f1486a09d2f6338f426407470ba686dcc3ae8d99900364999f413"
REQUIRED_BASELINE_SHA256 = "1f84210e3af9939f86bbfca901b5116f59606b03e13599e8dde6a9eb80b97672"
REQUIRED_SPLIT_SHA256 = "99cb772bf6cd93a4a1e925cfeda7e8adde5bb5a831eafaa840478425a39516e8"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()


def acceptance_policy() -> dict:
    return {
        "primary_endpoint": "TAP-mut per-requested accuracy, paired 7B minus 1.5B",
        "minimum_gain_pp": MIN_TAP_MUT_GAIN_PP,
        "minimum_ci95_low_pp_exclusive": MIN_CI95_LOW_PP,
        "maximum_answer_rate_regression_pp": MAX_ANSWER_RATE_REGRESSION_PP,
        "maximum_cap_hit_rate": MAX_CAP_HIT_RATE,
    }


def token_audit(items: list[dict], tokenizer, max_new_tokens: int) -> dict:
    report = {}
    for condition, prompt_key in (("TAP-ref", "prompt_ref"), ("TAP-mut", "prompt_mut")):
        lengths = []
        for item in items:
            rendered = tokenizer.apply_chat_template(
                [{"role": "user", "content": item[prompt_key]}],
                tokenize=False,
                add_generation_prompt=True,
            )
            lengths.append(len(tokenizer(rendered, add_special_tokens=False)["input_ids"]))
        maximum = max(lengths)
        report[condition] = {
            "minimum_prompt_tokens": min(lengths),
            "maximum_prompt_tokens": maximum,
            "maximum_prompt_plus_completion": maximum + max_new_tokens,
            "sequence_limit": TOKENIZER_MAX_LENGTH,
            "fits": maximum + max_new_tokens <= TOKENIZER_MAX_LENGTH,
        }
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--items", default="results/tap_train_items.jsonl")
    parser.add_argument("--baseline", default="results/tap_adapter_compare.json")
    parser.add_argument("--split", default="results/tap_pilot_split.json")
    parser.add_argument("--gate1-receipt", default="results/tap_capacity_gate_receipt.json")
    parser.add_argument("--out", default="results/tap_7b_capacity_protocol.json")
    args = parser.parse_args(argv)

    paths = {
        "items": ROOT / args.items,
        "baseline": ROOT / args.baseline,
        "split": ROOT / args.split,
        "gate1_receipt": ROOT / args.gate1_receipt,
    }
    required_hashes = {
        "items": REQUIRED_ITEMS_SHA256,
        "baseline": REQUIRED_BASELINE_SHA256,
        "split": REQUIRED_SPLIT_SHA256,
    }
    problems: list[str] = []
    for name, path in paths.items():
        if not path.is_file():
            problems.append(f"missing {name}: {path}")
    if problems:
        print("REFUSED: " + "; ".join(problems))
        return 2

    hashes = {name: sha256_file(path) for name, path in paths.items()}
    for name, expected in required_hashes.items():
        if hashes[name] != expected:
            problems.append(f"{name} sha256 {hashes[name]} != {expected}")

    items = [
        json.loads(line)
        for line in paths["items"].read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    try:
        validate_items(items)
    except RuntimeError as exc:
        problems.append(str(exc))
    split = json.loads(paths["split"].read_text(encoding="utf-8"))
    baseline = json.loads(paths["baseline"].read_text(encoding="utf-8"))
    try:
        baseline_indexed = validate_artifact(baseline, split)
    except ValueError as exc:
        problems.append(f"baseline invalid: {exc}")
        baseline_indexed = {}

    item_ids = {item["id"] for item in items}
    pilot_ids = set(split.get("pilot_ids", []))
    primary_ids = item_ids - pilot_ids
    if len(pilot_ids) != 16 or len(primary_ids) != 584:
        problems.append(
            f"split must be 584 primary + 16 pilot, got {len(primary_ids)} + {len(pilot_ids)}"
        )
    if baseline_indexed and set(baseline_indexed["base::TAP-ref"]) != item_ids:
        problems.append("baseline item IDs do not equal the frozen input IDs")
    if any(item.get("split") != "train" for item in items):
        problems.append("non-train row found")

    dirty = git("status", "--porcelain", "--untracked-files=all").splitlines()
    if dirty:
        problems.append(f"worktree is not clean before protocol freeze: {dirty[:5]}")

    from config.settings import immutable_revision_for
    if immutable_revision_for(MODEL_NAME) != MODEL_REVISION:
        problems.append("7B immutable revision is not configured")

    try:
        from huggingface_hub import HfApi
        hub_sha = HfApi().model_info(MODEL_NAME, revision=MODEL_REVISION).sha
    except Exception as exc:  # noqa: BLE001
        problems.append(f"cannot verify official Hub identity: {exc}")
        hub_sha = None
    if hub_sha != MODEL_REVISION:
        problems.append(f"Hub resolved {hub_sha!r}, expected {MODEL_REVISION}")

    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME, revision=MODEL_REVISION, trust_remote_code=True
        )
        tokens = token_audit(items, tokenizer, 128)
    except Exception as exc:  # noqa: BLE001
        problems.append(f"tokenizer/token audit failed: {exc}")
        tokens = {}
    if tokens and not all(row["fits"] for row in tokens.values()):
        problems.append("at least one rendered prompt exceeds the frozen sequence budget")

    free_disk = shutil.disk_usage(ROOT).free
    if free_disk < 25 * 1024 ** 3:
        problems.append(f"less than 25 GiB free disk: {free_disk}")
    try:
        import torch
        gpu = {
            "available": torch.cuda.is_available(),
            "name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "total_memory_bytes": (
                torch.cuda.get_device_properties(0).total_memory
                if torch.cuda.is_available() else 0
            ),
            "bf16_supported": (
                torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False
            ),
        }
    except Exception as exc:  # noqa: BLE001
        problems.append(f"GPU inspection failed: {exc}")
        gpu = {"available": False, "error": repr(exc)}
    if not gpu.get("available") or gpu.get("total_memory_bytes", 0) < 20 * 1024 ** 3:
        problems.append("the gate requires a CUDA GPU with at least 20 GiB")

    protocol = {
        "schema_version": "oneiros_tap_7b_capacity_protocol_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "ready" if not problems else "refused",
        "label": (
            "frozen before 7B generation; train-only quantized capacity diagnostic, "
            "not SFT, generalization, validation, final-test, or promotion evidence"
        ),
        "git": {
            "branch": git("branch", "--show-current"),
            "commit": git("rev-parse", "HEAD"),
            "clean_before_write": not dirty,
        },
        "candidate": {
            "model": MODEL_NAME,
            "model_revision": MODEL_REVISION,
            "official_hub_sha": hub_sha,
            "base_only": True,
            "adapter": None,
        },
        "baseline": {
            "model": BASELINE_MODEL,
            "model_revision": BASELINE_REVISION,
            "artifact": args.baseline,
            "artifact_sha256": hashes["baseline"],
            "gate1_receipt": args.gate1_receipt,
            "gate1_receipt_sha256": hashes["gate1_receipt"],
        },
        "panel": {
            "items": args.items,
            "items_sha256": hashes["items"],
            "total": len(item_ids),
            "primary": len(primary_ids),
            "pilot_excluded_from_primary": len(pilot_ids),
            "split": args.split,
            "split_sha256": hashes["split"],
            "dataset_counts": dict(Counter(item.get("benchmark") for item in items)),
            "all_rows_train": all(item.get("split") == "train" for item in items),
            "validation_read": False,
            "sealed_final_read": False,
        },
        "generation": {
            "conditions": ["TAP-ref", "TAP-mut"],
            "decoding": "greedy",
            "do_sample": False,
            "max_new_tokens": 128,
            "batch": 8,
            "tokenizer_max_length": TOKENIZER_MAX_LENGTH,
            "attention_implementation": "sdpa",
            "quantization": "NF4 4-bit, shared project runtime",
            "full_raw_output_retention": True,
            "atomic_checkpoint_per_condition": True,
            "token_audit": tokens,
        },
        "acceptance_policy": acceptance_policy(),
        "resources": {"free_disk_bytes": free_disk, "gpu": gpu},
        "source_files_sha256": source_bundle(),
        "problems": problems,
    }
    output = ROOT / args.out
    output.write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": protocol["status"], "problems": problems}, indent=2))
    print(f"written: {output}")
    return 0 if not problems else 2


if __name__ == "__main__":
    raise SystemExit(main())
