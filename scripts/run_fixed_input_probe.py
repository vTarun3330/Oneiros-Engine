"""Phase 3A generation: fixed-input value prediction, base vs arm A checkpoint 431.

Generation only (no training).  Every frozen item is prompted at levels A0, A3,
A4 and the answer-schema control, with greedy decoding, for the base model and
then for the same base with the arm A adapter.  Outputs are appended to one
JSONL per arm, keyed by (item, level), so a restarted run resumes without
repeating or rewriting finished items.  Scoring is a separate CPU step
(scripts/analyze_fixed_input_probe.py).

Run durably:  python scripts/gpu_run.py start --name phase3a_fixed_input -- \
    .venv-gpu/Scripts/python.exe scripts/run_fixed_input_probe.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

COHORT = "results/sft_root_cause_phase3a_cohort.json"
DESIGN = "results/sft_root_cause_phase3a_design_receipt.json"
OUT = "results/sft_root_cause/phase3a"
MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
REVISION = "2e1fd397ee46e1388853d2af2c993145b0f1098a"
ADAPTER = "checkpoints/local_sft_armA_baseline_successor_s42/sft_adapter"
ADAPTER_SHA256 = "e67dd599a37cbf2a738791c5cdf889cb4938a2ad53c991dca2fefae503b6f9e7"
#: arm -> (model, pinned revision).  Phase 3C adds the larger base model; its
#: snapshot is local and pinned in config.IMMUTABLE_MODEL_REVISIONS.
MODELS = {"base": (MODEL, REVISION), "arm_a_431": (MODEL, REVISION),
          "qwen7b_base": ("Qwen/Qwen2.5-Coder-7B-Instruct",
                          "c03e6d358207e414f1eca0bb1891e29f1db0e242")}
MAX_NEW_TOKENS = {"prefill": 48, "control": 96}
BATCH = 8


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", default="base,arm_a_431")
    parser.add_argument("--out", default=OUT)
    args = parser.parse_args(argv)
    arms = args.arms.split(",")
    if len({MODELS[a] for a in arms}) != 1:
        raise SystemExit("REFUSED: one run loads one base model; launch other models separately")
    import torch
    from engine.generator import Phi3Generator
    from engine.test_generation_prompt import format_chat_prompt
    from harness.fixed_input_probe import CONTROL_LEVEL, LEVELS, build_prompt

    design = json.loads((ROOT / DESIGN).read_text(encoding="utf-8"))
    cohort_path = ROOT / COHORT
    if sha(cohort_path) != design["cohort"]["sha256"]:
        raise SystemExit("REFUSED: the cohort differs from the frozen design receipt")
    if sha(ROOT / ADAPTER / "adapter_model.safetensors") != ADAPTER_SHA256:
        raise SystemExit("REFUSED: the arm A adapter changed")
    cohort = json.loads(cohort_path.read_text(encoding="utf-8"))
    records = {r["id"]: r for r in json.loads((ROOT / design["record_shard"]["path"])
                                              .read_text(encoding="utf-8"))
               if r["id"] in {f["record_id"] for f in cohort["functions"]}}
    work = []
    for function in cohort["functions"]:
        record = records[function["record_id"]]
        for item in function["items"]:
            for level in (*LEVELS, CONTROL_LEVEL):
                prompt = build_prompt(record, item["call"], item["expected_repr"], level)
                work.append({"key": f"{function['record_id']}::{item['input_kind']}::{level}",
                             "level": level, **prompt})
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    model_name, revision = MODELS[arms[0]]
    generator = Phi3Generator(model_name=model_name, model_revision=revision,
                              attention_implementation="sdpa")
    generator.load_model()
    if generator.model_revision != revision:
        raise SystemExit("REFUSED: loaded revision differs from the pinned one")
    tokenizer = generator.tokenizer
    tokenizer.padding_side = "left"
    for arm in arms:
        if arm == "arm_a_431":
            generator.load_lora_adapter(ROOT / ADAPTER)
        elif arm not in MODELS:
            raise SystemExit(f"unknown arm {arm}")
        generator.model.eval()
        path = out / f"generations_{arm}.jsonl"
        done = set()
        if path.exists():
            done = {json.loads(line)["key"] for line in path.read_text(encoding="utf-8")
                    .splitlines() if line.strip()}
        todo = [w for w in work if w["key"] not in done]
        started = time.time()
        print(f"[{arm}] {len(done)} done, {len(todo)} to generate", flush=True)
        for kind in ("prefill", "control"):
            batch_items = [w for w in todo if (w["level"] == CONTROL_LEVEL) == (kind == "control")]
            for start in range(0, len(batch_items), BATCH):
                chunk = batch_items[start:start + BATCH]
                texts = [format_chat_prompt(tokenizer, w["user"]) + w["prefill"] for w in chunk]
                encoded = tokenizer(texts, return_tensors="pt", padding=True,
                                    add_special_tokens=False).to(generator.model.device)
                with torch.no_grad():
                    output = generator.model.generate(
                        **encoded, max_new_tokens=MAX_NEW_TOKENS[kind], do_sample=False,
                        pad_token_id=tokenizer.pad_token_id)
                width = encoded["input_ids"].shape[1]
                with path.open("a", encoding="utf-8") as handle:
                    for w, row, mask in zip(chunk, output, encoded["attention_mask"]):
                        new = row[width:]
                        text = tokenizer.decode(new, skip_special_tokens=True)
                        generated = int((new != tokenizer.pad_token_id).sum())
                        handle.write(json.dumps({
                            "key": w["key"], "arm": arm, "level": w["level"], "output": text,
                            "prompt_tokens": int(mask.sum()), "new_tokens": generated,
                            "hit_token_limit": generated >= MAX_NEW_TOKENS[kind]}) + "\n")
                finished = start + len(chunk)
                if finished % (BATCH * 10) == 0 or finished == len(batch_items):
                    print(f"[{arm}] heartbeat {kind} {finished}/{len(batch_items)} "
                          f"{time.time() - started:.0f}s", flush=True)
        receipt = {"arm": arm, "model": model_name, "revision": revision,
                   "adapter_sha256": ADAPTER_SHA256 if arm == "arm_a_431" else None,
                   "cohort_sha256": design["cohort"]["sha256"], "decoding": "greedy",
                   "max_new_tokens": MAX_NEW_TOKENS, "generations": len(work),
                   "output": path.relative_to(ROOT).as_posix(), "output_sha256": sha(path),
                   "seconds": round(time.time() - started, 1),
                   "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        (out / f"completion_{arm}.json").write_text(json.dumps(receipt, indent=1),
                                                    encoding="utf-8")
        print(f"[{arm}] complete {json.dumps(receipt)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
