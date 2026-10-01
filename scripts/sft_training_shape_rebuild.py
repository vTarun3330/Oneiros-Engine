"""Usage: python scripts/sft_training_shape_rebuild.py <git-archive-export-of-816bbc2> <out.jsonl>

Rebuild the exact SFT training examples of checkpoints/local_sft_armA_baseline_successor_s42
with the TRAINING-COMMIT code (816bbc2, run from a git worktree), train split only.

A process audit hook HARD-BLOCKS opening any non-train corpus file (val / ablation_dev shards,
the combined records.json, splits.json, external_eval_index.json). Output: one JSON line per
unique example with its effective repeat count, plus a count summary checked against
sft_metadata.json. Writes only to the scratch directory given as argv[2].
"""
import hashlib
import json
import math
import os
from pathlib import Path
import sys

WT = Path(sys.argv[1]).resolve()
OUT = Path(sys.argv[2]).resolve()
MAIN = Path(__file__).resolve().parent.parent
CORPUS = MAIN / "data" / "corpus" / "v4_1_research_hardened_candidate"
BLOCKED = ("val.records.json", "ablation_dev.records.json", "external_eval_index.json")
opened = []


def hook(event, args):
    if event == "open" and args and isinstance(args[0], (str, bytes, os.PathLike)):
        p = os.fsdecode(args[0]).replace("\\", "/")
        if "/data/corpus/" in p:
            opened.append(p)
            name = p.rsplit("/", 1)[-1]
            if name in BLOCKED or (name in ("records.json", "splits.json")
                                   and "/development_view/" not in p):
                raise PermissionError(f"BLOCKED protected corpus file: {p}")


sys.addaudithook(hook)
os.chdir(WT)
sys.path.insert(0, str(WT))
from config import training_config  # noqa: E402
from scripts import train_on_dataset as trainer  # noqa: E402
from scripts import preflight_sft_run as pf  # noqa: E402

meta = json.loads((MAIN / "checkpoints/local_sft_armA_baseline_successor_s42/sft_run_config.json")
                  .read_text(encoding="utf-8"))
hp = meta["hyperparameters"]
trainer.PROMPT_INFORMATION_VARIANT = hp["prompt_information_variant"]
trainer.OUTPUT_INSTRUCTION_VARIANT = hp["output_instruction_variant"]
trainer.REQUIRE_SPLIT_ISOLATION = True
trainer.BASE_MODEL_NAME_OVERRIDE = meta["reproducibility"]["model_name"]
trainer.BASE_MODEL_REVISION_OVERRIDE = meta["reproducibility"]["model_revision"]
from transformers import AutoTokenizer  # noqa: E402
name, rev = trainer.resolved_base_model_identity()
tok = AutoTokenizer.from_pretrained(name, revision=rev, trust_remote_code=True,
                                    local_files_only=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

limits = {"prompt": hp["prompt_token_limit"], "selection": hp["selection_prompt_token_limit"],
          "repo_prompt": hp["repository_prompt_token_limit"],
          "completion": hp["generation_completion_token_limit"],
          "repo_completion": hp["repository_generation_completion_token_limit"],
          "config_repo_completion": training_config.sft_repository_completion_token_limit}
print("limits", limits, flush=True)
source = trainer.load_phase3_pairs(CORPUS, "train")
eligible, _ = trainer._filter_overlong_repository_completions(source)
compat = pf.compute_selection_compatibility(
    eligible, tok, selection_prompt_token_limit=limits["selection"],
    repository_prompt_token_limit=limits["repo_prompt"],
    repository_completion_token_limit=training_config.sft_repository_completion_token_limit)
selected = trainer.select_bounded_train_pairs(
    eligible, 2400, compatible_repository_ids=compat["compatible_repository_ids"],
    compatible_synthetic_ids=compat["compatible_synthetic_ids"],
    target_real_fraction=hp["real_target_fraction"], max_real_repeats=hp["max_real_repeats"],
    target_complex_fraction=hp["complex_target_fraction"])
pairs = {p["id"]: p for p in selected}
syn, repo, no_win = [], [], []
for i, pair in enumerate(selected, 1):
    mode = pair.get("execution_mode", trainer.FUNCTION_EXECUTION_MODE)
    if trainer.is_repository_execution_mode(mode):
        winners = trainer._repository_fragment_tests(pair.get("test_cases", []))
    else:
        tests = trainer.extract_dataset_tests(pair.get("test_cases", []), pair["entry_point"])
        winners, _ = trainer.evaluate_pair(tests, pair["golden_code"], pair["mutant_code"],
                                           pair["entry_point"])
    if not winners:
        no_win.append(pair["id"])
        continue
    prompt = trainer.build_pair_prompt(pair)
    (repo if trainer.is_repository_execution_mode(mode) else syn).extend(
        trainer.make_sft_data_point(pair, prompt, c) for c in winners[:3])
    if i % 400 == 0:
        print(f"{i}/{len(selected)} syn={len(syn)} repo={len(repo)}", flush=True)
syn, _ = trainer.filter_generation_compatible_sft_examples(
    syn, tok, limits["completion"], limits["repo_completion"], limits["prompt"],
    limits["repo_prompt"])
repo, _ = trainer.filter_generation_compatible_sft_examples(
    repo, tok, limits["completion"], limits["repo_completion"], limits["prompt"],
    limits["repo_prompt"])
syn, _ = trainer.deduplicate_sft_examples(syn)
repo, _ = trainer.deduplicate_sft_examples(repo)
frac = hp["real_target_fraction"]
desired = math.ceil(len(syn) * frac / (1.0 - frac))
eff_repo, sampling = trainer.balanced_repeat_examples(repo, desired, hp["max_real_repeats"],
                                                      "project")
repeats = {}
for dp in [*syn, *eff_repo]:
    k = hashlib.sha256((dp.prompt + "\x00" + dp.completion).encode()).hexdigest()
    repeats[k] = repeats.get(k, 0) + 1
with OUT.open("w", encoding="utf-8", newline="\n") as fh:
    for dp in [*syn, *repo]:
        k = hashlib.sha256((dp.prompt + "\x00" + dp.completion).encode()).hexdigest()
        p = pairs[dp.function_id]
        fh.write(json.dumps({"id": dp.function_id, "dataset": dp.dataset,
                             "execution_mode": dp.execution_mode, "project": dp.project,
                             "bug_family": dp.bug_family,
                             "complexity": (p.get("complexity") or {}).get("tier")
                             if isinstance(p.get("complexity"), dict) else p.get("complexity"),
                             "entry_point": p.get("entry_point"),
                             "completion": dp.completion, "prompt_sha256":
                             hashlib.sha256(dp.prompt.encode()).hexdigest(),
                             "prompt_tail": dp.prompt[-700:],
                             "effective_repeats": repeats[k]}) + "\n")
summary = {"selected_pairs": len(selected), "unique_synthetic": len(syn),
           "unique_repository": len(repo), "effective_repository": len(eff_repo),
           "effective_total": len(syn) + len(eff_repo), "records_without_winners": len(no_win),
           "corpus_files_opened": sorted(set(opened))}
summary["corpus_files_opened"] = sorted({p.split("/data/corpus/", 1)[1] for p in opened})
summary["examples_sha256"] = hashlib.sha256(OUT.read_bytes()).hexdigest()
summary["expected_from_sft_metadata"] = {"unique_synthetic": 5511, "unique_repository": 224,
                                         "effective_repository": 1378, "effective_total": 6889}
summary["exact_reproduction"] = all(summary[k] == v for k, v in
                                    summary["expected_from_sft_metadata"].items())
OUT.with_suffix(".summary.json").write_text(json.dumps(summary, indent=1) + "\n",
                                            encoding="utf-8")
print(json.dumps(summary, indent=1))
