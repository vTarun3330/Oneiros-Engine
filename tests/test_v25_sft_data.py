"""v2.5 SFT data path: exact pairs reach the actual Trainer unchanged and fully supervised;
pending, failed, duplicate and non-policy rows never enter. No weights are updated: the
integration test builds the real SFTTrainer around a tiny random CPU model and reads its own
training dataloader without taking an optimizer step."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import v25_sft_data as sd

ROOT = Path(__file__).resolve().parent.parent
R2 = ROOT / sd.R2
GOOD = {"status": "executed", "class": "semantic_kill", "fixed_valid": True,
        "rerun_agrees": True, "accepted": True, "module_sha256": "m"}
CAND = {"conversion": {"accepted": True, "module_sha256": "m"}}


def _tokenizer():
    from scripts.native_generated_tests_generate import CONTRACT
    try:
        return sd.token_counter(CONTRACT["base_model"], CONTRACT["base_revision"])
    except Exception:                                     # noqa: BLE001
        pytest.skip("base tokenizer not available locally")


# --- admission ------------------------------------------------------------------------------

def test_a_verified_positive_in_both_runs_is_admitted():
    assert sd.admit(CAND, [GOOD, dict(GOOD)], True) == []


@pytest.mark.parametrize("runs, selected, cand, fragment", [
    ([{**GOOD, "status": "pending_native_environment", "accepted": False}] * 2, True, CAND,
     "not executed"),
    ([{**GOOD, "class": "survived", "accepted": False}] * 2, True, CAND, "not a kill"),
    ([{**GOOD, "fixed_valid": False, "accepted": False}] * 2, True, CAND, "fixed revision"),
    ([GOOD, {**GOOD, "rerun_agrees": None}], True, CAND, "differ between runs"),
    ([GOOD, None], True, CAND, "missing from a repeatability run"),
    ([GOOD], True, CAND, "fewer than two"),
    ([GOOD, GOOD], False, CAND, "not chosen by the frozen selection"),
    ([GOOD, GOOD], True, {"conversion": {"accepted": False, "module_sha256": "m"}},
     "not policy-valid"),
    ([GOOD, GOOD], True, {"conversion": {"accepted": True, "module_sha256": "other"}},
     "different module"),
])
def test_pending_failed_or_non_policy_rows_are_refused(runs, selected, cand, fragment):
    problems = sd.admit(cand, runs, selected)
    assert any(fragment in p for p in problems), problems


def _pair(i, *, group="repo", repo="r1", lineage=None, function=None, canon=None,
          families=("f1",), rank=0):
    return {"pair_id": f"p{i}", "source_group": group, "repository": repo,
            "lineage": lineage or f"l{i}", "function": function or f"fn{i}",
            "canonical_test": canon or f"c{i}", "families": list(families), "rank": rank,
            "complexity": "simple"}


def test_duplicates_never_manufacture_balance_and_caps_hold():
    repo = [_pair(0, canon="same"), _pair(1, canon="same"),
            *[_pair(i, lineage="L", families=(f"f{i}",)) for i in range(2, 7)],
            *[_pair(i, function="F", families=(f"g{i}",)) for i in range(7, 12)]]
    syn = [_pair(100 + i, group=sd.SYNTHETIC_GROUP, families=(f"s{i}",)) for i in range(50)]
    mix = sd.select_mixture(repo, syn)
    report = sd.mixture_report(mix["pairs"])
    assert report["duplicate_canonical_tests"] == 0
    assert report["max_per_lineage"] <= 3 and report["max_per_function"] <= 3
    assert report["synthetic_pairs"] <= report["repository_pairs"]
    reasons = {d["reason"] for d in mix["dropped"]}
    assert {"duplicate canonical test", "lineage cap", "function cap",
            "synthetic replay budget"} <= reasons
    assert sd.select_mixture(repo, syn) == mix                   # deterministic


def test_family_cap_drops_synthetic_first():
    repo = [_pair(i, families=("a",)) for i in range(4)] + \
           [_pair(i, families=(f"b{i}",)) for i in range(4, 10)]
    syn = [_pair(100 + i, group=sd.SYNTHETIC_GROUP, families=("a",)) for i in range(10)]
    mix = sd.select_mixture(repo, syn)
    assert all(v <= 0.35 for v in sd.family_share(mix["pairs"]).values())
    assert any(d["reason"].startswith("family cap") for d in mix["dropped"])


def test_gate_fails_below_150_8_60():
    report = sd.mixture_report([_pair(i) for i in range(10)])
    assert report["training_gate_passed"] is False


def test_token_fit_never_truncates():
    assert sd.fit_problems({"prompt": 2049, "completion": 10, "combined": 2059})
    assert sd.fit_problems({"prompt": 1000, "completion": 1025, "combined": 2025})
    assert not sd.fit_problems({"prompt": 2048, "completion": 1024, "combined": 3072})


# --- trainer integration --------------------------------------------------------------------

def _real_selected_pair():
    if not (R2 / "selection_view_synthetic.jsonl").is_file():
        pytest.skip("local stage1_r2 artifacts not present")
    pairs, _ = sd.synthetic_pairs(R2)
    assert pairs, "no admitted synthetic pair"
    return pairs[0]


def _tiny_trainer(tmp_path, tok):
    import torch
    from transformers import Qwen2Config, Qwen2ForCausalLM
    from engine.sft_trainer_v25 import V25SFTTrainer
    trainer = V25SFTTrainer(model_name="Qwen/Qwen2.5-Coder-1.5B-Instruct",
                                output_dir=tmp_path, max_prompt_tokens=1024,
                                max_repository_prompt_tokens=1024, max_completion_tokens=1024,
                                max_repository_completion_tokens=1024)
    trainer.tokenizer = tok
    trainer.use_bf16 = False
    torch.manual_seed(0)
    trainer.model = Qwen2ForCausalLM(Qwen2Config(
        vocab_size=len(tok), hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=4096))
    return trainer


def test_a_real_selected_completion_reaches_the_actual_trainer_unchanged(tmp_path):
    from engine.sft_trainer import IGNORE_INDEX
    from engine.test_generation_prompt import format_chat_prompt
    tok, count = _tokenizer()
    pair = _real_selected_pair()
    trainer = _tiny_trainer(tmp_path, tok)
    dataset = trainer.prepare_dataset(sd.to_datapoints([pair]))
    hf = trainer._sft_trainer(dataset, trainer._sft_config({"epochs": 1, "optimizer_steps": 1}, use_cpu=True))
    batch = next(iter(hf.get_train_dataloader()))
    ids, labels = batch["input_ids"][0].tolist(), batch["labels"][0].tolist()
    supervised = [t for t, l in zip(ids, labels) if l != IGNORE_INDEX]
    assert supervised == [l for l in labels if l != IGNORE_INDEX]
    assert tok.decode(supervised) == pair["completion"] + tok.eos_token   # byte for byte
    prompt_ids = [t for t, l in zip(ids, labels) if l == IGNORE_INDEX]
    assert tok.decode(prompt_ids) == format_chat_prompt(tok, pair["prompt"])
    assert len(supervised) == count(pair)["completion"]
    assert "pytest_module_v1" in pair["prompt"] and pair["completion"].startswith("from ")


def test_trailing_whitespace_is_kept_and_leading_whitespace_refused(tmp_path):
    from engine.sft_trainer import IGNORE_INDEX
    tok, _ = _tokenizer()
    trainer = _tiny_trainer(tmp_path, tok)
    pair = {"pair_id": "x", "prompt": "### TEST\nExpected test format: pytest_module_v1",
            "completion": "from m import f\n\n\ndef test_f():\n    assert f() == 1\n\n",
            "repository": "r", "families": ["f"], "function": "fn",
            "source_group": sd.SYNTHETIC_GROUP, "dataset": "d"}
    record = trainer.prepare_dataset(sd.to_datapoints([pair]))[0]
    labels = [t for t in record["input_ids"][record["completion_start"]:]]
    assert tok.decode(labels) == pair["completion"] + tok.eos_token
    with pytest.raises(ValueError, match="starts with whitespace"):
        trainer.prepare_dataset(sd.to_datapoints([{**pair, "completion": " " + pair[
            "completion"]}]))
    assert IGNORE_INDEX == -100


def test_an_over_budget_prompt_refuses_and_is_never_compacted(tmp_path):
    tok, _ = _tokenizer()
    trainer = _tiny_trainer(tmp_path, tok)
    trainer.max_prompt_tokens = 20
    pair = {"pair_id": "x", "prompt": "word " * 200, "completion": "from m import f\n",
            "repository": "r", "families": ["f"], "function": "fn",
            "source_group": sd.SYNTHETIC_GROUP, "dataset": "d"}
    with pytest.raises(ValueError, match="never compacted"):
        trainer.prepare_dataset(sd.to_datapoints([pair]))


def test_mixed_task_kinds_are_refused(tmp_path):
    from engine.sft_trainer import SFTDataPoint
    tok, _ = _tokenizer()
    trainer = _tiny_trainer(tmp_path, tok)
    pair = {"pair_id": "x", "prompt": "p", "completion": "from m import f\n",
            "repository": "r", "families": ["f"], "function": "fn",
            "source_group": sd.SYNTHETIC_GROUP, "dataset": "d"}
    with pytest.raises(ValueError, match="cannot mix"):
        trainer.prepare_dataset(sd.to_datapoints([pair]) + [SFTDataPoint(prompt="p",
                                                                          completion="c")])


def test_family_cap_removal_never_leaves_excess_replay():
    repo = [_pair(0, families=("only",))]
    syn = [_pair(100, group=sd.SYNTHETIC_GROUP, families=("s",))]
    report = sd.mixture_report(sd.select_mixture(repo, syn)["pairs"])
    assert report["synthetic_pairs"] <= report["repository_pairs"]


def test_addendum2_exposure_never_cycles_a_small_corpus():
    from engine.sft_trainer_v25 import plan_v25_exposure
    small = plan_v25_exposure(300, 69_000)
    assert small["epochs"] == 3 and small["optimizer_steps"] == 3 * 19 == 57
    assert small["total_target_tokens_upper_bound"] <= 570_775
    big = plan_v25_exposure(6_889, 570_775)
    assert big["epochs"] == 1 and big["optimizer_steps"] == 431
    heavy = plan_v25_exposure(1_000, 250_000)
    assert heavy["epochs"] == 2                                   # token budget binds
    with pytest.raises(ValueError):
        plan_v25_exposure(10, 600_000)
    huge = plan_v25_exposure(4_000, 100_000)
    assert huge["optimizer_steps"] == 431 and huge["epochs"] == 3  # 431 ceiling binds


def test_trainer_refuses_a_plan_for_another_dataset_or_cycling(tmp_path):
    tok, _ = _tokenizer()
    trainer = _tiny_trainer(tmp_path, tok)
    pair = {"pair_id": "x", "prompt": "p", "completion": "from m import f\n",
            "repository": "r", "families": ["f"], "function": "fn",
            "source_group": sd.SYNTHETIC_GROUP, "dataset": "d"}
    with pytest.raises(ValueError, match="different dataset"):
        trainer.train(sd.to_datapoints([pair]), plan={"unique_examples": 2, "epochs": 1,
                                                      "optimizer_steps": 1})
    with pytest.raises(ValueError, match="beyond its frozen epochs"):
        trainer.train(sd.to_datapoints([pair]), plan={"unique_examples": 1, "epochs": 1,
                                                      "optimizer_steps": 5})
