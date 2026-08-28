import ast
import json
from pathlib import Path

import pytest
import torch

from engine.prompt_budget import PromptBudgetError, compact_unified_user_prompt
from engine.prompt_provenance import (
    build_repository_prompt_context,
    extract_non_gold_test_environment,
    prohibited_lineage_entries,
)
from engine.sft_trainer import CompletionOnlyDataCollator, SFTDataPoint
from engine.test_generation_prompt import (
    build_unified_user_prompt,
    format_chat_prompt,
)
from metrics.research_evaluation import (
    evaluate_candidate_slots,
    function_result,
    summarise_function_results,
)
from scripts.train_on_dataset import filter_generation_compatible_sft_examples


class _WordTokenizer:
    eos_token = "<eos>"
    pad_token_id = 0

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        assert tokenize is False and add_generation_prompt is True
        return "\n".join(item["content"] for item in messages) + "\n<assistant>"

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": text.split()}

    def pad(self, features, padding=True, return_tensors="pt"):
        width = max(len(item["input_ids"]) for item in features)
        ids, masks = [], []
        for item in features:
            size = len(item["input_ids"])
            ids.append(item["input_ids"] + [self.pad_token_id] * (width - size))
            masks.append(item["attention_mask"] + [0] * (width - size))
        return {
            "input_ids": torch.tensor(ids),
            "attention_mask": torch.tensor(masks),
        }


def test_gold_test_body_cannot_change_repository_support_context():
    first = """import pytest
CONST = 7
def helper(x):
    return x + CONST
def test_bug():
    assert helper(1) == 9
"""
    second = first.replace("assert helper(1) == 9", "assert helper(999) == -1")

    first_environment = extract_non_gold_test_environment(first)
    second_environment = extract_non_gold_test_environment(second)

    assert first_environment == second_environment
    assert "test_bug" not in first_environment[0]
    assert "helper" in first_environment[0]


def test_repository_context_has_only_allowed_field_lineage():
    context = build_repository_prompt_context(
        buggy_localized_source="import math\ndef public_api(x):\n    return x\n",
        specification="The public API should return the intended value.",
        execution_mode="repository_pytest_fragment",
        public_test_module_path="tests/test_api.py",
        buggy_test_environment_source="import pytest\ndef helper(): return 1\ndef test_api(): assert False",
    )

    assert context.localization_source in {
        "public_issue_metadata", "buggy_side_static_analysis"
    }
    assert prohibited_lineage_entries(context.field_lineage) == []
    assert "def test_api" not in context.support_context


def test_prompt_builder_rejects_oracle_only_keyword_inputs():
    with pytest.raises(TypeError):
        build_unified_user_prompt(
            code_under_test="def f(x): return x",
            execution_mode="function_assertion",
            target_symbols=["f"],
            reference_code="def f(x): return x + 1",
        )
    for forbidden in ("gold_patch", "oracle_result", "mutation_operator"):
        kwargs = {
            "code_under_test": "def f(x): return x",
            "execution_mode": "function_assertion",
            "target_symbols": ["f"],
            forbidden: "hidden",
        }
        with pytest.raises(TypeError):
            build_unified_user_prompt(**kwargs)


def test_section_compaction_preserves_required_fields_and_complete_units():
    tokenizer = _WordTokenizer()
    prompt = build_unified_user_prompt(
        code_under_test="def target(x):\n    return x\n",
        execution_mode="repository_pytest_fragment",
        specification="Return the supplied value without changing it.",
        support_context="\n\n".join(
            f"def helper_{index}(x):\n    return x + {index}" for index in range(60)
        ),
        target_symbols=["target"],
    )
    result = compact_unified_user_prompt(
        tokenizer, prompt, 300, format_chat_prompt
    )

    assert result.compacted is True
    assert result.final_token_count <= 300
    assert "### Behavioral specification" in result.user_prompt
    assert "Target symbol(s): `target`" in result.user_prompt
    assert "def target" in result.user_prompt
    assert "Return the supplied value" in result.user_prompt
    ast.parse(result.user_prompt.split("### Code under test", 1)[1].split("### Task", 1)[0])
    assert "def helper_0" not in result.user_prompt or "return x + 0" in result.user_prompt


def test_section_compaction_fails_closed_instead_of_slicing_target():
    tokenizer = _WordTokenizer()
    target = "def target(x):\n" + "\n".join(
        f"    value_{index} = x + {index}" for index in range(200)
    ) + "\n    return value_199"
    prompt = build_unified_user_prompt(
        code_under_test=target,
        execution_mode="function_assertion",
        specification="Return the computed value.",
        target_symbols=["target"],
    )
    with pytest.raises(PromptBudgetError, match="refusing token slicing"):
        compact_unified_user_prompt(tokenizer, prompt, 100, format_chat_prompt)


def test_sft_eligibility_excludes_required_target_that_cannot_fit_prompt_budget():
    tokenizer = _WordTokenizer()
    target = "def target(x):\n" + "\n".join(
        f"    value_{index} = x + {index}" for index in range(200)
    ) + "\n    return value_199"
    prompt = build_unified_user_prompt(
        code_under_test=target,
        execution_mode="function_assertion",
        specification="Return the computed value.",
        target_symbols=["target"],
    )
    example = SFTDataPoint(
        prompt=prompt,
        completion="assert target(0) == 199",
        function_id="over-budget-target",
    )

    retained, excluded = filter_generation_compatible_sft_examples(
        [example],
        tokenizer,
        max_completion_tokens=128,
        max_repository_completion_tokens=1024,
        max_prompt_tokens=100,
        max_repository_prompt_tokens=1024,
    )

    assert retained == []
    assert excluded[0]["record_id"] == "over-budget-target"
    assert excluded[0]["reason"] == "required_prompt_sections_exceed_mode_budget"


def test_completion_only_collator_masks_prompt_tokens():
    collator = CompletionOnlyDataCollator(_WordTokenizer())
    batch = collator([{
        "input_ids": [11, 12, 13, 14],
        "attention_mask": [1, 1, 1, 1],
        "completion_start": 2,
    }])

    assert batch["labels"].tolist() == [[-100, -100, 13, 14]]


def test_v4_1_ablation_dev_and_final_test_are_disjoint():
    root = Path(__file__).resolve().parent.parent
    corpus = root / "data" / "corpus" / "v4_1_research_hardened_candidate"
    splits = json.loads(corpus.joinpath("splits.json").read_text(encoding="utf-8"))
    manifest = json.loads(
        corpus.joinpath("ablation_dev_manifest.json").read_text(encoding="utf-8")
    )

    assert not (set(splits["ablation_dev"]) & set(splits["val"]))
    assert not (set(splits["ablation_dev"]) & set(splits["test"]))
    assert manifest["validation_overlap"] == 0
    assert manifest["test_overlap"] == 0


def test_kill_at_k_preserves_original_candidate_order_and_quality_layers():
    slots = [
        {"rank": 1, "parse_valid": False, "code": None},
        {"rank": 2, "parse_valid": True, "code": "assert f(0) == 1"},
        {"rank": 3, "parse_valid": True, "code": "assert f(2) == 3"},
    ]
    outcomes = evaluate_candidate_slots(
        slots,
        "def f(x):\n    return x + 1",
        "def f(x):\n    return x",
        "f",
    )
    result = function_result("f", "arithmetic", "f", outcomes, "fixture")
    summary = summarise_function_results([result], (1, 2, 4, 8))

    assert [item["rank"] for item in outcomes] == [1, 2, 3]
    assert summary["kill_at_k"]["1"]["rate"] == 0
    assert summary["kill_at_k"]["2"]["rate"] == 1
    assert summary["execution_valid_candidates"] >= summary["reference_valid_candidates"]
    assert "kill_at_k" in summary["source_metrics"]["fixture"]
