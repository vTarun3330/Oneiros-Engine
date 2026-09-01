from dataclasses import replace

import pytest

from metrics.research_evaluation import function_result, summarise_function_results
from scripts import audit_dataset_sampling as audit
from scripts.preflight_sft_run import _make_data_point
from scripts.train_on_dataset import (
    _record_to_pair, balanced_repeat_examples, build_pair_prompt, make_sft_data_point,
    summarize_train_pair_selection,
)
from utils.dataset_identity import DATASET_IDENTITY_POLICY, dataset_name_for_pair, dataset_name_from_source
from utils.sampling_audit import summarize_sampling_weights


def _record(dataset, index=0):
    return {
        "id": f"record-{index}", "task_type": "hidden_mutation_reproduction",
        "source": {"name": "oneiros_clean_mutations", "upstream": dataset},
        "entry_point": "f", "specification": "Return the successor.",
        "code_under_test": "def f(x): return x", "reference_code": "def f(x): return x + 1",
        "tests": [{"code": "assert f(1) == 2"}],
        "provenance": {"mutation_type": "arithmetic"},
        "quality": {"execution_mode": "function_assertion"},
    }


@pytest.mark.parametrize("source,expected", [
    ({"name": "oneiros_clean_mutations", "upstream": "humaneval"}, "humaneval"),
    ({"name": "oneiros_clean_mutations", "upstream": "mbpp"}, "mbpp"),
    ({"name": "bugsinpy_official_repository_fragment", "upstream": "BugsInPy"}, "BugsInPy"),
    ({"name": "official", "upstream": "SWE-bench Verified"}, "SWE-bench Verified"),
    ({"name": "manual_curated_examples"}, "manual_curated_examples"),
    ({"name": "fallback", "upstream": "  "}, "fallback"),
    ({"name": None, "upstream": []}, "unknown"),
    ("fixture", "fixture"), (None, "unknown"),
])
def test_upstream_dataset_label_resolution(source, expected):
    assert dataset_name_from_source(source) == expected


def test_labels_are_shared_by_training_and_preflight_but_never_change_prompts():
    mbpp = _record_to_pair(_record("mbpp"))
    humaneval = _record_to_pair(_record("humaneval"))
    assert mbpp["source_name"] == humaneval["source_name"] == "oneiros_clean_mutations"
    assert mbpp["dataset_name"] == "mbpp"
    assert humaneval["dataset_name"] == "humaneval"
    assert mbpp["dataset_identity_policy"] == DATASET_IDENTITY_POLICY
    assert build_pair_prompt(mbpp) == build_pair_prompt(humaneval)
    assert _make_data_point is make_sft_data_point
    point = _make_data_point(mbpp, "prompt", "completion")
    assert point.dataset == "mbpp"
    assert point.dataset_family == "mbpp::arithmetic"
    selection = summarize_train_pair_selection([mbpp, humaneval])
    assert selection["dataset_counts"] == {"humaneval": 1, "mbpp": 1}
    assert selection["ingestion_source_counts"] == {"oneiros_clean_mutations": 2}


def test_no_dataset_inference_from_record_ids_or_code():
    assert dataset_name_for_pair({"id": "mbpp_1", "mutant_code": "humaneval"}) == "unknown"
    assert dataset_name_for_pair({"source_name": "legacy_fixture"}) == "legacy_fixture"
    assert dataset_name_for_pair({"source": {}, "dataset_name": "guessed"}) == "unknown"


def _examples():
    return [make_sft_data_point(
        _record_to_pair(_record("mbpp" if index < 4 else "humaneval", index)),
        "prompt", f"completion-{index}",
    ) for index in range(5)]


def test_weight_report_detects_saturated_balancing_that_does_not_rebalance():
    examples = _examples()
    doubled, _ = balanced_repeat_examples(examples, 10, 2, "dataset")
    report = summarize_sampling_weights(examples, doubled)
    assert report["dataset_weights_changed"] is False
    assert report["per_dataset"]["mbpp"]["raw_weight"] == 0.8
    assert report["per_dataset"]["mbpp"]["effective_weight"] == 0.8
    assert report["per_dataset"]["humaneval"]["effective_examples"] == 2
    assert report["unique_examples"] == 5
    assert report["effective_examples"] == 10
    assert report["repeat_histogram"] == {"2": 5}
    assert report["extra_repetitions"] == 5


def test_non_saturated_fixture_reports_actual_weights_not_equal_weight_claim():
    examples = _examples()
    balanced, _ = balanced_repeat_examples(examples, 8, 3, "dataset")
    report = summarize_sampling_weights(examples, balanced)
    assert report["dataset_weights_changed"] is True
    assert report["per_dataset"]["humaneval"]["effective_weight"] == 0.375
    assert report["per_dataset"]["mbpp"]["effective_weight"] == 0.625
    assert sum(row["effective_examples"] for row in report["per_dataset"].values()) == 8
    assert sum(row["effective_examples"] for row in report["per_mutation_family"].values()) == 8
    assert report["max_repeat_count"] == 3
    assert report["weight_scope"].endswith("before_optimizer_padding")


def test_sampling_audit_rejects_new_examples_and_changed_labels():
    examples = _examples()
    with pytest.raises(ValueError, match="outside the raw pool"):
        summarize_sampling_weights(examples, [replace(examples[0], function_id="other")])
    with pytest.raises(ValueError, match="conflicting"):
        summarize_sampling_weights(examples, [replace(examples[0], dataset="other")])
    assert summarize_sampling_weights([], [])["effective_examples"] == 0


def _result(index, dataset, killed):
    return function_result(
        f"r{index}", "arithmetic", "f", [{
            "rank": 1, "code": "assert f(1) == 2", "parse_valid": True,
            "execution_valid": True, "reference_valid": True, "killed": killed,
        }], source_name="oneiros_clean_mutations", dataset_name=dataset,
    )


def test_dataset_macro_separates_benchmarks_with_the_same_ingestion_source():
    records = [_result(index, "mbpp", True) for index in range(9)]
    records.append(_result(9, "humaneval", False))
    result = summarise_function_results(records)
    assert result["function_kill_rate"] == 0.9
    assert result["equal_weight_source_macro"]["function_kill_rate"] == 0.9
    assert result["equal_weight_dataset_macro"]["function_kill_rate"] == 0.5
    assert result["equal_weight_dataset_macro"]["kill_at_k"]["1"] == 0.5
    assert result["equal_weight_dataset_macro"]["dataset_count"] == 2
    assert result["equal_weight_dataset_macro"]["status"] == "complete"
    assert result["dataset_metrics"]["mbpp"]["functions"] == 9


def test_missing_legacy_dataset_labels_do_not_get_silently_guessed_or_dropped():
    result = summarise_function_results([_result(0, "mbpp", True), _result(1, None, False)])
    macro = result["equal_weight_dataset_macro"]
    assert macro["status"] == "incomplete_dataset_labels"
    assert macro["unlabeled_functions"] == 1
    assert macro["function_kill_rate"] is None
    assert macro["kill_at_k"]["8"] is None
    assert result["function_validation_records"] == 2
    assert "unknown" in result["dataset_metrics"]


def test_corpus_label_audit_only_selects_training_records(tmp_path, monkeypatch):
    calls = []
    def load_pairs(path, split):
        calls.append(split)
        return [_record_to_pair(_record("mbpp", 0)), _record_to_pair(_record("humaneval", 1))]
    monkeypatch.setattr(audit, "load_phase3_pairs", load_pairs)
    monkeypatch.setattr(audit, "verify_corpus", lambda path: {
        "corpus_id": "fixture", "files": {"records.json": {"sha256": "fixture-hash"}},
    })
    monkeypatch.setattr(audit, "source_tree_sha256", lambda path: "source-hash")
    report = audit.build_audit(tmp_path)
    assert calls == ["train"]
    assert report["records_by_upstream_dataset"] == {"humaneval": 1, "mbpp": 1}
    assert report["sources_that_would_collapse_multiple_datasets"] == ["oneiros_clean_mutations"]
    assert report["labels_complete"] is True
    for row in report["sampling_diagnostic"]["treatments"].values():
        assert row["weights"]["dataset_weights_changed"] is False
        assert row["weights"]["repeat_histogram"] == {"2": 10}
