"""The O1 completion must reach the trainer unchanged, and change nothing else.

Two failures are guarded here, and both have real precedent in this project.

* Training on the corpus golden while reporting corrections.
  ``apply_relearning_round`` never reads a correction's completion; it appends
  the record's ordinary training pair. A sidecar routed through it trains on
  goldens and reports itself as correction supervision.
* Delivering the supervision to a sliver of the batch and calling it an arm.
  The multi-mutant run reached 76 of 800 selected pairs.

So: the exact bytes are asserted, and an absent sidecar is asserted to leave
the baseline byte-identical.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from harness.o1_sidecar import (
    SCHEMA, append_o1_examples, load_o1_sidecar,
)


class Example:
    """Stands in for SFTDataPoint; only ``completion`` matters here."""

    def __init__(self, prompt: str, completion: str, function_id: str = "x"):
        self.prompt = prompt
        self.completion = completion
        self.function_id = function_id

    def __eq__(self, other):
        return (self.prompt, self.completion, self.function_id) == (
            other.prompt, other.completion, other.function_id)

    def __repr__(self):
        return f"Example({self.function_id!r}, {self.completion!r})"


def _make(pair, prompt, completion):
    return Example(prompt, completion, pair["id"])


def _prompt(pair):
    return f"PROMPT<{pair['id']}>"


CORRECTED = "assert match_command('   command', 'command') == True"


def _row(record_id="mutation::rec_1", completion=CORRECTED, **over):
    row = {"record_id": record_id, "completion": completion, "repeats": 1,
           "verified": True, "kills_displayed_target": True,
           "source_dataset": "mbpp", "origin": "synthetic"}
    row.update(over)
    return row


def _pairs(*ids):
    return {value: {"id": value, "execution_mode": "function_assertion"}
            for value in ids}


def _sidecar_dir(tmp_path: Path, rows, **manifest_over):
    directory = tmp_path / "sidecar"
    directory.mkdir()
    path = directory / "train.sidecar.json"
    path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": SCHEMA,
        "sidecar_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "evaluation_split": "train",
        "sealed_final_test_accessed": False,
        "all_records_in_train_shard": True,
    }
    manifest.update(manifest_over)
    (directory / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return directory


# ------------------------------------- the completion arrives unchanged

def test_a_known_corrected_completion_reaches_the_trainer_byte_for_byte():
    baseline = [Example("p", "assert f(1) == 1", "mutation::other")]
    combined, report = append_o1_examples(
        baseline, [_row()], _pairs("mutation::rec_1"),
        make_data_point=_make, build_prompt=_prompt)
    assert report["appended_examples"] == 1
    appended = combined[-1]
    assert appended.completion == CORRECTED
    assert hashlib.sha256(appended.completion.encode()).hexdigest() == \
        hashlib.sha256(CORRECTED.encode()).hexdigest()


def test_the_corpus_golden_is_never_substituted():
    """The relearning path's actual behaviour, asserted not to happen here."""
    golden = "assert match_command('x', 'y') == False"
    pairs = _pairs("mutation::rec_1")
    pairs["mutation::rec_1"]["golden_code"] = golden
    pairs["mutation::rec_1"]["test_cases"] = [golden]
    combined, _ = append_o1_examples(
        [], [_row()], pairs, make_data_point=_make, build_prompt=_prompt)
    assert combined[0].completion == CORRECTED
    assert combined[0].completion != golden


def test_the_report_states_no_golden_was_substituted():
    _, report = append_o1_examples(
        [], [_row()], _pairs("mutation::rec_1"),
        make_data_point=_make, build_prompt=_prompt)
    assert report["golden_substituted"] is False
    assert "verbatim" in report["completion_source"]


# ---------------------------------- no sidecar means arm B equals arm A

def test_an_empty_sidecar_leaves_the_baseline_identical():
    baseline = [Example(f"p{i}", f"assert f({i}) == {i}", f"rec_{i}")
                for i in range(25)]
    combined, report = append_o1_examples(
        baseline, [], _pairs(), make_data_point=_make, build_prompt=_prompt)
    assert combined == baseline
    assert [e.completion for e in combined] == [e.completion for e in baseline]
    assert report["appended_examples"] == 0
    assert report["combined_examples"] == len(baseline)


def test_the_baseline_prefix_is_untouched_when_rows_are_appended():
    """Arm B must be arm A plus rows, not arm A rebalanced."""
    baseline = [Example(f"p{i}", f"assert f({i}) == {i}", f"rec_{i}")
                for i in range(40)]
    original = list(baseline)
    combined, _ = append_o1_examples(
        baseline, [_row(), _row("mutation::rec_2", "assert g(2) == 2")],
        _pairs("mutation::rec_1", "mutation::rec_2"),
        make_data_point=_make, build_prompt=_prompt)
    assert combined[:len(original)] == original
    assert len(combined) == len(original) + 2
    assert baseline == original, "the caller's baseline list was mutated"


# ------------------------------------------------- no repeats, no dupes

def test_a_completion_already_in_the_baseline_is_not_appended_again():
    baseline = [Example("p", CORRECTED, "mutation::rec_9")]
    combined, report = append_o1_examples(
        baseline, [_row()], _pairs("mutation::rec_1"),
        make_data_point=_make, build_prompt=_prompt)
    assert len(combined) == 1
    assert report["dropped"]["duplicate_completion"] == 1


def test_two_sidecar_rows_with_the_same_completion_collapse_to_one():
    rows = [_row("mutation::rec_1"), _row("mutation::rec_2")]
    combined, report = append_o1_examples(
        [], rows, _pairs("mutation::rec_1", "mutation::rec_2"),
        make_data_point=_make, build_prompt=_prompt)
    assert len(combined) == 1
    assert report["dropped"]["duplicate_completion"] == 1
    assert report["repeats_per_row"] == 1


def test_a_record_with_no_training_pair_is_reported_not_silently_skipped():
    """The multi-mutant failure: supervision that never reaches the batch."""
    combined, report = append_o1_examples(
        [], [_row("mutation::absent")], _pairs("mutation::rec_1"),
        make_data_point=_make, build_prompt=_prompt)
    assert combined == []
    assert report["records_not_in_training_pairs_count"] == 1
    assert "mutation::absent" in report["records_not_in_training_pairs"]


def test_a_row_over_the_token_budget_is_refused_not_truncated():
    combined, report = append_o1_examples(
        [], [_row()], _pairs("mutation::rec_1"),
        make_data_point=_make, build_prompt=_prompt,
        completion_fits=lambda pair, prompt, completion: False)
    assert combined == []
    assert report["dropped"]["token_budget"] == 1
    assert report["token_budget_rejections"][0]["record_id"] == "mutation::rec_1"


# ------------------------------------------------------- loader refusals

def test_a_tampered_sidecar_is_refused(tmp_path):
    directory = _sidecar_dir(tmp_path, [_row()])
    (directory / "train.sidecar.json").write_text(
        json.dumps([_row(completion="assert wrong()")]), encoding="utf-8")
    with pytest.raises(ValueError, match="not the rows that were verified"):
        load_o1_sidecar(directory)


def test_a_non_train_sidecar_is_refused(tmp_path):
    directory = _sidecar_dir(tmp_path, [_row()], evaluation_split="val")
    with pytest.raises(ValueError, match="only train-split"):
        load_o1_sidecar(directory)


def test_a_sidecar_not_denying_sealed_access_is_refused(tmp_path):
    directory = _sidecar_dir(tmp_path, [_row()],
                             sealed_final_test_accessed=True)
    with pytest.raises(ValueError, match="sealed-test"):
        load_o1_sidecar(directory)


def test_a_sidecar_not_asserting_train_shard_membership_is_refused(tmp_path):
    directory = _sidecar_dir(tmp_path, [_row()],
                             all_records_in_train_shard=False)
    with pytest.raises(ValueError, match="train shard"):
        load_o1_sidecar(directory)


def test_an_unverified_row_is_refused(tmp_path):
    directory = _sidecar_dir(tmp_path, [_row(verified=False)])
    with pytest.raises(ValueError, match="never a label"):
        load_o1_sidecar(directory)


def test_a_row_asking_for_repeats_is_refused(tmp_path):
    directory = _sidecar_dir(tmp_path, [_row(repeats=3)])
    with pytest.raises(ValueError, match="never repeats"):
        load_o1_sidecar(directory)


def test_an_empty_completion_is_refused(tmp_path):
    directory = _sidecar_dir(tmp_path, [_row(completion="   ")])
    with pytest.raises(ValueError, match="empty completion"):
        load_o1_sidecar(directory)


def test_a_wrong_schema_is_refused(tmp_path):
    directory = _sidecar_dir(tmp_path, [_row()], schema_version="something")
    with pytest.raises(ValueError, match="schema"):
        load_o1_sidecar(directory)


def test_a_clean_sidecar_loads(tmp_path):
    directory = _sidecar_dir(tmp_path, [_row(), _row("mutation::rec_2", "assert g()")])
    rows, manifest = load_o1_sidecar(directory)
    assert len(rows) == 2
    assert rows[0]["completion"] == CORRECTED
    assert manifest["evaluation_split"] == "train"


# --------------------------------------------- the trainer wiring itself

def test_the_trainer_exposes_o1_sidecar_and_defaults_it_off():
    from scripts import train_on_dataset as trainer
    assert hasattr(trainer, "O1_SIDECAR_PATH")
    assert trainer.O1_SIDECAR_PATH is None, (
        "a run with no --o1-sidecar must be arm A exactly")


def test_the_trainer_flag_exists_and_is_not_the_other_two_paths():
    source = Path(__file__).resolve().parent.parent / "scripts" / "train_on_dataset.py"
    text = source.read_text(encoding="utf-8")
    assert '"--o1-sidecar"' in text
    assert "O1_SIDECAR_PATH = args.o1_sidecar" in text
    # O1 must not be routed through either existing mechanism.
    assert "append_o1_examples" in text
    o1_block = text.split("if O1_SIDECAR_PATH:", 1)[1][:2000]
    assert "MULTI_MUTANT_COMPLETIONS" not in o1_block
    assert "apply_relearning_round" not in o1_block


def test_the_real_sft_data_point_carries_the_completion_verbatim():
    """Integration with the actual SFTDataPoint the trainer builds."""
    from scripts.train_on_dataset import make_sft_data_point
    pair = {"id": "mutation::rec_1", "project": "synthetic",
            "bug_family": "comparison", "group_id": "g",
            "execution_mode": "function_assertion",
            "source": {"name": "mbpp"}}
    point = make_sft_data_point(pair, "PROMPT", CORRECTED)
    assert point.completion == CORRECTED
    assert point.function_id == "mutation::rec_1"
