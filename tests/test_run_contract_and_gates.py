"""Generation settings must be binding, not merely recorded.

A manifest says what a run claimed. The resume contract decides what a run may
inherit. Before this, two runs differing only in sampling temperature produced
interchangeable checkpoints, because temperature was recorded nowhere at all -
so a resumed run could silently be a different experiment.
"""
from __future__ import annotations

import json

import pytest

from scripts.sequence_fit_gate import FUNCTION_MODE, execution_mode_of
from scripts.verify_successor_generation import REQUIRED_CONTRACT, verify

#: Every setting that changes what the model emits or how it is scored.
MUST_BE_BOUND = (
    "candidate_parse_mode",
    "retain_raw_output",
    "allow_test_function_candidates",
    "temperature",
    "top_p",
    "max_new_tokens",
    "candidates_per_function",
    "prompt_token_limit",
    "rendered_sequence_fit_policy",
    "base_model_name",
    "base_model_revision",
    "evaluator_source_sha256",
    "candidate_policy_source_sha256",
)


def test_the_contract_source_binds_every_generation_defining_setting():
    """Read the source rather than run it: importing needs a GPU stack."""
    from pathlib import Path
    source = (Path(__file__).resolve().parent.parent
              / "scripts" / "train_on_dataset.py").read_text(encoding="utf-8")
    block = source.split("run_contract = {", 1)[1].split("}", 1)[0]
    for field in MUST_BE_BOUND:
        assert f'"{field}"' in block, f"{field} is not bound into the run contract"


def test_the_contract_is_inside_the_resume_identity_not_beside_it():
    """If it is not in `context`, resume does not compare it."""
    from pathlib import Path
    source = (Path(__file__).resolve().parent.parent
              / "scripts" / "train_on_dataset.py").read_text(encoding="utf-8")
    returned = source.split("def _adapter_evaluation_context", 1)[1]
    returned = returned.split("return {", 1)[1].split("\n\n", 1)[0]
    assert '"run_contract": run_contract' in returned
    assert '"run_contract_sha256"' in returned


def test_resume_compares_the_whole_context():
    """The property the contract relies on."""
    from pathlib import Path
    source = (Path(__file__).resolve().parent.parent
              / "scripts" / "train_on_dataset.py").read_text(encoding="utf-8")
    assert 'if payload.get("context") != context:' in source


# --------------------------------------------------------------------------
# Sequence-fit gate
# --------------------------------------------------------------------------

def test_repository_records_are_not_gated_as_function_records():
    """They are held out of generation, so gating on them fails wrongly."""
    repository = {"quality": {"execution_mode": "repository_pytest_fragment"}}
    assert execution_mode_of(repository) != FUNCTION_MODE
    assert execution_mode_of({"quality": {}}) == FUNCTION_MODE
    assert execution_mode_of({}) == FUNCTION_MODE


def test_the_committed_gate_report_passed_on_the_generation_panel():
    from pathlib import Path
    report_path = (Path(__file__).resolve().parent.parent / "results"
                   / "v4_2_sequence_fit_gate_train_successor.json")
    if not report_path.exists():
        pytest.skip("gate not run in this checkout")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["gate_passed"] is True
    assert report["failure_count"] == 0
    assert report["generation_panel_size"] == 5595
    assert report["repository_records_held_not_generated_on"] == 457
    assert (report["max_rendered_prompt_plus_completion"]
            <= report["sequence_limit"])
    assert report["sealed_final_test_accessed"] is False


# --------------------------------------------------------------------------
# Successor verification
# --------------------------------------------------------------------------

def _artifact(tmp_path, contract=None, outcomes=None, **overrides):
    body = {
        "evaluation_split": "train",
        "final_test_measurement": False,
        "run_contract": dict(REQUIRED_CONTRACT) if contract is None else contract,
        "run_contract_sha256": "x",
        "function_validation_records": 1,
        "prompt_budget_failed_functions": 0,
        "function_results": [{"record_id": "r1",
                              "candidate_outcomes": outcomes or []}],
    }
    body.update(overrides)
    path = tmp_path / "successor.json"
    path.write_text(json.dumps(body), encoding="utf-8")
    return path


def _raw(text):
    import hashlib
    return {"raw_output": text,
            "raw_output_sha256": hashlib.sha256(text.encode()).hexdigest()}


def test_a_legacy_contract_is_rejected(tmp_path):
    contract = dict(REQUIRED_CONTRACT)
    contract["candidate_parse_mode"] = "first_assertion"
    contract["retain_raw_output"] = False
    path = _artifact(tmp_path, contract=contract, outcomes=[_raw("assert f(1)==1")])
    report = verify(path, "Qwen/Qwen2.5-Coder-1.5B-Instruct", "main")
    assert report["verified"] is False
    assert any("candidate_parse_mode" in p for p in report["problems"])
    assert any("retain_raw_output" in p for p in report["problems"])


def test_an_artifact_with_no_contract_at_all_is_rejected(tmp_path):
    path = _artifact(tmp_path, outcomes=[_raw("assert f(1)==1")])
    body = json.loads(path.read_text(encoding="utf-8"))
    del body["run_contract"]
    path.write_text(json.dumps(body), encoding="utf-8")
    report = verify(path, "Qwen/Qwen2.5-Coder-1.5B-Instruct", "main")
    assert report["verified"] is False
    assert any("predates the immutable contract" in p for p in report["problems"])


def test_a_tampered_raw_output_is_caught(tmp_path):
    outcome = _raw("assert f(1) == 1")
    outcome["raw_output"] = "assert f(1) == 2"   # text changed, hash not
    path = _artifact(tmp_path, outcomes=[outcome])
    report = verify(path, "Qwen/Qwen2.5-Coder-1.5B-Instruct", "main")
    assert report["raw_output_hash_mismatches"] == 1
    assert report["verified"] is False


def test_missing_raw_output_is_rejected_despite_the_flag(tmp_path):
    path = _artifact(tmp_path, outcomes=[{"raw_output_sha256": "abc"}])
    report = verify(path, "Qwen/Qwen2.5-Coder-1.5B-Instruct", "main")
    assert report["verified"] is False
    assert any("no candidate carries raw output" in p for p in report["problems"])


def test_prompt_budget_failures_are_reported_as_truncation(tmp_path):
    path = _artifact(tmp_path, outcomes=[_raw("assert f(1)==1")],
                     prompt_budget_failed_functions=3)
    report = verify(path, "Qwen/Qwen2.5-Coder-1.5B-Instruct", "main")
    assert report["no_runtime_token_truncation"] is False
    assert report["verified"] is False


def test_an_incomplete_artifact_is_rejected(tmp_path):
    path = _artifact(tmp_path, outcomes=[_raw("assert f(1)==1")],
                     function_validation_records=5595)
    report = verify(path, "Qwen/Qwen2.5-Coder-1.5B-Instruct", "main")
    assert report["verified"] is False
    assert any("incomplete" in p for p in report["problems"])


def test_the_sealed_split_is_refused(tmp_path):
    path = _artifact(tmp_path, outcomes=[_raw("assert f(1)==1")],
                     evaluation_split="test")
    with pytest.raises(SystemExit, match="sealed"):
        verify(path, "Qwen/Qwen2.5-Coder-1.5B-Instruct", "main")


# --------------------------------------------------------------------------
# Completion-truncation blocking. Predeclared before the run produced numbers.
# --------------------------------------------------------------------------

def _long(tokens=1200):
    return "assert f(" + ("1, " * tokens) + "1) == 1"


def test_outputs_at_the_completion_limit_are_flagged_and_block_the_build(tmp_path):
    outcomes = [_raw(_long()) for _ in range(10)]
    path = _artifact(tmp_path, outcomes=outcomes)
    report = verify(path, "Qwen/Qwen2.5-Coder-1.5B-Instruct", "main")
    assert report["suspected_completion_truncation"] == 10
    assert report["outputs_reaching_completion_limit_share"] == 1.0
    assert report["completion_limit_threshold_exceeded"] is True
    assert report["verified"] is False
    assert any("completion" in p and "blocked" in p for p in report["problems"])


def test_truncated_candidates_are_ineligible_for_oracle_labels(tmp_path):
    path = _artifact(tmp_path, outcomes=[_raw(_long()), _raw("assert f(1) == 1")])
    report = verify(path, "Qwen/Qwen2.5-Coder-1.5B-Instruct", "main")
    assert report["candidates_ineligible_for_oracle_labels"] == 1
    assert "suspected_completion_truncation" in report["ineligible_reasons"]


def test_unparseable_outputs_are_also_ineligible(tmp_path):
    path = _artifact(tmp_path, outcomes=[_raw("assert f(1 =="), _raw("assert f(1) == 1")])
    report = verify(path, "Qwen/Qwen2.5-Coder-1.5B-Instruct", "main")
    assert report["unparseable_outputs"] == 1
    assert report["candidates_ineligible_for_oracle_labels"] == 1
    assert "unparseable_raw_output" in report["ineligible_reasons"]


def test_a_truncated_and_unparseable_output_is_counted_once(tmp_path):
    """Truncation usually CAUSES the parse failure; double-counting would
    overstate how much data is unusable."""
    path = _artifact(tmp_path, outcomes=[_raw("assert f(" + ("1, " * 1200))])
    report = verify(path, "Qwen/Qwen2.5-Coder-1.5B-Instruct", "main")
    assert report["candidates_ineligible_for_oracle_labels"] == 1


def test_a_clean_run_below_the_threshold_verifies(tmp_path):
    outcomes = [_raw(f"assert f({i}) == {i}") for i in range(100)]
    path = _artifact(tmp_path, outcomes=outcomes)
    report = verify(path, "Qwen/Qwen2.5-Coder-1.5B-Instruct", "main")
    assert report["completion_limit_threshold_exceeded"] is False
    assert report["candidates_ineligible_for_oracle_labels"] == 0
    assert report["verified"] is True, report["problems"]


def test_the_threshold_is_predeclared_as_a_constant():
    from scripts.verify_successor_generation import MAX_COMPLETION_LIMIT_HIT_RATE
    assert 0 < MAX_COMPLETION_LIMIT_HIT_RATE < 0.10


# --------------------------------------------------------------------------
# The gate must predict the pipeline's admission decision, not merely measure
# prompt lengths. 1024 + 1024 = 2048 is refused against a 2048-token sequence
# whatever the actual prompts do, and the first gate passed that config.
# --------------------------------------------------------------------------

def test_the_gate_replicates_the_pipelines_declared_allocation_rule():
    import re
    from pathlib import Path
    source = (Path(__file__).resolve().parent.parent
              / "scripts" / "train_on_dataset.py").read_text(encoding="utf-8")
    # The pipeline refuses when total >= sequence_budget.
    assert "if total >= sequence_budget:" in source

    gate_source = (Path(__file__).resolve().parent.parent
                   / "scripts" / "sequence_fit_gate.py").read_text(encoding="utf-8")
    assert "declared_fits = declared_total < sequence_limit" in gate_source


def test_the_corrective_advice_never_suggests_shortening_the_output():
    from pathlib import Path
    gate_source = (Path(__file__).resolve().parent.parent
                   / "scripts" / "sequence_fit_gate.py").read_text(encoding="utf-8")
    assert "do NOT reduce the completion budget" in gate_source


def test_the_queued_command_is_defined_once():
    """Preflight validated a command the launch did not use: the flag was
    added to one and not the other."""
    from pathlib import Path
    chain = (Path(__file__).resolve().parent.parent / "scripts"
             / "queue_successor_train_generation_v2.sh").read_text(encoding="utf-8")
    assert "RUN_ARGS=(" in chain
    assert 'CMD="python scripts/train_on_dataset.py ${RUN_ARGS[*]}"' in chain
    assert '"${RUN_ARGS[@]}"' in chain
    # The settings must appear exactly once in EXECUTABLE content. Comment
    # lines are stripped first: the comment explaining this very bug mentions
    # the flag, and counting it would make the test fail on its own docstring.
    executable = chr(10).join(
        line for line in chain.splitlines() if not line.lstrip().startswith("#"))
    for flag in ("--max-sequence-tokens", "--generation-completion-token-limit",
                 "--candidate-parse-mode", "--retain-raw-output"):
        assert executable.count(flag) == 1, f"{flag} is written more than once"


def test_the_committed_gate_report_uses_an_admissible_allocation():
    import json
    from pathlib import Path
    path = (Path(__file__).resolve().parent.parent / "results"
            / "v4_2_sequence_fit_gate_train_successor.json")
    if not path.exists():
        return
    report = json.loads(path.read_text(encoding="utf-8"))
    assert report["declared_allocation_fits"] is True
    assert report["declared_prompt_plus_completion"] < report["sequence_limit"]
    assert report["completion_token_limit"] == 1024, "output length was reduced"
    assert report["gate_passed"] is True


# --------------------------------------------------------------------------
# Fenced output. This exact mistake has now been made twice on this project:
# once in the completion-truncation audit, once here. Parsing raw model text
# without unwrapping the fence scored 43678 of 44760 outputs unparseable
# (97.6%) against a pipeline parse rate of 97.37%.
# --------------------------------------------------------------------------

FENCED = "```python\nassert f(1) == 1\nassert f(2) == 2\n```"


def test_a_fenced_output_is_not_counted_as_unparseable(tmp_path):
    path = _artifact(tmp_path, outcomes=[_raw(FENCED)])
    report = verify(path, "Qwen/Qwen2.5-Coder-1.5B-Instruct", "main")
    assert report["unparseable_outputs"] == 0
    assert report["candidates_ineligible_for_oracle_labels"] == 0


def test_genuinely_broken_output_is_still_counted(tmp_path):
    """The unwrapper must not turn the check off altogether."""
    path = _artifact(tmp_path, outcomes=[_raw("```python\nassert f(1 ==\n```")])
    report = verify(path, "Qwen/Qwen2.5-Coder-1.5B-Instruct", "main")
    assert report["unparseable_outputs"] == 1


def test_the_verifier_uses_the_successor_parsers_own_unwrapper():
    """Not a private reimplementation that can drift from the real parser."""
    from pathlib import Path
    source = (Path(__file__).resolve().parent.parent
              / "scripts" / "verify_successor_generation.py").read_text(encoding="utf-8")
    assert "from scripts.analyze_parser_pilot import whole_output_of" in source
    assert "ast.parse(whole_output_of(raw))" in source


# --------------------------------------------------------------------------
# An artifact with nothing usable left must not pass a gate that licenses a
# build. The first verifier returned verified=true at a 97.6% ineligible rate.
# --------------------------------------------------------------------------

def test_a_mostly_ineligible_artifact_is_refused(tmp_path):
    outcomes = [_raw("assert f(1 ==") for _ in range(80)]
    outcomes += [_raw(f"assert f({i}) == {i}") for i in range(20)]
    path = _artifact(tmp_path, outcomes=outcomes)
    report = verify(path, "Qwen/Qwen2.5-Coder-1.5B-Instruct", "main")
    assert report["ineligible_ceiling_exceeded"] is True
    assert report["verified"] is False
    assert any("ineligible" in p for p in report["problems"])


def test_the_ineligible_ceiling_is_predeclared():
    from scripts.verify_successor_generation import MAX_INELIGIBLE_RATE
    assert 0 < MAX_INELIGIBLE_RATE < 1


def test_the_committed_verification_passed_on_real_numbers():
    import json
    from pathlib import Path
    path = (Path(__file__).resolve().parent.parent / "results"
            / "v4_2_successor_generation_verification.json")
    if not path.exists():
        return
    report = json.loads(path.read_text(encoding="utf-8"))
    assert report["verified"] is True
    assert report["candidates"] == report["candidates_with_raw_output"]
    assert report["raw_output_hash_mismatches"] == 0
    assert report["prompt_budget_failed_functions"] == 0
    assert report["completion_limit_threshold_exceeded"] is False
    assert report["ineligible_ceiling_exceeded"] is False
    # The fence bug would put this near 0.98.
    assert report["unparseable_share"] < 0.01


def test_ineligible_candidates_are_counted_individually(tmp_path):
    """Keying on (record_id, rank) alone collapsed 80 unusable candidates
    within one function into a single entry, because outcomes without a rank
    all mapped to rank 0."""
    outcomes = [_raw("assert f(1 ==") for _ in range(5)]
    path = _artifact(tmp_path, outcomes=outcomes)
    report = verify(path, "Qwen/Qwen2.5-Coder-1.5B-Instruct", "main")
    assert report["candidates_ineligible_for_oracle_labels"] == 5
