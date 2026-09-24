"""CPU tests for the execution-feedback repair loop (no model, synthetic targets)."""
from __future__ import annotations

import pytest

from harness.buggy_side_execution import execute_on_code_under_test
from harness.execution_feedback import (
    CATEGORIES, FORBIDDEN_TOKENS, REPAIRABLE, RETAINED, WELL_FORMED, _TEMPLATES,
    assert_no_hidden_content, build_feedback, classify_execution, classify_policy_failure,
)
from harness.tool_assisted_generation import (
    Budget, BudgetExceeded, FINAL_SLOTS, Journal, SEQUENCES_PER_TARGET, derived_seed,
    make_frozen_parser, permitted_view, run_target, select_final,
)

GOLDEN = "def pick(xs, i):\n    return sorted(xs)[i]  # GOLDEN-SENTINEL-LINE\n"
BUGGY = "def pick(xs, i):\n    return xs[i]\n"
GOLD_TEST = "assert pick([3, 1, 2], 0) == 1  # GOLD-TEST-SENTINEL"
SEED = 42


def _record(**overrides):
    record = {"id": "rec-1", "group_id": "lineage-1", "entry_point": "pick",
              "specification": "Return the i-th smallest element.",
              "prompt_code_under_test": BUGGY, "mutant_code": BUGGY,
              "golden_code": GOLDEN, "test_cases": [GOLD_TEST],
              "execution_mode": "function_assertion"}
    record.update(overrides)
    return record


class ScriptedSampler:
    """Deterministic stand-in for the model; counts every call."""

    def __init__(self, round1, canonical, repair):
        self.round1, self.canonical, self.repair = round1, canonical, repair
        self.calls = []
        self.slot_of_seed = {derived_seed(SEED, "rec-1", "round2", slot): slot
                             for slot in range(1, 9)}

    def __call__(self, view, additions, per_prompt, seed):
        assert "golden_code" not in view and "test_cases" not in view
        self.calls.append((tuple(additions), per_prompt, seed))
        if per_prompt == 8:
            texts = self.round1
        else:
            slot = self.slot_of_seed[seed]
            texts = [self.repair[slot] if additions[0] else self.canonical[slot]]
            if additions[0]:
                assert "SENTINEL" not in additions[0]
        return [[{"text": text, "input_tokens": 100, "output_tokens": 10,
                  "prompt_sha256": "p"} for text in texts]]


@pytest.fixture(scope="module")
def parse():
    return make_frozen_parser()


def _guarded_executor(code_under_test, candidate, timeout):
    assert "SENTINEL" not in code_under_test, "fixed code reached the executor"
    return execute_on_code_under_test(code_under_test, candidate, timeout)


ROUND1 = [
    "assert pick([3, 1, 2], 0) == 1",      # 1 fails on buggy -> retained
    "assert pick([3, 1, 2], 0) == 3",      # 2 passes on buggy -> retained
    "assert pick([3, 1, 2], 0 ==",         # 3 syntax -> repair
    "assert pick([3, 1, 2]) == 1",         # 4 wrong arity -> repair
    "assert pick([1], 5) == 1",            # 5 raises inside target -> retained
    "assert pick([3, 1, 2], 0) == 1",      # 6 duplicate of 1 -> repair
    "assert helper(1) == pick([1], 0)",    # 7 invented name -> repair
    "import os\nassert pick([1], 0) == 1",  # 8 shape/prohibited -> repair
]
CANONICAL = {slot: f"assert pick([{slot}, 0], 1) == {slot}" for slot in range(1, 9)}
REPAIR = {slot: f"assert pick([9, {slot}], 0) == {slot}" for slot in range(1, 9)}


def _run(parse, tmp_path=None, sampler=None, **kwargs):
    sampler = sampler or ScriptedSampler(ROUND1, CANONICAL, REPAIR)
    journal = Journal(tmp_path / "journal.jsonl" if tmp_path else None)
    result = run_target(_record(), sampler=sampler, parse=parse, execute=_guarded_executor,
                        base_seed=SEED, journal=journal, timeout=1.0, **kwargs)
    return result, sampler


def test_taxonomy_is_closed_and_templates_are_clean():
    assert set(REPAIRABLE).isdisjoint(RETAINED)
    assert set(WELL_FORMED) <= set(RETAINED)
    assert set(_TEMPLATES) == set(REPAIRABLE)
    for category in REPAIRABLE:
        assert_no_hidden_content(_TEMPLATES[category].format(detail="x"), [])
    for category in RETAINED:
        with pytest.raises(ValueError):
            build_feedback(category, "", [])


def test_feedback_is_hashable_reproducible_and_never_echoes_hidden_material():
    hidden = [GOLDEN, GOLD_TEST]
    first = build_feedback("malformed_invocation", "return sorted(xs)[i]  # GOLDEN-SENTINEL-LINE",
                           hidden)
    assert "SENTINEL" not in first["message"] and first["detail"] == ""
    again = build_feedback("malformed_invocation", "return sorted(xs)[i]  # GOLDEN-SENTINEL-LINE",
                           hidden)
    assert first == again and len(first["sha256"]) == 64
    with pytest.raises(ValueError):
        assert_no_hidden_content("the mutant was killed", [])
    with pytest.raises(ValueError):
        assert_no_hidden_content("return sorted(xs)[i]  # GOLDEN-SENTINEL-LINE", [GOLDEN])


def test_permitted_view_removes_fixed_code_and_gold_tests_and_refuses_protected():
    view = permitted_view(_record())
    assert "golden_code" not in view and "test_cases" not in view
    assert view["code_under_test"] == BUGGY
    with pytest.raises(PermissionError):
        permitted_view(_record(), protected_ids={"rec-1"})
    with pytest.raises(PermissionError):
        permitted_view(_record(), protected_ids={"lineage-1"})


def test_execution_classification_separates_target_candidate_and_infrastructure():
    run = lambda test: classify_execution(execute_on_code_under_test(BUGGY, test, 1.0))[0]
    assert run("assert pick([3, 1, 2], 0) == 1") == "observed_assertion_failure"
    assert run("assert pick([3, 1, 2], 0) == 3") == "executes_without_error"
    assert run("assert pick([1], 5) == 1") == "target_raised_exception"
    assert run("assert pick([1]) == 1") == "malformed_invocation"
    assert run("assert helper(1) == 1") == "unavailable_name"
    assert classify_execution({"status": "infrastructure_error", "origin": "harness"})[0] == \
        "infrastructure_failure"
    looping = "def pick(xs, i):\n    while True:\n        i += 1\n"
    assert classify_execution(execute_on_code_under_test(
        looping, "assert pick([1], 0) == 1", 0.3))[0] == "target_timeout"
    assert classify_policy_failure("syntax_error:bad")[0] == "invalid_python_syntax"
    assert classify_policy_failure("target_entry_point_not_called")[0] == \
        "missing_test_invocation"


def test_mutable_state_never_carries_between_executions():
    stateful = "def pick(xs, i, seen=[]):\n    seen.append(1)\n    return len(seen)\n"
    test = "assert pick([], 0) == 1"
    first = execute_on_code_under_test(stateful, test, 1.0)
    second = execute_on_code_under_test(stateful, test, 1.0)
    assert first["status"] == second["status"] == "pass"


def test_assertion_failures_are_retained_not_repaired(parse):
    result, _ = _run(parse)
    c = {cand["candidate_id"]: cand for cand in result["candidates"]["C"]}
    assert c["C|r1|1"]["category"] == "observed_assertion_failure"
    assert c["C|r2|1"]["status"] == "resample" and c["C|r2|1"]["feedback"] is None
    assert c["C|r1|2"]["category"] == "executes_without_error"
    assert c["C|r2|2"]["feedback"] is None
    assert c["C|r1|5"]["category"] == "target_raised_exception"
    assert c["C|r2|5"]["feedback"] is None
    final_ids = [slot["candidate_id"] for slot in result["final"]["C"]]
    assert "C|r1|1" in final_ids and "C|r1|5" in final_ids


def test_invalid_artifacts_are_repaired_with_lineage(parse):
    result, sampler = _run(parse)
    c = {cand["candidate_id"]: cand for cand in result["candidates"]["C"]}
    expected = {3: "invalid_python_syntax", 4: "malformed_invocation",
                6: "duplicate_candidate", 7: "unavailable_name",
                8: "invalid_candidate_shape"}
    for slot, category in expected.items():
        assert c[f"C|r1|{slot}"]["category"] == category
        child = c[f"C|r2|{slot}"]
        assert child["status"] == "repair"
        assert child["parent_candidate"] == f"rec-1|r1|{slot}"
        assert child["feedback"]["category"] == category
    repair_prompts = [call[0][0] for call in sampler.calls if call[0][0]]
    assert len(repair_prompts) == result["budget"]["repairs"] == 5
    for prompt in repair_prompts:
        for token in ("SENTINEL", "fails", "passes", "killed", "mutant"):
            assert token not in prompt


def test_raw_outputs_and_hashes_are_retained(parse):
    import hashlib
    result, _ = _run(parse)
    for arm in ("B", "C"):
        for cand in result["candidates"][arm]:
            assert cand["raw_output_sha256"] == hashlib.sha256(
                cand["raw_output"].encode()).hexdigest()
            for key in ("prompt_sha256", "model_call_number", "input_tokens",
                        "output_tokens", "wall_seconds", "generation_seed", "category"):
                assert key in cand


def test_compute_budget_is_identical_between_b_and_c(parse):
    result, _ = _run(parse)
    budget = result["budget"]
    assert budget["sequences"] == {"B": SEQUENCES_PER_TARGET, "C": SEQUENCES_PER_TARGET}
    assert budget["model_calls"]["B"] == budget["model_calls"]["C"] == 9
    assert len(result["candidates"]["B"]) == len(result["candidates"]["C"]) == 16


def test_b_and_c_are_identical_wherever_no_feedback_was_given(parse):
    result, _ = _run(parse)
    b = {cand["candidate_id"][2:]: cand for cand in result["candidates"]["B"]}
    for cand in result["candidates"]["C"]:
        if cand["feedback"] is None:
            assert cand["raw_output"] == b[cand["candidate_id"][2:]]["raw_output"]


def test_repair_budget_cannot_be_exceeded(parse):
    with pytest.raises(BudgetExceeded):
        _run(parse, budget=Budget(max_repairs_per_target=2))


def test_generated_token_cap_is_enforced(parse):
    class Long(ScriptedSampler):
        def __call__(self, *args):
            groups = super().__call__(*args)
            for group in groups:
                for item in group:
                    item["output_tokens"] = 5000
            return groups
    with pytest.raises(BudgetExceeded):
        _run(parse, sampler=Long(ROUND1, CANONICAL, REPAIR))


def test_eight_final_slots_are_deterministic(parse):
    first, _ = _run(parse)
    second, _ = _run(parse)
    for arm in ("B", "C"):
        assert len(first["final"][arm]) == FINAL_SLOTS
        assert first["final"][arm] == second["final"][arm]


def test_final_selection_ignores_fixed_side_and_kill_information(parse):
    result, _ = _run(parse)
    candidates = result["candidates"]["C"]
    baseline = [c["candidate_id"] for c in select_final(candidates)]
    poisoned = [dict(c, killed=(i % 2 == 0), reference_valid=(i % 3 == 0),
                     mutant_status="x") for i, c in enumerate(candidates)]
    assert [c["candidate_id"] for c in select_final(poisoned)] == baseline
    # Passing, failing and raising in the target are equally well-formed.
    ranks = {c["candidate_id"]: c["final_rank"] for c in select_final(candidates)}
    assert ranks["C|r1|1"] < ranks["C|r1|2"] < ranks["C|r1|5"]


def test_resume_replays_the_journal_without_new_model_calls(parse, tmp_path):
    first, sampler = _run(parse, tmp_path=tmp_path)
    assert sampler.calls
    replay_sampler = ScriptedSampler(ROUND1, CANONICAL, REPAIR)
    second, _ = _run(parse, tmp_path=tmp_path, sampler=replay_sampler)
    assert replay_sampler.calls == []
    assert first["final"] == second["final"]
    assert all(c["replayed_from_journal"] for c in second["candidates"]["C"])


def test_final_slots_score_through_the_unchanged_successor_evaluator(parse):
    from harness.sealed_final_evaluator import _slots_from_outputs
    from metrics.research_evaluation import evaluate_candidate_slots
    result, _ = _run(parse)
    final = result["final"]["C"]
    slots = _slots_from_outputs([s["raw_output"] for s in final], [s["code"] for s in final])
    outcomes = evaluate_candidate_slots(slots, GOLDEN, BUGGY, "pick", allow_test_function=True)
    assert len(outcomes) == FINAL_SLOTS
    assert any(o["killed"] for o in outcomes)  # the retained failing assertion kills


def test_every_category_is_declared():
    assert len(CATEGORIES) == len(set(CATEGORIES))
    assert "observed_assertion_failure" not in REPAIRABLE
    assert all(token == token.lower() for token in FORBIDDEN_TOKENS)


def test_over_budget_repair_prompt_falls_back_to_the_b_sample(parse):
    from harness.tool_assisted_generation import RepairPromptOverBudget

    class Tight(ScriptedSampler):
        def __call__(self, view, additions, per_prompt, seed):
            if additions[0]:
                raise RepairPromptOverBudget("repair prompt exceeds the prompt budget")
            return super().__call__(view, additions, per_prompt, seed)

    result, _ = _run(parse, sampler=Tight(ROUND1, CANONICAL, REPAIR))
    assert result["budget"]["repairs_undelivered_prompt_over_budget"] == [3, 4, 6, 7, 8]
    assert result["budget"]["sequences"] == {"B": 16, "C": 16}
    b = {c["candidate_id"][2:]: c["raw_output"] for c in result["candidates"]["B"]}
    for cand in result["candidates"]["C"]:
        assert cand["feedback"] is None
        assert cand["raw_output"] == b[cand["candidate_id"][2:]]
