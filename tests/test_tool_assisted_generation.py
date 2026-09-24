"""CPU tests for the execution-feedback repair loop and its sham-feedback control.

No model is loaded.  A scripted sampler stands in for generation; token
matching is tested both with a deterministic mock and on the real Qwen chat
template.
"""
from __future__ import annotations

import copy
import hashlib

import pytest

from harness.buggy_side_execution import execute_on_code_under_test
from harness.execution_feedback import (
    CATEGORIES, CLOSING, FORBIDDEN_TOKENS, PAD_UNIT, REPAIRABLE, RETAINED, SHAM_SENTENCE,
    WELL_FORMED, _TEMPLATES, assert_no_hidden_content, build_feedback, classify_execution,
    classify_policy_failure, repair_addition, sham_addition, sham_is_neutral,
)
from harness.tool_assisted_generation import (
    Budget, BudgetExceeded, FINAL_SLOTS, Journal, MatchedPair, MatchInfeasible,
    SEQUENCES_PER_TARGET, derived_seed, make_frozen_parser, permitted_view, run_target,
    select_final,
)

GOLDEN = "def pick(xs, i):\n    return sorted(xs)[i]  # GOLDEN-SENTINEL-LINE\n"
BUGGY = "def pick(xs, i):\n    return xs[i]\n"
GOLD_TEST = "assert pick([3, 1, 2], 0) == 1  # GOLD-TEST-SENTINEL"
SEED = 42
OUTCOME_WORDS = ("passes", "passed", "fails", "failed", "raised", "timed out", "killed",
                 "assertionerror", "mutant", "golden", "reference", "sentinel")


def _record(**overrides):
    record = {"id": "rec-1", "group_id": "lineage-1", "entry_point": "pick",
              "specification": "Return the i-th smallest element.",
              "prompt_code_under_test": BUGGY, "mutant_code": BUGGY,
              "golden_code": GOLDEN, "test_cases": [GOLD_TEST],
              "execution_mode": "function_assertion"}
    record.update(overrides)
    return record


def fake_tokens(addition):
    return 100 + (len(addition.split()) if addition else 0)


def mock_matcher(view, repair, echo):
    """Deterministic stand-in: pads the sham with neutral units to equal length."""
    target = fake_tokens(repair)
    unpadded = fake_tokens(sham_addition(echo, 0))
    if unpadded > target:
        raise MatchInfeasible("sham longer than repair")
    units = target - unpadded
    return MatchedPair(repair, sham_addition(echo, units), target, unpadded,
                       fake_tokens(sham_addition(echo, units)), units)


def infeasible_matcher(view, repair, echo):
    raise MatchInfeasible("forced")


class ScriptedSampler:
    """Deterministic stand-in for the model; counts every call."""

    def __init__(self, round1, canonical, repair, sham):
        self.round1, self.canonical, self.repair, self.sham = round1, canonical, repair, sham
        self.calls = []
        self.slot_of_seed = {derived_seed(SEED, "rec-1", "round2", slot): slot
                             for slot in range(1, 9)}

    def __call__(self, view, additions, per_prompt, seed):
        assert "golden_code" not in view and "test_cases" not in view
        self.calls.append((tuple(additions), per_prompt, seed))
        addition = additions[0]
        if per_prompt == 8:
            texts = self.round1
        else:
            slot = self.slot_of_seed[seed]
            if addition is None:
                texts = [self.canonical[slot]]
            elif SHAM_SENTENCE in addition:
                texts = [self.sham[slot]]
            else:
                texts = [self.repair[slot]]
        return [[{"text": text, "input_tokens": fake_tokens(addition),
                  "output_tokens": len(text), "prompt_sha256": "p"}
                 for text in texts]]


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
REPAIR_SLOTS = [3, 4, 6, 7, 8]
CANONICAL = {slot: f"assert pick([{slot}, 0], 1) == {slot}" for slot in range(1, 9)}
REPAIR = {slot: f"assert pick([9, {slot}], 0) == {slot}" for slot in range(1, 9)}
SHAM = {slot: f"assert pick([7, {slot}, 5], 1) == {slot}" for slot in range(1, 9)}


def _sampler():
    return ScriptedSampler(ROUND1, CANONICAL, REPAIR, SHAM)


def _run(parse, tmp_path=None, sampler=None, matcher=mock_matcher, **kwargs):
    sampler = sampler or _sampler()
    journal = Journal(tmp_path / "journal.jsonl" if tmp_path else None)
    result = run_target(_record(), sampler=sampler, parse=parse, execute=_guarded_executor,
                        matcher=matcher, base_seed=SEED, journal=journal, timeout=1.0,
                        **kwargs)
    return result, sampler


@pytest.fixture(scope="module")
def result(parse):
    return _run(parse)


def _by_slot(result, arm, round_):
    return {c["slot"]: c for c in result["candidates"][arm] if c["round"] == round_}


# --- taxonomy and feedback -------------------------------------------------

def test_taxonomy_is_closed_and_templates_are_clean():
    assert set(REPAIRABLE).isdisjoint(RETAINED)
    assert set(WELL_FORMED) <= set(RETAINED)
    assert set(_TEMPLATES) == set(REPAIRABLE)
    for category in REPAIRABLE:
        assert_no_hidden_content(_TEMPLATES[category].format(detail="x"), [])
    for category in RETAINED:
        with pytest.raises(ValueError):
            build_feedback(category, "", [])
    assert len(CATEGORIES) == len(set(CATEGORIES))
    assert all(token == token.lower() for token in FORBIDDEN_TOKENS)


def test_feedback_is_hashable_reproducible_and_never_echoes_hidden_material():
    hidden = [GOLDEN, GOLD_TEST]
    detail = "return sorted(xs)[i]  # GOLDEN-SENTINEL-LINE"
    first = build_feedback("malformed_invocation", detail, hidden)
    assert "SENTINEL" not in first["message"] and first["detail"] == ""
    assert first == build_feedback("malformed_invocation", detail, hidden)
    assert len(first["sha256"]) == 64
    with pytest.raises(ValueError):
        assert_no_hidden_content("the mutant was killed", [])


def test_sham_sentence_is_shorter_than_every_feedback_template_and_neutral():
    for template in _TEMPLATES.values():
        assert len(SHAM_SENTENCE.split()) < len(template.format(detail="").split())
    for word in OUTCOME_WORDS + tuple(CATEGORIES) + ("syntax", "error", "parse", "api"):
        assert word not in (SHAM_SENTENCE + PAD_UNIT + CLOSING).lower()


def test_permitted_view_removes_fixed_code_and_gold_tests_and_refuses_protected():
    view = permitted_view(_record())
    assert "golden_code" not in view and "test_cases" not in view
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


def test_mutable_state_never_carries_between_executions():
    stateful = "def pick(xs, i, seen=[]):\n    seen.append(1)\n    return len(seen)\n"
    first = execute_on_code_under_test(stateful, "assert pick([], 0) == 1", 1.0)
    second = execute_on_code_under_test(stateful, "assert pick([], 0) == 1", 1.0)
    assert first["status"] == second["status"] == "pass"


# --- required properties 1-10 ---------------------------------------------

def test_1_b_and_c_share_the_identical_round_one(result):
    res, _ = result
    b, c = _by_slot(res, "B", 1), _by_slot(res, "C", 1)
    assert all(b[s]["raw_output"] == c[s]["raw_output"] for s in range(1, 9))


def test_2_repair_and_sham_echo_the_identical_parent(result):
    res, _ = result
    b2, c2, r1 = _by_slot(res, "B", 2), _by_slot(res, "C", 2), _by_slot(res, "C", 1)
    assert sorted(res["budget"]["repairs_delivered"]) == REPAIR_SLOTS
    for slot in REPAIR_SLOTS:
        assert c2[slot]["status"] == "repair" and b2[slot]["status"] == "sham"
        assert c2[slot]["parent_candidate"] == b2[slot]["parent_candidate"] == f"rec-1|r1|{slot}"
        echo = r1[slot]["code"]
        prefix = "You previously answered:\n\n" + echo
        assert c2[slot]["prompt_addition"].startswith(prefix)
        assert b2[slot]["prompt_addition"].startswith(prefix)


def test_3_rendered_input_tokens_are_exactly_equal(result):
    res, _ = result
    b2, c2 = _by_slot(res, "B", 2), _by_slot(res, "C", 2)
    for slot in REPAIR_SLOTS:
        assert b2[slot]["input_tokens"] == c2[slot]["input_tokens"]
        assert b2[slot]["matched_input_tokens"] == c2[slot]["matched_input_tokens"]


def test_3b_real_qwen_template_matching_is_exact_and_uses_the_adapter_path():
    from harness.generation_adapter import build_prompts, successor_settings
    from harness.prompt_factory import prompt_factory
    from harness.tool_assisted_generation import make_token_matcher
    from transformers import AutoTokenizer
    settings = successor_settings()
    tokenizer = AutoTokenizer.from_pretrained(settings.base_model_name,
                                              revision=settings.base_model_revision)
    build_prompt = prompt_factory(settings.prompt_settings())
    matcher = make_token_matcher(tokenizer, settings, build_prompt)
    view = permitted_view(_record())
    echo = "assert pick([3, 1, 2]) == 1"
    for category in ("malformed_invocation", "missing_test_invocation", "duplicate_candidate",
                     "invalid_candidate_shape"):
        feedback = build_feedback(category, "pick() missing 1 required positional argument",
                                  [GOLDEN])
        pair = matcher(view, repair_addition(echo, feedback, [GOLDEN]), echo)
        assert pair.sham_tokens == pair.repair_tokens and pair.pad_units >= 0
        assert sham_is_neutral(pair.sham_addition, echo)
        ids, generable, failures = build_prompts(
            tokenizer, [view, view], settings, build_prompt,
            {0: pair.repair_addition, 1: pair.sham_addition})
        assert not failures and len(ids[0]) == len(ids[1]) == pair.repair_tokens


def test_4_sham_contains_no_category_or_execution_result(result):
    res, _ = result
    r1 = _by_slot(res, "B", 1)
    shams = [c for c in res["candidates"]["B"] if c["status"] == "sham"]
    assert len(shams) == len(REPAIR_SLOTS)
    for cand in shams:
        echo = r1[cand["slot"]]["code"]
        assert sham_is_neutral(cand["prompt_addition"], echo)
        instruction = cand["prompt_addition"].split(echo, 1)[1].lower()
        for word in OUTCOME_WORDS + tuple(CATEGORIES) + ("syntax", "error", "parse"):
            assert word not in instruction
        assert cand["feedback"] is None


def test_5_changing_c_category_changes_only_c_text():
    echo = "assert pick([3, 1, 2]) == 1"
    a = repair_addition(echo, build_feedback("malformed_invocation", "x", []), [])
    b = repair_addition(echo, build_feedback("duplicate_candidate", "", []), [])
    assert a != b
    sham_a = mock_matcher(None, a, echo).sham_addition
    sham_b = mock_matcher(None, b, echo).sham_addition
    assert sham_is_neutral(sham_a, echo) and sham_is_neutral(sham_b, echo)
    strip = lambda text: text.replace(PAD_UNIT, "").replace("\n", "")
    assert strip(sham_a) == strip(sham_b)


def test_6_infeasible_match_gives_both_arms_the_same_non_repair_fallback(parse):
    res, sampler = _run(parse, matcher=infeasible_matcher)
    assert res["budget"]["repairs_delivered"] == []
    assert sorted(int(s) for s in res["budget"]["repairs_skipped_match_infeasible"]) == \
        REPAIR_SLOTS
    b2, c2 = _by_slot(res, "B", 2), _by_slot(res, "C", 2)
    for slot in range(1, 9):
        assert b2[slot]["status"] == c2[slot]["status"] == "resample"
        assert b2[slot]["raw_output"] == c2[slot]["raw_output"]
    assert not any(call[0][0] for call in sampler.calls)
    assert res["budget"]["sequences"] == {"B": 16, "C": 16}


def test_7_output_tokens_are_recorded_but_not_claimed_equal(result):
    res, _ = result
    tokens = res["budget"]["tokens"]
    assert tokens["B"]["output"] != tokens["C"]["output"]
    assert "may differ" in res["control"]
    assert "exact compute" not in res["control"]


def _doctor(result, mutate):
    data = copy.deepcopy(result)
    mutate(data)
    from scripts.analyse_tool_assisted_pilot import pairing_problems
    return pairing_problems(data)


def test_8_analyzer_refuses_every_kind_of_mismatch(result):
    from scripts.analyse_tool_assisted_pilot import pairing_problems
    res, _ = result
    assert pairing_problems(res) == []

    def calls(d): d["budget"]["model_calls"]["C"] += 1
    def seqs(d): d["budget"]["sequences"]["B"] -= 1
    def tokens(d): _slot(d, "B", REPAIR_SLOTS[0])["input_tokens"] += 1
    def parents(d): _slot(d, "B", REPAIR_SLOTS[0])["parent_candidate"] = "rec-1|r1|1"
    def no_sham(d): _slot(d, "B", REPAIR_SLOTS[0])["status"] = "resample"
    def leaky(d):
        cand = _slot(d, "B", REPAIR_SLOTS[0])
        cand["prompt_addition"] = cand["prompt_addition"].replace(
            SHAM_SENTENCE, "Your previous test is not valid Python.")
    def unshared(d): _slot(d, "C", 1)["raw_output"] = "different"
    for mutate in (calls, seqs, tokens, parents, no_sham, leaky, unshared):
        assert _doctor(res, mutate), mutate.__name__


def _slot(data, arm, slot):
    return next(c for c in data["candidates"][arm] if c["round"] == 2 and c["slot"] == slot)


def test_8b_analyzer_refuses_mismatched_caps(result, tmp_path, monkeypatch):
    import json
    import scripts.analyse_tool_assisted_pilot as analysis
    res, _ = result
    loop = tmp_path / "loop"
    loop.mkdir()
    other = copy.deepcopy(res)
    other["record_id"] = "rec-2"
    other["budget"]["max_new_tokens"] = 512
    for data in (res, other):
        name = hashlib.sha256(data["record_id"].encode()).hexdigest() + ".json"
        (loop / name).write_text(json.dumps(data), encoding="utf-8")
    monkeypatch.setattr(analysis, "OUTPUT_DIR", tmp_path)
    accounting = analysis.loop_accounting({"record_ids": ["rec-1", "rec-2"]})
    assert accounting["request_budget_matched"] is False
    assert any("caps differ" in problem for problem in accounting["pairing_problems"])
    assert "output_tokens" in accounting and "ratio_c_over_b" in accounting["output_tokens"]


def test_9_hidden_material_and_outcomes_never_enter_sham_or_repair_prompts(result):
    res, sampler = result
    for additions, _, _ in sampler.calls:
        for addition in additions:
            if not addition:
                continue
            assert "SENTINEL" not in addition
            instruction = addition.split("\n\n", 2)[-1].lower()
            for word in OUTCOME_WORDS:
                assert word not in instruction


def test_10_resume_repeats_neither_member_of_a_matched_pair(parse, tmp_path):
    first, sampler = _run(parse, tmp_path=tmp_path)
    keys = [call for call in sampler.calls]
    assert len(keys) == 1 + len(REPAIR_SLOTS) * 2 + (8 - len(REPAIR_SLOTS))
    replay = _sampler()
    second, _ = _run(parse, tmp_path=tmp_path, sampler=replay)
    assert replay.calls == []
    assert first["final"] == second["final"]


# --- budgets, raw outputs, final slots -------------------------------------

def test_request_budgets_are_identical_between_b_and_c(result):
    res, _ = result
    budget = res["budget"]
    assert budget["sequences"] == {"B": SEQUENCES_PER_TARGET, "C": SEQUENCES_PER_TARGET}
    assert budget["model_calls"]["B"] == budget["model_calls"]["C"] == 9
    assert len(res["candidates"]["B"]) == len(res["candidates"]["C"]) == 16


def test_raw_outputs_and_hashes_are_retained(result):
    res, _ = result
    for arm in ("B", "C"):
        for cand in res["candidates"][arm]:
            assert cand["raw_output_sha256"] == hashlib.sha256(
                cand["raw_output"].encode()).hexdigest()


def test_repair_budget_and_token_cap_are_enforced(parse):
    with pytest.raises(BudgetExceeded):
        _run(parse, budget=Budget(max_repairs_per_target=2))

    class Long(ScriptedSampler):
        def __call__(self, *args):
            groups = super().__call__(*args)
            for group in groups:
                for item in group:
                    item["output_tokens"] = 5000
            return groups
    with pytest.raises(BudgetExceeded):
        _run(parse, sampler=Long(ROUND1, CANONICAL, REPAIR, SHAM))


def test_assertion_failures_and_target_exceptions_are_retained_not_repaired(result):
    res, _ = result
    r1, c2 = _by_slot(res, "C", 1), _by_slot(res, "C", 2)
    assert r1[1]["category"] == "observed_assertion_failure"
    assert r1[2]["category"] == "executes_without_error"
    assert r1[5]["category"] == "target_raised_exception"
    for slot in (1, 2, 5):
        assert c2[slot]["status"] == "resample" and c2[slot]["feedback"] is None


def test_eight_final_slots_are_deterministic(parse, result):
    first, _ = result
    second, _ = _run(parse)
    for arm in ("B", "C"):
        assert len(first["final"][arm]) == FINAL_SLOTS
        assert first["final"][arm] == second["final"][arm]


def test_final_selection_is_text_only_under_adversarial_fields(result):
    """Phase 6: fake kill, fixed-validity, assertion and execution fields change nothing."""
    res, _ = result
    for arm in ("B", "C"):
        candidates = res["candidates"][arm]
        baseline = [c["candidate_id"] for c in select_final(candidates)]
        poisoned = []
        for index, cand in enumerate(candidates):
            fake = dict(cand, killed=index % 2 == 0, reference_valid=index % 3 == 0,
                        mutant_status="assertion_error", reference_status="pass",
                        execution={"status": "pass" if index % 2 else "assertion_error",
                                   "origin": "candidate"},
                        category=CATEGORIES[index % len(CATEGORIES)])
            poisoned.append(fake)
        assert [c["candidate_id"] for c in select_final(poisoned)] == baseline
    # A clean pass, an assertion failure and a target exception rank in slot order.
    ranks = {c["candidate_id"]: c["final_rank"] for c in select_final(res["candidates"]["C"])}
    assert ranks["C|r1|1"] < ranks["C|r1|2"] < ranks["C|r1|5"]


def test_final_slots_score_through_the_unchanged_successor_evaluator(result):
    from harness.sealed_final_evaluator import _slots_from_outputs
    from metrics.research_evaluation import evaluate_candidate_slots
    res, _ = result
    final = res["final"]["C"]
    slots = _slots_from_outputs([s["raw_output"] for s in final], [s["code"] for s in final])
    outcomes = evaluate_candidate_slots(slots, GOLDEN, BUGGY, "pick", allow_test_function=True)
    assert len(outcomes) == FINAL_SLOTS and any(o["killed"] for o in outcomes)
