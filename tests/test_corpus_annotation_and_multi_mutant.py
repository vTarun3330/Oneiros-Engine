"""Tests for repository complexity, defect taxonomy, and multi-mutant builds."""
from __future__ import annotations

import pytest

from harness.candidate_policy import (
    executable_candidate,
    validate_function_assertion,
    validate_generated_test,
    validate_test_function,
)
from harness.corpus_annotation import (
    annotate_record,
    origin_group,
    record_test_framework,
)
from harness.multi_mutant_examples import (
    build_kill_matrix,
    build_multi_mutant_example,
    select_assertion_cover,
)
from harness.repository_complexity import analyze_repository_complexity
from harness.repository_defect_taxonomy import (
    FALLBACK_FAMILY,
    classify_repository_defect,
    diff_lines,
)


# --------------------------------------------------------------------------
# repository complexity
# --------------------------------------------------------------------------

def test_unparseable_region_is_measured_not_scored_as_simple():
    """A large broken slice must not be filed as ``simple`` by default."""
    broken = "\n".join(f"    x{index} = compute({index}) +" for index in range(80))
    metrics = analyze_repository_complexity(broken)
    assert metrics.parse_status == "unparsed"
    assert metrics.logical_lines >= 60
    assert metrics.tier == "complex"


def test_trivial_region_is_simple_and_parsed():
    metrics = analyze_repository_complexity("def f(a):\n    return a + 1\n")
    assert metrics.parse_status == "parsed"
    assert metrics.tier == "simple"
    assert metrics.parameter_count == 1


def test_state_and_exception_signals_are_counted():
    source = (
        "def handle(self, value):\n"
        "    try:\n"
        "        self.total += value\n"
        "        self.cache[value] = True\n"
        "    except KeyError:\n"
        "        raise ValueError('bad')\n"
    )
    metrics = analyze_repository_complexity(source)
    assert metrics.state_mutation_count >= 2
    assert metrics.exception_path_count >= 2


def test_file_banner_does_not_prevent_parsing():
    source = "# File: pkg/mod.py\ndef f(x):\n    return x\n"
    assert analyze_repository_complexity(source).parse_status == "parsed"


# --------------------------------------------------------------------------
# defect taxonomy
# --------------------------------------------------------------------------

def test_added_none_guard_is_input_validation():
    buggy = "def f(node):\n    return node.next.value\n"
    fixed = (
        "def f(node):\n"
        "    if node.next is not None:\n"
        "        return node.next.value\n"
        "    return None\n"
    )
    result = classify_repository_defect(buggy, fixed)
    assert result.primary_bug_family == "input_validation"
    assert "fix_adds_guard_condition" in result.evidence


def test_traceback_type_steers_family_when_diff_is_weak():
    buggy = "def f(items):\n    return items[3]\n"
    fixed = "def f(items):\n    return items[2]\n"
    evidence = {"buggy_output_tail": "E       IndexError: list index out of range"}
    result = classify_repository_defect(buggy, fixed, evidence)
    assert result.primary_bug_family == "indexing_and_data_structures"
    assert "traceback:IndexError" in result.evidence


def test_no_evidence_falls_back_explicitly_rather_than_guessing():
    result = classify_repository_defect("", "")
    assert result.primary_bug_family == FALLBACK_FAMILY
    assert result.classification_confidence == "none"
    assert result.family_scores == {}


def test_diff_lines_reports_both_sides():
    added, removed = diff_lines("a = 1\n", "a = 2\n")
    assert added == ["a = 2"]
    assert removed == ["a = 1"]


def test_classification_is_deterministic():
    buggy = "def f(x):\n    return x / 0\n"
    fixed = "def f(x):\n    if x == 0:\n        raise ValueError\n    return x / x\n"
    first = classify_repository_defect(buggy, fixed)
    second = classify_repository_defect(buggy, fixed)
    assert first.primary_bug_family == second.primary_bug_family
    assert first.family_scores == second.family_scores


# --------------------------------------------------------------------------
# annotation
# --------------------------------------------------------------------------

def _synthetic_record() -> dict:
    return {
        "id": "syn-1",
        "source": {"upstream": "humaneval"},
        "group_id": "function:semantic:abc",
        "entry_point": "f",
        "code_under_test": "def f(x):\n    return x - 1\n",
        "prompt_code_under_test": "def f(x):\n    return x - 1\n",
        "reference_code": "def f(x):\n    return x + 1\n",
        "provenance": {"mutation_type": "arithmetic"},
        "quality": {"execution_mode": "function_assertion"},
        "test_format": "assert_statement",
        "tests": [{"code": "assert f(1) == 2"}],
    }


def test_origin_group_rejects_an_unknown_upstream():
    """A new dataset must be assigned a group, not silently absorbed."""
    record = _synthetic_record()
    record["source"] = {"upstream": "some_new_benchmark"}
    with pytest.raises(ValueError):
        origin_group(record)


def test_test_framework_is_not_a_bug_family():
    record = _synthetic_record()
    assert record_test_framework(record) == "assert_statement"
    record["quality"]["execution_mode"] = "repository_pytest_fragment"
    assert record_test_framework(record) == "pytest"


def test_synthetic_annotation_uses_function_complexity_policy():
    annotation = annotate_record(_synthetic_record(), "train")
    assert annotation.origin_group == "synthetic_function"
    assert annotation.complexity_tier in {"simple", "moderate", "complex"}
    assert annotation.function_lineage == "function:semantic:abc"


def test_repository_targets_are_distinct_even_within_one_project():
    """Two defects in one project must count as two targets, not one."""
    base = {
        "id": "repo-1",
        "source": {"upstream": "BugsInPy"},
        "group_id": "project:bugsinpy:django",
        "entry_point": "",
        "code_under_test": "def f():\n    return 1\n",
        "reference_code": "def f():\n    return 2\n",
        "prompt_code_under_test": "def f():\n    return 1\n",
        "support_context": "",
        "quality": {"execution_mode": "repository_pytest_fragment"},
        "provenance": {"project": "django", "bug_id": "1"},
        "tests": [],
    }
    other = dict(base, id="repo-2", provenance={"project": "django", "bug_id": "2"})
    first = annotate_record(base, "train")
    second = annotate_record(other, "train")
    assert first.function_lineage != second.function_lineage


# --------------------------------------------------------------------------
# candidate policy shapes
# --------------------------------------------------------------------------

def test_frozen_single_assertion_policy_is_unchanged_by_the_new_shape():
    assertion = "assert f(1) == 2"
    assert validate_function_assertion(assertion, "f").valid
    # Enabling the wider shape must not alter the verdict on an assertion.
    assert validate_generated_test(assertion, "f").shape == "assertion"
    assert validate_generated_test(assertion, "f", True).shape == "assertion"


def test_test_function_is_rejected_unless_explicitly_allowed():
    code = "def test_f():\n    assert f(1) == 2\n"
    assert not validate_generated_test(code, "f").valid
    assert validate_generated_test(code, "f", True).shape == "test_function"


def test_test_function_policy_blocks_capabilities_and_bad_shapes():
    assert not validate_test_function(
        "def test_f():\n    import os\n    assert f(1) == 2\n", "f"
    ).valid
    assert not validate_test_function(
        "def test_f():\n    for i in range(3):\n        assert f(i) == i\n", "f"
    ).valid
    assert not validate_test_function("def test_f(fixture):\n    assert f(1) == 2\n", "f").valid
    assert not validate_test_function("def test_f():\n    pass\n", "f").valid
    assert not validate_test_function("def test_f():\n    assert g(1) == 2\n", "f").valid


def test_executable_candidate_appends_the_call_only_for_test_functions():
    code = "def test_f():\n    assert f(1) == 2\n"
    assert executable_candidate(code, "test_function").endswith("test_f()\n")
    assert executable_candidate("assert f(1) == 2", "assertion") == "assert f(1) == 2"


# --------------------------------------------------------------------------
# multi-mutant construction
# --------------------------------------------------------------------------

def _lineage() -> list[dict]:
    reference = "def clamp(v, lo, hi):\n    return max(lo, min(v, hi))\n"
    return [
        {
            "id": f"m{index}",
            "group_id": "function:semantic:clamp",
            "entry_point": "clamp",
            "reference_code": reference,
            "code_under_test": mutant,
            "prompt_code_under_test": mutant,
            "source": {"upstream": "humaneval"},
            "provenance": {"mutation_type": family},
            "tests": [{"code": test}],
        }
        for index, (mutant, family, test) in enumerate([
            ("def clamp(v, lo, hi):\n    return max(lo, min(v, hi - 1))\n",
             "boundary", "assert clamp(11, 0, 10) == 10"),
            ("def clamp(v, lo, hi):\n    return max(lo + 1, min(v, hi))\n",
             "boundary", "assert clamp(-1, 0, 10) == 0"),
        ])
    ]


def _classifier(tests, golden, mutant, timeout=5.0):
    """Execute assertions in-process against the two supplied programs."""
    rows = []
    for test in tests:
        def run(source: str) -> bool:
            namespace: dict = {}
            try:
                exec(source, namespace)  # noqa: S102 - test fixture
                exec(test, namespace)    # noqa: S102 - test fixture
                return True
            except Exception:
                return False

        golden_ok = run(golden)
        mutant_ok = run(mutant) if golden_ok else False
        rows.append({
            "test": test,
            "golden": {"ok": golden_ok, "status": "pass" if golden_ok else "assertion_error"},
            "mutant": {"ok": mutant_ok},
            "valid": golden_ok,
            "killed": golden_ok and not mutant_ok,
        })
    return rows


def test_kill_matrix_rejects_assertions_invalid_on_the_reference():
    records = _lineage()
    records[0]["tests"].append({"code": "assert clamp(5, 0, 10) == 999"})
    matrix = build_kill_matrix(records, classifier=_classifier)
    index = matrix.assertions.index("assert clamp(5, 0, 10) == 999")
    assert matrix.reference_valid[index] is False
    assert "reference_invalid" in matrix.rejected["assert clamp(5, 0, 10) == 999"]


def test_broad_example_is_verified_and_kills_the_displayed_target():
    example = build_multi_mutant_example(_lineage(), 0, classifier=_classifier)
    assert example is not None
    assert example.verified
    assert example.kills_displayed_target
    assert example.completion.startswith("def test_clamp_")
    assert example.mutants_killed == 2


def test_cover_starts_from_a_target_killing_assertion():
    records = _lineage()
    matrix = build_kill_matrix(records, classifier=_classifier)
    selected = select_assertion_cover(matrix, 1)
    assert matrix.killed[selected[0]][1] is True


def test_min_assertions_tops_up_without_breaking_target_coverage():
    records = _lineage()
    matrix = build_kill_matrix(records, classifier=_classifier)
    topped = select_assertion_cover(matrix, 0, max_assertions=8, min_assertions=2)
    assert len(topped) >= 2
    assert matrix.killed[topped[0]][0] is True


def test_lineage_without_a_target_killing_assertion_yields_no_example():
    """No honest label exists, so none is invented."""
    records = _lineage()
    for record in records:
        record["tests"] = [{"code": "assert clamp(5, 0, 10) == 5"}]
    assert build_multi_mutant_example(records, 0, classifier=_classifier) is None


def test_upstream_assertions_widen_the_candidate_pool():
    records = _lineage()
    matrix = build_kill_matrix(
        records, classifier=_classifier,
        extra_assertions=["assert clamp(5, 0, 10) == 5"],
    )
    assert "assert clamp(5, 0, 10) == 5" in matrix.assertions
