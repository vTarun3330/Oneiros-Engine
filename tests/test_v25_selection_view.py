"""Unique-first selection: deterministic per-function mutant set cover with a hard cap."""
from __future__ import annotations

from scripts import v25_selection_view as sv


def test_set_cover_prefers_tests_killing_more_mutants():
    tests = {"a": {"m1"}, "b": {"m1", "m2", "m3"}, "c": {"m4"}, "d": {"m3", "m4"}}
    assert sv.select_function(tests) == ["b", "d"]   # tie on new: d kills more overall


def test_cap_per_function_is_enforced():
    tests = {f"t{i}": {f"m{i}"} for i in range(10)}
    assert len(sv.select_function(tests)) == sv.PER_FUNCTION == 3


def test_selection_is_deterministic_and_never_repeats_a_test():
    tests = {"x": {"m1", "m2"}, "y": {"m1", "m2"}, "z": {"m3"}}
    first = sv.select_function(tests)
    assert first == sv.select_function(dict(reversed(list(tests.items()))))
    assert len(first) == len(set(first)) == 2


def test_no_test_is_taken_when_nothing_new_is_covered():
    assert sv.select_function({"a": {"m1"}, "b": {"m1"}}) == ["b"]   # hash tie-break
