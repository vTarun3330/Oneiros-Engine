from harness.candidate_policy import validate_function_assertion


def test_accepts_one_assert_that_calls_target():
    result = validate_function_assertion(
        "assert add(2, 3) == 5", "add"
    )
    assert result.valid


def test_rejects_missing_target_and_multiple_statements():
    assert not validate_function_assertion("assert True", "add").valid
    result = validate_function_assertion(
        "value = add(2, 3)\nassert value == 5", "add"
    )
    assert not result.valid
    assert result.reason == "candidate_must_be_one_assert"


def test_rejects_import_file_access_and_dunder_introspection():
    assert not validate_function_assertion(
        "import os\nassert add(1, 2) == 3", "add"
    ).valid
    assert not validate_function_assertion(
        "assert add(1, 2) == open('secret').read()", "add"
    ).valid
    assert not validate_function_assertion(
        "assert add(1, 2).__class__ is int", "add"
    ).valid


def test_rejects_unbounded_comprehensions_and_large_literals():
    assert not validate_function_assertion(
        "assert add(*[x for x in range(2)]) == 1", "add"
    ).valid
    assert not validate_function_assertion(
        f"assert add(1, 2) == {'x' * 5000!r}", "add"
    ).valid
