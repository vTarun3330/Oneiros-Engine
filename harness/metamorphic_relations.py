"""Assertions that kill a mutant WITHOUT knowing what the function should return.

The measured bottleneck is oracle prediction: on mbpp, 73% of the dominant
failure is a wrong expected value on an input that already reveals the bug, and
mbpp's one-sentence specification gives the model no way to know that value.
Every intervention tried so far attacks the model. This attacks the *form of
the assertion* instead.

A metamorphic relation asserts a relationship between two calls rather than the
value of one::

    assert max_of_three(1, 2, 3) == max_of_three(3, 2, 1)   # order cannot matter
    assert sort_list(x) == sort_list(sort_list(x))          # sorting twice is sorting
    assert len(sort_list([3, 1, 2])) == 3                   # length is preserved

If the mutant breaks ordering, the first kills it - and nothing had to know the
answer is 3. The oracle problem is sidestepped rather than solved.

WHAT THIS IS NOT. A relation is not guaranteed by the specification; it is a
hypothesis that either holds on the reference or does not. That is exactly the
standard every other candidate in this project is held to - reference-valid,
and kills the mutant - so a relation that passes it is a test of the same
quality as an assert-equals that passes it. Nothing here consults the
reference to CHOOSE a relation; relations are proposed blind from the shape of
the arguments and then verified.

LEAKAGE. Relations are built only from the entry point's name, the arity, and
argument values the model itself already wrote. The reference is never read.
"""
from __future__ import annotations

import ast
from typing import Any, Iterable

#: Relation families, each a short name and a template builder. Kept small and
#: readable on purpose: the point is to establish whether a ceiling EXISTS, not
#: to ship an exhaustive theory of metamorphic testing.
RELATION_NAMES = (
    "argument_permutation",
    "idempotence",
    "involution",
    "list_order_invariance",
    "list_reversal_invariance",
    "length_preservation",
    "type_preservation",
    "self_consistency_on_copy",
)


def _is_sequence_literal(source: str) -> bool:
    try:
        node = ast.parse(source, mode="eval").body
    except SyntaxError:
        return False
    return isinstance(node, (ast.List, ast.Tuple)) or (
        isinstance(node, ast.Constant) and isinstance(node.value, str))


def _is_list_literal(source: str) -> bool:
    try:
        node = ast.parse(source, mode="eval").body
    except SyntaxError:
        return False
    return isinstance(node, ast.List)


def call(entry_point: str, arguments: Iterable[str]) -> str:
    return entry_point + "(" + ", ".join(arguments) + ")"


def propose(entry_point: str, arguments: list[str]) -> list[dict[str, Any]]:
    """Value-free assertions for one argument tuple the model already used.

    Returns proposals, not verified tests. The caller executes each against the
    reference and the mutant; a proposal that fails on the reference is simply
    a wrong hypothesis and is discarded, exactly as a wrong expected value is.
    """
    if not entry_point or not arguments:
        return []

    proposals: list[dict[str, Any]] = []
    base = call(entry_point, arguments)

    def add(name: str, code: str) -> None:
        proposals.append({"relation": name, "code": code})

    # Order of the arguments cannot matter for a symmetric function. Only
    # adjacent swaps are tried: a full permutation set explodes with arity and
    # adds nothing a swap does not already detect.
    for index in range(len(arguments) - 1):
        swapped = list(arguments)
        swapped[index], swapped[index + 1] = swapped[index + 1], swapped[index]
        if swapped != arguments:
            add("argument_permutation",
                "assert " + base + " == " + call(entry_point, swapped))

    if len(arguments) == 1:
        only = arguments[0]
        # Applying the function twice changes nothing (sorting, normalising,
        # deduplicating, absolute value).
        add("idempotence",
            "assert " + call(entry_point, [base]) + " == " + base)
        # Applying it twice returns the input (reverse, negate, swap case).
        add("involution",
            "assert " + call(entry_point, [base]) + " == " + only)
        # Determinism. Weak, but it catches a mutant that mutates its argument
        # in place - a defect class the fuzzer's aliasing bug used to hide.
        add("self_consistency_on_copy",
            "assert " + base + " == " + base)

        if _is_sequence_literal(only):
            add("length_preservation",
                "assert len(" + base + ") == len(" + only + ")")
            add("list_reversal_invariance",
                "assert " + base + " == "
                + call(entry_point, [only + "[::-1]"]))
        if _is_list_literal(only):
            add("list_order_invariance",
                "assert " + base + " == "
                + call(entry_point, ["sorted(" + only + ")"]))
        add("type_preservation",
            "assert type(" + base + ") == type(" + only + ")")

    return proposals


def extract_argument_tuples(candidate: str, entry_point: str) -> list[list[str]]:
    """Every argument tuple this candidate passes to the function under test.

    Taking the model's own inputs is deliberate: 73% of them already probe an
    input where reference and mutant differ, so the inputs are the part of the
    candidate that is mostly RIGHT. Only the asserted value is wrong, and that
    is the part a relation replaces.
    """
    try:
        tree = ast.parse(candidate)
    except SyntaxError:
        return []
    tuples: list[list[str]] = []
    seen: set[tuple[str, ...]] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        function = node.func
        name = (
            function.id if isinstance(function, ast.Name)
            else function.attr if isinstance(function, ast.Attribute)
            else None
        )
        if name != entry_point or node.keywords:
            continue
        try:
            arguments = [ast.unparse(argument) for argument in node.args]
        except Exception:
            continue
        if not arguments:
            continue
        key = tuple(arguments)
        if key in seen:
            continue
        seen.add(key)
        tuples.append(arguments)
    return tuples
