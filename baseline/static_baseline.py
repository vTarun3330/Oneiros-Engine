"""
Static Template Baseline for Oneiros.

Generates tests using handcrafted templates and heuristics.
This is the simplest "smart" baseline — no learning, no fuzzing,
just expert-level templates applied to function signatures.

Represents what a developer might write as a first pass of tests.
"""
import re
import random
from typing import List, Dict, Any
from dataclasses import dataclass

from harness.safe_execution import execute_code


class StaticBaseline:
    """
    Template-based test generator.

    Strategy: parse signature, apply standard boundary-value
    templates. No randomness, no learning — pure heuristic.
    """

    def __init__(self, seed: int = 42):
        random.seed(seed)
        self.stats = {
            "total_generated": 0,
            "valid_generated": 0,
        }

    def generate_tests(
        self,
        golden_code: str,
        entry_point: str,
        num_tests: int = 10,
    ) -> List[str]:
        """Generate template-based tests for a function."""
        param_types = self._infer_types(golden_code, entry_point)
        calls = self._generate_template_calls(entry_point, param_types, num_tests)
        asserts = self._calls_to_asserts(golden_code, calls)
        self.stats["total_generated"] += len(asserts)
        self.stats["valid_generated"] += len(asserts)
        return asserts

    def _infer_types(self, code: str, entry_point: str) -> List[str]:
        """Infer parameter types from signature."""
        sig = re.search(
            rf'def\s+{re.escape(entry_point)}\s*\((.*?)\)', code, re.DOTALL
        )
        if not sig:
            return ["any"]
        params = sig.group(1)
        if not params.strip():
            return []

        types = []
        for p in params.split(","):
            p = p.strip()
            if not p or p == "self":
                continue
            hint = p.split(":")[-1].strip().lower() if ":" in p else ""
            if "list" in hint:
                types.append("list")
            elif "str" in hint:
                types.append("str")
            elif "float" in hint:
                types.append("float")
            elif "bool" in hint:
                types.append("bool")
            elif "int" in hint:
                types.append("int")
            elif "dict" in hint:
                types.append("dict")
            else:
                types.append("any")
        return types if types else ["any"]

    def _generate_template_calls(
        self, entry_point: str, param_types: List[str], num_tests: int
    ) -> List[str]:
        """Generate function calls from standard templates."""
        # Standard test values per type (deterministic, expert-chosen)
        templates = {
            "int": [0, 1, -1, 2, 10, 100, -100],
            "float": [0.0, 1.0, -1.0, 0.5, 100.0],
            "str": ['""', '"a"', '"abc"', '"hello"', '" "'],
            "bool": ["True", "False"],
            "list": ["[]", "[1]", "[1, 2, 3]", "[0, 0]", "[-1, 0, 1]",
                     "[1, 2, 3, 4, 5]"],
            "dict": ["{}", '{"a": 1}', '{"x": 0, "y": 1}'],
            "any": [0, 1, '""', "[]", "True", "None"],
        }

        from itertools import product
        value_lists = [templates.get(t, templates["any"]) for t in param_types]
        combos = list(product(*value_lists))

        # Deterministic order (no shuffling — this is a static baseline)
        calls = []
        for combo in combos[:num_tests]:
            args = ", ".join(str(v) for v in combo)
            calls.append(f"result = {entry_point}({args})")

        return calls

    def _calls_to_asserts(
        self, golden_code: str, calls: List[str]
    ) -> List[str]:
        """Execute calls against golden code and build assertions."""
        asserts = []
        for call in calls:
            ok, result, _ = execute_code(golden_code, call)
            if not ok or result is None:
                asserts.append(call)
                continue
            expected = (
                result["repr"]
                if isinstance(result, dict) and set(result) == {"repr", "type"}
                else repr(result)
            )
            call_expr = call.replace("result = ", "").strip()
            asserts.append(f"assert {call_expr} == {expected}")

        return asserts

    def get_stats(self) -> Dict[str, Any]:
        return self.stats.copy()


if __name__ == "__main__":
    print("=" * 60)
    print("Static Template Baseline Test")
    print("=" * 60)

    sample_code = '''
def has_close_elements(numbers, threshold):
    for idx, elem in enumerate(numbers):
        for idx2, elem2 in enumerate(numbers):
            if idx != idx2:
                distance = abs(elem - elem2)
                if distance < threshold:
                    return True
    return False
'''

    gen = StaticBaseline()
    tests = gen.generate_tests(sample_code, "has_close_elements", num_tests=10)

    print(f"\nGenerated {len(tests)} tests:")
    for i, t in enumerate(tests):
        print(f"  [{i+1}] {t[:90]}")

    print(f"\nStats: {gen.get_stats()}")
    print("=" * 60)
