"""
Grammar-Based Test Generator — Learn&Fuzz Baseline.

Implements a simplified Learn&Fuzz approach:
  1. Parse function signature to learn expected input grammar.
  2. Generate structurally valid inputs from the grammar.
  3. Use boundary-value analysis heuristics to maximize edge coverage.

This is the "middle-ground" baseline — smarter than random, but
not AI-powered. It proves how much value the LLM adds.
"""
import re
import ast
import random
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from harness.safe_execution import execute_code


@dataclass
class ParamGrammar:
    """Grammar rule for a single parameter."""
    name: str
    base_type: str          # int, float, str, list, bool, dict, any
    inner_type: str = ""    # for List[int] -> inner_type = "int"
    is_optional: bool = False
    default: str = ""


class GrammarLearner:
    """
    Learns input grammar from function signatures and docstrings.

    Produces a ParamGrammar for each parameter, then the generator
    uses those grammars to build valid, diverse test inputs.
    """

    def learn(self, code: str, entry_point: str) -> List[ParamGrammar]:
        """Extract parameter grammars from function code."""
        grammars = []

        # Find the def line
        sig_match = re.search(
            rf'def\s+{re.escape(entry_point)}\s*\((.*?)\)',
            code, re.DOTALL
        )
        if not sig_match:
            return [ParamGrammar(name="arg0", base_type="any")]

        params_str = sig_match.group(1)
        if not params_str.strip():
            return []

        for param in params_str.split(","):
            param = param.strip()
            if not param or param == "self":
                continue

            # Split name : type = default
            has_default = "=" in param
            default_val = ""
            if has_default:
                param_part, default_val = param.rsplit("=", 1)
                param = param_part.strip()
                default_val = default_val.strip()

            name = param.split(":")[0].strip()
            hint = param.split(":")[-1].strip() if ":" in param else ""

            base_type, inner = self._parse_type_hint(hint)

            grammars.append(ParamGrammar(
                name=name,
                base_type=base_type,
                inner_type=inner,
                is_optional="optional" in hint.lower() or "none" in hint.lower(),
                default=default_val,
            ))

        return grammars if grammars else [ParamGrammar(name="arg0", base_type="any")]

    def _parse_type_hint(self, hint: str) -> Tuple[str, str]:
        """Parse a type hint string into (base_type, inner_type)."""
        hint = hint.strip()
        h = hint.lower()

        # Check for container types: List[int], Tuple[str, ...]
        list_match = re.match(r'(?:list|List)\[(.+)\]', hint)
        if list_match:
            inner = list_match.group(1).strip().lower()
            return ("list", self._simple_type(inner))

        tuple_match = re.match(r'(?:tuple|Tuple)\[(.+)\]', hint)
        if tuple_match:
            return ("tuple", "")

        dict_match = re.match(r'(?:dict|Dict)\[(.+),\s*(.+)\]', hint)
        if dict_match:
            return ("dict", "")

        return (self._simple_type(h), "")

    def _simple_type(self, h: str) -> str:
        if "int" in h:
            return "int"
        if "float" in h:
            return "float"
        if "str" in h:
            return "str"
        if "bool" in h:
            return "bool"
        if "list" in h:
            return "list"
        if "dict" in h:
            return "dict"
        return "any"


class GrammarBaseline:
    """
    Grammar-based test generator (Learn&Fuzz style).

    Uses learned grammars + boundary-value analysis to produce
    structurally valid inputs that target edge cases.
    """

    # Boundary values per type — the core of boundary-value analysis
    BOUNDARY_VALUES = {
        "int": [0, 1, -1, 2, -2, 10, -10, 100, -100, 0, 1000],
        "float": [0.0, 1.0, -1.0, 0.5, -0.5, 0.1, 1e-10, 1e10],
        "str": ['""', '"a"', '"ab"', '"abc"', '"hello world"',
                '" "', '""', '"A"', '"123"', '"!@#"'],
        "bool": ["True", "False"],
        "list_int": ["[]", "[0]", "[1]", "[-1]", "[1, 2]",
                     "[1, 2, 3]", "[0, 0, 0]", "[1, -1]",
                     "[-1, 0, 1]", "[1, 2, 3, 4, 5]",
                     "list(range(10))", "[100]"],
        "list_float": ["[]", "[0.0]", "[1.0, 2.0]", "[0.5, -0.5]"],
        "list_str": ['[]', '["a"]', '["a", "b"]', '["abc", "def"]',
                     '["", "a"]', '["hello", "world"]'],
        "list_any": ["[]", "[1]", "[1, 2, 3]", "['a']", "[None]"],
        "dict": ["{}", '{"a": 1}', '{"key": "value"}',
                 '{"a": 1, "b": 2}'],
        "any": ["0", "1", "-1", '""', '"a"', "[]", "[1]", "True",
                "False", "None", "{}"],
        "tuple": ["()", "(1,)", "(1, 2)", "(1, 2, 3)"],
    }

    def __init__(self, seed: int = 42):
        random.seed(seed)
        self.learner = GrammarLearner()
        self.stats = {
            "total_generated": 0,
            "valid_generated": 0,
        }

    def generate_tests(
        self,
        golden_code: str,
        entry_point: str,
        num_tests: int = 20,
    ) -> List[str]:
        """
        Generate grammar-guided tests for a function.

        Returns list of assertion-based test strings.
        """
        grammars = self.learner.learn(golden_code, entry_point)

        # Generate diverse inputs from grammars
        raw_calls = self._generate_calls(entry_point, grammars, num_tests)

        # Execute against golden to get expected outputs, build asserts
        asserts = self._calls_to_asserts(golden_code, raw_calls)
        self.stats["total_generated"] += len(asserts)
        self.stats["valid_generated"] += len(asserts)

        return asserts

    def _generate_calls(
        self,
        entry_point: str,
        grammars: List[ParamGrammar],
        num_tests: int,
    ) -> List[str]:
        """Generate function call strings from grammars."""
        calls = []

        # Phase 1: boundary values cross-product (capped)
        from itertools import product
        value_lists = []
        for g in grammars:
            key = g.base_type
            if g.base_type == "list" and g.inner_type:
                key = f"list_{g.inner_type}"
            values = self.BOUNDARY_VALUES.get(key, self.BOUNDARY_VALUES["any"])
            if g.is_optional:
                values = list(values) + ["None"]
            value_lists.append(values)

        combos = list(product(*value_lists))
        random.shuffle(combos)

        for combo in combos[:num_tests]:
            args = ", ".join(str(v) for v in combo)
            calls.append(f"result = {entry_point}({args})")

        # Phase 2: if still need more, add random mutations
        while len(calls) < num_tests:
            args = []
            for g in grammars:
                key = g.base_type
                if g.base_type == "list" and g.inner_type:
                    key = f"list_{g.inner_type}"
                values = self.BOUNDARY_VALUES.get(key, self.BOUNDARY_VALUES["any"])
                args.append(str(random.choice(values)))
            calls.append(f"result = {entry_point}({', '.join(args)})")

        return calls[:num_tests]

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


# ── Convenience ───────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Grammar-Based (Learn&Fuzz) Baseline Test")
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

    gen = GrammarBaseline(seed=42)
    tests = gen.generate_tests(sample_code, "has_close_elements", num_tests=10)

    print(f"\nGenerated {len(tests)} tests:")
    for i, t in enumerate(tests):
        print(f"  [{i+1}] {t[:90]}")

    print(f"\nStats: {gen.get_stats()}")
    print("=" * 60)
