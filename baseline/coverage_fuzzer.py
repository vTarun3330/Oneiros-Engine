"""
Coverage-Guided Fuzzer Baseline for Oneiros.

Simulates Atheris-style coverage-guided fuzzing using Python's
sys.settrace to track branch coverage. Mutates inputs that reach
new code paths, prioritising them for further exploration.

This is the strongest non-AI baseline — it systematically explores
branches but has no semantic understanding of the code.
"""
import ast
import sys
import random
import re
import time
from typing import List, Dict, Any, Tuple, Set
from dataclasses import dataclass, field

from engine.bug_discovery import run_in_sandbox
from harness.safe_execution import execute_code


@dataclass
class CoverageResult:
    """Tracks which lines a test execution covered."""
    lines_hit: Set[int] = field(default_factory=set)
    branches_hit: Set[Tuple[int, int]] = field(default_factory=set)
    crashed: bool = False
    error: str = ""


class CoverageTracer:
    """Lightweight coverage tracer using sys.settrace."""

    def __init__(self):
        self.lines: Set[int] = set()
        self.branches: Set[Tuple[int, int]] = set()
        self._prev_line: int = -1

    def trace(self, frame, event, arg):
        if event == "line":
            lineno = frame.f_lineno
            self.lines.add(lineno)
            if self._prev_line >= 0:
                self.branches.add((self._prev_line, lineno))
            self._prev_line = lineno
        return self.trace

    def get_result(self) -> CoverageResult:
        return CoverageResult(
            lines_hit=self.lines.copy(),
            branches_hit=self.branches.copy(),
        )


class CoverageGuidedFuzzer:
    """
    Coverage-guided test generator (Atheris-style).

    Strategy:
      1. Start with a seed corpus of simple inputs.
      2. Run each input, track coverage.
      3. If an input reaches NEW lines/branches, save it.
      4. Mutate saved inputs to explore deeper.
      5. Repeat for a fixed number of iterations.
    """

    def __init__(self, seed: int = 42, max_iterations: int = 50):
        random.seed(seed)
        self.max_iterations = max_iterations
        self.global_coverage: Set[int] = set()
        self.global_branches: Set[Tuple[int, int]] = set()
        self.corpus: List[str] = []     # interesting inputs (test strings)
        self.stats = {
            "iterations": 0,
            "new_coverage_finds": 0,
            "total_lines_covered": 0,
            "total_branches_covered": 0,
        }

    # ── Seed generation ───────────────────────────────────────

    def _infer_param_types(self, code: str, entry_point: str) -> List[str]:
        """Infer parameter types from function signature and body."""
        types = []
        # Try to find signature
        sig_match = re.search(
            rf'def\s+{re.escape(entry_point)}\s*\((.*?)\)', code
        )
        if not sig_match:
            return ["any"]

        params_str = sig_match.group(1)
        if not params_str.strip():
            return []

        for param in params_str.split(","):
            param = param.strip()
            if not param or param == "self":
                continue
            hint = param.split(":")[-1].strip() if ":" in param else ""
            hint_lower = hint.lower()
            if "list" in hint_lower:
                types.append("list")
            elif "str" in hint_lower:
                types.append("str")
            elif "float" in hint_lower:
                types.append("float")
            elif "bool" in hint_lower:
                types.append("bool")
            elif "dict" in hint_lower:
                types.append("dict")
            elif "int" in hint_lower:
                types.append("int")
            else:
                types.append("any")
        return types

    def _make_seed_values(self, ptype: str) -> List[str]:
        """Produce a small set of seed values for a type."""
        if ptype == "int":
            return ["0", "1", "-1", "2", "100", "-100"]
        elif ptype == "float":
            return ["0.0", "1.0", "-1.0", "0.5", "1e10"]
        elif ptype == "str":
            return ['""', '"a"', '"hello"', '"abc"', '" "']
        elif ptype == "list":
            return ["[]", "[1]", "[1, 2, 3]", "[0]", "[-1, 0, 1]",
                    "[1, 2, 3, 4, 5]"]
        elif ptype == "bool":
            return ["True", "False"]
        elif ptype == "dict":
            return ["{}", '{"a": 1}', '{"key": "val"}']
        else:  # any
            return ["0", '""', "[]", "None", "True", "1"]

    def _build_seed_corpus(
        self, code: str, entry_point: str
    ) -> List[str]:
        """Build initial test corpus from type hints."""
        ptypes = self._infer_param_types(code, entry_point)
        if not ptypes:
            return [f"result = {entry_point}()"]

        # Cross-product of first few seeds per param (capped)
        from itertools import product
        all_seeds = [self._make_seed_values(pt) for pt in ptypes]
        combos = list(product(*all_seeds))
        random.shuffle(combos)
        combos = combos[:30]  # cap at 30 seeds

        corpus = []
        for combo in combos:
            args = ", ".join(combo)
            corpus.append(f"result = {entry_point}({args})")
        return corpus

    # ── Mutation of inputs ────────────────────────────────────

    def _mutate_input(self, test_code: str) -> str:
        """Mutate a test input to explore new paths."""
        mutations = [
            self._mutate_number,
            self._mutate_list,
            self._mutate_string,
            self._mutate_bool,
            self._swap_arg,
        ]
        return random.choice(mutations)(test_code)

    def _mutate_number(self, code: str) -> str:
        nums = list(re.finditer(r'(?<![a-zA-Z_])(-?\d+)', code))
        if not nums:
            return code
        m = random.choice(nums)
        old = int(m.group())
        new = random.choice([old + 1, old - 1, old * 2, 0, 1, -1,
                             old + random.randint(-10, 10)])
        return code[:m.start()] + str(new) + code[m.end():]

    def _mutate_list(self, code: str) -> str:
        lists = list(re.finditer(r'\[([^\[\]]*)\]', code))
        if not lists:
            return code
        m = random.choice(lists)
        action = random.choice(["empty", "add", "duplicate", "reverse_str"])
        if action == "empty":
            return code[:m.start()] + "[]" + code[m.end():]
        elif action == "add":
            content = m.group(1).strip()
            extra = str(random.randint(-5, 5))
            new_content = f"{content}, {extra}" if content else extra
            return code[:m.start()] + f"[{new_content}]" + code[m.end():]
        elif action == "duplicate":
            return code[:m.start()] + m.group() + " + " + m.group() + code[m.end():]
        return code

    def _mutate_string(self, code: str) -> str:
        strings = list(re.finditer(r'"([^"]*)"', code))
        if not strings:
            return code
        m = random.choice(strings)
        new = random.choice(['""', '"a"', '"xyz"', '" "', '"123"', '"!@#"'])
        return code[:m.start()] + new + code[m.end():]

    def _mutate_bool(self, code: str) -> str:
        if "True" in code:
            return code.replace("True", "False", 1)
        if "False" in code:
            return code.replace("False", "True", 1)
        return code

    def _swap_arg(self, code: str) -> str:
        match = re.search(r'\((.+)\)', code)
        if not match:
            return code
        args = match.group(1).split(",")
        if len(args) < 2:
            return code
        i, j = random.sample(range(len(args)), 2)
        args[i], args[j] = args[j], args[i]
        return code[:match.start(1)] + ",".join(args) + code[match.end(1):]

    # ── Core fuzzing loop ─────────────────────────────────────

    def _run_with_coverage(
        self, function_code: str, test_code: str
    ) -> CoverageResult:
        """Execute test_code against function_code and measure coverage."""
        outcome = run_in_sandbox(
            function_code, test_code, "assert", timeout_seconds=0.5
        )
        return CoverageResult(
            lines_hit=set(outcome.lines),
            branches_hit=set(outcome.branches),
            crashed=not outcome.success,
            error=(
                "" if outcome.success
                else f"{outcome.error_type}: {outcome.error_message}"
            ),
        )

    def generate_tests(
        self,
        golden_code: str,
        entry_point: str,
        num_tests: int = 20,
    ) -> List[str]:
        """
        Generate coverage-guided tests for a function.

        Returns list of test code strings (assert-based).
        """
        # Build seed corpus
        seeds = self._build_seed_corpus(golden_code, entry_point)
        self.corpus = []
        self.global_coverage = set()
        self.global_branches = set()

        # Phase 1: run seeds, keep interesting ones
        for test_code in seeds:
            cov = self._run_with_coverage(golden_code, test_code)
            new_lines = cov.lines_hit - self.global_coverage
            new_branches = cov.branches_hit - self.global_branches
            if new_lines or new_branches:
                self.corpus.append(test_code)
                self.global_coverage |= cov.lines_hit
                self.global_branches |= cov.branches_hit
                self.stats["new_coverage_finds"] += 1

        # Phase 2: mutate interesting inputs
        iterations = 0
        while iterations < self.max_iterations and len(self.corpus) < num_tests * 2:
            iterations += 1
            if not self.corpus:
                break
            parent = random.choice(self.corpus)
            child = self._mutate_input(parent)

            # Validate syntax
            try:
                compile(child, "<test>", "exec")
            except SyntaxError:
                continue

            cov = self._run_with_coverage(golden_code, child)
            new_lines = cov.lines_hit - self.global_coverage
            new_branches = cov.branches_hit - self.global_branches
            if new_lines or new_branches:
                self.corpus.append(child)
                self.global_coverage |= cov.lines_hit
                self.global_branches |= cov.branches_hit
                self.stats["new_coverage_finds"] += 1

        self.stats["iterations"] += iterations
        self.stats["total_lines_covered"] = len(self.global_coverage)
        self.stats["total_branches_covered"] = len(self.global_branches)

        # Convert corpus to assert-based tests
        return self._corpus_to_asserts(golden_code, self.corpus[:num_tests])

    def _corpus_to_asserts(
        self, golden_code: str, corpus: List[str]
    ) -> List[str]:
        """Convert raw call statements into assertion tests."""
        asserts = []
        for test_code in corpus:
            ok, result, _ = execute_code(golden_code, test_code)
            if not ok or result is None:
                asserts.append(test_code)
                continue
            expected = (
                result["repr"]
                if isinstance(result, dict) and set(result) == {"repr", "type"}
                else repr(result)
            )
            call_expr = test_code.replace("result = ", "").strip()
            asserts.append(f"assert {call_expr} == {expected}")

        return asserts

    def get_stats(self) -> Dict[str, Any]:
        return self.stats.copy()


# ── Convenience ───────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Coverage-Guided Fuzzer Baseline Test")
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

    fuzzer = CoverageGuidedFuzzer(seed=42, max_iterations=100)
    tests = fuzzer.generate_tests(sample_code, "has_close_elements", num_tests=10)

    print(f"\nGenerated {len(tests)} tests:")
    for i, t in enumerate(tests):
        print(f"  [{i+1}] {t[:80]}")

    print(f"\nStats: {fuzzer.get_stats()}")
    print("=" * 60)
