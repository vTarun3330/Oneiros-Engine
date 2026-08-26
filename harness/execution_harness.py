"""
Execution Harness for running test cases against functions.

This module handles test execution, result collection, and
differential testing between golden and mutant functions.
"""
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum

sys.path.insert(0, str(Path(__file__).parent.parent))

from harness.dataset_loader import TargetFunction
from harness.mutation_engine import Mutant
from harness.safe_execution import execute_code


class TestResult(Enum):
    """Possible outcomes of test execution."""
    PASS = "pass"                    # Test passed (no bug found)
    FAIL = "fail"                    # Test found a bug (assertion failed)
    ERROR = "error"                  # Test caused an error
    TIMEOUT = "timeout"              # Test timed out
    CRASH = "crash"                  # Test caused a crash


@dataclass
class ExecutionResult:
    """Result of executing a test case."""
    test_id: str
    target_id: str
    result: TestResult
    output: str = ""
    error_message: str = ""
    execution_time: float = 0.0

    # Differential testing results
    golden_output: Any = None
    mutant_output: Any = None
    outputs_differ: bool = False

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["result"] = self.result.value
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExecutionResult":
        data["result"] = TestResult(data["result"])
        return cls(**data)

    def is_bug_found(self) -> bool:
        """Check if this result indicates a bug was found."""
        # An execution error is not evidence of a defect unless the reference
        # succeeded and differential testing classified the difference as FAIL.
        return self.result == TestResult.FAIL or self.outputs_differ


@dataclass
class TestCase:
    """Represents a test case to execute."""
    id: str
    code: str                        # The test code to run
    target_function: str             # Name of function being tested
    inputs: Dict[str, Any] = field(default_factory=dict)  # Test inputs
    expected_output: Any = None      # Expected output (if known)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ExecutionHarness:
    """
    Harness for executing test cases against Python functions.
    """

    def __init__(self, timeout_seconds: float = 5.0):
        """
        Initialize the execution harness.

        Args:
            timeout_seconds: Maximum time for test execution
        """
        self.timeout = timeout_seconds
        self.execution_count = 0

    def _create_execution_namespace(
        self,
        function_code: str
    ) -> Dict[str, Any]:
        """
        Create a clean namespace with the function defined.

        Args:
            function_code: The function code to execute

        Returns:
            Namespace dictionary
        """
        raise RuntimeError(
            "In-process namespaces are disabled; use execute_test so code runs "
            "in the isolated execution worker."
        )

    def execute_test(
        self,
        test_code: str,
        function_code: str,
        entry_point: str,
        test_id: str = None
    ) -> ExecutionResult:
        """
        Execute a single test against a function.

        Args:
            test_code: The test code to run
            function_code: The function being tested
            entry_point: Name of the function to call
            test_id: Identifier for the test

        Returns:
            ExecutionResult with the outcome
        """
        test_id = test_id or f"test_{self.execution_count}"
        self.execution_count += 1

        start_time = time.time()
        ok, output, error = execute_code(function_code, test_code, self.timeout)
        execution_time = time.time() - start_time
        if ok:
            result = TestResult.PASS
        elif error == "TIMEOUT" or "TIMEOUT" in error:
            result = TestResult.TIMEOUT
        elif error.startswith("AssertionError"):
            result = TestResult.FAIL
        else:
            result = TestResult.ERROR
        return ExecutionResult(
            test_id=test_id,
            target_id=entry_point,
            result=result,
            output="" if output is None else str(output),
            error_message=error,
            execution_time=execution_time,
        )

    def execute_test_batch(
        self,
        tests: List[TestCase],
        function_code: str,
        entry_point: str
    ) -> List[ExecutionResult]:
        """
        Execute a batch of tests against a function.

        Args:
            tests: List of test cases
            function_code: The function being tested
            entry_point: Name of the function to call

        Returns:
            List of ExecutionResults
        """
        results = []

        for test in tests:
            result = self.execute_test(
                test_code=test.code,
                function_code=function_code,
                entry_point=entry_point,
                test_id=test.id
            )
            results.append(result)

        return results

    def differential_test(
        self,
        test_code: str,
        golden_function: TargetFunction,
        mutant: Mutant,
        test_id: str = None
    ) -> ExecutionResult:
        """
        Execute differential test between golden and mutant.

        Args:
            test_code: Test code that exercises the function
            golden_function: The correct function
            mutant: The mutant to test
            test_id: Identifier for the test

        Returns:
            ExecutionResult with differential info
        """
        test_id = test_id or f"diff_test_{self.execution_count}"
        self.execution_count += 1

        start_time = time.time()

        # Execute on golden function
        golden_result = self._safe_execute(
            test_code,
            golden_function.code,
            golden_function.entry_point
        )

        # Execute on mutant
        mutant_result = self._safe_execute(
            test_code,
            mutant.code,
            mutant.entry_point
        )

        execution_time = time.time() - start_time

        # Check if outputs differ
        outputs_differ = False
        if golden_result["success"] and mutant_result["success"]:
            outputs_differ = golden_result["output"] != mutant_result["output"]
        elif golden_result["success"] != mutant_result["success"]:
            outputs_differ = True

        # Determine result
        if outputs_differ:
            result = TestResult.FAIL  # Found a difference (bug exposed)
        elif not mutant_result["success"]:
            result = TestResult.ERROR
        else:
            result = TestResult.PASS

        return ExecutionResult(
            test_id=test_id,
            target_id=mutant.id,
            result=result,
            output=str(mutant_result.get("output", "")),
            error_message=mutant_result.get("error", ""),
            execution_time=execution_time,
            golden_output=golden_result.get("output"),
            mutant_output=mutant_result.get("output"),
            outputs_differ=outputs_differ
        )

    def _safe_execute(
        self,
        test_code: str,
        function_code: str,
        entry_point: str
    ) -> Dict[str, Any]:
        """
        Safely execute code and capture result.

        Returns:
            Dict with success, output, and error fields
        """
        success, output, error = execute_code(
            function_code, test_code, self.timeout
        )
        return {
            "success": success,
            "output": output,
            "error": error or None,
        }

    def batch_differential_test(
        self,
        tests: List[str],
        golden_function: TargetFunction,
        mutant: Mutant
    ) -> List[ExecutionResult]:
        """
        Run multiple differential tests.

        Args:
            tests: List of test code strings
            golden_function: The correct function
            mutant: The mutant to test

        Returns:
            List of ExecutionResults
        """
        results = []

        for i, test_code in enumerate(tests):
            result = self.differential_test(
                test_code=test_code,
                golden_function=golden_function,
                mutant=mutant,
                test_id=f"batch_diff_{mutant.id}_{i}"
            )
            results.append(result)

        return results

    def label_results(
        self,
        results: List[ExecutionResult]
    ) -> Tuple[List[ExecutionResult], List[ExecutionResult]]:
        """
        Label results as winners (found bugs) and losers (didn't find bugs).

        Args:
            results: List of execution results

        Returns:
            Tuple of (winners, losers)
        """
        winners = []
        losers = []

        for result in results:
            if result.is_bug_found():
                winners.append(result)
            else:
                losers.append(result)

        return winners, losers


class TestGenerator:
    """
    Generates basic test cases for functions.
    Used for seed memory initialization.
    """

    def __init__(self):
        pass

    def generate_simple_tests(
        self,
        function: TargetFunction,
        num_tests: int = 5
    ) -> List[TestCase]:
        """
        Generate simple test cases for a function.

        Args:
            function: The target function
            num_tests: Number of tests to generate

        Returns:
            List of TestCase objects
        """
        tests = []
        # Determine entry point name (TargetFunction uses entry_point, SystemLevelFunction uses name)
        entry_point = getattr(function, 'entry_point', getattr(function, 'name', 'unknown'))

        # Parse function signature for parameters
        params = self._extract_parameters(function.signature)

        # Generate tests based on parameter types
        for i in range(num_tests):
            inputs = self._generate_inputs(params, seed=i)
            test_code = self._create_test_code(
                entry_point,
                inputs
            )

            tests.append(TestCase(
                id=f"seed_test_{function.id}_{i}",
                code=test_code,
                target_function=entry_point,
                inputs=inputs
            ))

        return tests

    def _extract_parameters(self, signature: str) -> List[Dict[str, Any]]:
        """Extract parameter info from signature."""
        params = []

        # Simple regex to find parameters
        import re
        match = re.search(r'\((.*?)\)', signature)
        if not match:
            return params

        param_str = match.group(1)
        if not param_str.strip():
            return params

        for p in param_str.split(','):
            p = p.strip()
            if not p:
                continue

            # Handle type hints
            if ':' in p:
                name, type_hint = p.split(':', 1)
                name = name.strip()
                type_hint = type_hint.split('=')[0].strip()
            else:
                name = p.split('=')[0].strip()
                type_hint = "Any"

            # Infer type from hint
            inferred_type = self._infer_type(type_hint)

            params.append({
                "name": name,
                "type_hint": type_hint,
                "inferred_type": inferred_type
            })

        return params

    def _infer_type(self, type_hint: str) -> str:
        """Infer Python type from type hint."""
        type_hint = type_hint.lower()

        if 'int' in type_hint:
            return 'int'
        elif 'float' in type_hint:
            return 'float'
        elif 'str' in type_hint:
            return 'str'
        elif 'bool' in type_hint:
            return 'bool'
        elif 'list' in type_hint:
            return 'list'
        elif 'dict' in type_hint:
            return 'dict'
        else:
            return 'any'

    def _generate_inputs(
        self,
        params: List[Dict[str, Any]],
        seed: int = 0
    ) -> Dict[str, Any]:
        """Generate input values for parameters."""
        import random
        random.seed(seed)

        inputs = {}

        for param in params:
            name = param["name"]
            ptype = param["inferred_type"]

            if ptype == 'int':
                inputs[name] = random.randint(-10, 100)
            elif ptype == 'float':
                inputs[name] = round(random.uniform(-10, 100), 2)
            elif ptype == 'str':
                strings = ["", "a", "hello", "test123", "  spaces  "]
                inputs[name] = random.choice(strings)
            elif ptype == 'bool':
                inputs[name] = random.choice([True, False])
            elif ptype == 'list':
                inputs[name] = [random.randint(0, 10) for _ in range(random.randint(0, 5))]
            elif ptype == 'dict':
                inputs[name] = {"key": random.randint(0, 10)}
            else:
                inputs[name] = random.randint(0, 10)

        return inputs

    def _create_test_code(
        self,
        entry_point: str,
        inputs: Dict[str, Any]
    ) -> str:
        """Create test code string."""
        # Format arguments
        args = ", ".join(
            f"{k}={repr(v)}" for k, v in inputs.items()
        )

        return f"""
# Auto-generated test case
try:
    result = {entry_point}({args})
except Exception as e:
    result = f"ERROR: {{type(e).__name__}}: {{e}}"
"""


def create_seed_tests(functions: List[TargetFunction]) -> List[TestCase]:
    """
    Create seed test cases for memory initialization.

    Args:
        functions: List of target functions

    Returns:
        List of TestCase objects
    """
    generator = TestGenerator()
    all_tests = []

    for func in functions:
        tests = generator.generate_simple_tests(func, num_tests=3)
        all_tests.extend(tests)

    return all_tests


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Execution Harness")
    print("=" * 60)

    # Test with a simple function
    sample_function = """
def add(a, b):
    \"\"\"Add two numbers.\"\"\"
    return a + b
"""

    # Create a test
    test_code = """
result = add(2, 3)
assert result == 5, f"Expected 5, got {result}"
"""

    harness = ExecutionHarness()

    print("\n1. Testing passing test...")
    result = harness.execute_test(
        test_code=test_code,
        function_code=sample_function,
        entry_point="add",
        test_id="test_add_pass"
    )
    print(f"   Result: {result.result.value}")

    print("\n2. Testing failing test...")
    failing_test = """
result = add(2, 3)
assert result == 6, f"Expected 6, got {result}"
"""
    result = harness.execute_test(
        test_code=failing_test,
        function_code=sample_function,
        entry_point="add",
        test_id="test_add_fail"
    )
    print(f"   Result: {result.result.value}")
    print(f"   Error: {result.error_message}")

    print("\n3. Testing buggy mutant...")
    buggy_function = """
def add(a, b):
    \"\"\"Add two numbers.\"\"\"
    return a - b  # Bug: should be +
"""

    # Create mock objects
    from dataclasses import dataclass

    @dataclass
    class MockGolden:
        code: str = sample_function
        entry_point: str = "add"

    @dataclass
    class MockMutant:
        id: str = "mutant_1"
        code: str = buggy_function
        entry_point: str = "add"

    diff_result = harness.differential_test(
        test_code="result = add(5, 3)",
        golden_function=MockGolden(),
        mutant=MockMutant()
    )
    print(f"   Result: {diff_result.result.value}")
    print(f"   Golden output: {diff_result.golden_output}")
    print(f"   Mutant output: {diff_result.mutant_output}")
    print(f"   Outputs differ: {diff_result.outputs_differ}")

    print("\n" + "=" * 60)
    print("Execution Harness tests complete!")
    print("=" * 60)
