"""
Random Baseline for Oneiros Engine.

This module implements a random test generation baseline
for comparison against the Oneiros learning approach.
"""
import random
import string
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import get_training_functions, get_testing_functions
from config.system_functions import SystemLevelFunction


@dataclass
class RandomTest:
    """A randomly generated test case."""
    id: str
    input_code: str
    function_id: str
    is_valid: bool = True


class RandomBaseline:
    """
    Random test generation baseline.

    Generates random inputs without any learning or memory.
    Used as a comparison point for Oneiros.
    """

    def __init__(self, seed: int = None):
        """
        Initialize random baseline.

        Args:
            seed: Random seed for reproducibility
        """
        if seed is not None:
            random.seed(seed)

        self.stats = {
            "total_generated": 0,
            "valid_generated": 0
        }

    def _random_int(self, low: int = -100, high: int = 100) -> int:
        """Generate random integer."""
        return random.randint(low, high)

    def _random_float(self, low: float = -100.0, high: float = 100.0) -> float:
        """Generate random float."""
        return round(random.uniform(low, high), 2)

    def _random_string(self, max_len: int = 20) -> str:
        """Generate random string."""
        length = random.randint(0, max_len)
        return ''.join(random.choices(string.ascii_letters + string.digits + ' ', k=length))

    def _random_list(self, max_len: int = 10) -> List[Any]:
        """Generate random list."""
        length = random.randint(0, max_len)
        return [self._random_int() for _ in range(length)]

    def _random_dict(self, max_keys: int = 5) -> Dict[str, Any]:
        """Generate random dictionary."""
        num_keys = random.randint(0, max_keys)
        return {
            f"key_{i}": self._random_int()
            for i in range(num_keys)
        }

    def _random_bool(self) -> bool:
        """Generate random boolean."""
        return random.choice([True, False])

    def _random_none_or_value(self, value: Any) -> Any:
        """Randomly return None or the value."""
        return random.choice([None, value])

    def generate_for_pandas_merge(self) -> str:
        """Generate random input for pandas.merge wrapper."""
        templates = [
            "result = merge_wrapper({}, {})",
            "result = merge_wrapper({{'a': [1, 2]}}, {{'b': [3, 4]}})",
            "result = merge_wrapper({{'key': [1]}}, {{'key': [2]}}, on='key')",
            "result = merge_wrapper({{'x': []}}, {{'y': []}}, how='outer')",
            f"result = merge_wrapper({{'col': {self._random_list(5)}}}, {{'col': {self._random_list(5)}}})",
            "result = merge_wrapper(None, {})",
            "result = merge_wrapper({}, None)",
        ]
        return random.choice(templates)

    def generate_for_json_loads(self) -> str:
        """Generate random input for json.loads wrapper."""
        templates = [
            'result = json_loads_wrapper("{}")',
            'result = json_loads_wrapper("[]")',
            'result = json_loads_wrapper("null")',
            'result = json_loads_wrapper("{}")',  # Empty dict
            'result = json_loads_wrapper("invalid json")',
            f'result = json_loads_wrapper(\'{{"key": {self._random_int()}}}\')',
            'result = json_loads_wrapper("")',
            'result = json_loads_wrapper(None)',
        ]
        return random.choice(templates)

    def generate_for_datetime_strptime(self) -> str:
        """Generate random input for datetime.strptime wrapper."""
        templates = [
            "result = datetime_parse_wrapper('2024-01-15', '%Y-%m-%d')",
            "result = datetime_parse_wrapper('invalid', '%Y-%m-%d')",
            "result = datetime_parse_wrapper('2024-02-29', '%Y-%m-%d')",  # Leap year
            "result = datetime_parse_wrapper('', '%Y-%m-%d')",
            "result = datetime_parse_wrapper('2024-13-01', '%Y-%m-%d')",  # Invalid month
            f"result = datetime_parse_wrapper('{random.randint(1900,2100)}-{random.randint(1,12):02d}-{random.randint(1,31):02d}', '%Y-%m-%d')",
            "result = datetime_parse_wrapper(None, '%Y-%m-%d')",
        ]
        return random.choice(templates)

    def generate_for_os_path_join(self) -> str:
        """Generate random input for os.path.join wrapper."""
        templates = [
            "result = path_join_wrapper('a', 'b', 'c')",
            "result = path_join_wrapper('', '')",
            "result = path_join_wrapper('/absolute', 'relative')",
            "result = path_join_wrapper('relative', '/absolute')",  # Tricky case
            f"result = path_join_wrapper('{self._random_string(10)}', '{self._random_string(10)}')",
            "result = path_join_wrapper()",
            "result = path_join_wrapper(None)",
        ]
        return random.choice(templates)

    def generate_for_re_match(self) -> str:
        """Generate random input for re.match wrapper."""
        templates = [
            "result = regex_match_wrapper(r'\\d+', '123abc')",
            "result = regex_match_wrapper(r'[a-z]+', 'hello')",
            "result = regex_match_wrapper(r'invalid[', 'test')",  # Invalid regex
            "result = regex_match_wrapper('', '')",
            "result = regex_match_wrapper(r'.*', None)",
            f"result = regex_match_wrapper(r'.+', '{self._random_string(20)}')",
        ]
        return random.choice(templates)

    def generate_for_function(
        self,
        func: SystemLevelFunction,
        num_tests: int = 1
    ) -> List[RandomTest]:
        """
        Generate random tests for a specific function.

        Args:
            func: Target function
            num_tests: Number of tests to generate

        Returns:
            List of RandomTest objects
        """
        tests = []

        # Map function IDs to generators
        generators = {
            "sys_pandas_merge": self.generate_for_pandas_merge,
            "sys_json_loads": self.generate_for_json_loads,
            "sys_datetime_strptime": self.generate_for_datetime_strptime,
            "sys_os_path_join": self.generate_for_os_path_join,
            "sys_re_match": self.generate_for_re_match,
        }

        generator = generators.get(func.id, self._generate_generic)

        for i in range(num_tests):
            if func.id in generators:
                input_code = generator()
            else:
                input_code = self._generate_generic(func)

            # Simple validity check
            is_valid = True
            try:
                compile(input_code, '<string>', 'exec')
            except SyntaxError:
                is_valid = False

            test = RandomTest(
                id=f"random_{func.id}_{self.stats['total_generated']}",
                input_code=input_code,
                function_id=func.id,
                is_valid=is_valid
            )

            self.stats["total_generated"] += 1
            if is_valid:
                self.stats["valid_generated"] += 1

            tests.append(test)

        return tests

    def _generate_generic(self, func: SystemLevelFunction = None) -> str:
        """Generate generic random call."""
        if func:
            entry_point = func.signature.split('(')[0].split()[-1]
            args = []
            for _ in range(random.randint(0, 3)):
                arg_type = random.choice(['int', 'str', 'list', 'dict', 'none'])
                if arg_type == 'int':
                    args.append(str(self._random_int()))
                elif arg_type == 'str':
                    args.append(f"'{self._random_string(10)}'")
                elif arg_type == 'list':
                    args.append(str(self._random_list(5)))
                elif arg_type == 'dict':
                    args.append(str(self._random_dict(3)))
                else:
                    args.append('None')

            return f"result = {entry_point}({', '.join(args)})"

        return "result = None"

    def run_baseline(
        self,
        functions: List[SystemLevelFunction],
        tests_per_function: int = 10
    ) -> Dict[str, List[RandomTest]]:
        """
        Run baseline test generation on all functions.

        Args:
            functions: List of target functions
            tests_per_function: Tests to generate per function

        Returns:
            Dict mapping function_id to list of tests
        """
        results = {}

        for func in functions:
            tests = self.generate_for_function(func, tests_per_function)
            results[func.id] = tests

        return results

    def get_stats(self) -> Dict[str, int]:
        """Get generation statistics."""
        return self.stats.copy()


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Random Baseline")
    print("=" * 60)

    baseline = RandomBaseline(seed=42)

    # Get testing functions
    testing_funcs = get_testing_functions()

    print(f"\nGenerating random tests for {len(testing_funcs)} functions...")

    results = baseline.run_baseline(testing_funcs, tests_per_function=5)

    for func_id, tests in results.items():
        print(f"\n{func_id}:")
        for test in tests[:3]:  # Show first 3
            print(f"  - {test.input_code[:60]}...")
            print(f"    Valid: {test.is_valid}")

    print(f"\nStats: {baseline.get_stats()}")

    print("\n" + "=" * 60)
    print("Random baseline test complete!")
    print("=" * 60)
