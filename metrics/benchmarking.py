"""
Benchmarking Module for Oneiros Engine.

This module provides metrics and evaluation for measuring
the effectiveness of test generation and bug discovery.
"""
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict, field
from collections import defaultdict

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import benchmark_config, DATA_DIR


@dataclass
class BenchmarkResult:
    """Result of benchmarking a test generation approach."""
    approach_name: str

    # Bug Discovery Metrics
    total_bugs: int = 0
    bugs_found: int = 0
    bug_discovery_rate: float = 0.0      # bugs_found / total_bugs
    unique_bugs_found: int = 0

    # Test Generation Metrics
    tests_generated: int = 0
    valid_tests: int = 0
    validity_rate: float = 0.0           # valid_tests / tests_generated

    # Efficiency Metrics
    tests_per_bug: float = 0.0           # tests_generated / bugs_found
    execution_time: float = 0.0          # Total time in seconds
    time_per_bug: float = 0.0            # execution_time / bugs_found

    # Memory Metrics
    memory_size: int = 0
    duplicates_filtered: int = 0
    novelty_rate: float = 0.0            # unique / total

    # Per-function breakdown
    per_function_results: Dict[str, Dict] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def summary(self) -> str:
        return f"""
{self.approach_name} Benchmark Results
{'='*50}
Bug Discovery Rate: {self.bug_discovery_rate:.2%} ({self.bugs_found}/{self.total_bugs})
Validity Rate: {self.validity_rate:.2%} ({self.valid_tests}/{self.tests_generated})
Tests per Bug: {self.tests_per_bug:.1f}
Time per Bug: {self.time_per_bug:.2f}s
Novelty Rate: {self.novelty_rate:.2%}
"""


class Benchmarker:
    """
    Benchmarking system for comparing test generation approaches.

    Metrics:
    1. Bug Discovery Rate (BDR): % of unique bugs found
    2. Validity Rate: % of syntactically valid tests
    3. Tests per Bug: Efficiency measure
    4. Time per Bug: Speed measure
    5. Novelty Rate: % of tests that are semantically unique
    """

    def __init__(self):
        self.results: List[BenchmarkResult] = []
        self.current_benchmark: Optional[BenchmarkResult] = None

        # Per-mutant tracking
        self.mutant_status: Dict[str, bool] = {}  # mutant_id -> killed?
        self.test_per_mutant: Dict[str, List[str]] = defaultdict(list)

    def start_benchmark(self, approach_name: str, total_bugs: int) -> None:
        """Start a new benchmark run."""
        self.current_benchmark = BenchmarkResult(
            approach_name=approach_name,
            total_bugs=total_bugs
        )
        self.start_time = time.time()
        self.mutant_status = {}
        self.test_per_mutant = defaultdict(list)

    def record_test(
        self,
        test_id: str,
        function_id: str,
        is_valid: bool,
        found_bug: bool,
        mutant_id: str = None,
        is_novel: bool = True
    ) -> None:
        """Record a test execution result."""
        if not self.current_benchmark:
            raise ValueError("No benchmark started. Call start_benchmark first.")

        self.current_benchmark.tests_generated += 1

        if is_valid:
            self.current_benchmark.valid_tests += 1

        if not is_novel:
            self.current_benchmark.duplicates_filtered += 1

        if found_bug and mutant_id:
            if mutant_id not in self.mutant_status or not self.mutant_status[mutant_id]:
                self.mutant_status[mutant_id] = True
                self.current_benchmark.bugs_found += 1
                self.current_benchmark.unique_bugs_found += 1
            self.test_per_mutant[mutant_id].append(test_id)

        # Update per-function stats
        if function_id not in self.current_benchmark.per_function_results:
            self.current_benchmark.per_function_results[function_id] = {
                "tests": 0,
                "valid": 0,
                "bugs_found": 0
            }

        self.current_benchmark.per_function_results[function_id]["tests"] += 1
        if is_valid:
            self.current_benchmark.per_function_results[function_id]["valid"] += 1
        if found_bug:
            self.current_benchmark.per_function_results[function_id]["bugs_found"] += 1

    def finish_benchmark(self, memory_stats: Dict = None) -> BenchmarkResult:
        """Finish the benchmark and compute final metrics."""
        if not self.current_benchmark:
            raise ValueError("No benchmark started.")

        result = self.current_benchmark
        result.execution_time = time.time() - self.start_time

        # Compute derived metrics
        if result.total_bugs > 0:
            result.bug_discovery_rate = result.bugs_found / result.total_bugs

        if result.tests_generated > 0:
            result.validity_rate = result.valid_tests / result.tests_generated

        if result.bugs_found > 0:
            result.tests_per_bug = result.tests_generated / result.bugs_found
            result.time_per_bug = result.execution_time / result.bugs_found
        else:
            result.tests_per_bug = float('inf')
            result.time_per_bug = float('inf')

        # Memory stats
        if memory_stats:
            result.memory_size = memory_stats.get("current_size", 0)
            result.duplicates_filtered = memory_stats.get("duplicates_filtered", 0)
            total_attempts = result.tests_generated + result.duplicates_filtered
            if total_attempts > 0:
                result.novelty_rate = (total_attempts - result.duplicates_filtered) / total_attempts

        self.results.append(result)
        self.current_benchmark = None

        return result

    def compare_approaches(self) -> Dict[str, Any]:
        """Compare all benchmarked approaches."""
        if len(self.results) < 2:
            return {"message": "Need at least 2 benchmark runs to compare"}

        comparison = {
            "approaches": [],
            "best_bug_discovery": None,
            "most_efficient": None,
            "fastest": None
        }

        best_bdr = 0
        best_efficiency = float('inf')
        best_time = float('inf')

        for result in self.results:
            comparison["approaches"].append({
                "name": result.approach_name,
                "bug_discovery_rate": result.bug_discovery_rate,
                "tests_per_bug": result.tests_per_bug,
                "time_per_bug": result.time_per_bug
            })

            if result.bug_discovery_rate > best_bdr:
                best_bdr = result.bug_discovery_rate
                comparison["best_bug_discovery"] = result.approach_name

            if result.tests_per_bug < best_efficiency:
                best_efficiency = result.tests_per_bug
                comparison["most_efficient"] = result.approach_name

            if result.time_per_bug < best_time:
                best_time = result.time_per_bug
                comparison["fastest"] = result.approach_name

        return comparison

    def save_results(self, path: Path = None) -> Path:
        """Save all benchmark results to disk."""
        path = path or (DATA_DIR / "benchmarks")
        path.mkdir(parents=True, exist_ok=True)

        results_file = path / "benchmark_results.json"

        data = {
            "results": [r.to_dict() for r in self.results],
            "comparison": self.compare_approaches()
        }

        with open(results_file, 'w') as f:
            json.dump(data, f, indent=2, default=str)

        print(f"Saved benchmark results to {results_file}")
        return results_file


def calculate_mutation_score(killed: int, total: int) -> float:
    """Calculate mutation score (% of mutants killed)."""
    if total == 0:
        return 0.0
    return killed / total


def calculate_coverage_increase(
    baseline_coverage: float,
    new_coverage: float
) -> float:
    """Calculate coverage improvement."""
    return new_coverage - baseline_coverage


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Benchmarking Module")
    print("=" * 60)

    benchmarker = Benchmarker()

    # Simulate Oneiros benchmark
    print("\n1. Running Oneiros benchmark simulation...")
    benchmarker.start_benchmark("Oneiros", total_bugs=100)

    # Simulate test results
    for i in range(50):
        benchmarker.record_test(
            test_id=f"test_{i}",
            function_id="sys_pandas_merge",
            is_valid=i % 10 != 0,  # 90% valid
            found_bug=i % 5 == 0,  # 20% find bugs
            mutant_id=f"mutant_{i % 20}",
            is_novel=i % 3 != 0
        )

    result = benchmarker.finish_benchmark(memory_stats={"current_size": 40, "duplicates_filtered": 10})
    print(result.summary())

    # Simulate Random baseline
    print("\n2. Running Random baseline simulation...")
    benchmarker.start_benchmark("Random", total_bugs=100)

    for i in range(100):
        benchmarker.record_test(
            test_id=f"random_{i}",
            function_id="sys_pandas_merge",
            is_valid=i % 20 != 0,  # 95% valid but...
            found_bug=i % 20 == 0,  # Only 5% find bugs
            mutant_id=f"mutant_{i % 50}",
            is_novel=True
        )

    result = benchmarker.finish_benchmark()
    print(result.summary())

    # Compare
    print("\n3. Comparison:")
    comparison = benchmarker.compare_approaches()
    print(f"   Best Bug Discovery: {comparison['best_bug_discovery']}")
    print(f"   Most Efficient: {comparison['most_efficient']}")

    # Save
    benchmarker.save_results()

    print("\n" + "=" * 60)
    print("Benchmarking test complete!")
    print("=" * 60)
