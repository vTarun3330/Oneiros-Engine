"""
Dataset Loader for HumanEval and MBPP datasets.

This module handles loading, parsing, and selecting target functions
from HumanEval and MBPP benchmarks for the Unified Evaluation Harness.
"""
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
import re

try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    dataset_config,
    GOLDEN_DIR,
    HUMANEVAL_DIR,
    MBPP_DIR,
    DATA_DIR
)


@dataclass
class TargetFunction:
    """Represents a target function for testing."""
    id: str                          # Unique identifier
    name: str                        # Function name
    source: str                      # 'humaneval' or 'mbpp'
    code: str                        # Complete function code
    docstring: str                   # Function docstring
    signature: str                   # Function signature
    test_cases: List[str] = field(default_factory=list)  # Example test cases
    complexity_score: int = 0        # Estimated complexity (1-10)
    category: str = "general"        # Category (loops, validation, etc.)
    entry_point: str = ""            # Entry point function name

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TargetFunction":
        return cls(**data)

    def get_hash(self) -> str:
        """Get a unique hash for this function."""
        return hashlib.md5(self.code.encode()).hexdigest()[:8]


class DatasetLoader:
    """
    Loads and manages HumanEval and MBPP datasets.
    """

    def __init__(self):
        if not DATASETS_AVAILABLE:
            raise ImportError(
                "The 'datasets' library is required. "
                "Install with: pip install datasets"
            )

        self.humaneval_data: List[Dict] = []
        self.mbpp_data: List[Dict] = []
        self.selected_functions: List[TargetFunction] = []

    def load_humaneval(self, force_refresh: bool = False) -> List[Dict]:
        """
        Load the HumanEval dataset.

        Args:
            force_refresh: If True, download fresh data even if cached

        Returns:
            List of HumanEval problems
        """
        cache_file = HUMANEVAL_DIR / "humaneval_cache.json"

        # Try loading from cache
        if cache_file.exists() and not force_refresh:
            with open(cache_file, 'r', encoding='utf-8') as f:
                self.humaneval_data = json.load(f)
            print(f"Loaded {len(self.humaneval_data)} HumanEval problems from cache")
            return self.humaneval_data

        # Download from HuggingFace
        print("Downloading HumanEval dataset...")
        dataset = load_dataset("openai_humaneval", split="test")

        self.humaneval_data = []
        for item in dataset:
            self.humaneval_data.append({
                "task_id": item["task_id"],
                "prompt": item["prompt"],
                "canonical_solution": item["canonical_solution"],
                "test": item["test"],
                "entry_point": item["entry_point"]
            })

        # Cache the data
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(self.humaneval_data, f, indent=2)

        print(f"Downloaded and cached {len(self.humaneval_data)} HumanEval problems")
        return self.humaneval_data

    def load_mbpp(self, force_refresh: bool = False) -> List[Dict]:
        """
        Load the MBPP dataset.

        Args:
            force_refresh: If True, download fresh data even if cached

        Returns:
            List of MBPP problems
        """
        cache_file = MBPP_DIR / "mbpp_cache.json"

        # Try loading from cache
        if cache_file.exists() and not force_refresh:
            with open(cache_file, 'r', encoding='utf-8') as f:
                self.mbpp_data = json.load(f)
            print(f"Loaded {len(self.mbpp_data)} MBPP problems from cache")
            return self.mbpp_data

        # Download from HuggingFace
        print("Downloading MBPP dataset...")
        dataset = load_dataset("mbpp", split="test")

        self.mbpp_data = []
        for item in dataset:
            self.mbpp_data.append({
                "task_id": item["task_id"],
                "text": item["text"],  # Problem description
                "code": item["code"],  # Solution code
                "test_list": item["test_list"],  # Test assertions
                "test_setup_code": item.get("test_setup_code", ""),
                "challenge_test_list": item.get("challenge_test_list", [])
            })

        # Cache the data
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(self.mbpp_data, f, indent=2)

        print(f"Downloaded and cached {len(self.mbpp_data)} MBPP problems")
        return self.mbpp_data

    def _extract_function_name(self, code: str) -> str:
        """Extract the main function name from code."""
        match = re.search(r'def\s+(\w+)\s*\(', code)
        return match.group(1) if match else "unknown"

    def _extract_signature(self, code: str) -> str:
        """Extract function signature from code."""
        match = re.search(r'(def\s+\w+\s*\([^)]*\))', code)
        return match.group(1) if match else ""

    def _extract_docstring(self, code: str) -> str:
        """Extract docstring from code."""
        # Match triple-quoted strings after function definition
        match = re.search(r'def[^:]+:\s*(?:\'\'\'|""")(.*?)(?:\'\'\'|""")', code, re.DOTALL)
        return match.group(1).strip() if match else ""

    def _estimate_complexity(self, code: str) -> int:
        """
        Estimate code complexity on a scale of 1-10.

        Factors considered:
        - Number of lines
        - Nested loops/conditions
        - Recursive calls
        - Data structure operations
        """
        lines = code.strip().split('\n')
        num_lines = len(lines)

        # Count complexity indicators
        loop_count = len(re.findall(r'\b(for|while)\b', code))
        condition_count = len(re.findall(r'\b(if|elif|else)\b', code))
        recursion = 1 if self._extract_function_name(code) in code.split('def', 1)[-1] else 0
        list_ops = len(re.findall(r'\[(.*?)\]', code))
        dict_ops = len(re.findall(r'\{(.*?)\}', code))

        # Calculate score
        score = (
            min(num_lines / 5, 3) +  # Line count contribution (max 3)
            min(loop_count, 2) +      # Loop contribution (max 2)
            min(condition_count / 2, 2) +  # Condition contribution (max 2)
            recursion * 2 +           # Recursion bonus
            min((list_ops + dict_ops) / 3, 1)  # Data structure contribution (max 1)
        )

        return max(1, min(10, int(score)))

    def _categorize_function(self, code: str, docstring: str) -> str:
        """Categorize function based on code patterns and description."""
        text = (code + " " + docstring).lower()

        if any(word in text for word in ['sort', 'order', 'arrange']):
            return "sorting"
        elif any(word in text for word in ['search', 'find', 'locate', 'index']):
            return "searching"
        elif any(word in text for word in ['valid', 'check', 'verify', 'is_']):
            return "data_validation"
        elif any(word in text for word in ['string', 'str', 'text', 'word', 'char']):
            return "string_manipulation"
        elif any(word in text for word in ['list', 'array', 'element', 'append']):
            return "list_operations"
        elif any(word in text for word in ['sum', 'product', 'factorial', 'prime', 'math']):
            return "mathematical"
        elif any(word in text for word in ['dict', 'map', 'key', 'value']):
            return "dictionary_operations"
        elif any(word in text for word in ['recursive', 'recursion']):
            return "recursive"
        elif re.search(r'\b(for|while)\b.*\b(for|while)\b', code):
            return "complex_loops"
        else:
            return "general"

    def _parse_humaneval_function(self, item: Dict) -> TargetFunction:
        """Parse a HumanEval item into a TargetFunction."""
        # Combine prompt and solution for complete function
        full_code = item["prompt"] + item["canonical_solution"]

        return TargetFunction(
            id=f"humaneval_{item['task_id'].replace('/', '_')}",
            name=item["entry_point"],
            source="humaneval",
            code=full_code,
            docstring=self._extract_docstring(item["prompt"]),
            signature=self._extract_signature(item["prompt"]),
            test_cases=[item["test"]],
            complexity_score=self._estimate_complexity(full_code),
            category=self._categorize_function(full_code, item["prompt"]),
            entry_point=item["entry_point"]
        )

    def _parse_mbpp_function(self, item: Dict) -> TargetFunction:
        """Parse an MBPP item into a TargetFunction."""
        code = item["code"]
        func_name = self._extract_function_name(code)

        return TargetFunction(
            id=f"mbpp_{item['task_id']}",
            name=func_name,
            source="mbpp",
            code=code,
            docstring=item["text"],
            signature=self._extract_signature(code),
            test_cases=item["test_list"],
            complexity_score=self._estimate_complexity(code),
            category=self._categorize_function(code, item["text"]),
            entry_point=func_name
        )

    def select_target_functions(
        self,
        n: int = None,
        min_complexity: int = 3,
        max_complexity: int = 8,
        categories: List[str] = None
    ) -> List[TargetFunction]:
        """
        Select diverse target functions for the evaluation harness.

        Args:
            n: Number of functions to select (default from config)
            min_complexity: Minimum complexity score
            max_complexity: Maximum complexity score
            categories: Specific categories to prioritize

        Returns:
            List of selected TargetFunction objects
        """
        n = n or dataset_config.num_target_functions
        categories = categories or dataset_config.priority_categories

        # Ensure data is loaded
        if not self.humaneval_data:
            self.load_humaneval()
        if not self.mbpp_data:
            self.load_mbpp()

        # Parse all functions
        all_functions: List[TargetFunction] = []

        for item in self.humaneval_data:
            try:
                func = self._parse_humaneval_function(item)
                if min_complexity <= func.complexity_score <= max_complexity:
                    all_functions.append(func)
            except Exception as e:
                print(f"Warning: Failed to parse HumanEval {item.get('task_id')}: {e}")

        for item in self.mbpp_data:
            try:
                func = self._parse_mbpp_function(item)
                if min_complexity <= func.complexity_score <= max_complexity:
                    all_functions.append(func)
            except Exception as e:
                print(f"Warning: Failed to parse MBPP {item.get('task_id')}: {e}")

        print(f"Found {len(all_functions)} functions in complexity range [{min_complexity}, {max_complexity}]")

        # Select diverse functions across categories
        selected: List[TargetFunction] = []
        category_counts: Dict[str, int] = {cat: 0 for cat in categories}
        category_counts["other"] = 0

        # First pass: prioritize specified categories
        for func in sorted(all_functions, key=lambda f: -f.complexity_score):
            if len(selected) >= n:
                break

            cat = func.category if func.category in categories else "other"
            max_per_category = max(2, n // len(categories))

            if category_counts[cat] < max_per_category:
                selected.append(func)
                category_counts[cat] += 1

        # Second pass: fill remaining slots
        for func in all_functions:
            if len(selected) >= n:
                break
            if func not in selected:
                selected.append(func)

        self.selected_functions = selected[:n]

        # Print selection summary
        print(f"\nSelected {len(self.selected_functions)} target functions:")
        for cat in set(f.category for f in self.selected_functions):
            count = sum(1 for f in self.selected_functions if f.category == cat)
            print(f"  - {cat}: {count}")

        return self.selected_functions

    def save_golden_functions(self, functions: List[TargetFunction] = None) -> Path:
        """
        Save selected functions to the golden/ directory.

        Args:
            functions: Functions to save (defaults to selected_functions)

        Returns:
            Path to the saved file
        """
        functions = functions or self.selected_functions
        if not functions:
            raise ValueError("No functions to save. Call select_target_functions() first.")

        # Save metadata
        metadata_file = GOLDEN_DIR / "golden_functions.json"
        metadata = [f.to_dict() for f in functions]

        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)

        # Save individual function files
        for func in functions:
            func_file = GOLDEN_DIR / f"{func.id}.py"
            with open(func_file, 'w', encoding='utf-8') as f:
                f.write(f"# {func.id}\n")
                f.write(f"# Source: {func.source}\n")
                f.write(f"# Category: {func.category}\n")
                f.write(f"# Complexity: {func.complexity_score}\n\n")
                f.write(func.code)
                if func.test_cases:
                    f.write("\n\n# Test Cases:\n")
                    for test in func.test_cases:
                        f.write(f"# {test}\n")

        print(f"\nSaved {len(functions)} golden functions to {GOLDEN_DIR}")
        return metadata_file

    def load_golden_functions(self) -> List[TargetFunction]:
        """
        Load previously saved golden functions.

        Returns:
            List of TargetFunction objects
        """
        metadata_file = GOLDEN_DIR / "golden_functions.json"

        if not metadata_file.exists():
            raise FileNotFoundError(
                f"No golden functions found at {metadata_file}. "
                "Run select_target_functions() and save_golden_functions() first."
            )

        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)

        self.selected_functions = [TargetFunction.from_dict(m) for m in metadata]
        print(f"Loaded {len(self.selected_functions)} golden functions")

        return self.selected_functions


def generate_golden_functions(n: int = 25) -> List[TargetFunction]:
    """
    Convenience function to generate golden functions.

    Args:
        n: Number of functions to select

    Returns:
        List of selected functions
    """
    loader = DatasetLoader()
    loader.load_humaneval()
    loader.load_mbpp()
    functions = loader.select_target_functions(n=n)
    loader.save_golden_functions()
    return functions


if __name__ == "__main__":
    # Test the dataset loader
    print("=" * 60)
    print("Testing Dataset Loader")
    print("=" * 60)

    loader = DatasetLoader()

    print("\n1. Loading HumanEval...")
    humaneval = loader.load_humaneval()
    print(f"   Sample: {humaneval[0]['task_id']}")

    print("\n2. Loading MBPP...")
    mbpp = loader.load_mbpp()
    print(f"   Sample: {mbpp[0]['task_id']}")

    print("\n3. Selecting target functions...")
    functions = loader.select_target_functions(n=25)

    print("\n4. Saving golden functions...")
    loader.save_golden_functions()

    print("\n5. Sample function details:")
    sample = functions[0]
    print(f"   ID: {sample.id}")
    print(f"   Name: {sample.name}")
    print(f"   Category: {sample.category}")
    print(f"   Complexity: {sample.complexity_score}")
    print(f"   Code preview: {sample.code[:100]}...")

    print("\n" + "=" * 60)
    print("Dataset loading complete!")
    print("=" * 60)
