"""
Unified Dataset Loader for Oneiros Engine.

This module provides a unified interface to load training data from multiple sources:
- HumanEval (164 problems)
- MBPP (974 problems)
- BugsInPy (400+ real bugs)
- System Functions (60 library functions)

All sources are normalized to a common UnifiedFunction schema.
"""
import json
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
import re

# Configuration
DATA_DIR = Path(__file__).parent.parent / "data"
HUMANEVAL_CACHE = DATA_DIR / "humaneval" / "humaneval_cache.json"
MBPP_CACHE = DATA_DIR / "mbpp" / "mbpp_cache.json"
BUGSINPY_METADATA = DATA_DIR / "bugsinpy" / "bugsinpy_metadata.json"
GOLDEN_DIR = DATA_DIR / "golden"
UNIFIED_CACHE = DATA_DIR / "unified_dataset.json"


@dataclass
class UnifiedFunction:
    """
    Unified representation of a function from any dataset source.

    This provides a common schema that works for training regardless
    of whether the function comes from HumanEval, MBPP, BugsInPy, or system libs.
    """
    id: str                           # Unique identifier
    source: str                       # humaneval/mbpp/bugsinpy/system
    code: str                         # The function code
    signature: str                    # Function signature line
    entry_point: str                  # Function name to call
    docstring: str                    # Description/docstring
    test_cases: List[str] = field(default_factory=list)  # Test assertions
    category: str = "general"         # Problem category
    complexity_score: int = 5         # Difficulty 1-10

    # Optional fields for bugsinpy
    buggy_code: Optional[str] = None  # Buggy version (if from bugsinpy)
    fixed_code: Optional[str] = None  # Fixed version (if from bugsinpy)
    bug_description: Optional[str] = None  # Bug explanation

    # Optional fields for system functions
    library: Optional[str] = None     # Library name (e.g., "pandas")
    edge_cases: List[str] = field(default_factory=list)  # Known edge cases

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UnifiedFunction':
        """Create from dictionary."""
        return cls(**data)


def extract_signature(code: str) -> str:
    """Extract function signature from code."""
    match = re.search(r'def\s+\w+\s*\([^)]*\)(?:\s*->\s*[^:]+)?:', code)
    if match:
        return match.group(0).rstrip(':')
    return ""


def extract_docstring(code: str) -> str:
    """Extract docstring from code."""
    # Match triple-quoted strings after def
    match = re.search(r'"""(.*?)"""', code, re.DOTALL)
    if match:
        return match.group(1).strip()
    match = re.search(r"'''(.*?)'''", code, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""


def load_humaneval() -> List[UnifiedFunction]:
    """
    Load all HumanEval problems.

    Returns:
        List of UnifiedFunction objects from HumanEval dataset
    """
    if not HUMANEVAL_CACHE.exists():
        print(f"Warning: HumanEval cache not found at {HUMANEVAL_CACHE}")
        return []

    with open(HUMANEVAL_CACHE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    functions = []
    for item in data:
        task_id = item.get('task_id', '')
        # Create unified ID
        uid = f"humaneval_{task_id.replace('/', '_')}"

        # Combine prompt and canonical solution for full code
        prompt = item.get('prompt', '')
        solution = item.get('canonical_solution', '')
        full_code = prompt + solution

        # Extract test cases
        test_code = item.get('test', '')
        test_cases = [test_code] if test_code else []

        func = UnifiedFunction(
            id=uid,
            source="humaneval",
            code=full_code,
            signature=extract_signature(prompt),
            entry_point=item.get('entry_point', ''),
            docstring=extract_docstring(prompt),
            test_cases=test_cases,
            category="coding_challenge",
            complexity_score=6
        )
        functions.append(func)

    return functions


def load_mbpp() -> List[UnifiedFunction]:
    """
    Load all MBPP problems.

    Returns:
        List of UnifiedFunction objects from MBPP dataset
    """
    if not MBPP_CACHE.exists():
        print(f"Warning: MBPP cache not found at {MBPP_CACHE}")
        return []

    with open(MBPP_CACHE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    functions = []
    for item in data:
        task_id = str(item.get('task_id', ''))
        uid = f"mbpp_{task_id}"

        code = item.get('code', '')
        text = item.get('text', '')

        # Combine test lists
        test_list = item.get('test_list', [])
        challenge_tests = item.get('challenge_test_list', [])
        all_tests = test_list + challenge_tests

        # Extract entry point from code
        sig_match = re.search(r'def\s+(\w+)\s*\(', code)
        entry_point = sig_match.group(1) if sig_match else ''

        func = UnifiedFunction(
            id=uid,
            source="mbpp",
            code=code,
            signature=extract_signature(code),
            entry_point=entry_point,
            docstring=text,  # MBPP uses 'text' as description
            test_cases=all_tests,
            category="basic_programming",
            complexity_score=5
        )
        functions.append(func)

    return functions


def load_bugsinpy() -> List[UnifiedFunction]:
    """
    Load all BugsInPy bugs.

    Returns:
        List of UnifiedFunction objects from BugsInPy dataset
    """
    if not BUGSINPY_METADATA.exists():
        print(f"Warning: BugsInPy metadata not found at {BUGSINPY_METADATA}")
        return []

    with open(BUGSINPY_METADATA, 'r', encoding='utf-8') as f:
        data = json.load(f)

    functions = []
    for item in data:
        uid = f"bugsinpy_{item.get('project', '')}_{item.get('bug_id', '')}"

        fixed_code = item.get('fixed_code', '')
        buggy_code = item.get('buggy_code', '')

        # Parse test cases
        test_cases = item.get('test_cases', [])
        if isinstance(test_cases, str):
            test_cases = [test_cases]

        func = UnifiedFunction(
            id=uid,
            source="bugsinpy",
            code=fixed_code,  # Use fixed version as the "golden" code
            signature=extract_signature(fixed_code),
            entry_point=item.get('entry_point', ''),
            docstring=item.get('description', ''),
            test_cases=test_cases,
            category=item.get('category', 'bug_fix'),
            complexity_score=7,
            buggy_code=buggy_code,
            fixed_code=fixed_code,
            bug_description=item.get('description', '')
        )
        functions.append(func)

    return functions


def load_system_functions() -> List[UnifiedFunction]:
    """
    Load system-level functions from config.

    Returns:
        List of UnifiedFunction objects from system functions
    """
    try:
        from config.system_functions import TRAINING_FUNCTIONS, TESTING_FUNCTIONS

        functions = []
        all_funcs = TRAINING_FUNCTIONS + TESTING_FUNCTIONS

        for sf in all_funcs:
            func = UnifiedFunction(
                id=sf.id,
                source="system",
                code=sf.wrapper_code or "",
                signature=sf.signature,
                entry_point=sf.name.split('.')[-1],  # e.g., "merge" from "pandas.merge"
                docstring=sf.docstring,
                test_cases=[],  # System functions use edge_cases instead
                category=sf.category,
                complexity_score=sf.complexity_score,
                library=sf.library,
                edge_cases=sf.edge_cases
            )
            functions.append(func)

        return functions
    except ImportError as e:
        print(f"Warning: Could not import system functions: {e}")
        return []


def load_golden_functions() -> List[UnifiedFunction]:
    """
    Load pre-processed golden functions.

    Returns:
        List of UnifiedFunction objects from golden directory
    """
    golden_json = GOLDEN_DIR / "golden_functions.json"
    if not golden_json.exists():
        return []

    with open(golden_json, 'r', encoding='utf-8') as f:
        data = json.load(f)

    functions = []
    for item in data:
        func = UnifiedFunction(
            id=item.get('id', ''),
            source=item.get('source', 'golden'),
            code=item.get('code', ''),
            signature=item.get('signature', ''),
            entry_point=item.get('entry_point', ''),
            docstring=item.get('docstring', ''),
            test_cases=item.get('test_cases', []),
            category=item.get('category', 'general'),
            complexity_score=item.get('complexity_score', 5)
        )
        functions.append(func)

    return functions


class UnifiedDatasetLoader:
    """
    Main loader for the unified dataset.

    Combines data from all sources into a single training dataset.
    """

    def __init__(self, use_cache: bool = True):
        """
        Initialize the loader.

        Args:
            use_cache: Whether to use cached unified dataset if available
        """
        self.use_cache = use_cache
        self._dataset: List[UnifiedFunction] = []
        self._loaded = False

    def load(self, sources: List[str] = None) -> List[UnifiedFunction]:
        """
        Load the unified dataset.

        Args:
            sources: List of sources to include.
                     Options: ['humaneval', 'mbpp', 'bugsinpy', 'system', 'golden']
                     Default: all sources

        Returns:
            List of UnifiedFunction objects
        """
        if sources is None:
            sources = ['humaneval', 'mbpp', 'bugsinpy', 'system']

        # Check cache first
        if self.use_cache and UNIFIED_CACHE.exists():
            print("Loading from cache...")
            with open(UNIFIED_CACHE, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self._dataset = [UnifiedFunction.from_dict(d) for d in data]
            self._loaded = True
            return self._dataset

        # Load from each source
        self._dataset = []

        if 'humaneval' in sources:
            funcs = load_humaneval()
            print(f"Loaded {len(funcs)} HumanEval problems")
            self._dataset.extend(funcs)

        if 'mbpp' in sources:
            funcs = load_mbpp()
            print(f"Loaded {len(funcs)} MBPP problems")
            self._dataset.extend(funcs)

        if 'bugsinpy' in sources:
            funcs = load_bugsinpy()
            print(f"Loaded {len(funcs)} BugsInPy bugs")
            self._dataset.extend(funcs)

        if 'system' in sources:
            funcs = load_system_functions()
            print(f"Loaded {len(funcs)} system functions")
            self._dataset.extend(funcs)

        if 'golden' in sources:
            funcs = load_golden_functions()
            print(f"Loaded {len(funcs)} golden functions")
            self._dataset.extend(funcs)

        self._loaded = True
        return self._dataset

    def save_cache(self) -> Path:
        """Save the unified dataset to cache."""
        if not self._loaded:
            self.load()

        UNIFIED_CACHE.parent.mkdir(parents=True, exist_ok=True)
        with open(UNIFIED_CACHE, 'w', encoding='utf-8') as f:
            data = [func.to_dict() for func in self._dataset]
            json.dump(data, f, indent=2)

        print(f"Saved unified dataset to {UNIFIED_CACHE}")
        return UNIFIED_CACHE

    def get_by_source(self, source: str) -> List[UnifiedFunction]:
        """Get functions from a specific source."""
        if not self._loaded:
            self.load()
        return [f for f in self._dataset if f.source == source]

    def get_by_category(self, category: str) -> List[UnifiedFunction]:
        """Get functions by category."""
        if not self._loaded:
            self.load()
        return [f for f in self._dataset if f.category == category]

    def get_training_split(self, test_ratio: float = 0.1) -> tuple:
        """
        Split dataset into training and testing.

        Args:
            test_ratio: Fraction of data for testing

        Returns:
            Tuple of (train_functions, test_functions)
        """
        if not self._loaded:
            self.load()

        import random
        data = self._dataset.copy()
        random.shuffle(data)

        split_idx = int(len(data) * (1 - test_ratio))
        return data[:split_idx], data[split_idx:]

    def summary(self) -> Dict[str, Any]:
        """Get dataset summary statistics."""
        if not self._loaded:
            self.load()

        by_source = {}
        by_category = {}
        total_tests = 0

        for func in self._dataset:
            by_source[func.source] = by_source.get(func.source, 0) + 1
            by_category[func.category] = by_category.get(func.category, 0) + 1
            total_tests += len(func.test_cases)

        return {
            "total_functions": len(self._dataset),
            "by_source": by_source,
            "by_category": by_category,
            "total_test_cases": total_tests,
            "avg_tests_per_function": total_tests / max(len(self._dataset), 1)
        }


def get_unified_dataset(sources: List[str] = None) -> List[UnifiedFunction]:
    """
    Convenience function to get the unified dataset.

    Args:
        sources: List of sources to include

    Returns:
        List of UnifiedFunction objects
    """
    loader = UnifiedDatasetLoader(use_cache=False)
    return loader.load(sources)


if __name__ == "__main__":
    print("=" * 60)
    print("Unified Dataset Loader")
    print("=" * 60)

    loader = UnifiedDatasetLoader(use_cache=False)
    dataset = loader.load()

    print("\n" + "=" * 60)
    print("Dataset Summary")
    print("=" * 60)

    summary = loader.summary()
    print(f"\nTotal Functions: {summary['total_functions']}")

    print("\nBy Source:")
    for source, count in summary['by_source'].items():
        print(f"  {source}: {count}")

    print(f"\nTotal Test Cases: {summary['total_test_cases']}")
    print(f"Avg Tests per Function: {summary['avg_tests_per_function']:.2f}")

    # Save cache
    print("\n" + "-" * 60)
    cache_path = loader.save_cache()
    print(f"Cache saved to: {cache_path}")

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)
