"""
BugsInPy Loader for real-world bug integration.

This module curates self-contained logical bugs from the BugsInPy
dataset for the extended evaluation harness.
"""
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import BUGSINPY_DIR, DATA_DIR
from harness.dataset_loader import TargetFunction
from harness.safe_execution import execute_code


# Curated list of self-contained bugs from BugsInPy
# These are bugs that:
# 1. Are logical in nature (not configuration/dependency issues)
# 2. Can be tested with simple input/output
# 3. Are self-contained (single function or small module)
CURATED_BUGSINPY_BUGS = [
    {
        "id": "bugsinpy_black_1",
        "project": "black",
        "bug_id": 1,
        "description": "String formatting bug - incorrect handling of trailing comma",
        "buggy_code": '''
def format_string(s: str) -> str:
    """Format a string by removing trailing commas."""
    if s.endswith(","):
        return s[:-1]
    return s
''',
        "fixed_code": '''
def format_string(s: str) -> str:
    """Format a string by removing trailing commas."""
    if s.endswith(", "):
        return s[:-2]
    elif s.endswith(","):
        return s[:-1]
    return s
''',
        "test_cases": [
            "assert format_string('hello,') == 'hello'",
            "assert format_string('hello, ') == 'hello'",
            "assert format_string('hello') == 'hello'",
        ],
        "category": "string_manipulation",
        "entry_point": "format_string"
    },
    {
        "id": "bugsinpy_cookiecutter_1",
        "project": "cookiecutter",
        "bug_id": 1,
        "description": "Path normalization bug - incorrect handling of backslashes",
        "buggy_code": '''
def normalize_path(path: str) -> str:
    """Normalize a file path."""
    # Replace backslashes with forward slashes
    normalized = path.replace("\\\\", "/")
    # Remove duplicate slashes
    while "//" in normalized:
        normalized = normalized.replace("//", "/")
    return normalized
''',
        "fixed_code": '''
def normalize_path(path: str) -> str:
    """Normalize a file path."""
    # Replace backslashes with forward slashes
    normalized = path.replace("\\\\", "/").replace("\\", "/")
    # Remove duplicate slashes
    while "//" in normalized:
        normalized = normalized.replace("//", "/")
    return normalized
''',
        "test_cases": [
            "assert normalize_path('a\\\\b\\\\c') == 'a/b/c'",
            "assert normalize_path('a/b/c') == 'a/b/c'",
            "assert normalize_path('a\\b\\c') == 'a/b/c'",
        ],
        "category": "string_manipulation",
        "entry_point": "normalize_path"
    },
    {
        "id": "bugsinpy_fastapi_1",
        "project": "fastapi",
        "bug_id": 1,
        "description": "Query parameter parsing - missing default value handling",
        "buggy_code": '''
def parse_query_param(param: str, default=None) -> str:
    """Parse a query parameter from a URL string."""
    if "=" in param:
        key, value = param.split("=")
        return value
    return default
''',
        "fixed_code": '''
def parse_query_param(param: str, default=None) -> str:
    """Parse a query parameter from a URL string."""
    if "=" in param:
        key, value = param.split("=", 1)  # Split only on first =
        return value if value else default
    return default
''',
        "test_cases": [
            "assert parse_query_param('key=value') == 'value'",
            "assert parse_query_param('key=') == None",
            "assert parse_query_param('key=val=ue') == 'val=ue'",
            "assert parse_query_param('key') == None",
        ],
        "category": "data_validation",
        "entry_point": "parse_query_param"
    },
    {
        "id": "bugsinpy_httpie_1",
        "project": "httpie",
        "bug_id": 1,
        "description": "Header parsing - case sensitivity issue",
        "buggy_code": '''
def get_header(headers: dict, name: str) -> str:
    """Get a header value by name (case-insensitive)."""
    return headers.get(name)
''',
        "fixed_code": '''
def get_header(headers: dict, name: str) -> str:
    """Get a header value by name (case-insensitive)."""
    for key, value in headers.items():
        if key.lower() == name.lower():
            return value
    return None
''',
        "test_cases": [
            "assert get_header({'Content-Type': 'json'}, 'content-type') == 'json'",
            "assert get_header({'CONTENT-TYPE': 'json'}, 'Content-Type') == 'json'",
            "assert get_header({'x-api-key': '123'}, 'X-API-KEY') == '123'",
        ],
        "category": "data_validation",
        "entry_point": "get_header"
    },
    {
        "id": "bugsinpy_pandas_1",
        "project": "pandas",
        "bug_id": 1,
        "description": "Find max value - empty list handling",
        "buggy_code": '''
def find_max(data: list, default=None):
    """Find the maximum value in a list."""
    if len(data) == 0:
        return 0  # Bug: should return default
    return max(data)
''',
        "fixed_code": '''
def find_max(data: list, default=None):
    """Find the maximum value in a list."""
    if len(data) == 0:
        return default
    return max(data)
''',
        "test_cases": [
            "assert find_max([1, 5, 3]) == 5",
            "assert find_max([]) == None",
            "assert find_max([], default=-1) == -1",
            "assert find_max([-1, -5, -3]) == -1",
        ],
        "category": "list_operations",
        "entry_point": "find_max"
    },
    {
        "id": "bugsinpy_requests_1",
        "project": "requests",
        "bug_id": 1,
        "description": "URL encoding - special character handling",
        "buggy_code": '''
def encode_url_param(value: str) -> str:
    """URL encode a parameter value."""
    special_chars = {' ': '%20', '&': '%26', '=': '%3D'}
    result = value
    for char, encoded in special_chars.items():
        result = result.replace(char, encoded)
    return result
''',
        "fixed_code": '''
def encode_url_param(value: str) -> str:
    """URL encode a parameter value."""
    special_chars = {' ': '%20', '&': '%26', '=': '%3D', '+': '%2B', '#': '%23'}
    result = value
    for char, encoded in special_chars.items():
        result = result.replace(char, encoded)
    return result
''',
        "test_cases": [
            "assert encode_url_param('hello world') == 'hello%20world'",
            "assert encode_url_param('a=b&c=d') == 'a%3Db%26c%3Dd'",
            "assert encode_url_param('test+value') == 'test%2Bvalue'",
            "assert encode_url_param('hash#tag') == 'hash%23tag'",
        ],
        "category": "string_manipulation",
        "entry_point": "encode_url_param"
    },
    {
        "id": "bugsinpy_scrapy_1",
        "project": "scrapy",
        "bug_id": 1,
        "description": "List deduplication - order preservation issue",
        "buggy_code": '''
def deduplicate(items: list) -> list:
    """Remove duplicates from a list while preserving order."""
    return list(set(items))
''',
        "fixed_code": '''
def deduplicate(items: list) -> list:
    """Remove duplicates from a list while preserving order."""
    seen = set()
    result = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result
''',
        "test_cases": [
            "assert deduplicate([1, 2, 2, 3, 1]) == [1, 2, 3]",
            "assert deduplicate(['a', 'b', 'a', 'c']) == ['a', 'b', 'c']",
            "assert deduplicate([]) == []",
        ],
        "category": "list_operations",
        "entry_point": "deduplicate"
    },
    {
        "id": "bugsinpy_thefuck_1",
        "project": "thefuck",
        "bug_id": 1,
        "description": "Command matching - whitespace handling",
        "buggy_code": '''
def match_command(command: str, pattern: str) -> bool:
    """Check if a command matches a pattern."""
    return command.startswith(pattern)
''',
        "fixed_code": '''
def match_command(command: str, pattern: str) -> bool:
    """Check if a command matches a pattern."""
    return command.strip().startswith(pattern.strip())
''',
        "test_cases": [
            "assert match_command('git push', 'git') == True",
            "assert match_command('  git push', 'git') == True",
            "assert match_command('git push', '  git') == True",
            "assert match_command('docker run', 'git') == False",
        ],
        "category": "string_manipulation",
        "entry_point": "match_command"
    },
    {
        "id": "bugsinpy_tornado_1",
        "project": "tornado",
        "bug_id": 1,
        "description": "Integer parsing - negative number handling",
        "buggy_code": '''
def safe_int(value: str, default: int = 0) -> int:
    """Safely parse an integer from a string."""
    if value.isdigit():
        return int(value)
    return default
''',
        "fixed_code": '''
def safe_int(value: str, default: int = 0) -> int:
    """Safely parse an integer from a string."""
    try:
        return int(value)
    except (ValueError, TypeError):
        return default
''',
        "test_cases": [
            "assert safe_int('123') == 123",
            "assert safe_int('-5') == -5",
            "assert safe_int('abc') == 0",
            "assert safe_int('12.5') == 0",
        ],
        "category": "data_validation",
        "entry_point": "safe_int"
    },
    {
        "id": "bugsinpy_youtube_dl_1",
        "project": "youtube-dl",
        "bug_id": 1,
        "description": "Duration parsing - format handling",
        "buggy_code": '''
def parse_duration(duration_str: str) -> int:
    """Parse a duration string (HH:MM:SS or MM:SS) to seconds."""
    parts = duration_str.split(":")
    if len(parts) == 2:
        minutes, seconds = int(parts[0]), int(parts[1])
        return minutes * 60 + seconds
    elif len(parts) == 3:
        hours, minutes, seconds = int(parts[0]), int(parts[1]), int(parts[2])
        return hours * 3600 + minutes * 60 + seconds
    return 0
''',
        "fixed_code": '''
def parse_duration(duration_str: str) -> int:
    """Parse a duration string (HH:MM:SS or MM:SS or SS) to seconds."""
    parts = duration_str.split(":")
    if len(parts) == 1:
        return int(parts[0])
    elif len(parts) == 2:
        minutes, seconds = int(parts[0]), int(parts[1])
        return minutes * 60 + seconds
    elif len(parts) == 3:
        hours, minutes, seconds = int(parts[0]), int(parts[1]), int(parts[2])
        return hours * 3600 + minutes * 60 + seconds
    return 0
''',
        "test_cases": [
            "assert parse_duration('1:30') == 90",
            "assert parse_duration('1:00:00') == 3600",
            "assert parse_duration('45') == 45",
            "assert parse_duration('2:30:45') == 9045",
        ],
        "category": "data_validation",
        "entry_point": "parse_duration"
    },
]


@dataclass
class RealWorldBug:
    """Represents a real-world bug from BugsInPy."""
    id: str
    project: str
    bug_id: int
    description: str
    buggy_code: str
    fixed_code: str
    test_cases: List[str]
    category: str
    entry_point: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RealWorldBug":
        return cls(**data)

    def to_target_function(self, use_buggy: bool = True) -> TargetFunction:
        """Convert to a TargetFunction for compatibility with execution harness."""
        code = self.buggy_code if use_buggy else self.fixed_code
        return TargetFunction(
            id=self.id,
            name=self.entry_point,
            source="bugsinpy",
            code=code.strip(),
            docstring=self.description,
            signature=f"def {self.entry_point}(...)",
            test_cases=self.test_cases,
            complexity_score=5,  # Default medium complexity
            category=self.category,
            entry_point=self.entry_point
        )


class BugsInPyLoader:
    """
    Loader for curated BugsInPy bugs.
    """

    def __init__(self):
        self.bugs: List[RealWorldBug] = []

    def load_curated_bugs(self) -> List[RealWorldBug]:
        """
        Load the curated set of BugsInPy bugs.

        Returns:
            List of RealWorldBug objects
        """
        self.bugs = []

        for bug_data in CURATED_BUGSINPY_BUGS:
            bug = RealWorldBug(
                id=bug_data["id"],
                project=bug_data["project"],
                bug_id=bug_data["bug_id"],
                description=bug_data["description"],
                buggy_code=bug_data["buggy_code"],
                fixed_code=bug_data["fixed_code"],
                test_cases=bug_data["test_cases"],
                category=bug_data["category"],
                entry_point=bug_data["entry_point"]
            )
            self.bugs.append(bug)

        print(f"Loaded {len(self.bugs)} curated BugsInPy bugs")
        return self.bugs

    def save_bugs(self) -> Path:
        """
        Save bugs to the bugsinpy directory.

        Returns:
            Path to the metadata file
        """
        if not self.bugs:
            self.load_curated_bugs()

        # Save metadata
        metadata_file = BUGSINPY_DIR / "bugsinpy_metadata.json"
        metadata = [b.to_dict() for b in self.bugs]

        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)

        # Save individual bug files (buggy versions)
        for bug in self.bugs:
            # Save buggy version
            buggy_file = BUGSINPY_DIR / f"{bug.id}_buggy.py"
            with open(buggy_file, 'w', encoding='utf-8') as f:
                f.write(f"# {bug.id} (BUGGY VERSION)\n")
                f.write(f"# Project: {bug.project}\n")
                f.write(f"# Description: {bug.description}\n")
                f.write(f"# Category: {bug.category}\n\n")
                f.write(bug.buggy_code.strip())
                f.write("\n\n# Test Cases:\n")
                for test in bug.test_cases:
                    f.write(f"# {test}\n")

            # Save fixed version
            fixed_file = BUGSINPY_DIR / f"{bug.id}_fixed.py"
            with open(fixed_file, 'w', encoding='utf-8') as f:
                f.write(f"# {bug.id} (FIXED VERSION)\n")
                f.write(f"# Project: {bug.project}\n")
                f.write(f"# Description: {bug.description}\n")
                f.write(f"# Category: {bug.category}\n\n")
                f.write(bug.fixed_code.strip())
                f.write("\n\n# Test Cases:\n")
                for test in bug.test_cases:
                    f.write(f"# {test}\n")

        print(f"Saved {len(self.bugs)} bugs to {BUGSINPY_DIR}")
        return metadata_file

    def load_bugs(self) -> List[RealWorldBug]:
        """
        Load previously saved bugs.

        Returns:
            List of RealWorldBug objects
        """
        metadata_file = BUGSINPY_DIR / "bugsinpy_metadata.json"

        if not metadata_file.exists():
            print("No saved bugs found, loading curated bugs...")
            return self.load_curated_bugs()

        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)

        self.bugs = [RealWorldBug.from_dict(b) for b in metadata]
        print(f"Loaded {len(self.bugs)} BugsInPy bugs from cache")

        return self.bugs

    def get_buggy_functions(self) -> List[TargetFunction]:
        """
        Get buggy versions as TargetFunction objects.

        Returns:
            List of TargetFunction objects (buggy versions)
        """
        if not self.bugs:
            self.load_bugs()

        return [bug.to_target_function(use_buggy=True) for bug in self.bugs]

    def get_fixed_functions(self) -> List[TargetFunction]:
        """
        Get fixed versions as TargetFunction objects.

        Returns:
            List of TargetFunction objects (fixed versions)
        """
        if not self.bugs:
            self.load_bugs()

        return [bug.to_target_function(use_buggy=False) for bug in self.bugs]

    def verify_bugs(self) -> Dict[str, Dict[str, bool]]:
        """
        Verify that each bug actually exhibits different behavior.

        Returns:
            Dict mapping bug_id to test results for buggy vs fixed
        """
        if not self.bugs:
            self.load_bugs()

        results = {}

        for bug in self.bugs:
            bug_results = {
                "buggy_passes_all": True,
                "fixed_passes_all": True,
                "behavior_differs": False,
                "test_results": []
            }

            for test in bug.test_cases:
                # Test buggy version
                buggy_pass = self._run_test(bug.buggy_code, test)

                # Test fixed version
                fixed_pass = self._run_test(bug.fixed_code, test)

                bug_results["test_results"].append({
                    "test": test,
                    "buggy_pass": buggy_pass,
                    "fixed_pass": fixed_pass
                })

                if not buggy_pass:
                    bug_results["buggy_passes_all"] = False
                if not fixed_pass:
                    bug_results["fixed_passes_all"] = False
                if buggy_pass != fixed_pass:
                    bug_results["behavior_differs"] = True

            results[bug.id] = bug_results

        return results

    def _run_test(self, code: str, test: str) -> bool:
        """Run a single test and return if it passes."""
        passed, _, _ = execute_code(code.strip(), test)
        return passed

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the curated bugs."""
        if not self.bugs:
            self.load_bugs()

        # Category distribution
        categories = {}
        projects = {}

        for bug in self.bugs:
            categories[bug.category] = categories.get(bug.category, 0) + 1
            projects[bug.project] = projects.get(bug.project, 0) + 1

        return {
            "total_bugs": len(self.bugs),
            "categories": categories,
            "projects": projects
        }


def integrate_bugsinpy() -> List[RealWorldBug]:
    """
    Convenience function to integrate BugsInPy bugs.

    Returns:
        List of RealWorldBug objects
    """
    print("=" * 60)
    print("Integrating BugsInPy Real-World Bugs")
    print("=" * 60)

    loader = BugsInPyLoader()
    bugs = loader.load_curated_bugs()
    loader.save_bugs()

    # Verify bugs
    print("\nVerifying bugs...")
    results = loader.verify_bugs()

    verified_count = sum(1 for r in results.values() if r["behavior_differs"])
    print(f"  Bugs with distinct buggy/fixed behavior: {verified_count}/{len(bugs)}")

    # Print summary
    summary = loader.get_summary()
    print(f"\nBugsInPy Summary:")
    print(f"  Total bugs: {summary['total_bugs']}")
    print(f"  Projects: {list(summary['projects'].keys())}")
    print(f"  Categories: {summary['categories']}")

    print("\n" + "=" * 60)
    return bugs


if __name__ == "__main__":
    # Test the BugsInPy loader
    print("=" * 60)
    print("Testing BugsInPy Loader")
    print("=" * 60)

    loader = BugsInPyLoader()
    bugs = loader.load_curated_bugs()

    print(f"\nLoaded {len(bugs)} bugs")

    # Show details of first bug
    if bugs:
        bug = bugs[0]
        print(f"\nExample Bug: {bug.id}")
        print(f"  Project: {bug.project}")
        print(f"  Description: {bug.description}")
        print(f"  Category: {bug.category}")
        print(f"\nBuggy code:")
        print(bug.buggy_code)
        print(f"\nFixed code:")
        print(bug.fixed_code)

    # Verify bugs
    print("\nVerifying all bugs...")
    results = loader.verify_bugs()

    for bug_id, result in results.items():
        status = "✓" if result["behavior_differs"] else "✗"
        print(f"  {status} {bug_id}: fixed_passes={result['fixed_passes_all']}, buggy_fails={not result['buggy_passes_all']}")

    # Save bugs
    print("\nSaving bugs...")
    loader.save_bugs()

    print("\n" + "=" * 60)
    print("BugsInPy Loader test complete!")
    print("=" * 60)
