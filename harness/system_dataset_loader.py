"""
System-Level Dataset Loader for Oneiros Engine.

This module handles loading and preparing system-level functions for training
and testing. It works with wrapper functions that can be mutated by mutmut.
"""
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
import subprocess
import tempfile
import shutil

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.system_functions import (
    SystemLevelFunction,
    TRAINING_FUNCTIONS,
    TESTING_FUNCTIONS,
    get_training_functions,
    get_testing_functions
)
from config import DATA_DIR, GOLDEN_DIR, MUTANTS_DIR


@dataclass
class SystemLevelDataset:
    """Represents the complete system-level dataset."""
    training_functions: List[SystemLevelFunction]
    testing_functions: List[SystemLevelFunction]
    mutants: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def total_training_count(self) -> int:
        return len(self.training_functions)

    @property
    def total_testing_count(self) -> int:
        return len(self.testing_functions)

    @property
    def total_mutants(self) -> int:
        return len(self.mutants)

    def summary(self) -> Dict[str, Any]:
        return {
            "training_functions": self.total_training_count,
            "testing_functions": self.total_testing_count,
            "total_mutants": self.total_mutants,
            "libraries": list(set(f.library for f in self.training_functions))
        }


class SystemLevelDatasetLoader:
    """
    Loader for system-level functions dataset.

    This loader:
    1. Loads predefined system-level functions
    2. Generates wrapper files for mutation
    3. Runs mutmut to generate mutants
    4. Prepares training/testing splits
    """

    def __init__(self):
        self.training_functions = get_training_functions()
        self.testing_functions = get_testing_functions()
        self.wrappers_dir = GOLDEN_DIR / "system_wrappers"
        self.mutants_data: List[Dict[str, Any]] = []

    def setup_directories(self) -> None:
        """Create necessary directories."""
        self.wrappers_dir.mkdir(parents=True, exist_ok=True)
        (MUTANTS_DIR / "system_mutants").mkdir(parents=True, exist_ok=True)

    def generate_wrapper_files(self) -> Path:
        """
        Generate individual Python files for each wrapper function.
        These files will be mutated by mutmut.

        Returns:
            Path to the wrappers directory
        """
        self.setup_directories()

        for func in self.training_functions:
            file_path = self.wrappers_dir / f"{func.id}.py"
            content = f'''"""
{func.name} - System-Level Wrapper for Oneiros Engine
Library: {func.library}
Complexity: {func.complexity_score}/10

Edge Cases to Test:
{chr(10).join(f"  - {ec}" for ec in func.edge_cases)}

Expected Bug Types:
{chr(10).join(f"  - {bt}" for bt in func.bug_types)}
"""
{func.import_statement}

{func.wrapper_code.strip()}


def get_function_metadata():
    """Return metadata about this function."""
    return {{
        "id": "{func.id}",
        "name": "{func.name}",
        "library": "{func.library}",
        "signature": "{func.signature}"
    }}
'''
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)

        # Create __init__.py
        init_content = '''"""System-level wrapper functions for Oneiros Engine."""
from pathlib import Path
import importlib.util

def load_wrapper(func_id: str):
    """Dynamically load a wrapper function by ID."""
    wrapper_path = Path(__file__).parent / f"{func_id}.py"
    if not wrapper_path.exists():
        raise FileNotFoundError(f"Wrapper not found: {func_id}")

    spec = importlib.util.spec_from_file_location(func_id, wrapper_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
'''
        with open(self.wrappers_dir / "__init__.py", 'w', encoding='utf-8') as f:
            f.write(init_content)

        print(f"Generated {len(self.training_functions)} wrapper files in {self.wrappers_dir}")
        return self.wrappers_dir

    def generate_mutants_for_function(self, func: SystemLevelFunction, max_mutants: int = 20) -> List[Dict[str, Any]]:
        """
        Generate mutants for a single function using mutmut.

        Args:
            func: The function to mutate
            max_mutants: Maximum number of mutants to generate

        Returns:
            List of mutant dictionaries
        """
        wrapper_file = self.wrappers_dir / f"{func.id}.py"
        if not wrapper_file.exists():
            raise FileNotFoundError(f"Wrapper file not found: {wrapper_file}")

        mutants = []

        # Read original code
        with open(wrapper_file, 'r', encoding='utf-8') as f:
            original_code = f.read()

        # Create temp directory for mutmut
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            temp_file = temp_path / f"{func.id}.py"

            # Copy file to temp
            shutil.copy(wrapper_file, temp_file)

            try:
                # Run mutmut
                result = subprocess.run(
                    ["mutmut", "run", "--paths-to-mutate", str(temp_file), "--no-progress"],
                    capture_output=True,
                    text=True,
                    cwd=temp_dir,
                    timeout=60
                )

                # Get mutant results
                result = subprocess.run(
                    ["mutmut", "results"],
                    capture_output=True,
                    text=True,
                    cwd=temp_dir
                )

                # Parse mutants (simplified - in production, use mutmut's JSON output)
                mutant_count = min(max_mutants, 20)  # Limit mutants

                for i in range(1, mutant_count + 1):
                    show_result = subprocess.run(
                        ["mutmut", "show", str(i)],
                        capture_output=True,
                        text=True,
                        cwd=temp_dir
                    )

                    if show_result.returncode == 0 and show_result.stdout:
                        mutants.append({
                            "id": f"{func.id}_mutant_{i}",
                            "function_id": func.id,
                            "function_name": func.name,
                            "mutant_number": i,
                            "original_code": original_code,
                            "mutated_code": show_result.stdout,
                            "library": func.library
                        })

            except subprocess.TimeoutExpired:
                print(f"  Warning: Timeout generating mutants for {func.id}")
            except FileNotFoundError:
                print(f"  Warning: mutmut not found. Install with: pip install mutmut")
                # Generate synthetic mutants as fallback
                mutants = self._generate_synthetic_mutants(func, original_code, max_mutants)

        return mutants

    def _generate_synthetic_mutants(self, func: SystemLevelFunction, original_code: str, count: int) -> List[Dict[str, Any]]:
        """
        Generate synthetic mutants when mutmut is not available.
        Uses simple string replacements for common mutation patterns.
        """
        import re

        mutation_patterns = [
            (r'==', '!='),
            (r'!=', '=='),
            (r'>=', '>'),
            (r'<=', '<'),
            (r'>', '>='),
            (r'<', '<='),
            (r'\+', '-'),
            (r'-', '+'),
            (r'\*', '/'),
            (r'/', '*'),
            (r'True', 'False'),
            (r'False', 'True'),
            (r'and', 'or'),
            (r'or', 'and'),
            (r'return ', 'return None  # '),
        ]

        mutants = []
        for i, (pattern, replacement) in enumerate(mutation_patterns[:count]):
            mutated = re.sub(pattern, replacement, original_code, count=1)
            if mutated != original_code:
                mutants.append({
                    "id": f"{func.id}_synthetic_{i+1}",
                    "function_id": func.id,
                    "function_name": func.name,
                    "mutant_number": i + 1,
                    "original_code": original_code,
                    "mutated_code": mutated,
                    "library": func.library,
                    "mutation_type": f"{pattern} -> {replacement}"
                })

        return mutants

    def generate_all_mutants(self, mutants_per_function: int = 15) -> List[Dict[str, Any]]:
        """
        Generate mutants for all training functions.

        Args:
            mutants_per_function: Number of mutants per function

        Returns:
            List of all mutant dictionaries
        """
        # First, ensure wrapper files exist
        self.generate_wrapper_files()

        all_mutants = []

        print(f"\nGenerating mutants for {len(self.training_functions)} functions...")
        for i, func in enumerate(self.training_functions):
            print(f"  [{i+1}/{len(self.training_functions)}] {func.name}...", end=" ")

            mutants = self.generate_mutants_for_function(func, mutants_per_function)
            all_mutants.extend(mutants)

            print(f"{len(mutants)} mutants")

        self.mutants_data = all_mutants
        print(f"\nTotal mutants generated: {len(all_mutants)}")

        return all_mutants

    def save_dataset(self) -> Path:
        """
        Save the complete dataset to disk.

        Returns:
            Path to the saved dataset file
        """
        output_file = DATA_DIR / "system_level_dataset.json"

        dataset = {
            "training_functions": [f.to_dict() for f in self.training_functions],
            "testing_functions": [f.to_dict() for f in self.testing_functions],
            "mutants": self.mutants_data,
            "summary": {
                "training_count": len(self.training_functions),
                "testing_count": len(self.testing_functions),
                "mutant_count": len(self.mutants_data),
                "libraries": list(set(f.library for f in self.training_functions))
            }
        }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=2)

        print(f"\nDataset saved to {output_file}")
        return output_file

    def load_dataset(self) -> SystemLevelDataset:
        """
        Load a previously saved dataset.

        Returns:
            SystemLevelDataset object
        """
        dataset_file = DATA_DIR / "system_level_dataset.json"

        if not dataset_file.exists():
            raise FileNotFoundError(
                f"Dataset not found at {dataset_file}. "
                "Run generate_all_mutants() first."
            )

        with open(dataset_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Convert back to objects
        training = [SystemLevelFunction(**f) for f in data["training_functions"]]
        testing = [SystemLevelFunction(**f) for f in data["testing_functions"]]

        return SystemLevelDataset(
            training_functions=training,
            testing_functions=testing,
            mutants=data["mutants"]
        )

    def get_dataset_summary(self) -> Dict[str, Any]:
        """Get a summary of the current dataset state."""
        return {
            "training_functions": len(self.training_functions),
            "testing_functions": len(self.testing_functions),
            "mutants_generated": len(self.mutants_data),
            "wrappers_dir": str(self.wrappers_dir),
            "libraries": sorted(set(f.library for f in self.training_functions)),
            "training_by_library": {
                lib: sum(1 for f in self.training_functions if f.library == lib)
                for lib in set(f.library for f in self.training_functions)
            }
        }


def generate_system_level_dataset(mutants_per_function: int = 15) -> SystemLevelDataset:
    """
    Convenience function to generate the complete system-level dataset.

    Args:
        mutants_per_function: Number of mutants per training function

    Returns:
        Complete SystemLevelDataset
    """
    loader = SystemLevelDatasetLoader()

    print("=" * 60)
    print("System-Level Dataset Generation")
    print("=" * 60)

    # Generate wrapper files
    loader.generate_wrapper_files()

    # Generate mutants
    loader.generate_all_mutants(mutants_per_function)

    # Save dataset
    loader.save_dataset()

    # Load and return
    return loader.load_dataset()


if __name__ == "__main__":
    print("=" * 60)
    print("Testing System-Level Dataset Loader")
    print("=" * 60)

    loader = SystemLevelDatasetLoader()

    print("\n1. Dataset Summary:")
    summary = loader.get_dataset_summary()
    for key, value in summary.items():
        print(f"   {key}: {value}")

    print("\n2. Generating wrapper files...")
    loader.generate_wrapper_files()

    print("\n3. Generating mutants (this may take a while)...")
    loader.generate_all_mutants(mutants_per_function=10)

    print("\n4. Saving dataset...")
    loader.save_dataset()

    print("\n5. Final Summary:")
    final_summary = loader.get_dataset_summary()
    for key, value in final_summary.items():
        print(f"   {key}: {value}")

    print("\n" + "=" * 60)
    print("Dataset generation complete!")
    print("=" * 60)
