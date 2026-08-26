"""
Mutation Engine for generating buggy variants using mutmut.

This module creates mutants (buggy versions) of golden functions
to build the Unified Evaluation Harness.
"""
import json
import subprocess
import tempfile
import shutil
import os
import re
import ast
import random
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
from copy import deepcopy

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    dataset_config,
    GOLDEN_DIR,
    MUTANTS_DIR,
    DATA_DIR
)
from harness.dataset_loader import TargetFunction, DatasetLoader


class MutationType(Enum):
    """Types of mutations that can be applied."""
    ARITHMETIC_OPERATOR = "arithmetic_operator"      # + -> -, * -> /
    COMPARISON_OPERATOR = "comparison_operator"      # < -> <=, == -> !=
    LOGICAL_OPERATOR = "logical_operator"            # and -> or
    BOUNDARY_VALUE = "boundary_value"                # off-by-one errors
    RETURN_VALUE = "return_value"                    # return True -> return False
    CONDITION_NEGATION = "condition_negation"        # if x -> if not x
    REMOVE_STATEMENT = "remove_statement"            # delete a line
    CONSTANT_REPLACEMENT = "constant_replacement"    # 0 -> 1, "" -> "x"
    VARIABLE_SWAP = "variable_swap"                  # swap variable names


@dataclass
class Mutation:
    """Represents a single mutation applied to code."""
    id: str
    type: MutationType
    original_code: str
    mutated_code: str
    location: Dict[str, int]  # {"line": n, "col": m}
    description: str

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["type"] = self.type.value
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Mutation":
        data["type"] = MutationType(data["type"])
        return cls(**data)


@dataclass
class Mutant:
    """Represents a mutated version of a target function."""
    id: str
    parent_id: str              # ID of the golden function
    parent_name: str            # Name of the original function
    code: str                   # Full mutated code
    mutation: Mutation          # The applied mutation
    entry_point: str            # Function to call for testing

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["mutation"] = self.mutation.to_dict()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Mutant":
        data["mutation"] = Mutation.from_dict(data["mutation"])
        return cls(**data)


class MutationEngine:
    """
    Engine for generating mutations in Python code.

    Uses a combination of AST-based mutations and pattern-based
    replacements to create subtle bugs.
    """

    # Mutation patterns
    ARITHMETIC_SWAPS = [
        ('+', '-'), ('-', '+'), ('*', '/'), ('/', '*'),
        ('//', '/'), ('/', '//'), ('%', '/'), ('**', '*')
    ]

    COMPARISON_SWAPS = [
        ('<', '<='), ('<=', '<'), ('>', '>='), ('>=', '>'),
        ('==', '!='), ('!=', '=='), ('<', '>'), ('>', '<'),
        ('<=', '>='), ('>=', '<='), ('==', 'is'), ('is', '==')
    ]

    LOGICAL_SWAPS = [
        (' and ', ' or '), (' or ', ' and '),
        ('True', 'False'), ('False', 'True'),
        (' not ', ' ')
    ]

    BOUNDARY_MUTATIONS = [
        (r'\b(\d+)\b', lambda m: str(int(m.group(1)) + 1)),  # n -> n+1
        (r'\b(\d+)\b', lambda m: str(int(m.group(1)) - 1)),  # n -> n-1
        (r'range\((\w+)\)', r'range(\1 - 1)'),               # range(n) -> range(n-1)
        (r'range\((\w+)\)', r'range(\1 + 1)'),               # range(n) -> range(n+1)
        (r'\[:(\w+)\]', r'[:\1-1]'),                         # [:n] -> [:n-1]
        (r'\[(\w+):\]', r'[\1+1:]'),                         # [n:] -> [n+1:]
    ]

    def __init__(self, seed: int = 42):
        """Initialize the mutation engine."""
        random.seed(seed)
        self.mutation_counter = 0

    def _get_mutation_id(self) -> str:
        """Generate a unique mutation ID."""
        self.mutation_counter += 1
        return f"mut_{self.mutation_counter:04d}"

    def _find_operator_positions(
        self,
        code: str,
        operators: List[Tuple[str, str]]
    ) -> List[Tuple[int, str, str]]:
        """
        Find positions of operators that can be mutated.

        Returns:
            List of (position, original, replacement) tuples
        """
        positions = []
        for orig, repl in operators:
            # Find all occurrences
            idx = 0
            while True:
                idx = code.find(orig, idx)
                if idx == -1:
                    break
                # Avoid replacing within strings or comments
                if not self._is_in_string_or_comment(code, idx):
                    positions.append((idx, orig, repl))
                idx += len(orig)
        return positions

    def _is_in_string_or_comment(self, code: str, pos: int) -> bool:
        """Check if position is inside a string or comment."""
        # Simple heuristic - count quotes before position
        before = code[:pos]

        # Check if in a comment
        last_newline = before.rfind('\n')
        line_start = last_newline + 1 if last_newline != -1 else 0
        line_before_pos = before[line_start:]
        if '#' in line_before_pos:
            return True

        # Check if in a string (very simplified)
        single_quotes = before.count("'") - before.count("\\'")
        double_quotes = before.count('"') - before.count('\\"')
        triple_single = before.count("'''")
        triple_double = before.count('"""')

        # Odd number of quotes means we're inside a string
        in_single = (single_quotes - triple_single * 3) % 2 == 1
        in_double = (double_quotes - triple_double * 3) % 2 == 1

        return in_single or in_double

    def _get_line_number(self, code: str, pos: int) -> int:
        """Get line number for a position in code."""
        return code[:pos].count('\n') + 1

    def mutate_arithmetic(self, code: str) -> List[Mutation]:
        """Generate arithmetic operator mutations."""
        mutations = []
        positions = self._find_operator_positions(code, self.ARITHMETIC_SWAPS)

        for pos, orig, repl in positions:
            mutated = code[:pos] + repl + code[pos + len(orig):]

            mutations.append(Mutation(
                id=self._get_mutation_id(),
                type=MutationType.ARITHMETIC_OPERATOR,
                original_code=orig,
                mutated_code=repl,
                location={"line": self._get_line_number(code, pos), "col": pos},
                description=f"Changed '{orig}' to '{repl}'"
            ))

        return mutations

    def mutate_comparison(self, code: str) -> List[Mutation]:
        """Generate comparison operator mutations."""
        mutations = []
        positions = self._find_operator_positions(code, self.COMPARISON_SWAPS)

        for pos, orig, repl in positions:
            mutations.append(Mutation(
                id=self._get_mutation_id(),
                type=MutationType.COMPARISON_OPERATOR,
                original_code=orig,
                mutated_code=repl,
                location={"line": self._get_line_number(code, pos), "col": pos},
                description=f"Changed '{orig}' to '{repl}'"
            ))

        return mutations

    def mutate_logical(self, code: str) -> List[Mutation]:
        """Generate logical operator mutations."""
        mutations = []
        positions = self._find_operator_positions(code, self.LOGICAL_SWAPS)

        for pos, orig, repl in positions:
            mutations.append(Mutation(
                id=self._get_mutation_id(),
                type=MutationType.LOGICAL_OPERATOR,
                original_code=orig,
                mutated_code=repl,
                location={"line": self._get_line_number(code, pos), "col": pos},
                description=f"Changed '{orig}' to '{repl}'"
            ))

        return mutations

    def mutate_boundary(self, code: str) -> List[Mutation]:
        """Generate boundary value mutations (off-by-one errors)."""
        mutations = []

        for pattern, replacement in self.BOUNDARY_MUTATIONS:
            if callable(replacement):
                # For lambda replacements
                for match in re.finditer(pattern, code):
                    if not self._is_in_string_or_comment(code, match.start()):
                        try:
                            new_val = replacement(match)
                            mutations.append(Mutation(
                                id=self._get_mutation_id(),
                                type=MutationType.BOUNDARY_VALUE,
                                original_code=match.group(0),
                                mutated_code=new_val,
                                location={"line": self._get_line_number(code, match.start()), "col": match.start()},
                                description=f"Boundary mutation: '{match.group(0)}' to '{new_val}'"
                            ))
                        except:
                            pass
            else:
                # For string replacements
                for match in re.finditer(pattern, code):
                    if not self._is_in_string_or_comment(code, match.start()):
                        new_val = re.sub(pattern, replacement, match.group(0))
                        mutations.append(Mutation(
                            id=self._get_mutation_id(),
                            type=MutationType.BOUNDARY_VALUE,
                            original_code=match.group(0),
                            mutated_code=new_val,
                            location={"line": self._get_line_number(code, match.start()), "col": match.start()},
                            description=f"Boundary mutation: '{match.group(0)}' to '{new_val}'"
                        ))

        return mutations

    def mutate_return_value(self, code: str) -> List[Mutation]:
        """Generate return value mutations."""
        mutations = []

        # Find return statements
        return_pattern = r'return\s+(.+?)(?:\n|$)'

        for match in re.finditer(return_pattern, code):
            if self._is_in_string_or_comment(code, match.start()):
                continue

            return_val = match.group(1).strip()
            line = self._get_line_number(code, match.start())

            # Generate alternative return values based on type
            alternatives = []

            if return_val in ('True', 'False'):
                alternatives.append('False' if return_val == 'True' else 'True')
            elif return_val == 'None':
                alternatives.extend(['0', '[]', '""'])
            elif return_val.isdigit():
                alternatives.append(str(int(return_val) + 1))
                alternatives.append(str(int(return_val) - 1))
            elif return_val == '[]':
                alternatives.append('None')
            elif return_val == '{}':
                alternatives.append('None')
            elif return_val.startswith('['):
                alternatives.append('[]')
            elif return_val.startswith('"') or return_val.startswith("'"):
                alternatives.append('""')
            else:
                # Generic - try negating or returning None
                alternatives.append('None')

            for alt in alternatives[:2]:  # Limit to 2 alternatives per return
                mutations.append(Mutation(
                    id=self._get_mutation_id(),
                    type=MutationType.RETURN_VALUE,
                    original_code=f"return {return_val}",
                    mutated_code=f"return {alt}",
                    location={"line": line, "col": match.start()},
                    description=f"Changed return value from '{return_val}' to '{alt}'"
                ))

        return mutations

    def mutate_condition_negation(self, code: str) -> List[Mutation]:
        """Generate condition negation mutations."""
        mutations = []

        # Find if/elif/while conditions
        condition_pattern = r'(if|elif|while)\s+(.+?):'

        for match in re.finditer(condition_pattern, code):
            if self._is_in_string_or_comment(code, match.start()):
                continue

            keyword = match.group(1)
            condition = match.group(2).strip()
            line = self._get_line_number(code, match.start())

            # Negate the condition
            if condition.startswith('not '):
                negated = condition[4:]
            elif ' and ' in condition or ' or ' in condition:
                negated = f"not ({condition})"
            else:
                negated = f"not {condition}"

            mutations.append(Mutation(
                id=self._get_mutation_id(),
                type=MutationType.CONDITION_NEGATION,
                original_code=f"{keyword} {condition}:",
                mutated_code=f"{keyword} {negated}:",
                location={"line": line, "col": match.start()},
                description=f"Negated condition: '{condition}' to '{negated}'"
            ))

        return mutations

    def apply_mutation(self, code: str, mutation: Mutation) -> str:
        """Apply a mutation at its recorded source offset.

        Applying the first matching string can mutate a different occurrence
        when a literal or operator appears more than once in a function.
        """
        offset = mutation.location["col"]
        original = mutation.original_code
        if code[offset : offset + len(original)] != original:
            raise ValueError(
                f"Mutation location no longer matches {original!r} at offset {offset}"
            )
        return code[:offset] + mutation.mutated_code + code[offset + len(original):]
    def generate_all_mutations(self, code: str) -> List[Mutation]:
        """Generate all possible mutations for a piece of code."""
        all_mutations = []

        all_mutations.extend(self.mutate_arithmetic(code))
        all_mutations.extend(self.mutate_comparison(code))
        all_mutations.extend(self.mutate_logical(code))
        all_mutations.extend(self.mutate_boundary(code))
        all_mutations.extend(self.mutate_return_value(code))
        all_mutations.extend(self.mutate_condition_negation(code))

        return all_mutations

    def generate_mutants(
        self,
        function: TargetFunction,
        num_mutants: int = None
    ) -> List[Mutant]:
        """
        Generate mutant versions of a target function.

        Args:
            function: The golden function to mutate
            num_mutants: Number of mutants to generate (default from config)

        Returns:
            List of Mutant objects
        """
        num_mutants = num_mutants or dataset_config.mutants_per_function

        # Generate all possible mutations
        all_mutations = self.generate_all_mutations(function.code)

        if not all_mutations:
            print(f"Warning: No mutations possible for {function.id}")
            return []

        # Select diverse mutations
        selected = self._select_diverse_mutations(all_mutations, num_mutants)

        # Create mutant objects
        mutants = []
        for i, mutation in enumerate(selected):
            mutated_code = self.apply_mutation(function.code, mutation)

            # Verify the mutated code is valid Python
            if not self._is_valid_python(mutated_code):
                continue

            mutant = Mutant(
                id=f"{function.id}_mutant_{i+1:03d}",
                parent_id=function.id,
                parent_name=function.name,
                code=mutated_code,
                mutation=mutation,
                entry_point=function.entry_point
            )
            mutants.append(mutant)

        return mutants

    def _select_diverse_mutations(
        self,
        mutations: List[Mutation],
        n: int
    ) -> List[Mutation]:
        """Select diverse mutations across different types."""
        if len(mutations) <= n:
            return mutations

        # Group by type
        by_type: Dict[MutationType, List[Mutation]] = {}
        for m in mutations:
            if m.type not in by_type:
                by_type[m.type] = []
            by_type[m.type].append(m)

        # Select from each type proportionally
        selected = []
        types = list(by_type.keys())
        per_type = max(1, n // len(types))

        for t in types:
            type_mutations = by_type[t]
            random.shuffle(type_mutations)
            selected.extend(type_mutations[:per_type])

        # Fill remaining slots randomly
        remaining = [m for m in mutations if m not in selected]
        random.shuffle(remaining)
        selected.extend(remaining[:n - len(selected)])

        return selected[:n]

    def _is_valid_python(self, code: str) -> bool:
        """Check if code is valid Python."""
        try:
            ast.parse(code)
            return True
        except SyntaxError:
            return False


class MutantHarness:
    """
    Manages the collection of mutants for the evaluation harness.
    """

    def __init__(self):
        self.engine = MutationEngine()
        self.mutants: List[Mutant] = []
        self.golden_functions: List[TargetFunction] = []

    def build_harness(
        self,
        functions: List[TargetFunction] = None
    ) -> List[Mutant]:
        """
        Build the complete mutant harness from golden functions.

        Args:
            functions: List of golden functions (loads from file if None)

        Returns:
            List of all generated mutants
        """
        # Load golden functions if not provided
        if functions is None:
            loader = DatasetLoader()
            functions = loader.load_golden_functions()

        self.golden_functions = functions
        self.mutants = []

        print(f"\nGenerating mutants for {len(functions)} golden functions...")
        print("-" * 50)

        for func in functions:
            mutants = self.engine.generate_mutants(func)
            self.mutants.extend(mutants)
            print(f"  {func.id}: {len(mutants)} mutants")

        print("-" * 50)
        print(f"Total mutants generated: {len(self.mutants)}")

        return self.mutants

    def save_harness(self) -> Path:
        """
        Save the mutant harness to disk.

        Returns:
            Path to the metadata file
        """
        if not self.mutants:
            raise ValueError("No mutants to save. Run build_harness() first.")

        # Save metadata
        metadata_file = MUTANTS_DIR / "mutants_metadata.json"
        metadata = [m.to_dict() for m in self.mutants]

        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)

        # Save individual mutant files
        for mutant in self.mutants:
            mutant_file = MUTANTS_DIR / f"{mutant.id}.py"
            with open(mutant_file, 'w', encoding='utf-8') as f:
                f.write(f"# Mutant: {mutant.id}\n")
                f.write(f"# Parent: {mutant.parent_id}\n")
                f.write(f"# Mutation Type: {mutant.mutation.type.value}\n")
                f.write(f"# Description: {mutant.mutation.description}\n")
                f.write(f"# Entry Point: {mutant.entry_point}\n\n")
                f.write(mutant.code)

        # Save summary
        summary = self._generate_summary()
        summary_file = MUTANTS_DIR / "harness_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)

        print(f"\nSaved {len(self.mutants)} mutants to {MUTANTS_DIR}")
        return metadata_file

    def load_harness(self) -> List[Mutant]:
        """
        Load a previously saved mutant harness.

        Returns:
            List of Mutant objects
        """
        metadata_file = MUTANTS_DIR / "mutants_metadata.json"

        if not metadata_file.exists():
            raise FileNotFoundError(
                f"No harness found at {metadata_file}. "
                "Run build_harness() and save_harness() first."
            )

        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)

        self.mutants = [Mutant.from_dict(m) for m in metadata]
        print(f"Loaded {len(self.mutants)} mutants")

        return self.mutants

    def _generate_summary(self) -> Dict[str, Any]:
        """Generate a summary of the harness."""
        # Count mutations by type
        type_counts = {}
        for mutant in self.mutants:
            t = mutant.mutation.type.value
            type_counts[t] = type_counts.get(t, 0) + 1

        # Count mutants per parent
        parent_counts = {}
        for mutant in self.mutants:
            p = mutant.parent_id
            parent_counts[p] = parent_counts.get(p, 0) + 1

        return {
            "total_mutants": len(self.mutants),
            "total_golden_functions": len(self.golden_functions),
            "mutation_type_distribution": type_counts,
            "mutants_per_function": parent_counts,
            "average_mutants_per_function": len(self.mutants) / max(1, len(self.golden_functions))
        }

    def get_mutant(self, mutant_id: str) -> Optional[Mutant]:
        """Get a specific mutant by ID."""
        for m in self.mutants:
            if m.id == mutant_id:
                return m
        return None

    def get_mutants_for_function(self, function_id: str) -> List[Mutant]:
        """Get all mutants for a specific golden function."""
        return [m for m in self.mutants if m.parent_id == function_id]


def build_evaluation_harness() -> Tuple[List[TargetFunction], List[Mutant]]:
    """
    Convenience function to build the complete evaluation harness.

    Returns:
        Tuple of (golden_functions, mutants)
    """
    from harness.dataset_loader import generate_golden_functions

    print("=" * 60)
    print("Building Unified Evaluation Harness")
    print("=" * 60)

    # Step 1: Generate golden functions
    print("\n[Phase 1] Generating golden functions...")
    functions = generate_golden_functions(n=dataset_config.num_target_functions)

    # Step 2: Generate mutants
    print("\n[Phase 2] Generating mutants...")
    harness = MutantHarness()
    mutants = harness.build_harness(functions)
    harness.save_harness()

    print("\n" + "=" * 60)
    print("Evaluation Harness Complete!")
    print(f"  - Golden functions: {len(functions)}")
    print(f"  - Mutants: {len(mutants)}")
    print("=" * 60)

    return functions, mutants


if __name__ == "__main__":
    # Test the mutation engine
    print("=" * 60)
    print("Testing Mutation Engine")
    print("=" * 60)

    # Sample code to mutate
    sample_code = '''
def is_prime(n):
    """Check if a number is prime."""
    if n < 2:
        return False
    for i in range(2, int(n ** 0.5) + 1):
        if n % i == 0:
            return False
    return True
'''

    print("\nOriginal code:")
    print(sample_code)

    engine = MutationEngine()
    mutations = engine.generate_all_mutations(sample_code)

    print(f"\nGenerated {len(mutations)} possible mutations:")
    for m in mutations[:10]:
        print(f"  [{m.type.value}] Line {m.location['line']}: {m.description}")

    if len(mutations) > 10:
        print(f"  ... and {len(mutations) - 10} more")

    # Apply first mutation
    if mutations:
        mutated = engine.apply_mutation(sample_code, mutations[0])
        print(f"\nExample mutation applied ({mutations[0].description}):")
        print(mutated)

    print("\n" + "=" * 60)
    print("Building full evaluation harness...")
    print("=" * 60)

    functions, mutants = build_evaluation_harness()
