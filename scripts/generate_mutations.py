"""
Mutation Generator v3 for Oneiros Engine.

Fixes from v2:
  - Skips def line (no mutating type annotations like ->)
  - Skips docstrings completely
  - Skips import lines
  - Skips comment-only lines
  - Fixed off_by_one regex (no broken backreferences)
  - Only mutates actual executable Python logic

Target: 10,000 clean mutation pairs
"""
import json
import re
import ast
import random
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Tuple

DATA_DIR = Path(__file__).parent.parent / "data"


@dataclass
class MutationPair:
    id: str
    source: str
    golden_code: str
    mutant_code: str
    entry_point: str
    test_cases: List[str]
    mutation_type: str
    mutation_description: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def get_logic_lines(code: str) -> List[int]:
    """
    Return line indices (0-based) that are actual executable logic.
    Excludes: def line, imports, docstrings, comments, blank lines.
    """
    lines = code.split('\n')
    logic_indices = []
    in_docstring = False
    past_def = False

    for i, line in enumerate(lines):
        stripped = line.strip()

        # Skip blank lines
        if not stripped:
            continue

        # Skip import lines
        if stripped.startswith('from ') or stripped.startswith('import '):
            continue

        # Skip def line (contains -> type annotation)
        if stripped.startswith('def '):
            past_def = True
            continue

        # Handle docstrings
        if '"""' in stripped or "'''" in stripped:
            # Count triple quotes
            tq_count = stripped.count('"""') + stripped.count("'''")
            if tq_count >= 2:
                # Single-line docstring like """text"""
                continue
            else:
                in_docstring = not in_docstring
                continue

        if in_docstring:
            continue

        # Skip comment-only lines
        if stripped.startswith('#'):
            continue

        # Skip decorator lines
        if stripped.startswith('@'):
            continue

        # This is a logic line!
        if past_def:
            logic_indices.append(i)

    return logic_indices


def mutate_line(line: str, ops: List[Tuple]) -> List[Tuple[str, str, str]]:
    """
    Apply mutation operators to a single line.
    Returns list of (mutated_line, mutation_type, description).
    """
    results = []
    for pattern, replacement, mtype, desc in ops:
        matches = list(re.finditer(pattern, line))
        for match in matches:
            mutated = line[:match.start()] + replacement + line[match.end():]
            if mutated != line:
                results.append((mutated, mtype, f"{desc}"))
    return results


def mutate_return(line: str) -> List[Tuple[str, str, str]]:
    """
    Specifically handle return-value mutations.
    Preserves leading whitespace properly.
    """
    stripped = line.lstrip()
    indent = line[:len(line) - len(stripped)]
    results = []

    swaps = [
        ('return True',  'return False', 'return True -> False'),
        ('return False', 'return True',  'return False -> True'),
        ('return 0',     'return 1',     'return 0 -> 1'),
        ('return 1',     'return 0',     'return 1 -> 0'),
        ('return []',    'return None',  'return [] -> None'),
        ('return None',  'return 0',     'return None -> 0'),
        ('return ""',    'return None',  'return \"\" -> None'),
    ]

    for old, new, desc in swaps:
        if stripped.startswith(old):
            mutated = indent + stripped.replace(old, new, 1)
            if mutated != line:
                results.append((mutated, 'return', desc))

    return results


# ── Mutation operators (applied to individual logic lines only) ──

LINE_MUTATIONS = [
    # Arithmetic (avoid matching -> type annotations)
    (r'(?<![->])\+(?!=)', '-', 'arithmetic', '+ -> -'),
    (r'(?<![<>!=])-(?![>])', '+', 'arithmetic', '- -> +'),
    (r'\*(?!\*)', '/', 'arithmetic', '* -> /'),
    (r'(?<!\*)\/(?!\/)', '*', 'arithmetic', '/ -> *'),
    (r'\/\/', '%', 'arithmetic', '// -> %'),
    (r'(?<!\/)%', '//', 'arithmetic', '% -> //'),
    (r'\*\*', '*', 'arithmetic', '** -> *'),

    # Comparison (will NOT match -> since we skip def lines)
    (r'<=', '>=', 'comparison', '<= -> >='),
    (r'(?<!=)<(?!=)', '>', 'comparison', '< -> >'),
    (r'>=', '<=', 'comparison', '>= -> <='),
    (r'(?<!=)>(?!=)', '<', 'comparison', '> -> <'),
    (r'==', '!=', 'comparison', '== -> !='),
    (r'!=', '==', 'comparison', '!= -> =='),

    # Logical
    (r'\band\b', 'or', 'logical', 'and -> or'),
    (r'\bor\b', 'and', 'logical', 'or -> and'),
    (r'\bnot ', '', 'negate_removal', 'remove not'),

    # Boolean (only in code, not docstrings - guaranteed by line filtering)
    (r'\bTrue\b', 'False', 'boolean', 'True -> False'),
    (r'\bFalse\b', 'True', 'boolean', 'False -> True'),

    # Boundary
    (r'(?<!\d\.)(?<!\d)\b0\b(?!\.)', '1', 'boundary', '0 -> 1'),
    (r'(?<!\d\.)(?<!\d)\b1\b(?!\.\d)', '0', 'boundary', '1 -> 0'),
    (r'(?<!\d\.)(?<!\d)\b1\b(?!\.\d)', '2', 'boundary', '1 -> 2'),

    # Index
    (r'\[0\]', '[1]', 'index', '[0] -> [1]'),
    (r'\[-1\]', '[-2]', 'index', '[-1] -> [-2]'),
    (r'\[-1\]', '[0]', 'index', '[-1] -> [0]'),

    # Membership
    (r' in ', ' not in ', 'membership', 'in -> not in'),
    (r' not in ', ' in ', 'membership', 'not in -> in'),
    (r' is not ', ' is ', 'identity', 'is not -> is'),
    (r' is ', ' is not ', 'identity', 'is -> is not'),

    # String/collection
    (r'\.strip\(\)', '', 'string', 'remove .strip()'),
    (r'\.lower\(\)', '.upper()', 'string', '.lower -> .upper'),
    (r'\.upper\(\)', '.lower()', 'string', '.upper -> .lower'),
    (r'\.append\(', '.extend([', 'collection', '.append -> .extend'),
]

# Off-by-one mutations (applied differently - full line match)
def apply_off_by_one(line: str) -> List[Tuple[str, str, str]]:
    """Apply off-by-one mutations with proper string substitution."""
    results = []

    # range(len(x)) -> range(len(x) - 1)
    m = re.search(r'range\(len\((\w+)\)\)', line)
    if m:
        var = m.group(1)
        mutated = line[:m.start()] + f'range(len({var}) - 1)' + line[m.end():]
        results.append((mutated, 'off_by_one', f'range(len({var})) -> range(len({var})-1)'))

    # range(n) -> range(n - 1) where n is a variable
    m = re.search(r'range\(([a-zA-Z_]\w*)\)', line)
    if m and m.group(1) != 'len':
        var = m.group(1)
        mutated = line[:m.start()] + f'range({var} - 1)' + line[m.end():]
        results.append((mutated, 'off_by_one', f'range({var}) -> range({var}-1)'))

    # range(a, b) -> range(a, b - 1)
    m = re.search(r'range\((\w+),\s*(\w+)\)', line)
    if m:
        a, b = m.group(1), m.group(2)
        mutated = line[:m.start()] + f'range({a}, {b} - 1)' + line[m.end():]
        results.append((mutated, 'off_by_one', f'range({a},{b}) -> range({a},{b}-1)'))

    # i + 1 -> i
    m = re.search(r'(\w+)\s*\+\s*1\b', line)
    if m:
        var = m.group(1)
        mutated = line[:m.start()] + var + line[m.end():]
        results.append((mutated, 'off_by_one', f'{var}+1 -> {var}'))

    # i - 1 -> i
    m = re.search(r'(\w+)\s*-\s*1\b', line)
    if m:
        var = m.group(1)
        mutated = line[:m.start()] + var + line[m.end():]
        results.append((mutated, 'off_by_one', f'{var}-1 -> {var}'))

    return results


def is_valid_python(code: str) -> bool:
    """Quick syntax check."""
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def is_clean_mutant(code: str, mutant: str) -> bool:
    """Check mutant is valid Python and doesn't introduce regex artifacts."""
    if not is_valid_python(mutant):
        return False
    if mutant == code:
        return False
    # Reject if mutation introduced backslash sequences not in original
    for seq in ['\\1', '\\2', '\\3']:
        if seq in mutant and seq not in code:
            return False
    return True


def generate_mutations_for_function(code: str) -> List[Tuple[str, str, str]]:
    """
    Generate all valid mutations for a function.
    Every mutant is validated with ast.parse and artifact checks.
    Returns list of (full_mutated_code, mutation_type, description).
    """
    lines = code.split('\n')
    logic_indices = get_logic_lines(code)
    results = []

    for li in logic_indices:
        line = lines[li]

        # Apply regex mutations
        for mutated_line, mtype, desc in mutate_line(line, LINE_MUTATIONS):
            new_lines = lines.copy()
            new_lines[li] = mutated_line
            full_mutant = '\n'.join(new_lines)
            if is_clean_mutant(code, full_mutant):
                results.append((full_mutant, mtype, f"{desc} (line {li+1})"))

        # Apply off-by-one mutations
        for mutated_line, mtype, desc in apply_off_by_one(line):
            new_lines = lines.copy()
            new_lines[li] = mutated_line
            full_mutant = '\n'.join(new_lines)
            if is_clean_mutant(code, full_mutant):
                results.append((full_mutant, mtype, f"{desc} (line {li+1})"))

        # Apply return-value mutations
        for mutated_line, mtype, desc in mutate_return(line):
            new_lines = lines.copy()
            new_lines[li] = mutated_line
            full_mutant = '\n'.join(new_lines)
            if is_clean_mutant(code, full_mutant):
                results.append((full_mutant, mtype, f"{desc} (line {li+1})"))

    return results


def load_dataset() -> List[Dict]:
    path = DATA_DIR / "unified_dataset.json"
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def generate(target: int = 10000) -> List[MutationPair]:
    print("=" * 60)
    print("MUTATION GENERATOR v3 (body-only, no annotations)")
    print(f"Target: {target:,} pairs")
    print("=" * 60)

    dataset = load_dataset()
    print(f"\nLoaded {len(dataset)} functions")

    pairs: List[MutationPair] = []
    count = 0
    type_counts: Dict[str, int] = {}

    # First pass
    for i, func in enumerate(dataset):
        if len(pairs) >= target:
            break

        code = func.get('code', '')
        if not code or len(code.strip()) < 30:
            continue

        mutations = generate_mutations_for_function(code)
        random.shuffle(mutations)

        for mutant_code, mtype, desc in mutations:
            if len(pairs) >= target:
                break
            count += 1
            type_counts[mtype] = type_counts.get(mtype, 0) + 1
            pairs.append(MutationPair(
                id=f"{func['id']}_mut_{count:05d}",
                source=func.get('source', 'unknown'),
                golden_code=code,
                mutant_code=mutant_code,
                entry_point=func.get('entry_point', ''),
                test_cases=func.get('test_cases', []),
                mutation_type=mtype,
                mutation_description=desc,
            ))

        if (i + 1) % 200 == 0:
            print(f"  [{i+1}/{len(dataset)}] {len(pairs):,} pairs")

    # Second pass with compound mutations if needed
    if len(pairs) < target:
        print(f"\n  First pass: {len(pairs):,}. Running compound pass...")
        random.shuffle(dataset)
        for func in dataset:
            if len(pairs) >= target:
                break
            code = func.get('code', '')
            if not code or len(code.strip()) < 30:
                continue
            mutations = generate_mutations_for_function(code)
            if len(mutations) >= 2:
                for j in range(min(5, len(mutations))):
                    if len(pairs) >= target:
                        break
                    m1_code, t1, d1 = mutations[j]
                    secondary = generate_mutations_for_function(m1_code)
                    if secondary:
                        m2_code, t2, d2 = random.choice(secondary)
                        count += 1
                        type_counts['compound'] = type_counts.get('compound', 0) + 1
                        pairs.append(MutationPair(
                            id=f"{func['id']}_cmp_{count:05d}",
                            source=func.get('source', 'unknown'),
                            golden_code=code,
                            mutant_code=m2_code,
                            entry_point=func.get('entry_point', ''),
                            test_cases=func.get('test_cases', []),
                            mutation_type='compound',
                            mutation_description=f"{d1} AND {d2}",
                        ))

    print(f"\nGenerated: {len(pairs):,} pairs")
    print("\nBy mutation type:")
    for t, c in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"  {t:20s} {c:>6,}")

    return pairs

def save(pairs: List[MutationPair]):
    out = DATA_DIR / "mutation_pairs.json"
    with open(out, 'w', encoding='utf-8') as f:
        json.dump([p.to_dict() for p in pairs], f, indent=2)
    print(f"\nSaved to: {out}")

    identical = sum(1 for p in pairs if p.golden_code == p.mutant_code)
    print(f"Sanity check - identical pairs: {identical} (should be 0)")

    sources = {}
    for p in pairs:
        sources[p.source] = sources.get(p.source, 0) + 1
    print("\nBy source:")
    for s, c in sorted(sources.items()):
        print(f"  {s}: {c:,}")

if __name__ == "__main__":
    pairs = generate(target=10000)
    save(pairs)
    print(f"\nDone! {len(pairs):,} clean mutation pairs.")
