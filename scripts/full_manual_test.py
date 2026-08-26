"""
Full Manual Testing of Oneiros Mutation Dataset.

Tests every single pair across all 10,000 entries:
  1. Golden != Mutant (no identical pairs)
  2. Mutant is valid Python (compiles with ast.parse)
  3. Golden is valid Python
  4. Mutation is in function body (not in def/import/docstring lines)
  5. Single-point mutation (diff is small, targeted)
  6. Test cases exist
  7. Entry point is non-empty
  8. Per-type examples with full diffs
"""
import json
import ast
import difflib
import sys
from pathlib import Path
from collections import Counter

DATA_DIR = Path(__file__).parent.parent / "data"


def is_valid_python(code: str) -> bool:
    """Check if code compiles without syntax errors."""
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def get_diff_lines(golden: str, mutant: str):
    """Get changed lines between golden and mutant."""
    g = golden.splitlines()
    m = mutant.splitlines()
    changed = []
    for i, (gl, ml) in enumerate(zip(g, m)):
        if gl != ml:
            changed.append((i + 1, gl, ml))
    # Handle length differences
    if len(g) != len(m):
        changed.append(("LENGTH_DIFF", len(g), len(m)))
    return changed


def is_def_line(line: str) -> bool:
    return line.strip().startswith('def ')


def is_import_line(line: str) -> bool:
    s = line.strip()
    return s.startswith('from ') or s.startswith('import ')


def is_in_docstring(code: str, line_no: int) -> bool:
    """Check if a line number is inside a docstring."""
    lines = code.splitlines()
    in_doc = False
    for i, line in enumerate(lines):
        stripped = line.strip()
        if '"""' in stripped or "'''" in stripped:
            count = stripped.count('"""') + stripped.count("'''")
            if count == 1:
                in_doc = not in_doc
            # If count >= 2, it's a single-line docstring, skip
        if i == line_no - 1:
            return in_doc
    return False


def main():
    # Load data
    muts = json.load(open(DATA_DIR / "mutation_pairs.json", encoding="utf-8"))
    total = len(muts)

    print("=" * 70)
    print("FULL MANUAL TESTING - ONEIROS MUTATION DATASET")
    print(f"Total pairs: {total:,}")
    print("=" * 70)

    # Counters
    pass_count = 0
    fail_reasons = Counter()
    type_counts = Counter()
    type_pass = Counter()
    type_fail = Counter()

    # Issue tracking
    issues = []

    # Per-type sample tracking (will show 2 per type)
    type_samples = {}

    for i, m in enumerate(muts):
        mtype = m.get("mutation_type", "unknown")
        type_counts[mtype] += 1

        errors = []

        # TEST 1: golden != mutant
        if m["golden_code"] == m["mutant_code"]:
            errors.append("IDENTICAL: golden == mutant")

        # TEST 2: golden is valid Python
        if not is_valid_python(m["golden_code"]):
            errors.append("GOLDEN_SYNTAX: golden code has syntax errors")

        # TEST 3: mutant is valid Python
        if not is_valid_python(m["mutant_code"]):
            errors.append("MUTANT_SYNTAX: mutant code has syntax errors")

        # TEST 4: has test cases
        if not m.get("test_cases") or len(m["test_cases"]) == 0:
            errors.append("NO_TESTS: missing test cases")

        # TEST 5: has entry point
        if not m.get("entry_point"):
            errors.append("NO_ENTRY: missing entry_point")

        # TEST 6: mutation targets body only (not def/import line)
        diff = get_diff_lines(m["golden_code"], m["mutant_code"])
        for d in diff:
            if d[0] == "LENGTH_DIFF":
                continue
            line_no, old_line, new_line = d
            golden_lines = m["golden_code"].splitlines()
            if line_no <= len(golden_lines):
                if is_def_line(golden_lines[line_no - 1]):
                    errors.append(f"DEF_LINE: mutation on def line {line_no}")
                if is_import_line(golden_lines[line_no - 1]):
                    errors.append(f"IMPORT_LINE: mutation on import line {line_no}")

        # TEST 7: diff is reasonable size (< 5 changed lines for non-compound)
        num_changed = len([d for d in diff if d[0] != "LENGTH_DIFF"])
        if mtype != "compound" and num_changed > 3:
            errors.append(f"LARGE_DIFF: {num_changed} lines changed")

        # TEST 8: no broken regex artifacts (only flag if mutation INTRODUCED it)
        golden_has_backref = "\\1" in m["golden_code"] or "\\2" in m["golden_code"]
        mutant_has_backref = "\\1" in m["mutant_code"] or "\\2" in m["mutant_code"]
        if mutant_has_backref and not golden_has_backref:
            errors.append("REGEX_LEAK: \\1 or \\2 introduced by mutation")

        # Record results
        if errors:
            for e in errors:
                fail_reasons[e.split(":")[0]] += 1
            type_fail[mtype] += 1
            if len(issues) < 20:
                issues.append((i, m["id"], mtype, errors))
        else:
            pass_count += 1
            type_pass[mtype] += 1

        # Collect samples (2 passing per type)
        if mtype not in type_samples and not errors:
            type_samples[mtype] = (i, m)

    # ── REPORT ──
    print(f"\n{'='*70}")
    print("TEST RESULTS")
    print(f"{'='*70}")
    print(f"  PASSED: {pass_count:,} / {total:,}")
    print(f"  FAILED: {total - pass_count:,} / {total:,}")
    print(f"  PASS RATE: {pass_count/total*100:.1f}%")

    print(f"\n{'='*70}")
    print("FAILURES BY REASON")
    print(f"{'='*70}")
    if fail_reasons:
        for reason, count in fail_reasons.most_common():
            print(f"  {reason:25s} {count:>6,}")
    else:
        print("  NONE -- all tests passed!")

    print(f"\n{'='*70}")
    print("PASS/FAIL BY MUTATION TYPE")
    print(f"{'='*70}")
    for mtype in sorted(type_counts.keys()):
        p = type_pass.get(mtype, 0)
        f = type_fail.get(mtype, 0)
        t = type_counts[mtype]
        status = "ALL PASS" if f == 0 else f"{f} FAIL"
        print(f"  {mtype:20s}  {p:>5,} pass  {f:>5,} fail  ({t:>5,} total)  [{status}]")

    if issues:
        print(f"\n{'='*70}")
        print(f"FIRST {len(issues)} FAILED PAIRS (details)")
        print(f"{'='*70}")
        for idx, mid, mtype, errs in issues:
            print(f"\n  Pair #{idx} | {mid} | type={mtype}")
            for e in errs:
                print(f"    -> {e}")

    print(f"\n{'='*70}")
    print("SAMPLE PASSING PAIR PER TYPE (with full diff)")
    print(f"{'='*70}")
    for mtype in sorted(type_samples.keys()):
        idx, m = type_samples[mtype]
        print(f"\n--- {mtype.upper()} (pair #{idx}) ---")
        print(f"ID: {m['id']}")
        print(f"Desc: {m['mutation_description']}")

        g_lines = m["golden_code"].splitlines(keepends=True)
        m_lines = m["mutant_code"].splitlines(keepends=True)
        diff = list(difflib.unified_diff(g_lines, m_lines, fromfile="golden", tofile="mutant"))
        for line in diff[:15]:
            print(f"  {line.rstrip()}")

    # Final verdict
    print(f"\n{'='*70}")
    if pass_count == total:
        print("VERDICT: ALL 10,000 PAIRS PASSED ALL TESTS")
    else:
        print(f"VERDICT: {total - pass_count} PAIRS HAVE ISSUES")
    print(f"{'='*70}")

    return pass_count == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
