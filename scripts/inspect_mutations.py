"""Manually inspect 5 diverse mutation pairs with full code."""
import json
import difflib

muts = json.load(open("data/mutation_pairs.json", encoding="utf-8"))

# Pick diverse examples: different types, different sources
targets = {
    "comparison": None,
    "arithmetic": None,
    "boundary": None,
    "return": None,
    "off_by_one": None,
}

for m in muts:
    t = m["mutation_type"]
    if t in targets and targets[t] is None:
        # Prefer short functions for readability
        if len(m["golden_code"]) < 400:
            targets[t] = m

for mtype, m in targets.items():
    if m is None:
        continue
    print("=" * 70)
    print(f"TYPE: {m['mutation_type']}  |  DESC: {m['mutation_description']}")
    print(f"ID: {m['id']}  |  SOURCE: {m['source']}")
    print(f"ENTRY POINT: {m['entry_point']}")
    print(f"TEST CASES: {len(m['test_cases'])} tests")
    print("=" * 70)

    print("\n--- GOLDEN CODE ---")
    print(m["golden_code"])

    print("\n--- MUTANT CODE ---")
    print(m["mutant_code"])

    print("\n--- DIFF ---")
    g_lines = m["golden_code"].splitlines(keepends=True)
    m_lines = m["mutant_code"].splitlines(keepends=True)
    diff = difflib.unified_diff(g_lines, m_lines, fromfile="golden", tofile="mutant")
    for line in diff:
        print(line.rstrip())

    if m["test_cases"]:
        print("\n--- FIRST TEST CASE ---")
        print(m["test_cases"][0][:300])

    print("\n\n")
