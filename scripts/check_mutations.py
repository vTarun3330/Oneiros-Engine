"""Show real mutation examples with exact diffs."""
import json
import difflib

muts = json.load(open("data/mutation_pairs.json", encoding="utf-8"))

# Show 3 examples with clear diffs
for idx in [0, 100, 500]:
    m = muts[idx]
    print(f"{'='*60}")
    print(f"PAIR #{idx}")
    print(f"ID:   {m['id']}")
    print(f"TYPE: {m['mutation_type']}")
    print(f"DESC: {m['mutation_description']}")
    print(f"{'='*60}")

    golden_lines = m["golden_code"].splitlines(keepends=True)
    mutant_lines = m["mutant_code"].splitlines(keepends=True)

    diff = list(difflib.unified_diff(
        golden_lines, mutant_lines,
        fromfile="GOLDEN", tofile="MUTANT", lineterm=""
    ))

    if diff:
        print("DIFF:")
        for line in diff[:30]:
            print(line.rstrip())
    else:
        print("WARNING: golden == mutant (no difference found!)")

    print(f"\nGolden length: {len(m['golden_code'])} chars")
    print(f"Mutant length: {len(m['mutant_code'])} chars")
    print(f"Are they equal? {m['golden_code'] == m['mutant_code']}")
    print()

# Count how many are actually identical
identical = sum(1 for m in muts if m["golden_code"] == m["mutant_code"])
print(f"\n{'='*60}")
print(f"TOTAL: {len(muts)} pairs")
print(f"IDENTICAL (golden==mutant): {identical}")
print(f"ACTUALLY DIFFERENT: {len(muts) - identical}")
print(f"{'='*60}")
