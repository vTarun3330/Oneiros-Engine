"""Check if REGEX_LEAK failures are false positives (backslash already in golden)."""
import json

muts = json.load(open("data/mutation_pairs.json", encoding="utf-8"))

# Check all pairs that have \1 or \2 in mutant
regex_leaks = []
for i, m in enumerate(muts):
    if "\\1" in m["mutant_code"] or "\\2" in m["mutant_code"]:
        in_golden = "\\1" in m["golden_code"] or "\\2" in m["golden_code"]
        regex_leaks.append({
            "idx": i,
            "id": m["id"],
            "type": m["mutation_type"],
            "in_golden_too": in_golden,
        })

print(f"Total REGEX_LEAK pairs: {len(regex_leaks)}")
print(f"Also in golden:         {sum(1 for r in regex_leaks if r['in_golden_too'])}")
print(f"Only in mutant:         {sum(1 for r in regex_leaks if not r['in_golden_too'])}")

print("\nDetails:")
for r in regex_leaks[:10]:
    print(f"  #{r['idx']} {r['id']} type={r['type']} in_golden={r['in_golden_too']}")
