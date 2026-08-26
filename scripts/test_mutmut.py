"""Quick test of mutmut's mutation API."""
from mutmut.file_mutation import create_mutations, combine_mutations_to_source

code = '''def is_even(n):
    if n % 2 == 0:
        return True
    return False
'''

source_cst, mutations = create_mutations(code)
print(f"Mutmut generated {len(mutations)} mutations from a simple function")
print()

for i, m in enumerate(mutations):
    result = combine_mutations_to_source(source_cst, [m])
    print(f"--- Mutation {i} ---")
    if isinstance(result, tuple):
        src, names = result
        print(f"Type: tuple, src type: {type(src)}, names: {names}")
        print(src[:300])
    else:
        print(f"Type: {type(result)}")
        print(str(result)[:300])
    print()
    if i >= 4:
        break
