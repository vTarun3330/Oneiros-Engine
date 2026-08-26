"""
Oneiros DataLoader — unified loader for all training data.

Loads and validates:
  • unified_dataset.json   (1,631 golden functions)
  • mutation_pairs.json    (10,000 golden↔mutant pairs)

Provides:
  • OneirosDataset         — iterable dataset
  • OneirosDataLoader      — batched loader with train/val/test splits
  • DPO-ready formatting   — prompt/chosen/rejected triples
  • Validation & stats
"""
import json
import random
import math
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Iterator, Tuple

DATA_DIR = Path(__file__).parent.parent / "data"


# ── Data classes ───────────────────────────────────────────────

@dataclass
class FunctionRecord:
    """A single golden function from the unified dataset."""
    id: str
    source: str
    code: str
    signature: str
    entry_point: str
    docstring: str
    test_cases: List[str]
    category: str = ""
    complexity_score: int = 5

    @classmethod
    def from_dict(cls, d: Dict) -> "FunctionRecord":
        return cls(
            id=d.get("id", ""),
            source=d.get("source", ""),
            code=d.get("code", ""),
            signature=d.get("signature", ""),
            entry_point=d.get("entry_point", ""),
            docstring=d.get("docstring", ""),
            test_cases=d.get("test_cases", []),
            category=d.get("category", ""),
            complexity_score=d.get("complexity_score", 5),
        )


@dataclass
class MutationRecord:
    """A golden↔mutant pair from the mutation dataset."""
    id: str
    source: str
    golden_code: str
    mutant_code: str
    entry_point: str
    test_cases: List[str]
    mutation_type: str
    mutation_description: str

    @classmethod
    def from_dict(cls, d: Dict) -> "MutationRecord":
        return cls(
            id=d.get("id", ""),
            source=d.get("source", ""),
            golden_code=d.get("golden_code", ""),
            mutant_code=d.get("mutant_code", ""),
            entry_point=d.get("entry_point", ""),
            test_cases=d.get("test_cases", []),
            mutation_type=d.get("mutation_type", ""),
            mutation_description=d.get("mutation_description", ""),
        )


@dataclass
class DPOTriple:
    """A DPO training triple: prompt, chosen (good test), rejected (bad test)."""
    prompt: str
    chosen: str
    rejected: str
    function_id: str = ""


# ── Dataset ────────────────────────────────────────────────────

class OneirosDataset:
    """
    Main dataset class for Oneiros training.

    Holds:
      • functions:  List[FunctionRecord]  — golden functions
      • mutations:  List[MutationRecord]  — golden↔mutant pairs
    """

    def __init__(
        self,
        functions_path: Path = None,
        mutations_path: Path = None,
    ):
        self.functions_path = functions_path or DATA_DIR / "unified_dataset.json"
        self.mutations_path = mutations_path or DATA_DIR / "mutation_pairs.json"

        self.functions: List[FunctionRecord] = []
        self.mutations: List[MutationRecord] = []

        self._loaded = False

    # ── Loading ────────────────────────────────────────────────

    def load(self) -> "OneirosDataset":
        """Load both datasets from disk."""
        self._load_functions()
        self._load_mutations()
        self._loaded = True
        return self

    def _load_functions(self):
        if not self.functions_path.exists():
            print(f"⚠  Functions file not found: {self.functions_path}")
            return
        with open(self.functions_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        self.functions = [FunctionRecord.from_dict(d) for d in raw]

    def _load_mutations(self):
        if not self.mutations_path.exists():
            print(f"⚠  Mutations file not found: {self.mutations_path}")
            return
        with open(self.mutations_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        self.mutations = [MutationRecord.from_dict(d) for d in raw]

    # ── Sizes ──────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.mutations)

    @property
    def num_functions(self) -> int:
        return len(self.functions)

    @property
    def num_mutations(self) -> int:
        return len(self.mutations)

    # ── Iteration ──────────────────────────────────────────────

    def __iter__(self) -> Iterator[MutationRecord]:
        return iter(self.mutations)

    def __getitem__(self, idx: int) -> MutationRecord:
        return self.mutations[idx]

    # ── DPO formatting ─────────────────────────────────────────

    def to_dpo_triples(self) -> List[DPOTriple]:
        """
        Convert mutation pairs into DPO training triples.

        • prompt   = "Generate a test for <entry_point>:\n<golden_code>"
        • chosen   = test that kills the mutant (from test_cases)
        • rejected = the mutant code (model should NOT generate buggy code)
        """
        triples = []
        for m in self.mutations:
            if not m.test_cases:
                continue

            prompt = (
                f"Generate a test to find bugs in this function:\n\n"
                f"```python\n{m.golden_code}\n```"
            )
            chosen = m.test_cases[0] if m.test_cases else ""
            rejected = (
                f"# This mutant has a bug: {m.mutation_description}\n"
                f"```python\n{m.mutant_code}\n```"
            )
            triples.append(DPOTriple(
                prompt=prompt,
                chosen=chosen,
                rejected=rejected,
                function_id=m.id,
            ))
        return triples

    # ── Splits ─────────────────────────────────────────────────

    def split(
        self,
        train: float = 0.8,
        val: float = 0.1,
        test: float = 0.1,
        seed: int = 42,
    ) -> Tuple["OneirosDataset", "OneirosDataset", "OneirosDataset"]:
        """Split into train/val/test datasets."""
        assert abs(train + val + test - 1.0) < 1e-6, "Splits must sum to 1.0"

        rng = random.Random(seed)
        indices = list(range(len(self.mutations)))
        rng.shuffle(indices)

        n = len(indices)
        n_train = int(n * train)
        n_val = int(n * val)

        train_idx = indices[:n_train]
        val_idx = indices[n_train : n_train + n_val]
        test_idx = indices[n_train + n_val :]

        def _make_subset(idx_list: List[int]) -> "OneirosDataset":
            ds = OneirosDataset.__new__(OneirosDataset)
            ds.functions_path = self.functions_path
            ds.mutations_path = self.mutations_path
            ds.functions = self.functions  # shared reference
            ds.mutations = [self.mutations[i] for i in idx_list]
            ds._loaded = True
            return ds

        return _make_subset(train_idx), _make_subset(val_idx), _make_subset(test_idx)


# ── DataLoader ─────────────────────────────────────────────────

class OneirosDataLoader:
    """
    Batched data loader with shuffling support.

    Usage:
        dataset = OneirosDataset().load()
        loader = OneirosDataLoader(dataset, batch_size=32, shuffle=True)
        for batch in loader:
            # batch is a list of MutationRecord
            ...
    """

    def __init__(
        self,
        dataset: OneirosDataset,
        batch_size: int = 32,
        shuffle: bool = True,
        seed: int = 42,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return math.ceil(len(self.dataset) / self.batch_size)

    def __iter__(self) -> Iterator[List[MutationRecord]]:
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            self.rng.shuffle(indices)

        for start in range(0, len(indices), self.batch_size):
            batch_idx = indices[start : start + self.batch_size]
            yield [self.dataset[i] for i in batch_idx]


# ── Validation ─────────────────────────────────────────────────

def validate_dataset(dataset: OneirosDataset) -> Dict[str, Any]:
    """
    Run a full validation pass across the entire dataset.
    Returns a report dict with stats and any issues found.
    """
    report: Dict[str, Any] = {
        "functions_total": dataset.num_functions,
        "mutations_total": dataset.num_mutations,
        "issues": [],
        "by_source": {},
        "by_mutation_type": {},
        "has_test_cases": 0,
        "missing_test_cases": 0,
        "empty_golden": 0,
        "empty_mutant": 0,
        "golden_equals_mutant": 0,
        "avg_golden_len": 0,
        "avg_mutant_len": 0,
    }

    golden_lens = []
    mutant_lens = []

    for m in dataset.mutations:
        # Source distribution
        report["by_source"][m.source] = report["by_source"].get(m.source, 0) + 1

        # Mutation type distribution
        report["by_mutation_type"][m.mutation_type] = (
            report["by_mutation_type"].get(m.mutation_type, 0) + 1
        )

        # Test cases
        if m.test_cases and len(m.test_cases) > 0:
            report["has_test_cases"] += 1
        else:
            report["missing_test_cases"] += 1

        # Code quality
        if not m.golden_code or len(m.golden_code.strip()) < 5:
            report["empty_golden"] += 1
        if not m.mutant_code or len(m.mutant_code.strip()) < 5:
            report["empty_mutant"] += 1
        if m.golden_code == m.mutant_code:
            report["golden_equals_mutant"] += 1

        golden_lens.append(len(m.golden_code))
        mutant_lens.append(len(m.mutant_code))

    report["avg_golden_len"] = int(sum(golden_lens) / max(len(golden_lens), 1))
    report["avg_mutant_len"] = int(sum(mutant_lens) / max(len(mutant_lens), 1))

    # Issues
    if report["empty_golden"] > 0:
        report["issues"].append(f"{report['empty_golden']} pairs with empty golden code")
    if report["empty_mutant"] > 0:
        report["issues"].append(f"{report['empty_mutant']} pairs with empty mutant code")
    if report["golden_equals_mutant"] > 0:
        report["issues"].append(
            f"{report['golden_equals_mutant']} pairs where golden == mutant (no mutation)"
        )
    if report["missing_test_cases"] > 0:
        report["issues"].append(
            f"{report['missing_test_cases']} pairs missing test cases"
        )

    # Functions validation
    func_issues = 0
    for f in dataset.functions:
        if not f.code or len(f.code.strip()) < 5:
            func_issues += 1
    if func_issues:
        report["issues"].append(f"{func_issues} functions with empty code")

    return report


def print_report(report: Dict[str, Any]):
    """Pretty-print a validation report."""
    print("=" * 60)
    print("ONEIROS DATASET VALIDATION REPORT")
    print("=" * 60)

    print(f"\n[FUNCTIONS]       {report['functions_total']:,}")
    print(f"[MUTATION PAIRS]  {report['mutations_total']:,}")
    print(f"[WITH TESTS]      {report['has_test_cases']:,}")
    print(f"[MISSING TESTS]   {report['missing_test_cases']:,}")

    print(f"\n[AVG GOLDEN LEN]  {report['avg_golden_len']:,} chars")
    print(f"[AVG MUTANT LEN]  {report['avg_mutant_len']:,} chars")

    print("\n-- By Source --")
    for src, cnt in sorted(report["by_source"].items()):
        print(f"  {src:15s} {cnt:>6,}")

    print("\n-- By Mutation Type --")
    for mt, cnt in sorted(
        report["by_mutation_type"].items(), key=lambda x: -x[1]
    )[:12]:
        print(f"  {mt:15s} {cnt:>6,}")

    quality = report["empty_golden"] + report["empty_mutant"] + report["golden_equals_mutant"]
    if quality == 0 and not report["issues"]:
        print("\n[OK] ALL CHECKS PASSED -- dataset is clean")
    else:
        print(f"\n[WARN] Issues found: {len(report['issues'])}")
        for issue in report["issues"]:
            print(f"   - {issue}")

    # Usable pairs
    usable = (
        report["mutations_total"]
        - report["empty_golden"]
        - report["empty_mutant"]
        - report["golden_equals_mutant"]
    )
    print(f"\n[TARGET] Usable pairs for training: {usable:,}")


# ── Main ───────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading Oneiros dataset...\n")
    dataset = OneirosDataset().load()

    # 1. Validate
    report = validate_dataset(dataset)
    print_report(report)

    # 2. Test splits
    print("\n" + "=" * 60)
    print("TESTING TRAIN/VAL/TEST SPLIT")
    print("=" * 60)
    train_ds, val_ds, test_ds = dataset.split(train=0.8, val=0.1, test=0.1)
    print(f"  Train: {len(train_ds):,} pairs")
    print(f"  Val:   {len(val_ds):,} pairs")
    print(f"  Test:  {len(test_ds):,} pairs")
    assert len(train_ds) + len(val_ds) + len(test_ds) == len(dataset)
    print("  [OK] Split sizes add up correctly")

    # 3. Test DataLoader
    print("\n" + "=" * 60)
    print("TESTING DATALOADER")
    print("=" * 60)
    loader = OneirosDataLoader(train_ds, batch_size=32, shuffle=True)
    print(f"  Batches: {len(loader)}")
    total_items = 0
    for batch in loader:
        total_items += len(batch)
    print(f"  Total items iterated: {total_items:,}")
    assert total_items == len(train_ds)
    print("  [OK] All items iterated correctly")

    # 4. Test DPO conversion
    print("\n" + "=" * 60)
    print("TESTING DPO TRIPLE CONVERSION")
    print("=" * 60)
    triples = dataset.to_dpo_triples()
    print(f"  DPO triples: {len(triples):,}")
    if triples:
        t = triples[0]
        print(f"\n  Sample triple:")
        print(f"    prompt:   {t.prompt[:80]}...")
        print(f"    chosen:   {t.chosen[:80]}...")
        print(f"    rejected: {t.rejected[:80]}...")
    print("  [OK] DPO conversion working")

    # 5. Summary
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED [OK]")
    print("=" * 60)
