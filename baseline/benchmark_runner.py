"""
Baseline Benchmark Runner for Oneiros (v7 — Memory-Safe).

Baselines:
  0. TestCases — only the dataset's existing test_cases
  1. Random   — random inputs (no dataset tests)
  2. Static   — boundary-value templates (no dataset tests)
  3. Grammar  — type-aware boundary analysis (no dataset tests)
  4. Coverage — output-diversity fuzzing (no dataset tests)
"""
import json
import time
import sys
import re
import gc
import random
import warnings
from pathlib import Path
from typing import List, Dict, Any, Tuple
from collections import Counter
from itertools import product as _product

warnings.filterwarnings("ignore", category=SyntaxWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, str(Path(__file__).parent.parent))
from harness.training_data import extract_dataset_assertions
from harness.safe_execution import classify_assertions, execute_code

DATA_DIR    = Path(__file__).parent.parent / "data"
RESULTS_DIR = Path(__file__).parent.parent / "results"

EXEC_TIMEOUT = 0.5  # seconds per single exec


def safe_exec(function_code: str, test_code: str):
    """Execute in a fresh restricted process with a parent-enforced timeout."""
    return execute_code(function_code, test_code, EXEC_TIMEOUT)


def kills_mutant(test_code, golden_code, mutant_code):
    outcome = classify_assertions(
        [test_code], golden_code, mutant_code, EXEC_TIMEOUT
    )[0]
    return bool(outcome["killed"])


def make_assert(golden_code, call):
    ok, result, _ = safe_exec(golden_code, call)
    if ok and result is not None:
        expr = call.replace("result = ", "").strip()
        expected = (
            result["repr"]
            if isinstance(result, dict) and set(result) == {"repr", "type"}
            else repr(result)
        )
        return f"assert {expr} == {expected}"
    return call


# ── Extract usable tests from dataset test_cases ─────────────

def extract_dataset_tests(test_cases: List[str], entry_point: str) -> List[str]:
    """Extract valid assertions, including assertions split over lines."""
    return extract_dataset_assertions(test_cases, entry_point)


# ── Param parsing ────────────────────────────────────────────

def _parse_params(code, entry_point):
    sig = re.search(rf'def\s+{re.escape(entry_point)}\s*\((.*?)\)',
                    code, re.DOTALL)
    if not sig:
        return []
    raw = sig.group(1)
    if not raw.strip():
        return []
    types = []
    for p in raw.split(","):
        p = p.strip()
        if not p:
            continue
        hint = p.split(":")[-1].strip().lower() if ":" in p else ""
        hint = hint.split("=")[0].strip()
        if "list" in hint:      types.append("list")
        elif "str" in hint:     types.append("str")
        elif "float" in hint:   types.append("float")
        elif "bool" in hint:    types.append("bool")
        elif "int" in hint:     types.append("int")
        else:                   types.append("any")
    return types


# ── 0. Dataset Test Cases ONLY Baseline ────────────────────────

class TestCasesBaseline:
    def generate_tests(self, golden_code, entry_point, test_cases, num_tests=10):
        return extract_dataset_tests(test_cases, entry_point)[:num_tests]

# ── 1. Random Baseline ───────────────────────────────────────

class RandomBaseline:
    VALS = {
        "int":   lambda: str(random.randint(-10, 10)),
        "float": lambda: str(round(random.uniform(-10, 10), 2)),
        "str":   lambda: repr("".join(random.choices("abc", k=random.randint(0, 4)))),
        "bool":  lambda: random.choice(["True", "False"]),
        "list":  lambda: str([random.randint(0, 5) for _ in range(random.randint(0, 4))]),
        "any":   lambda: random.choice(["0", "1", "-1", '""', "[]", "True"]),
    }

    def generate_tests(self, golden_code, entry_point, test_cases, num_tests=10):
        tests = []
        ptypes = _parse_params(golden_code, entry_point)
        if ptypes:
            for _ in range(num_tests):
                args = ", ".join(self.VALS.get(t, self.VALS["any"])() for t in ptypes)
                call = f"result = {entry_point}({args})"
                tests.append(make_assert(golden_code, call))
        return tests[:num_tests]


# ── 2. Static Baseline ──────────────────────────────────────

class StaticBaseline:
    VALS = {
        "int":   [0, 1, -1, 2, 10],
        "float": [0.0, 1.0, -1.0, 0.5, 0.01],
        "str":   ['""', '"a"', '"abc"', '"hello"'],
        "bool":  ["True", "False"],
        "list":  ["[]", "[1]", "[1, 2, 3]", "[0, 0]", "[-1, 0, 1]"],
        "any":   ["0", "1", '""', "[]", "True", "-1"],
    }

    def generate_tests(self, golden_code, entry_point, test_cases, num_tests=10):
        tests = []
        ptypes = _parse_params(golden_code, entry_point)
        if ptypes:
            vlists = [self.VALS.get(t, self.VALS["any"]) for t in ptypes]
            combos = list(_product(*vlists))[:num_tests]
            for combo in combos:
                args = ", ".join(str(v) for v in combo)
                call = f"result = {entry_point}({args})"
                tests.append(make_assert(golden_code, call))
        return tests[:num_tests]


# ── 3. Grammar Baseline (Learn&Fuzz) ────────────────────────

class GrammarBaseline:
    BOUNDARY = {
        "int":   [0, 1, -1, 2, -2, 10, -10, 100],
        "float": [0.0, 1.0, -1.0, 0.5, -0.5, 0.1, 0.01, 1e-6],
        "str":   ['""', '"a"', '"ab"', '"abc"', '"hello world"', '" "'],
        "bool":  ["True", "False"],
        "list":  ["[]", "[0]", "[1]", "[-1]", "[1, 2]", "[1, 2, 3]",
                  "[0, 0, 0]", "[1, -1]", "[100]"],
        "any":   ["0", "1", "-1", '""', "[]", "[1]", "True", "None"],
    }

    def generate_tests(self, golden_code, entry_point, test_cases, num_tests=10):
        tests = []
        ptypes = _parse_params(golden_code, entry_point)
        if ptypes:
            vlists = [self.BOUNDARY.get(t, self.BOUNDARY["any"]) for t in ptypes]
            combos = list(_product(*vlists))
            random.shuffle(combos)
            for combo in combos[:num_tests]:
                args = ", ".join(str(v) for v in combo)
                call = f"result = {entry_point}({args})"
                tests.append(make_assert(golden_code, call))
        return tests[:num_tests]


# ── 4. Coverage Baseline (Atheris-style) ─────────────────────

class CoverageBaseline:
    def generate_tests(self, golden_code, entry_point, test_cases, num_tests=10):
        tests = []
        ptypes = _parse_params(golden_code, entry_point)
        if not ptypes:
            return tests

        vals = ["0", "1", "-1", "2", "[]", "[1]", "[1,2,3]", '""',
                '"a"', '"abc"', "True", "False", "100", "[-1,0,1]",
                "[0,0]", "0.5", "[1,1,1]"]
        seeds = []
        for _ in range(15):
            args = ", ".join(random.choices(vals, k=len(ptypes)))
            seeds.append(f"result = {entry_point}({args})")

        seen_outputs = {}
        for call in seeds:
            ok, result, _ = safe_exec(golden_code, call)
            if ok:
                key = repr(result)
                if key not in seen_outputs:
                    seen_outputs[key] = call

        interesting = list(seen_outputs.values())
        for _ in range(num_tests * 2):
            if not interesting:
                break
            parent = random.choice(interesting)
            child = self._mutate(parent)
            try:
                compile(child, "<t>", "exec")
            except SyntaxError:
                continue
            ok, result, _ = safe_exec(golden_code, child)
            if ok:
                key = repr(result)
                if key not in seen_outputs:
                    seen_outputs[key] = child
                    interesting.append(child)
            if len(seen_outputs) >= num_tests:
                break

        for call in list(seen_outputs.values())[:num_tests]:
            tests.append(make_assert(golden_code, call))
        return tests[:num_tests]

    def _mutate(self, code):
        nums = list(re.finditer(r'(?<![a-zA-Z_])(-?\d+)', code))
        if nums:
            m = random.choice(nums)
            old = int(m.group())
            new = random.choice([old+1, old-1, 0, 1, -1, old*2, -old])
            return code[:m.start()] + str(new) + code[m.end():]
        return code


# ── Benchmark Runner ──────────────────────────────────────────

def log(msg):
    print(msg, flush=True)


def run_benchmark(sample_size=None, tests_per_mutant=10, seed=42):
    random.seed(seed)

    mutation_file = DATA_DIR / "mutation_pairs.json"
    log(f"Loading {mutation_file.name} ...")
    all_pairs = json.load(open(mutation_file, encoding="utf-8"))

    if sample_size and sample_size < len(all_pairs):
        sample = random.sample(all_pairs, sample_size)
    else:
        sample = all_pairs

    log("=" * 70)
    log("ONEIROS BASELINE BENCHMARK (v7 — Memory-Safe)")
    log(f"Mutant pairs: {len(sample)}  |  Tests per mutant: {tests_per_mutant}")
    log("=" * 70)

    baselines = {
        "TestCases": TestCasesBaseline(),
        "Random":    RandomBaseline(),
        "Static":    StaticBaseline(),
        "Grammar":   GrammarBaseline(),
        "Coverage":  CoverageBaseline(),
    }

    results = {}

    for bl_name, bl in baselines.items():
        log(f"\n{'='*50}")
        log(f"  {bl_name} Baseline")
        log(f"{'='*50}")
        t0 = time.time()
        random.seed(seed)

        killed = 0
        survived = 0
        total_tests = 0
        skipped_generation = 0
        skipped_compile = 0
        kill_by_type = Counter()
        total_by_type = Counter()

        for i, pair in enumerate(sample):
            golden = pair["golden_code"]
            mutant = pair["mutant_code"]
            entry  = pair["entry_point"]
            mtype  = pair["mutation_type"]
            tc     = pair.get("test_cases", []) if bl_name == "TestCases" else []
            total_by_type[mtype] += 1

            try:
                tests = bl.generate_tests(golden, entry, tc,
                                          num_tests=tests_per_mutant)
            except Exception:
                skipped_generation += 1
                tests = []

            mutant_killed = False
            for test in tests:
                total_tests += 1
                try:
                    compile(test, "<t>", "exec")
                except SyntaxError:
                    skipped_compile += 1
                    continue
                if kills_mutant(test, golden, mutant):
                    mutant_killed = True
                    break

            if mutant_killed:
                killed += 1
                kill_by_type[mtype] += 1
            else:
                survived += 1

            # Progress logging every 500 pairs
            if (i + 1) % 500 == 0 or (i + 1) == len(sample):
                elapsed = time.time() - t0
                rate = killed / (i + 1) * 100
                log(f"    [{i+1:>5}/{len(sample)}]  killed={killed:<5} "
                    f"rate={rate:>5.1f}%  elapsed={elapsed:>6.1f}s")
                if skipped_generation > 0 or skipped_compile > 0:
                    log(f"        (Skips: Gen={skipped_generation}, Comp={skipped_compile})")

            # Memory cleanup every 500 pairs
            if (i + 1) % 500 == 0:
                gc.collect()

        elapsed = time.time() - t0
        kill_rate = killed / len(sample)

        results[bl_name] = {
            "kill_rate": kill_rate,
            "killed": killed,
            "survived": survived,
            "total_tests": total_tests,
            "skipped_generation": skipped_generation,
            "skipped_compile_tests": skipped_compile,
            "wall_time": elapsed,
            "kill_by_type": dict(kill_by_type),
            "total_by_type": dict(total_by_type),
        }

        log(f"  >> {bl_name}: {kill_rate:.1%} kill rate "
            f"({killed}/{len(sample)})  {elapsed:.1f}s")
        log(f"  >> Skipped: {skipped_generation} gen errors, "
            f"{skipped_compile} compile errors")

        # Force GC between baselines
        gc.collect()

    # ── Final Report ──────────────────────────────────────────

    log("\n" + "=" * 70)
    log("FINAL RESULTS")
    log("=" * 70)
    log(f"{'Baseline':<12} {'Kill Rate':>10} {'Killed':>8} "
        f"{'Tests':>8} {'Skips':>16} {'Time':>8}")
    log("-" * 75)
    for name, r in results.items():
        skips = f"{r['skipped_generation']:>4}/{r['skipped_compile_tests']:<4}"
        log(f"{name:<12} {r['kill_rate']:>9.1%} {r['killed']:>8} "
            f"{r['total_tests']:>8} {skips:>16} {r['wall_time']:>7.1f}s")

    log(f"\n{'='*70}")
    log("KILL RATE BY MUTATION TYPE")
    log(f"{'='*70}")
    all_types = sorted(set().union(
        *(set(r["total_by_type"].keys()) for r in results.values())
    ))
    header = f"{'Type':<16}" + "".join(f"{n:>14}" for n in results.keys())
    log(header)
    log("-" * len(header))
    for mtype in all_types:
        row = f"{mtype:<16}"
        for name, r in results.items():
            total = r["total_by_type"].get(mtype, 0)
            k = r["kill_by_type"].get(mtype, 0)
            if total > 0:
                pct = f"{k/total:.0%}"
                row += f"  {k:>4}/{total:<4} {pct:>4} "
            else:
                row += f"{'---':>14}"
        log(row)

    best = max(results.items(), key=lambda x: x[1]["kill_rate"])
    log(f"\nBest Baseline: {best[0]} ({best[1]['kill_rate']:.1%} kill rate)")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "baseline_benchmark_10k.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    log(f"Saved to {out}")
    log("=" * 70)

    return results


if __name__ == "__main__":
    run_benchmark(tests_per_mutant=8)
