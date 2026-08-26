"""
Oneiros Engine - Main Entry Point

A self-improving AI framework for autonomous test generation
and bug discovery in Python software.
"""
import argparse
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    dataset_config,
    model_config,
    memory_config,
    training_config,
    benchmark_config,
    DATA_DIR,
    GOLDEN_DIR,
    MUTANTS_DIR,
    BUGSINPY_DIR
)


def cmd_generate_data(args):
    """Generate the unified evaluation harness."""
    from harness.dataset_loader import DatasetLoader, generate_golden_functions
    from harness.mutation_engine import MutantHarness, build_evaluation_harness
    from harness.bugsinpy_loader import integrate_bugsinpy

    print("=" * 60)
    print("Oneiros Engine - Data Generation")
    print("=" * 60)

    if args.full:
        # Build complete harness
        functions, mutants = build_evaluation_harness()

        # Also integrate BugsInPy
        print("\n")
        bugs = integrate_bugsinpy()
    else:
        # Just load/generate golden functions
        loader = DatasetLoader()

        if args.refresh:
            print("\nDownloading fresh datasets...")
            loader.load_humaneval(force_refresh=True)
            loader.load_mbpp(force_refresh=True)
        else:
            loader.load_humaneval()
            loader.load_mbpp()

        print(f"\nSelecting {args.num_functions} target functions...")
        functions = loader.select_target_functions(n=args.num_functions)
        loader.save_golden_functions()

        if args.mutants:
            print("\nGenerating mutants...")
            harness = MutantHarness()
            mutants = harness.build_harness(functions)
            harness.save_harness()

        if args.bugsinpy:
            print("\n")
            bugs = integrate_bugsinpy()

    print("\nData generation complete!")
    return 0


def cmd_bugsinpy(args):
    """Integrate BugsInPy real-world bugs."""
    from harness.bugsinpy_loader import BugsInPyLoader, integrate_bugsinpy

    if args.verify:
        print("=" * 60)
        print("Verifying BugsInPy Bugs")
        print("=" * 60)

        loader = BugsInPyLoader()
        loader.load_curated_bugs()
        results = loader.verify_bugs()

        print("\nVerification Results:")
        for bug_id, result in results.items():
            status = "✓" if result["behavior_differs"] else "✗"
            fixed_ok = "✓" if result["fixed_passes_all"] else "✗"
            print(f"  {status} {bug_id}")
            print(f"      Fixed passes all tests: {fixed_ok}")
            print(f"      Behavior differs: {result['behavior_differs']}")

            if args.verbose:
                for tr in result["test_results"]:
                    test_status = "PASS" if tr["fixed_pass"] else "FAIL"
                    buggy_status = "PASS" if tr["buggy_pass"] else "FAIL"
                    print(f"        - {tr['test'][:50]}...")
                    print(f"          Fixed: {test_status}, Buggy: {buggy_status}")
    else:
        bugs = integrate_bugsinpy()

    return 0


def cmd_test_harness(args):
    """Test the execution harness."""
    from harness.execution_harness import ExecutionHarness, TestGenerator
    from harness.dataset_loader import DatasetLoader
    from harness.mutation_engine import MutantHarness

    print("=" * 60)
    print("Oneiros Engine - Harness Test")
    print("=" * 60)

    # Load data
    loader = DatasetLoader()
    try:
        functions = loader.load_golden_functions()
    except FileNotFoundError:
        print("No golden functions found. Run 'generate-data' first.")
        return 1

    harness = MutantHarness()
    try:
        mutants = harness.load_harness()
    except FileNotFoundError:
        print("No mutants found. Run 'generate-data --mutants' first.")
        return 1

    # Test execution
    exec_harness = ExecutionHarness()
    test_gen = TestGenerator()

    # Pick a sample function and its mutants
    sample_func = functions[0]
    sample_mutants = harness.get_mutants_for_function(sample_func.id)[:3]

    print(f"\nTesting function: {sample_func.name}")
    print(f"Testing against {len(sample_mutants)} mutants")

    # Generate tests
    tests = test_gen.generate_simple_tests(sample_func, num_tests=5)

    # Run differential tests
    for mutant in sample_mutants:
        print(f"\n  Mutant: {mutant.id}")
        print(f"  Mutation: {mutant.mutation.description}")

        for test in tests:
            result = exec_harness.differential_test(
                test_code=test.code,
                golden_function=sample_func,
                mutant=mutant
            )
            status = "BUG FOUND" if result.is_bug_found() else "pass"
            print(f"    {test.id}: {status}")

    print("\nHarness test complete!")
    return 0


def cmd_info(args):
    """Display information about the current dataset."""
    import json

    print("=" * 60)
    print("Oneiros Engine - Dataset Information")
    print("=" * 60)

    # Check golden functions
    golden_file = GOLDEN_DIR / "golden_functions.json"
    if golden_file.exists():
        with open(golden_file, 'r') as f:
            golden = json.load(f)
        print(f"\nGolden Functions: {len(golden)}")

        # Category breakdown
        categories = {}
        for func in golden:
            cat = func.get('category', 'unknown')
            categories[cat] = categories.get(cat, 0) + 1

        print("  Categories:")
        for cat, count in sorted(categories.items()):
            print(f"    - {cat}: {count}")
    else:
        print("\nGolden Functions: Not generated yet")

    # Check mutants
    summary_file = MUTANTS_DIR / "harness_summary.json"
    if summary_file.exists():
        with open(summary_file, 'r') as f:
            summary = json.load(f)

        print(f"\nMutants: {summary['total_mutants']}")
        print(f"Average per function: {summary['average_mutants_per_function']:.1f}")

        print("  Mutation Types:")
        for mtype, count in sorted(summary['mutation_type_distribution'].items()):
            print(f"    - {mtype}: {count}")
    else:
        print("\nMutants: Not generated yet")

    # Check BugsInPy
    bugsinpy_file = BUGSINPY_DIR / "bugsinpy_metadata.json"
    if bugsinpy_file.exists():
        with open(bugsinpy_file, 'r') as f:
            bugs = json.load(f)

        print(f"\nBugsInPy Real-World Bugs: {len(bugs)}")

        # Project breakdown
        projects = {}
        categories = {}
        for bug in bugs:
            proj = bug.get('project', 'unknown')
            cat = bug.get('category', 'unknown')
            projects[proj] = projects.get(proj, 0) + 1
            categories[cat] = categories.get(cat, 0) + 1

        print("  Projects:")
        for proj, count in sorted(projects.items()):
            print(f"    - {proj}: {count}")

        print("  Categories:")
        for cat, count in sorted(categories.items()):
            print(f"    - {cat}: {count}")
    else:
        print("\nBugsInPy: Not integrated yet")

    # Total count
    print("\n" + "-" * 40)
    total_bugs = 0
    if summary_file.exists():
        total_bugs += summary['total_mutants']
    if bugsinpy_file.exists():
        total_bugs += len(bugs)
    print(f"Total Bugs in Harness: {total_bugs}")

    return 0


def cmd_run(args):
    """Run the Oneiros learning loop."""
    from oneiros_loop import OneirosLoop, LoopConfig

    print("=" * 60)
    print("Oneiros Engine - Learning Loop")
    print("=" * 60)

    config = LoopConfig(
        num_iterations=args.iterations,
        tests_per_iteration=8,
        dpo_train_every=3,
        save_every=args.iterations,
        use_mock_generator=args.mock,
        verbose=args.verbose,
        max_functions_per_iter=0  # use all training functions
    )

    loop = OneirosLoop(config)
    results = loop.run()

    return 0


def cmd_system(args):
    """Generate system-level function dataset."""
    from harness.system_dataset_loader import generate_system_level_dataset

    print("=" * 60)
    print("Oneiros Engine - System-Level Dataset Generation")
    print("=" * 60)

    dataset = generate_system_level_dataset(
        mutants_per_function=args.mutants_per_function
    )

    print(f"\nDataset Summary:")
    print(f"  Training functions: {dataset.total_training_count}")
    print(f"  Testing functions: {dataset.total_testing_count}")
    print(f"  Total mutants: {dataset.total_mutants}")

    return 0


def cmd_benchmark(args):
    """Run benchmarks comparing approaches."""
    from metrics.benchmarking import Benchmarker
    from baseline.random_baseline import RandomBaseline
    from config import get_testing_functions
    from harness.execution_harness import ExecutionHarness

    print("=" * 60)
    print("Oneiros Engine - Benchmarking")
    print("=" * 60)

    testing_funcs = get_testing_functions()
    benchmarker = Benchmarker()

    if args.baseline_only:
        print("\nRunning Random Baseline only...")
        baseline = RandomBaseline(seed=42)
        benchmarker.start_benchmark("Random", total_bugs=100)

        tests = baseline.run_baseline(testing_funcs, tests_per_function=20)

        for func_id, func_tests in tests.items():
            for test in func_tests:
                benchmarker.record_test(
                    test_id=test.id,
                    function_id=test.function_id,
                    is_valid=test.is_valid,
                    found_bug=False,  # Would need execution
                    is_novel=True
                )

        result = benchmarker.finish_benchmark()
        print(result.summary())
    else:
        print("\nFull benchmark requires running both Oneiros and baselines.")
        print("Use --baseline-only for quick baseline test.")

    benchmarker.save_results()

    return 0


def cmd_analyze(args):
    """Analyze user-supplied code with contracts, references, fuzzing, and optional LLM proposals."""
    import ast
    import json
    from engine.bug_discovery import analyze_function, read_contract_tests

    source_path = Path(args.file)
    source = source_path.read_text(encoding="utf-8")
    reference_source = (
        Path(args.reference_file).read_text(encoding="utf-8")
        if args.reference_file else None
    )
    contract_tests = read_contract_tests(Path(args.tests_file)) if args.tests_file else []
    suggestions = []

    if args.llm:
        from engine.generator import Phi3Generator

        tree = ast.parse(source)
        function = next(
            (
                node for node in tree.body
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == args.entry_point
            ),
            None,
        )
        if function is None:
            raise ValueError(f"Entry point '{args.entry_point}' was not found in {source_path}")
        generator = Phi3Generator()
        generator.load_model()
        if args.adapter:
            from peft import PeftModel
            generator.model = PeftModel.from_pretrained(generator.model, args.adapter)
        generated = generator.generate(
            function_signature=f"def {args.entry_point}{ast.unparse(function.args)}",
            docstring=ast.get_docstring(function) or args.spec,
            function_id=args.entry_point,
            edge_cases=[args.spec] if args.spec else [],
            num_samples=args.llm_samples,
        )
        suggestions = [test.input_code for test in generated if test.is_valid]

    report = analyze_function(
        source,
        args.entry_point,
        contract_tests=contract_tests,
        reference_source=reference_source,
        suggested_tests=suggestions,
        max_probes=args.probes,
        timeout_seconds=args.timeout,
    )
    payload = report.to_dict()
    rendered = json.dumps(payload, indent=2)
    print(rendered)
    if args.output:
        Path(args.output).write_text(rendered + "\n", encoding="utf-8")
        print(f"Saved analysis report to {args.output}")
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Oneiros Engine - Self-improving test generation"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # generate-data command
    gen_parser = subparsers.add_parser(
        "generate-data",
        help="Generate the unified evaluation harness"
    )
    gen_parser.add_argument(
        "-n", "--num-functions",
        type=int,
        default=dataset_config.num_target_functions,
        help=f"Number of target functions (default: {dataset_config.num_target_functions})"
    )
    gen_parser.add_argument(
        "--mutants",
        action="store_true",
        help="Also generate mutants"
    )
    gen_parser.add_argument(
        "--full",
        action="store_true",
        help="Generate complete harness (functions + mutants + bugsinpy)"
    )
    gen_parser.add_argument(
        "--bugsinpy",
        action="store_true",
        help="Also integrate BugsInPy bugs"
    )
    gen_parser.add_argument(
        "--refresh",
        action="store_true",
        help="Force re-download of datasets"
    )
    gen_parser.set_defaults(func=cmd_generate_data)

    # bugsinpy command
    bugsinpy_parser = subparsers.add_parser(
        "bugsinpy",
        help="Integrate BugsInPy real-world bugs"
    )
    bugsinpy_parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify bugs have different behavior"
    )
    bugsinpy_parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed test results"
    )
    bugsinpy_parser.set_defaults(func=cmd_bugsinpy)

    # test-harness command
    test_parser = subparsers.add_parser(
        "test-harness",
        help="Test the execution harness"
    )
    test_parser.set_defaults(func=cmd_test_harness)

    # info command
    info_parser = subparsers.add_parser(
        "info",
        help="Display dataset information"
    )
    info_parser.set_defaults(func=cmd_info)

    # run command (NEW) - Run the Oneiros learning loop
    run_parser = subparsers.add_parser(
        "run",
        help="Run the Oneiros learning loop"
    )
    run_parser.add_argument(
        "-n", "--iterations",
        type=int,
        default=5,
        help="Number of loop iterations (default: 5)"
    )
    run_parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock generator (no GPU required)"
    )
    run_parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        default=True,
        help="Show detailed output"
    )
    run_parser.set_defaults(func=cmd_run)

    # system command (NEW) - Generate system-level dataset
    system_parser = subparsers.add_parser(
        "system",
        help="Generate system-level function dataset"
    )
    system_parser.add_argument(
        "--mutants-per-function",
        type=int,
        default=15,
        help="Mutants per function (default: 15)"
    )
    system_parser.set_defaults(func=cmd_system)

    # benchmark command (NEW) - Run benchmarks
    bench_parser = subparsers.add_parser(
        "benchmark",
        help="Run benchmarks comparing Oneiros vs baselines"
    )
    bench_parser.add_argument(
        "--baseline-only",
        action="store_true",
        help="Run only the random baseline"
    )
    bench_parser.set_defaults(func=cmd_benchmark)

    # analyze command: evidence-backed bug discovery for arbitrary user code
    analyze_parser = subparsers.add_parser(
        "analyze",
        help="Analyze a user function with contracts, differential tests, and optional LLM proposals",
    )
    analyze_parser.add_argument("--file", required=True, help="Python file containing the function to analyze")
    analyze_parser.add_argument("--entry-point", required=True, help="Top-level function name to analyze")
    analyze_parser.add_argument("--reference-file", help="Optional known-correct implementation for differential testing")
    analyze_parser.add_argument("--tests-file", help="Optional Python/JSON file of user contract assertions")
    analyze_parser.add_argument("--spec", default="", help="Short behavior specification supplied to the LLM")
    analyze_parser.add_argument("--probes", type=int, default=48, help="Maximum type-guided boundary probes")
    analyze_parser.add_argument("--timeout", type=float, default=1.0, help="Per-probe sandbox timeout in seconds")
    analyze_parser.add_argument("--llm", action="store_true", help="Generate additional test proposals with Phi-3")
    analyze_parser.add_argument("--adapter", help="Optional trained Oneiros adapter path for --llm")
    analyze_parser.add_argument("--llm-samples", type=int, default=8, help="Number of LLM test proposals")
    analyze_parser.add_argument("--output", help="Optional path for the JSON analysis report")
    analyze_parser.set_defaults(func=cmd_analyze)

    # Parse and execute
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 0

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
