"""
Oneiros Engine - Main Learning Loop

This script implements the complete Oneiros learning loop:
1. Load system-level functions
2. Generate test inputs with Phi-3
3. Execute tests and collect results
4. Classify as Winners/Losers with Oracle
5. Store Winners in FAISS memory
6. Train Phi-3 with DPO on preferences
7. Loop with improved model
"""
import sys
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import configuration
from config import (
    dataset_config,
    model_config,
    memory_config,
    training_config,
    DATA_DIR,
    get_training_functions,
    get_testing_functions,
)

# Import harness components
from harness.execution_harness import ExecutionHarness, ExecutionResult
from harness.system_dataset_loader import SystemLevelDatasetLoader

# Import engine components (lazy load to handle import errors)
ENGINE_AVAILABLE = True
try:
    from engine import (
        FAISSMemory,
        Phi3Generator,
        MockGenerator,
        FeedbackOracle,
        DPOTrainer,
        DPODataPoint,
    )
except ImportError as e:
    ENGINE_AVAILABLE = False
    print(f"Warning: Engine not fully available: {e}")
    print("Some components require: pip install faiss-cpu sentence-transformers transformers peft trl")


@dataclass
class LoopConfig:
    """Configuration for the Oneiros learning loop."""
    num_iterations: int = 10
    tests_per_iteration: int = 8
    dpo_train_every: int = 3       # Train DPO every N iterations
    save_every: int = 5            # Save checkpoint every N iterations
    use_mock_generator: bool = True  # Use mock for testing
    max_functions_per_iter: int = 0  # 0 = use all training functions
    verbose: bool = True


class OneirosLoop:
    """
    Main Oneiros Learning Loop.

    Implements the complete cycle:
    Generate → Execute → Evaluate → Store → Train → Repeat
    """

    def __init__(self, config: LoopConfig = None):
        """
        Initialize the Oneiros loop.

        Args:
            config: Loop configuration
        """
        self.config = config or LoopConfig()

        # Load functions
        print("Loading system-level functions...")
        self.training_functions = get_training_functions()
        self.testing_functions = get_testing_functions()

        print(f"  Training: {len(self.training_functions)} functions")
        print(f"  Testing: {len(self.testing_functions)} functions")

        # Initialize components
        self.harness = ExecutionHarness(timeout_seconds=5.0)
        self.memory = None
        self.generator = None
        self.oracle = None
        self.trainer = None

        # Statistics
        self.stats = {
            "iterations": 0,
            "total_tests_generated": 0,
            "total_bugs_found": 0,
            "total_winners": 0,
            "total_losers": 0,
            "dpo_trainings": 0
        }

        # Results storage
        self.all_winners = []
        self.all_losers = []

    def setup(self) -> None:
        """Initialize all components."""
        print("\nSetting up Oneiros Engine...")

        if not ENGINE_AVAILABLE:
            print("  Warning: Using mock components due to missing dependencies")
            self.config.use_mock_generator = True

        # Initialize FAISS Memory
        try:
            self.memory = FAISSMemory()
            print("  ✓ FAISS Memory initialized")
        except Exception as e:
            print(f"  ✗ FAISS Memory failed: {e}")
            self.memory = None

        # Initialize Generator
        if self.config.use_mock_generator:
            self.generator = MockGenerator()
            print("  ✓ Mock Generator initialized (for testing)")
        else:
            try:
                self.generator = Phi3Generator()
                print("  ✓ Phi-3 Generator initialized")
            except Exception as e:
                print(f"  ✗ Phi-3 failed, using mock: {e}")
                self.generator = MockGenerator()

        # Initialize Oracle
        if self.memory:
            self.oracle = FeedbackOracle(self.memory)
            print("  ✓ Feedback Oracle initialized")

            # Create seed memory
            from harness.execution_harness import create_seed_tests
            seed_limit = min(100, len(self.training_functions))
            seed_tests = create_seed_tests(self.training_functions[:seed_limit])
            for t in seed_tests:
                # Need to run it to see if it's a winner? Actually just add as generic examples
                self.memory.add(t.code, t.target_function, found_bug=False)
            print(f"  ✓ Seed memory initialized with {len(seed_tests)} basic tests")
        else:
            print("  ✗ Oracle requires memory")

        # Initialize DPO Trainer
        if not self.config.use_mock_generator:
            try:
                self.trainer = DPOTrainer()
                print("  ✓ DPO Trainer initialized")
            except Exception as e:
                print(f"  ✗ DPO Trainer failed: {e}")

        print("\nSetup complete!")

    def _generate_tests(self, func, num_tests: int) -> List[Dict[str, Any]]:
        """Generate test inputs for a function."""
        # Get memory examples for this function
        memory_examples = []
        if self.memory:
            memory_examples = self.memory.get_for_prompt(func.id, k=3)

        # Generate tests
        generated = self.generator.generate(
            function_signature=func.signature,
            docstring=func.docstring,
            function_id=func.id,
            edge_cases=func.edge_cases,
            memory_examples=memory_examples,
            library=func.library,
            num_samples=num_tests
        )

        return [{
            "id": g.id,
            "input": g.input_code,
            "function_id": g.function_id,
            "is_valid": g.is_valid
        } for g in generated]

    def _execute_test(self, test: Dict[str, Any], func) -> Dict[str, Any]:
        """Execute a test and return result with bug info."""
        # Get wrapper code for the function
        wrapper_code = func.wrapper_code

        # Execute the test
        result = self.harness.execute_test(
            test_code=test["input"],
            function_code=wrapper_code,
            entry_point=func.signature.split('(')[0].split()[-1],
            test_id=test["id"]
        )

        # Determine if bug was found
        found_bug = result.result.value in ["fail", "error"]

        return {
            **test,
            "found_bug": found_bug,
            "execution_result": result.result.value,
            "error_message": result.error_message
        }

    def _evaluate_and_store(self, executed_tests: List[Dict[str, Any]]) -> None:
        """Evaluate tests with oracle and store winners in memory."""
        if not self.oracle:
            return

        for test in executed_tests:
            result = self.oracle.evaluate(
                test_input=test["input"],
                found_bug=test.get("found_bug", False),
                is_valid=test.get("is_valid", True),
                function_id=test.get("function_id", "")
            )

            if result.is_winner():
                self.all_winners.append(test)
                self.stats["total_winners"] += 1

                # Add to memory
                if self.memory:
                    self.memory.add(
                        test_input=test["input"],
                        function_id=test["function_id"],
                        found_bug=test.get("found_bug", False)
                    )

                if test.get("found_bug"):
                    self.stats["total_bugs_found"] += 1
            else:
                self.all_losers.append(test)
                self.stats["total_losers"] += 1

    def run_iteration(self, iteration: int) -> Dict[str, Any]:
        """Run a single iteration of the loop."""
        if self.config.verbose:
            print(f"\n{'='*50}")
            print(f"Iteration {iteration + 1}/{self.config.num_iterations}")
            print('='*50)

        iteration_stats = {
            "iteration": iteration + 1,
            "tests_generated": 0,
            "bugs_found": 0,
            "winners": 0,
            "losers": 0
        }

        # For each training function (use all unless limited by config)
        func_limit = self.config.max_functions_per_iter or len(self.training_functions)
        for func in self.training_functions[:func_limit]:
            if self.config.verbose:
                print(f"\n  Testing: {func.name}")

            # 1. Generate tests
            tests = self._generate_tests(func, num_tests=self.config.tests_per_iteration)
            iteration_stats["tests_generated"] += len(tests)
            self.stats["total_tests_generated"] += len(tests)

            if self.config.verbose:
                print(f"    Generated {len(tests)} tests")

            # 2. Execute tests
            executed = []
            for test in tests:
                result = self._execute_test(test, func)
                executed.append(result)

                if result.get("found_bug"):
                    iteration_stats["bugs_found"] += 1

            # 3. Evaluate and store
            self._evaluate_and_store(executed)

        # Update iteration stats
        iteration_stats["winners"] = self.stats["total_winners"]
        iteration_stats["losers"] = self.stats["total_losers"]

        self.stats["iterations"] = iteration + 1

        if self.config.verbose:
            print(f"\n  Results: {iteration_stats['bugs_found']} bugs, "
                  f"{self.stats['total_winners']} winners, "
                  f"{self.stats['total_losers']} losers")

        return iteration_stats

    def run(self) -> Dict[str, Any]:
        """Run the complete learning loop."""
        print("\n" + "="*60)
        print("ONEIROS ENGINE - Starting Learning Loop")
        print("="*60)

        self.setup()

        start_time = time.time()

        for i in range(self.config.num_iterations):
            self.run_iteration(i)

            # DPO Training Loop Integration
            if (i + 1) % self.config.dpo_train_every == 0 and self.trainer:
                if self.config.verbose:
                    print(f"\n  [Triggering structured DPO Training (Iteration {i+1})]")

                # Fetch recent winners and losers
                recent_winners = self.all_winners[-self.config.tests_per_iteration * 6:]
                recent_losers = self.all_losers[-self.config.tests_per_iteration * 6:]

                if len(recent_winners) > 0 and len(recent_losers) > 0:
                    from engine.dpo_trainer import create_dpo_pairs_from_results

                    pairs = create_dpo_pairs_from_results(
                        winners=recent_winners,
                        losers=recent_losers,
                        prompt_template="Generate test for {function_id}"
                    )

                    if len(pairs) > 0:
                        try:
                            # Train with optimized batch size for stability
                            print(f"    -> Training on {len(pairs)} preference pairs...")
                            train_config = self.trainer.train(pairs, num_epochs=1, batch_size=min(4, len(pairs)))

                            # Save and reload LoRA weights to generation model
                            adapter_path = self.trainer.save_adapter()
                            self.generator.load_lora_adapter(adapter_path)
                            self.stats["dpo_trainings"] += 1
                            print(f"    -> DPO Training complete. Loss: {train_config.get('loss', 'N/A')}")
                        except Exception as e:
                            print(f"    -> DPO Training failed: {e}")
                            import traceback
                            traceback.print_exc()
                else:
                    if self.config.verbose:
                        print("    -> Not enough winners/losers to pair for DPO")

            # Optional: Save checkpoint
            if (i + 1) % self.config.save_every == 0:
                self.save_checkpoint(i + 1)

        elapsed = time.time() - start_time

        # Final summary
        print("\n" + "="*60)
        print("ONEIROS ENGINE - Loop Complete")
        print("="*60)
        print(f"\nFinal Statistics:")
        print(f"  Iterations: {self.stats['iterations']}")
        print(f"  Tests Generated: {self.stats['total_tests_generated']}")
        print(f"  Bugs Found: {self.stats['total_bugs_found']}")
        print(f"  Winners: {self.stats['total_winners']}")
        print(f"  Losers: {self.stats['total_losers']}")
        print(f"  DPO Trainings: {self.stats['dpo_trainings']}")
        print(f"  Time: {elapsed:.2f}s")

        return self.stats

    def save_checkpoint(self, iteration: int) -> None:
        """Save current state to disk."""
        checkpoint_dir = DATA_DIR / "checkpoints" / f"iter_{iteration}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Save stats
        with open(checkpoint_dir / "stats.json", 'w') as f:
            json.dump(self.stats, f, indent=2)

        # Save memory
        if self.memory:
            self.memory.save(checkpoint_dir / "memory")

        print(f"  [Checkpoint saved at iteration {iteration}]")


def main():
    """Run the Oneiros learning loop."""
    config = LoopConfig(
        num_iterations=10,        # 10 iterations for full training
        tests_per_iteration=8,    # 8 tests per function per iteration
        dpo_train_every=3,        # DPO fires at iter 3, 6, 9
        save_every=5,
        use_mock_generator=False, # real Phi-3
        verbose=True,
        max_functions_per_iter=0  # 0 = use all 60 training functions
    )

    loop = OneirosLoop(config)
    results = loop.run()

    return results


if __name__ == "__main__":
    main()
