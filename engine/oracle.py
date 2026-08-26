"""
Feedback Oracle for Oneiros Engine.

This module classifies test results as Winners or Losers based on:
1. Bug Detection: Did the test find a bug (output differs)?
2. Novelty: Is the test semantically novel (FAISS distance)?
3. Validity: Is the test syntactically valid?
"""
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.memory import FAISSMemory


class TestLabel(Enum):
    """Classification labels for test cases."""
    WINNER = "winner"     # Found bug OR novel
    LOSER = "loser"       # Redundant OR invalid
    UNKNOWN = "unknown"   # Could not classify


@dataclass
class FeedbackResult:
    """Result of oracle feedback on a test."""
    test_id: str
    label: TestLabel
    found_bug: bool
    is_novel: bool
    is_valid: bool
    similarity_score: float
    reason: str

    def is_winner(self) -> bool:
        return self.label == TestLabel.WINNER


class FeedbackOracle:
    """
    Oracle that classifies tests as Winners or Losers.

    Classification Rules:
    - WINNER: Test found bug OR test is novel (FAISS distance > threshold)
    - LOSER: Test is redundant (high similarity) OR invalid syntax
    """

    def __init__(
        self,
        memory: FAISSMemory,
        novelty_threshold: float = None
    ):
        """
        Initialize the oracle.

        Args:
            memory: FAISS memory for novelty checking
            novelty_threshold: Override threshold from memory config
        """
        self.memory = memory
        self.novelty_threshold = novelty_threshold or memory.novelty_threshold

        # Statistics
        self.stats = {
            "total_evaluated": 0,
            "winners": 0,
            "losers": 0,
            "bugs_found": 0,
            "novel_tests": 0,
            "invalid_tests": 0
        }

    def evaluate(
        self,
        test_input: str,
        found_bug: bool,
        is_valid: bool = True,
        function_id: str = ""
    ) -> FeedbackResult:
        """
        Evaluate a single test and classify as Winner/Loser.

        Args:
            test_input: The test input code
            found_bug: Whether the test found a bug
            is_valid: Whether syntax is valid
            function_id: ID of target function

        Returns:
            FeedbackResult with classification
        """
        self.stats["total_evaluated"] += 1

        # Invalid tests are always losers
        if not is_valid:
            self.stats["losers"] += 1
            self.stats["invalid_tests"] += 1
            return FeedbackResult(
                test_id=f"eval_{self.stats['total_evaluated']}",
                label=TestLabel.LOSER,
                found_bug=False,
                is_novel=False,
                is_valid=False,
                similarity_score=0.0,
                reason="Invalid syntax"
            )

        # Check novelty using FAISS
        is_novel, similarity = self.memory.is_novel(test_input)

        # Bug-finding tests are always winners
        if found_bug:
            self.stats["winners"] += 1
            self.stats["bugs_found"] += 1
            return FeedbackResult(
                test_id=f"eval_{self.stats['total_evaluated']}",
                label=TestLabel.WINNER,
                found_bug=True,
                is_novel=is_novel,
                is_valid=True,
                similarity_score=similarity,
                reason="Found bug"
            )

        # Novel tests without bugs are also winners (exploration)
        if is_novel:
            self.stats["winners"] += 1
            self.stats["novel_tests"] += 1
            return FeedbackResult(
                test_id=f"eval_{self.stats['total_evaluated']}",
                label=TestLabel.WINNER,
                found_bug=False,
                is_novel=True,
                is_valid=True,
                similarity_score=similarity,
                reason="Novel exploration"
            )

        # Redundant tests are losers
        self.stats["losers"] += 1
        return FeedbackResult(
            test_id=f"eval_{self.stats['total_evaluated']}",
            label=TestLabel.LOSER,
            found_bug=False,
            is_novel=False,
            is_valid=True,
            similarity_score=similarity,
            reason=f"Redundant (similarity: {similarity:.3f})"
        )

    def evaluate_batch(
        self,
        tests: List[Dict[str, Any]]
    ) -> Tuple[List[FeedbackResult], List[FeedbackResult]]:
        """
        Evaluate multiple tests and split into winners/losers.

        Args:
            tests: List of test dicts with 'input', 'found_bug', 'is_valid', 'function_id'

        Returns:
            Tuple of (winners, losers) lists
        """
        winners = []
        losers = []

        for test in tests:
            result = self.evaluate(
                test_input=test.get("input", ""),
                found_bug=test.get("found_bug", False),
                is_valid=test.get("is_valid", True),
                function_id=test.get("function_id", "")
            )

            if result.is_winner():
                winners.append(result)
            else:
                losers.append(result)

        return winners, losers

    def create_dpo_pairs(
        self,
        winners: List[FeedbackResult],
        losers: List[FeedbackResult],
        tests: Dict[str, str]
    ) -> List[Dict[str, str]]:
        """
        Create preference pairs for DPO training.

        Args:
            winners: List of winner FeedbackResults
            losers: List of loser FeedbackResults
            tests: Dict mapping test_id to test input code

        Returns:
            List of dicts with 'chosen' and 'rejected' keys
        """
        pairs = []

        # Pair each winner with each loser
        for winner in winners:
            for loser in losers:
                if winner.test_id in tests and loser.test_id in tests:
                    pairs.append({
                        "chosen": tests[winner.test_id],
                        "rejected": tests[loser.test_id],
                        "winner_reason": winner.reason,
                        "loser_reason": loser.reason
                    })

        return pairs

    def get_stats(self) -> Dict[str, int]:
        """Get oracle statistics."""
        return self.stats.copy()

    def reset_stats(self) -> None:
        """Reset statistics."""
        self.stats = {
            "total_evaluated": 0,
            "winners": 0,
            "losers": 0,
            "bugs_found": 0,
            "novel_tests": 0,
            "invalid_tests": 0
        }


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Feedback Oracle")
    print("=" * 60)

    # Create mock memory
    from engine.memory import FAISSMemory
    memory = FAISSMemory()

    # Add some existing tests to memory
    memory.add("result = merge_wrapper({'a': [1]}, {'b': [2]})", "sys_pandas_merge")
    memory.add("result = json_loads_wrapper('{}')", "sys_json_loads")

    # Create oracle
    oracle = FeedbackOracle(memory)

    print("\n1. Testing bug-finding input (should be WINNER)...")
    result = oracle.evaluate(
        test_input="result = merge_wrapper({}, {}, on='missing')",
        found_bug=True,
        function_id="sys_pandas_merge"
    )
    print(f"   Label: {result.label.value}")
    print(f"   Reason: {result.reason}")

    print("\n2. Testing novel input (should be WINNER)...")
    result = oracle.evaluate(
        test_input="result = datetime_parse_wrapper('2024-02-29', '%Y-%m-%d')",
        found_bug=False,
        function_id="sys_datetime_strptime"
    )
    print(f"   Label: {result.label.value}")
    print(f"   Reason: {result.reason}")

    print("\n3. Testing redundant input (should be LOSER)...")
    result = oracle.evaluate(
        test_input="result = merge_wrapper({'a': [1]}, {'b': [2]})",  # Same as in memory
        found_bug=False,
        function_id="sys_pandas_merge"
    )
    print(f"   Label: {result.label.value}")
    print(f"   Reason: {result.reason}")

    print("\n4. Testing invalid syntax (should be LOSER)...")
    result = oracle.evaluate(
        test_input="result = merge_wrapper(",  # Invalid
        found_bug=False,
        is_valid=False,
        function_id="sys_pandas_merge"
    )
    print(f"   Label: {result.label.value}")
    print(f"   Reason: {result.reason}")

    print("\n5. Stats:")
    stats = oracle.get_stats()
    for k, v in stats.items():
        print(f"   {k}: {v}")

    print("\n" + "=" * 60)
    print("Oracle test complete!")
    print("=" * 60)
