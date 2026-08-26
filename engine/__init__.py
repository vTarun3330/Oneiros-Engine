"""
Oneiros Engine Core Module.

This module provides the main components for the Oneiros learning loop:
- FAISSMemory: Semantic memory for test inputs
- Phi3Generator: Test case generator
- FeedbackOracle: Winner/Loser classifier
- DPOTrainer: Direct Preference Optimization trainer
"""

from .memory import FAISSMemory, MemoryEntry
from .generator import Phi3Generator, GeneratedTest, MockGenerator
from .oracle import FeedbackOracle, FeedbackResult, TestLabel
from .dpo_trainer import DPOTrainer, DPODataPoint, create_dpo_pairs_from_results

__all__ = [
    # Memory
    "FAISSMemory",
    "MemoryEntry",

    # Generator
    "Phi3Generator",
    "GeneratedTest",
    "MockGenerator",

    # Oracle
    "FeedbackOracle",
    "FeedbackResult",
    "TestLabel",

    # DPO
    "DPOTrainer",
    "DPODataPoint",
    "create_dpo_pairs_from_results",
]
