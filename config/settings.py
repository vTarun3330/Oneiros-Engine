"""
Oneiros Engine Configuration Settings
"""
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional


# Base paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
GOLDEN_DIR = DATA_DIR / "golden"
MUTANTS_DIR = DATA_DIR / "mutants"
HUMANEVAL_DIR = DATA_DIR / "humaneval"
MBPP_DIR = DATA_DIR / "mbpp"
BUGSINPY_DIR = DATA_DIR / "bugsinpy"

# Immutable Phase 3 corpus used by the current SFT-first pipeline. Keep this
# value centralized so local preflight, Modal training, evaluation, and audit
# commands cannot silently select different corpora.
CANONICAL_CORPUS_VERSION = "v3_final_candidate"

# Ensure directories exist
for dir_path in [DATA_DIR, GOLDEN_DIR, MUTANTS_DIR, HUMANEVAL_DIR, MBPP_DIR, BUGSINPY_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)


@dataclass
class DatasetConfig:
    """Configuration for dataset generation."""
    # =========================================================================
    # SYSTEM-LEVEL TESTING CONFIGURATION
    # =========================================================================

    # Training set: 60 system-level functions across 14 libraries
    num_training_functions: int = 60

    # Testing set: 10 system-level functions (diverse subset)
    num_testing_functions: int = 10

    # Number of mutants per function
    mutants_per_function: int = 15

    # Estimated total dataset size: 60 * 15 = 900 mutants
    estimated_dataset_size: int = 900

    # Execution timeout for system-level functions (longer than unit tests)
    execution_timeout_seconds: int = 5

    # Legacy settings (kept for compatibility)
    num_target_functions: int = 25  # Deprecated: use num_training_functions
    min_lines: int = 5
    max_lines: int = 50

    # System-level function categories
    priority_categories: List[str] = field(default_factory=lambda: [
        "dataframe_operations",    # pandas
        "numerical_computing",     # numpy
        "json_parsing",            # json
        "datetime_parsing",        # datetime
        "path_manipulation",       # os
        "regex_operations",        # re
        "collection_operations",   # collections
        "combinatorial",           # itertools
        "functional_programming",  # functools
        "cryptographic",           # hashlib
        "encoding",                # base64
        "url_parsing",             # urllib
        "mathematical",            # math
        "string_operations",       # string
    ])

    # Libraries covered (14 standard/scientific Python libraries)
    target_libraries: List[str] = field(default_factory=lambda: [
        "pandas",
        "numpy",
        "json",
        "datetime",
        "os",
        "re",
        "collections",
        "itertools",
        "functools",
        "hashlib",
        "base64",
        "urllib",
        "math",
        "string",
    ])


@dataclass
class ModelConfig:
    """Configuration for the generative model."""
    # Base model
    model_name: str = "microsoft/Phi-3-mini-4k-instruct"
    # Immutable Hugging Face snapshot used for Phase 3 reproducibility.
    model_revision: str = os.getenv(
        "ONEIROS_MODEL_REVISION",
        "f39ac1d28e925b323eae81227eaba4464caced4e",
    )

    # QLoRA settings
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj"
    ])

    # One numerical base-model identity for generation, SFT, and DPO.  A
    # previous mismatch used FP16+nested quantization for deployed inference
    # but BF16+non-nested quantization for training checkpoint validation.
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = False
    attention_implementation: str = "eager"

    # Generation settings
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    batch_size: int = 16


@dataclass
class MemoryConfig:
    """Configuration for FAISS memory module."""
    # Embedding dimension (sentence-transformers default)
    embedding_dim: int = 384

    # Embedding model
    embedding_model: str = "all-MiniLM-L6-v2"

    # Novelty threshold (cosine similarity)
    novelty_threshold: float = 0.85

    # Maximum memory size
    max_memory_size: int = 10000


@dataclass
class TrainingConfig:
    """Configuration for DPO training."""
    # Number of training iterations
    num_iterations: int = 100

    # Tests per iteration
    tests_per_iteration: int = 16

    # DPO hyperparameters
    beta: float = 0.1               # KL constraint strength (higher = less divergence from base)
    learning_rate: float = 2e-5     # legacy/default learning rate
    max_grad_norm: float = 1.0      # legacy/default gradient clipping
    # DPO repeatedly updates a quantized policy and is more sensitive to
    # numerical overflow than SFT.  Keep its settings explicit and lower.
    dpo_learning_rate: float = 1e-5
    dpo_max_grad_norm: float = 0.5

    # SFT hyperparameters (Phase 1: teach format before DPO).  The V3 run
    # reached its best held-out kill rate after only 100 optimizer steps while
    # 2e-4 remained effectively undiminished through step 600.  Use a bounded
    # one-epoch schedule so token loss cannot keep improving by drifting away
    # from the mutation-killing objective.
    sft_learning_rate: float = 5e-5
    sft_epochs: int = 1
    sft_batch_size: int = 1           # micro-batch size for SFT (reduced to 1 to fit VRAM)
    sft_warmup_steps: int = 25
    sft_checkpoint_steps: int = 50
    # Keep the legacy default explicit. Targeted experiments can select
    # constant_with_warmup without silently changing resumable older runs.
    sft_lr_scheduler_type: str = "cosine"
    sft_min_function_kill_rate: float = 0.50
    # A monitored run must contain at least two scheduled validations so a
    # one-point terminal fluctuation cannot masquerade as a learning trend.
    sft_min_monitor_checkpoints: int = 2
    # These must match the live generation interface.  Canonical examples
    # remain in the corpus when they exceed a training gate; they are recorded
    # as phase-specific exclusions rather than truncated into a different test.
    sft_prompt_token_limit: int = 512
    sft_completion_token_limit: int = 128
    # Repository fragments are executable test bodies rather than the single
    # concise assertion emitted for function-level inference.  Preserve the
    # 128-token function contract while allowing verified repository tests to
    # use the remaining SFT context window (512 prompt + 1,024 completion).
    sft_repository_completion_token_limit: int = 1024

    # Training epochs per iteration
    epochs_per_iteration: int = 1

    # Checkpointing
    save_every: int = 10
    checkpoint_dir: Path = PROJECT_ROOT / "checkpoints"


@dataclass
class BenchmarkConfig:
    """Configuration for benchmarking."""
    # Benchmark duration in seconds (24 hours default)
    duration_seconds: int = 86400

    # Metrics to track
    metrics: List[str] = field(default_factory=lambda: [
        "bug_discovery_rate",
        "time_to_expose",
        "code_coverage",
        "semantic_diversity"
    ])

    # Results directory
    results_dir: Path = PROJECT_ROOT / "results"


# Global configuration instances
dataset_config = DatasetConfig()
model_config = ModelConfig()
memory_config = MemoryConfig()
training_config = TrainingConfig()
benchmark_config = BenchmarkConfig()

# Ensure checkpoint and results directories exist
training_config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
benchmark_config.results_dir.mkdir(parents=True, exist_ok=True)
