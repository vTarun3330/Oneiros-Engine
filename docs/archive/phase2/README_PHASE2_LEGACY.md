# Oneiros Engine

**A Research Engine for Mutation-Guided Automated Test Generation**

Oneiros is an SFT-first research engine that trains Phi-3-mini-4k-instruct to generate mutation-killing Python assertions. Direct Preference Optimization (DPO) is an optional second stage and is permitted only when it improves a locked validation panel without unacceptable regressions.

## Phase 3 source of truth (August 2026)

The active pipeline is:

`canonical V4 unified-prompt corpus -> verified SFT -> locked validation -> optional DPO -> final sealed test evaluation`

- The canonical corpus contains 8,237 records with group-disjoint train/validation/test splits and 150 fail-closed V3 exclusions.
- HumanEval, MBPP, BugsInPy, and SWE-bench use one model-visible prompt schema. Dataset identity, reference code, mutation metadata, patches, and oracle labels remain hidden.
- Function prompts use a 512-token budget; repository prompts use 1,024 tokens for specification, target code, and verified native execution context while keeping the same section order.
- Generated function-level candidates must be exactly one bounded assertion that calls the target entry point.
- Candidate execution uses a fresh restricted process, temporary working directory, hard parent timeout, and POSIX resource limits where available.
- The base model is pinned to Hugging Face revision `f39ac1d28e925b323eae81227eaba4464caced4e`; new artifacts record source, dependency, runtime, model, corpus, adapter, and evaluation-panel fingerprints.
- Repository-fragment records are held for native project-environment system testing and are not included in the function-level kill rate.
- The previously reported 67/100 smoke result is historical. Because the evaluator has now been hardened, it must be reproduced before it is used as the current headline result. The final test split remains sealed until model selection is complete.

Sections below describe both the current engine and some legacy Phase 2 components. When they conflict, this Phase 3 source-of-truth section and `scripts/train_on_dataset.py` are authoritative.

For a concise folder map, setup, artifact policy, and end-to-end execution
order, see [`REPOSITORY_GUIDE.md`](REPOSITORY_GUIDE.md). Corpus provenance and
split controls are documented in
[`dataset_documentation.md`](dataset_documentation.md).

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Project Structure](#project-structure)
- [The Learning Loop](#the-learning-loop)
- [Core Components](#core-components)
  - [Engine Modules](#engine-modules)
  - [Harness Modules](#harness-modules)
  - [Configuration](#configuration)
- [Dataset](#dataset)
- [Baseline Benchmarks](#baseline-benchmarks)
  - [TestCases (Human Expert)](#1-testcases--human-expert-baseline--827)
  - [Random](#2-random--floor-baseline--287)
  - [Static](#3-static--template-baseline--304)
  - [Grammar (Learn&Fuzz)](#4-grammar--learnfuzz-baseline--335)
  - [Coverage (Atheris-style)](#5-coverage--atheris-style-baseline--498)
- [Benchmarking Results](#benchmarking-results)
- [Setup & Installation](#setup--installation)
- [Running in Google Colab](#running-in-google-colab)
- [Key Configuration Parameters](#key-configuration-parameters)
- [Scripts & Utilities](#scripts--utilities)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                      ONEIROS ENGINE                             │
│                                                                 │
│  ┌──────────┐    ┌──────────────┐    ┌──────────────────────┐   │
│  │  Config   │    │  System-Level│    │  Unified Dataset     │   │
│  │ settings  │───▶│  Functions   │    │  (HumanEval + MBPP   │   │
│  │          │    │  (60 train,  │    │   + BugsInPy)        │   │
│  │          │    │   10 test)   │    │  1,631 functions     │   │
│  └──────────┘    └──────┬───────┘    │  10,000 mutant pairs │   │
│                         │            └──────────┬───────────┘   │
│                         ▼                       │               │
│  ┌──────────────────────────────────────────────┼───────────┐   │
│  │              LEARNING LOOP                   │           │   │
│  │                                              │           │   │
│  │  ┌─────────────┐    ┌──────────────┐         │           │   │
│  │  │ Phi-3 Model │───▶│  Execution   │         │           │   │
│  │  │ (Generator) │    │  Harness     │         │           │   │
│  │  │ + QLoRA     │    │  (5s timeout)│         │           │   │
│  │  └──────┬──────┘    └──────┬───────┘         │           │   │
│  │         │                  │                  │           │   │
│  │         │           ┌──────▼───────┐          │           │   │
│  │         │           │  Feedback    │          │           │   │
│  │         │           │  Oracle      │          │           │   │
│  │         │           │  (W/L Label) │          │           │   │
│  │         │           └──────┬───────┘          │           │   │
│  │         │                  │                  │           │   │
│  │    ┌────▼────┐      ┌──────▼───────┐   ┌─────▼────────┐  │   │
│  │    │  FAISS  │◀─────│  Winner/     │   │  DPO Trainer │  │   │
│  │    │  Memory │      │  Loser Split │──▶│  (LoRA Fine- │  │   │
│  │    │         │      └──────────────┘   │   Tuning)    │  │   │
│  │    └─────────┘                         └──────────────┘  │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              BASELINE BENCHMARKS                         │   │
│  │  Random │ Static │ Grammar │ Coverage │ TestCases        │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### Two Datasets, Two Purposes

Oneiros uses **two separate datasets** that serve different roles:

| Dataset | What it contains | Where it's used | Purpose |
|---------|-----------------|-----------------|--------|
| **Unified Dataset** (10,000 mutation pairs) | 1,631 golden functions from HumanEval + MBPP + BugsInPy, mutated into 10,000 golden/buggy pairs | **Baseline Benchmarking** (Phase 3) | Measures kill rate of non-AI baselines to establish the performance spectrum (28.7%–82.7%) |
| **System-Level Functions** (70 functions) | 60 training + 10 testing functions from real Python libraries (pandas, json, re, datetime, os, collections, etc.) | **AI Learning Loop** (Phase 4) | The actual targets that Phi-3 learns to generate tests for during DPO training |

**Why two datasets?**
- The **Unified Dataset** provides the scientific benchmark. Its 10,000 pairs with known bugs let us objectively measure how good any test generator is (including non-AI baselines) using the kill rate metric.
- The **System-Level Functions** are the real-world targets. These are complex library functions (like `pandas.DataFrame.merge` or `json.loads`) that represent what a software engineer actually needs to test. The AI learns on these because they require deep semantic understanding.

**The 70 System-Level Functions (Live Targets)**
These are **100% real, correct library functions** that were manually curated and wrapped (e.g., `pandas.DataFrame.merge`, `json.loads`, `re.findall`). They contain **no artificial bugs**. They are stored in `config/system_functions.py` complete with signatures, docstrings, and known edge cases.

**How they connect:** The baselines are measured on the Unified Dataset to get the "ceiling" (82.7%). Then Oneiros trains on the System-Level Functions. After training, Oneiros is evaluated on held-out functions to see if it approaches the human-expert ceiling.

**Why train on System Functions instead of the Unified Dataset?**
- **Fair Evaluation:** If we train the AI directly on the 10,000 mutation pairs and evaluate its kill rate on the exact same pairs, the result is meaningless—the model could simply memorize the answers.
- **Organic Bug Hunting:** By practicing on real, correct library code (Phase 4), the AI learns a *skill*: "how to explore edge cases." If a test causes a crash here, it's a true, organic bug or unhandled edge case in the library itself, not an artificial mutation.
- **The Flow:** The baselines establish the "scoreboard" on the Unified Dataset (Phase 3). The AI trains its testing skills on the System-Level Functions (Phase 4). To test generalization, we could then evaluate the educated AI against held-out functions or the Unified Dataset itself.

### The Role of FAISS

**FAISS** (Facebook AI Similarity Search) is Oneiros's long-term memory. It serves three critical functions:

**1. Novelty Detection ("Have I seen this before?")**
- Every generated test is converted into a 384-dimensional vector using `sentence-transformers/all-MiniLM-L6-v2`
- FAISS computes the **cosine distance** between this vector and all stored vectors
- If the distance exceeds the `novelty_threshold` (default: 0.7), the test is considered **novel**
- Novel tests are classified as Winners even if they didn't find a bug — this encourages exploration

**2. Prompt Enrichment ("What worked before?")**
- Before generating tests for a function, the generator queries FAISS for the **k=3 nearest** successful tests
- These are injected into the Phi-3 prompt as "examples of good tests"
- This gives the AI a starting point: *"Tests like these worked before, try to generate something similar but different"*

**3. Deduplication ("Don't repeat yourself")**
- Without FAISS, the AI would generate the same test over and over (LLMs are prone to repetition)
- By checking similarity before storing, FAISS ensures the memory only contains diverse, high-quality tests

**Worked Example:**
```
Iteration 1: AI generates "merge({}, {}, on='id')"  →  FAISS stores it (distance=1.0, novel)
Iteration 2: AI generates "merge({}, {}, on='id')"  →  FAISS rejects  (distance=0.05, duplicate)
Iteration 2: AI generates "merge(df, df, how='cross')" → FAISS stores (distance=0.82, novel)
Iteration 3: Generator prompt now includes both stored examples as inspiration
```

### Coverage: What We Use and What We Don't

There are two different "coverage" concepts in this project:

| Concept | Tool | Role in Oneiros |
|---------|------|----------------|
| **Coverage Baseline** | `baseline/coverage_fuzzer.py` using `sys.settrace()` | An **active fuzzer** that generates tests by tracking which code branches are hit and evolving inputs to explore new branches. This is our strongest non-AI baseline (49.8% kill rate). |
| **coverage.py** | Listed in `requirements.txt` | A **passive measurement tool**. It does NOT generate tests. It simply reports which lines of code were executed during a test run. We use it only for optional reporting, not as a core component. |

**Key distinction:** Our coverage *baseline* is an Atheris-style fuzzer that *actively generates* new test inputs. Python's `coverage.py` module is a *passive observer* that just watches and reports. They are fundamentally different tools that happen to share the word "coverage."

---

## Project Structure

```
oneiros/
├── README.md                        # Project overview and research context
├── REPOSITORY_GUIDE.md              # Current setup and execution order
├── dataset_documentation.md         # Canonical corpus and leakage policy
├── requirements.txt                 # Runtime dependencies
├── requirements-dev.txt             # Runtime plus local test dependency
├── main.py                          # Extended local entry point
├── oneiros_loop.py                  # Legacy exploratory learning loop
│
├── engine/                          # Generator, SFT, DPO, memory, and oracle
│   ├── generator.py                 # Phi-3 candidate generation and policy gate
│   ├── sft_trainer.py               # Verified supervised fine-tuning
│   ├── dpo_trainer.py               # Optional mutation-aware DPO
│   ├── prompt_budget.py             # Shared prompt/completion budgets
│   ├── model_runtime.py             # Pinned runtime/model configuration
│   ├── memory.py                    # FAISS semantic memory
│   ├── oracle.py                    # Winner/loser classification
│   └── bug_discovery.py             # Sandboxed bug discovery helpers
│
├── harness/                         # Corpus, mutation, and execution pipeline
│   ├── corpus.py                    # Manifest/hash/split verification
│   ├── candidate_policy.py          # Bounded target-assertion policy
│   ├── safe_execution.py            # Restricted child-process execution
│   ├── execution_harness.py         # Differential evaluation harness
│   ├── mutation_engine.py           # Mutation generation
│   ├── bugsinpy_v2.py               # BugsInPy repository evidence
│   └── swebench_verified.py         # SWE-bench Verified adapter
│
├── baseline/                        # Non-AI comparison methods
│   ├── random_baseline.py
│   ├── static_baseline.py
│   ├── grammar_baseline.py
│   ├── coverage_fuzzer.py
│   └── benchmark_runner.py
│
├── config/                          # Canonical version and run configuration
│   ├── settings.py
│   └── system_functions.py
│
├── scripts/                         # Build, audit, preflight, train, evaluate
│   ├── train_on_dataset.py          # Authoritative SFT/DPO pipeline
│   ├── modal_train.py               # Cloud GPU launcher
│   ├── preflight_sft_run.py         # Exact GPU-free preflight
│   ├── modal_calibrate_sft_adapters.py
│   └── build_corpus_v*.py
│
├── metrics/                         # Evaluation aggregation and scoring
├── tests/                           # Hardened unit/regression test suite
├── utils/                           # Logging, embeddings, provenance hashes
│
├── data/                            # Local corpora; manifest tracked, data ignored
├── checkpoints/                     # Local adapters/weights; ignored
├── results/                         # Local experiment outputs; ignored
├── Oneiros.ipynb                    # Colab-oriented project notebook
└── colab_run.ipynb                  # Streamlined Colab launcher
```

---

## The Learning Loop (Detailed Workflow)

The Oneiros loop follows a **Generate → Execute → Evaluate → Store → Train → Repeat** cycle. Below is a step-by-step walkthrough of what happens when you run `python oneiros_loop.py`.

### Step 1: Data Loading & Initialization

**Loading the System-Level Functions:**
```
oneiros_loop.py  →  config/__init__.py  →  config/system_functions.py
```
- `get_training_functions()` returns **60 functions** (e.g., `pandas.DataFrame.merge`, `json.loads`, `re.findall`)
- `get_testing_functions()` returns **10 held-out functions** (never seen during training, used for final evaluation)
- Each function object contains: `name`, `signature`, `docstring`, `wrapper_code`, `edge_cases`, `library`

**Initializing the Engine Components:**
1. **FAISS Memory** is created with an empty vector index (384 dimensions, cosine similarity)
2. **Phi-3 Generator** is initialized (model weights are lazy-loaded on first generation call)
3. **Feedback Oracle** is created and linked to FAISS memory
4. **Seed Memory**: A rule-based `TestGenerator` creates **3 simple tests per function** (180 total) using type inference from signatures. These are added to FAISS as the initial "experience" so the AI has examples to reference from iteration 1.
5. **DPO Trainer** is initialized with the tokenizer (model loaded on first training call)

### Step 2: Test Generation (per function, per iteration)

For each of the target functions (default: 10 per iteration):

1. **Memory Query**: FAISS retrieves the **k=3 most similar** previous winner tests for this function
2. **Prompt Construction**: The generator builds a structured prompt:
   ```
   You are a test case generator for Python's pandas library.
   Function: merge(left, right, how='inner', on=None)
   Description: Merge DataFrame objects with a database-style join.
   Edge cases: empty DataFrames, missing join keys, duplicate columns

   Examples of good tests from memory:
   1. merge({}, {}, on='missing_key')
   2. merge(df, df, how='cross')

   Generate a Python test input that could find bugs.
   ```
3. **Model Inference**: Phi-3 generates **8 test inputs** (configurable) using temperature sampling
4. **Parsing**: Raw model output is parsed into structured test objects with validity checks

### Step 3: Test Execution

- The **Execution Harness** creates a clean Python namespace with `exec()`
- It injects the function's `wrapper_code` into the namespace
- It runs each generated test string against the function with a **5-second timeout**
- Results are classified:
  - `PASS`: Test ran successfully, no error (function behaves normally)
  - `FAIL`: An `AssertionError` was raised (bug detected)
  - `ERROR`: Any other exception (e.g., `TypeError`, `ValueError` — often indicates an edge case hit)
  - `TIMEOUT`: Execution exceeded 5 seconds (possible infinite loop)

### Step 4: Oracle Evaluation & Memory Storage

The **Feedback Oracle** applies a three-rule classification:

```
                    ┌─── Invalid syntax? ──→ LOSER (rejected)
                    │
Test Result ────────┼─── Found bug (FAIL/ERROR)? ──→ WINNER (stored in FAISS)
                    │
                    └─── Novel? (FAISS distance > 0.7) ──→ WINNER (exploration)
                         │
                         └── Not novel (duplicate) ──→ LOSER (rejected)
```

- **Bug-finding tests** are always Winners — they detected a real issue
- **Novel tests** are Winners even without finding a bug — they explore new testing strategies
- **Redundant tests** (too similar to existing memory) are Losers — the AI shouldn't repeat itself
- All Winners are immediately added to FAISS memory, enriching future prompts

### Step 5: DPO Training (periodic — every N iterations)

**When it triggers:** Every `dpo_train_every` iterations (default: 2), the system creates preference pairs from recent Winners and Losers:

```
Preference Pair = (Prompt, Chosen, Rejected)

Prompt:   "Generate test for pandas.DataFrame.merge"
Chosen:   "merge({}, {}, on='missing')"     ← Winner: found a KeyError bug
Rejected: "merge({'a': [1]}, {'b': [2]})"   ← Loser: boring, redundant
```

**Training mechanics:**
1. Collects the most recent **48 tests** (6 × `tests_per_iteration`) to find pairs
2. Pairs are created only when a Winner and Loser exist for the **same function**
3. TRL's `DPOTrainer` fine-tunes only the **LoRA adapter weights** (not the full 3.8B parameters)
4. The updated adapter is saved and **hot-swapped** into the generator
5. The next iteration uses the improved model immediately

**Expected behavior across iterations:**
```
Iteration 2 → DPO fires → Loss: 0.84
Iteration 4 → DPO fires → Loss: 0.71  ← model is learning
Iteration 6 → DPO fires → Loss: 0.58  ← decreasing loss = improvement
```

### Step 6: Checkpoint
- Saves the LoRA adapter, FAISS memory index, and statistics JSON to `data/checkpoints/iter_N/`
- Checkpoints allow resuming training or evaluating the model at any point

---

## Core Components

### Engine Modules

| Module | File | Purpose |
|--------|------|---------|
| **Phi-3 Generator** | `engine/generator.py` | Generates test inputs using Phi-3-mini-4k-instruct with QLoRA 4-bit quantization. Includes prompt construction with function signatures, docstrings, edge cases, and FAISS memory examples. Supports hot-swapping LoRA adapters after DPO training. |
| **FAISS Memory** | `engine/memory.py` | Semantic memory using FAISS vector index with `all-MiniLM-L6-v2` sentence embeddings. Stores successful test cases and provides novelty detection (cosine distance thresholding) and k-nearest retrieval for prompt enrichment. |
| **Feedback Oracle** | `engine/oracle.py` | Classifies test results as Winners or Losers. A test is a Winner if it found a bug OR is semantically novel. A test is a Loser if it is redundant (high similarity to existing memory entries) or has invalid syntax. |
| **DPO Trainer** | `engine/dpo_trainer.py` | Fine-tunes Phi-3 using Direct Preference Optimization. Takes Winner/Loser pairs, constructs a HuggingFace Dataset, and trains LoRA adapters using TRL's DPOTrainer. Supports iterative training with adapter saving/loading. |

### Harness Modules

| Module | File | Purpose |
|--------|------|---------|
| **Execution Harness** | `harness/execution_harness.py` | Safely executes generated Python test code in sandboxed namespaces with configurable timeouts. Supports both single-function testing and differential testing (golden vs mutant). |
| **Mutation Engine** | `harness/mutation_engine.py` | Generates code mutants using 12 mutation types (boundary, arithmetic, comparison, off-by-one, return, boolean, index, logical, membership, negate_removal, identity, string). All mutants are AST-validated. |
| **Dataset Loader** | `harness/dataset_loader.py` | Loads `unified_dataset.json` and `mutation_pairs.json`, providing `TargetFunction` and `Mutant` objects for the execution pipeline. |
| **DataLoader** | `harness/dataloader.py` | PyTorch-compatible DataLoader with 80/10/10 train/val/test splits and DPO triple conversion for training. |
| **System Dataset Loader** | `harness/system_dataset_loader.py` | Loads system-level functions (pandas, json, regex, etc.) for the Phase 4 learning loop. |

### Configuration

| File | Purpose |
|------|---------|
| `config/settings.py` | Dataclass-based configuration for model (Phi-3, QLoRA params), memory (FAISS dimensions, thresholds), training (DPO beta, learning rate, epochs), and dataset settings. |
| `config/system_functions.py` | Defines 70 system-level functions (60 training + 10 testing) from pandas, json, datetime, os, re, collections, urllib, and more. Each function includes signature, docstring, wrapper code, and edge cases. |

---

## Dataset

The Oneiros dataset is a **mutation-testing dataset** built from three high-quality sources:

| Source | Functions | Origin | Purpose |
|--------|:---------:|--------|---------|
| **HumanEval** | 164 | OpenAI | Hand-crafted problems with docstrings and tests |
| **MBPP** | 974 | Google | Short self-contained functions, broad coverage |
| **BugsInPy** | 493 | Academic | Real bugs from 17 OSS projects (pandas, flask, keras) |
| **Total** | **1,631** | | |

### Mutation Pairs (10,000)

Each golden function is mutated using 12 targeted mutation types to create controlled, labeled bug variants:

| Mutation Type | Count | % | Example |
|:--------------|------:|--:|---------|
| boundary | 4,162 | 41.6% | `0` → `1`, `1` → `2` |
| arithmetic | 2,588 | 25.9% | `+` → `-`, `*` → `/` |
| comparison | 1,157 | 11.6% | `<` → `>`, `==` → `!=` |
| off_by_one | 781 | 7.8% | `range(len(x))` → `range(len(x)-1)` |
| return | 323 | 3.2% | `return True` → `return False` |
| boolean | 299 | 3.0% | `True` → `False` |
| index | 264 | 2.6% | `[0]` → `[1]`, `[-1]` → `[-2]` |
| logical | 249 | 2.5% | `and` → `or` |
| membership | 77 | 0.8% | `in` → `not in` |
| negate_removal | 70 | 0.7% | `if not x:` → `if x:` |
| identity | 17 | 0.2% | `is` → `is not` |
| string | 13 | 0.1% | `.lower()` → `.upper()` |

### Quality Assurance

All 10,000 pairs pass an 8-point automated validation suite (100% pass rate):
1. Golden code ≠ Mutant code
2. Golden compiles with `ast.parse()`
3. Mutant compiles with `ast.parse()`
4. At least 1 test case exists
5. Entry point is specified
6. Mutations target body-only (not `def` lines or imports)
7. Diff size ≤ 3 lines
8. No regex artifacts introduced

---

## Baseline Benchmarks

Five non-AI baselines establish the performance spectrum from "pure luck" to "human expert." Each baseline generates tests for every function in the Unified Dataset, and those tests are run against all 10,000 mutation pairs using the `benchmark_runner.py` script.

### How Benchmarking Works (All Baselines)

The benchmarking process is identical for every baseline — only the test *generation* method changes:

```
For each of the 10,000 mutation pairs:
  1. Baseline generates up to 8 test inputs for the golden function
  2. Each test is executed against the GOLDEN (correct) code → capture output
  3. Same test is executed against the MUTANT (buggy) code  → capture output
  4. If outputs DIFFER → mutant is KILLED (bug detected)
  5. If outputs MATCH  → mutant SURVIVED (bug escaped)

Kill Rate = (Killed Mutants / 10,000) × 100
```

This is called **Differential Testing**: we don't need to know the "right answer" — we just need to see that the golden and mutant versions produce *different* results.

---

### 1. TestCases — Human Expert Baseline (82.7%)

**What it is:** The original human-written unit tests from the library authors (e.g., the `assert` statements that ship with HumanEval and MBPP).

**How it works:**
- Each mutation pair in the Unified Dataset already includes `test_cases` written by the original function authors
- We run these expert tests against both the golden and mutant code
- If any test passes on the golden but fails on the mutant → kill

**Why 82.7% and not 100%:**
- **Equivalent mutants**: Some mutations don't change observable behavior (e.g., changing `i < 5` to `i <= 4` when `i` is always an integer). No test can detect these because the output is identical.
- **Blind spots**: Even expert developers don't test every possible edge case. A change deep in a rarely-tested branch may escape human-written tests.

**Why it's our ceiling:** These tests represent decades of collective engineering expertise. If the AI can match 82.7%, it has achieved **human-level test generation ability**.

---

### 2. Random — Floor Baseline (28.7%)

**What it is:** Pure random input generation with zero knowledge of the target function.

**How it works (step by step):**
1. For each function, generates completely random values: random integers, random strings, random lists, random booleans
2. Does **not** read the function signature or type hints
3. Constructs function calls like: `result = some_function(42, "xkcd", [True, -7])`
4. Most calls crash immediately with `TypeError` because the inputs are incompatible
5. Occasionally, a random input happens to trigger a bug by sheer luck

**Why it still kills 28.7%:** Many mutants are fragile — even garbage input causes them to crash differently than the golden code. For example, if the golden code raises `ValueError` but the mutant raises `TypeError` on the same garbage input, that counts as a kill.

**Why it matters:** This is the absolute floor. Any approach that scores below 28.7% is literally worse than random noise.

---

### 3. Static — Template Baseline (30.4%)

**What it is:** Deterministic, handcrafted boundary-value templates. No randomness, no learning.

**How it works (step by step):**
1. Parses the function signature to detect parameter types (`int`, `str`, `list`, etc.)
2. For each type, applies a fixed set of "expert-chosen" edge values:
   - `int` → `[0, 1, -1, 2, 10, 100, -100]`
   - `str` → `["", "a", "abc", "hello", " "]`
   - `list` → `[[], [1], [1,2,3], [0,0], [-1,0,1]]`
   - `bool` → `[True, False]`
3. Generates all combinations (cross-product) of these values across parameters
4. Runs each combination against the golden code to capture the expected output
5. Builds assertion tests: `assert function(0, "") == expected_result`

**Why it only reaches 30.4%:** It tries the same values for *every* function. It doesn't understand that `pandas.merge` needs DataFrames while `json.loads` needs strings. The templates are generic, not function-specific.

---

### 4. Grammar — Learn&Fuzz Baseline (33.5%)

**What it is:** Structure-aware fuzzing using learned input grammars. Inspired by Microsoft's Learn&Fuzz paper (Godefroid et al., 2017).

**How it works (step by step):**
1. **Grammar Learning** — The `GrammarLearner` class parses the function signature:
   ```python
   def merge(left: DataFrame, right: DataFrame, how: str = 'inner') -> DataFrame
   ```
   Produces grammar rules:
   ```
   ParamGrammar(name="left",  base_type="dict", inner_type="")
   ParamGrammar(name="right", base_type="dict", inner_type="")
   ParamGrammar(name="how",   base_type="str",  inner_type="", default="'inner'")
   ```

2. **Boundary-Value Selection** — Each grammar rule maps to a curated list of edge values:
   - `list` with `inner_type="int"` → `[[], [0], [1], [-1], [1,2], [1,2,3], [0,0,0], ...]`
   - `str` → `["", "a", "ab", "abc", "hello world", " ", "123", "!@#"]`

3. **Cross-Product Generation** — All combinations of boundary values are generated and capped at `num_tests`

4. **Golden Execution** — Each call is run against the golden code to capture expected output, then converted to assertions

**How it differs from Static:** Grammar understands *container types* (e.g., `List[int]` gets integer lists, not random strings). Static treats every `list` parameter the same way.

**Why it matters:** It proves that even type-aware generation without semantic understanding only reaches 33.5%. The gap between Grammar (33.5%) and TestCases (82.7%) is the "semantic gap" that Oneiros's AI must close.

---

### 5. Coverage — Atheris-style Baseline (49.8%)

**What it is:** Coverage-guided fuzzing that emulates the core algorithm of Google's Atheris security fuzzer.

**How it works (step by step):**

1. **Instrumentation** — Uses Python's `sys.settrace()` to hook into every line execution:
   ```python
   class CoverageTracer:
       def trace(self, frame, event, arg):
           lineno = frame.f_lineno
           self.lines.add(lineno)                        # track line coverage
           self.branches.add((self._prev_line, lineno))   # track branch transitions
   ```

2. **Seed Corpus** — Builds initial test inputs using type inference (similar to Grammar), capped at 30 seeds

3. **Phase 1 — Seed Evaluation:**
   - Runs each seed against the golden function with tracing enabled
   - Records which lines and branches were hit
   - Seeds that reach **new lines** are saved to the "corpus"

4. **Phase 2 — Evolutionary Mutation (up to 50 iterations):**
   - Picks a random parent from the corpus
   - Applies one of 5 mutation strategies:
     - `_mutate_number`: Change a number by ±1, ×2, or to 0/-1
     - `_mutate_list`: Empty, extend, or duplicate a list
     - `_mutate_string`: Replace with a different string value
     - `_mutate_bool`: Flip True↔False
     - `_swap_arg`: Swap the order of two arguments
   - Runs the mutated input with coverage tracking
   - If it hits a **new branch** → save to corpus and keep mutating from there
   - If no new coverage → discard

5. **Assert Generation** — Final corpus entries are executed against golden code to capture expected outputs and build assertion tests

**Comparison with industry tools:**

| Tool | Type | Generates Tests? | Tracks Coverage? | Evolves Inputs? |
|------|------|:---:|:---:|:---:|
| **Pylint** | Static Analyzer | ✗ | ✗ | ✗ |
| **coverage.py** | Passive Reporter | ✗ | ✓ (reports only) | ✗ |
| **Atheris (Google)** | Active Fuzzer | ✓ | ✓ (C++ instrumentation) | ✓ |
| **Our Coverage Baseline** | Active Fuzzer | ✓ | ✓ (`sys.settrace()`) | ✓ |

**How close to real Atheris:**
- ✅ **Same core algorithm:** Feedback-driven evolutionary exploration
- ✅ **Same coverage metric:** Branch coverage (line transitions)
- ✅ **Same mutation strategy:** Random perturbation of interesting inputs
- ⚠️ **Speed difference:** Atheris (C++ LLVM): millions of tests/sec. Ours (Python tracing): hundreds/sec
- ⚠️ **Scope:** Atheris instruments at bytecode level (can detect memory bugs). Ours operates at Python source level.

For our benchmark of 10,000 function-level mutation pairs, the speed difference doesn't affect kill rate results — both approaches have enough iterations to converge on the same coverage.

---

## Benchmarking Results

Full benchmark on **10,000 mutation pairs** with **8 tests per mutant maximum**:

| Baseline | Kill Rate | Mutants Killed | Total Tests | Time |
|----------|-----------|----------------|-------------|------|
| **TestCases** | **82.7%** | 8,274 | 58,499 | 245s |
| **Coverage** | 49.8% | 4,983 | 16,229 | 1,261s |
| **Grammar** | 33.5% | 3,352 | 63,949 | 784s |
| **Static** | 30.4% | 3,037 | 56,612 | 550s |
| **Random** | 28.7% | 2,871 | 79,258 | 683s |

**Key observations:**
- Random generates the **most tests** (79K) but kills the fewest — quantity ≠ quality
- Coverage generates the **fewest tests** (16K) but kills 49.8% — intelligent selection matters
- The gap from Coverage (49.8%) to TestCases (82.7%) = **32.9 percentage points** — this is the "semantic understanding gap" that only AI or human expertise can bridge

---

## Kill Rate: What It Is and Why It Matters

### Definition

**Kill Rate** (also called **Mutation Score**) measures how effective a set of tests is at detecting bugs. It is the standard metric in mutation testing research.

### Formula

```
Kill Rate = (Number of Killed Mutants / Total Number of Mutants) × 100
```

### How a Mutant is "Killed"

A mutant is killed when a test produces **different output** on the golden (correct) code vs. the mutant (buggy) code:

```python
# Golden code                    # Mutant code (boundary mutation: < → <=)
def is_prime(n):                 def is_prime(n):
    if n < 2:                        if n <= 2:        # ← BUG
        return False                     return False
    ...                              ...
    return True                      return True

# Test: is_prime(2)
# Golden: n < 2 → False → continues → returns True  ✓
# Mutant: n <= 2 → True → returns False              ✗ (different!)
# Result: MUTANT KILLED — the test detected the bug
```

### Worked Example with Our Baselines

Consider a mutant where `+` is changed to `-` in a sum function:

| Baseline | Test Generated | Golden Output | Mutant Output | Killed? |
|----------|---------------|---------------|---------------|:-------:|
| Random | `sum_func("hello", [])` | `TypeError` | `TypeError` | ✗ (same error) |
| Static | `sum_func(0, 0)` | `0` | `0` | ✗ (`0+0 == 0-0`) |
| Grammar | `sum_func(1, 2)` | `3` | `-1` | ✓ (`1+2 ≠ 1-2`) |
| Coverage | `sum_func(-1, 1)` | `0` | `-2` | ✓ |
| TestCases | `assert sum_func(3, 4) == 7` | `7` | `-1` | ✓ |

### Why Kill Rate Matters

1. **It measures real bug-finding ability**, not just code coverage or line count
2. **It's objective and reproducible** — the same mutants produce the same results every time
3. **It provides a single comparable number** across all baselines and the AI
4. **It's the standard metric** used in mutation testing research (Jia & Harman, 2011)

### How It's Used in Oneiros

- **Phase 3** established the baseline spectrum: Random (28.7%) → TestCases (82.7%)
- **Phase 4** trains the AI to generate tests. After training, we evaluate the AI's tests against the same 10,000 mutants
- **The goal**: If Oneiros achieves a kill rate close to 82.7%, it proves the AI has learned to generate tests at human-expert quality

---

## Setup & Installation

### Local verification (GPU-free)
```bash
cd oneiros
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

The canonical corpus is distributed separately. Before preflight or training,
place it at `data/corpus/v4_unified_prompt_candidate/` and verify that its hashes match
the tracked manifest.

### Google Colab (for full AI training loop — requires T4 GPU or better)
```bash
pip install torch transformers peft trl accelerate
pip install faiss-cpu sentence-transformers
pip install bitsandbytes>=0.46.1
pip install datasets mutmut pandas
```

---

## Running in Google Colab

1. Upload the `oneiros/` folder to Google Drive
2. Open a Colab notebook with GPU runtime (T4 recommended)
3. Mount Google Drive and run:

```python
%cd /content/drive/MyDrive/Capstone/oneiros
!pip install -U bitsandbytes>=0.46.1
!python oneiros_loop.py
```

**Minimum GPU Requirements:**
- ~10 GB VRAM for Phi-3 in 4-bit quantization + DPO training
- Google Colab T4 (15 GB) is sufficient

---

## Key Configuration Parameters

Defined in `oneiros_loop.py` (`LoopConfig`) and `config/settings.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_iterations` | 6 | Number of learning loop iterations |
| `tests_per_iteration` | 8 | Tests generated per function per iteration |
| `dpo_train_every` | 2 | DPO training frequency (every N iterations) |
| `save_every` | 2 | Checkpoint save frequency |
| `model_name` | `microsoft/Phi-3-mini-4k-instruct` | Base LLM for generation |
| `lora_r` | 16 | LoRA rank |
| `lora_alpha` | 32 | LoRA scaling factor |
| `learning_rate` | 5e-5 | DPO training learning rate |
| `beta` | 0.1 | DPO beta (preference strength) |
| `novelty_threshold` | 0.7 | FAISS cosine distance for novelty detection |
| `execution_timeout` | 5.0s | Maximum time per test execution |

---

## Scripts & Utilities

| Script | Purpose |
|--------|---------|
| `scripts/download_datasets.py` | Downloads HumanEval and MBPP from HuggingFace Hub |
| `scripts/download_bugsinpy.py` | Downloads and processes BugsInPy dataset |
| `scripts/generate_mutations.py` | Generates 10,000 mutation pairs using 12 mutation types with AST validation |
| `scripts/full_manual_test.py` | Runs 8-point validation suite across all 10K pairs |
| `scripts/check_mutations.py` | Quick sanity check on mutation pair integrity |
| `scripts/check_regex_leak.py` | Detects broken regex sequences introduced by mutations |
| `scripts/inspect_mutations.py` | Detailed inspection of individual mutation pairs |
| `baseline/benchmark_runner.py` | Runs all 5 baselines against the full dataset and outputs results |

---

## Future Work

The current canonical V4 corpus provides 8,237 behaviorally verified
records with group-disjoint train/validation/test splits. The next work is to:

1. train a larger balanced SFT adapter on the eligible training split;
2. run the predeclared five-seed base/SFT/feedback/diversity validation plan;
3. run leave-one-mutation-family-out training and validation;
4. start DPO only after SFT passes the repeated full-validation quality gate;
5. compare SFT and DPO by unique function coverage, regressions, redundancy, and
   Kill@1/2/4/8 rather than one aggregate smoke percentage;
6. execute untouched BugsInPy/SWE-bench tasks in native isolated repository
   environments; and
7. open the sealed test split once, only after the policy and paper tables are
   frozen.

The exact evaluation protocol and Modal commands are maintained in the Phase 3
research runbook linked below.

---

## References

Phase 3 research evaluation is specified in
[`RESEARCH_EVALUATION_AND_MODAL_RUNBOOK.md`](RESEARCH_EVALUATION_AND_MODAL_RUNBOOK.md).
The defensible novelty claim and current related-work comparison are recorded in
[`RELATED_WORK_AND_RESEARCH_GAP_2026.md`](RELATED_WORK_AND_RESEARCH_GAP_2026.md).

- **DPO:** Rafailov et al., "Direct Preference Optimization: Your Language Model is Secretly a Reward Model" (2023)
- **Phi-3:** Microsoft, "Phi-3 Technical Report" (2024)
- **FAISS:** Johnson et al., "Billion-scale similarity search with GPUs" (2019)
- **HumanEval:** Chen et al., "Evaluating Large Language Models Trained on Code" (2021)
- **MBPP:** Austin et al., "Program Synthesis with Large Language Models" (2021)
- **BugsInPy:** Widyasari et al., "BugsInPy: A Database of Existing Bugs in Python Programs" (2020)
- **Atheris:** Google, "Atheris: A Coverage-Guided Python Fuzzing Engine" (2021)
- **Learn&Fuzz:** Godefroid et al., "Learn&Fuzz: Machine Learning for Input Fuzzing" (2017)
