# Oneiros Engine - Google Colab Notebook
# =======================================
# This notebook runs the complete Oneiros learning loop on Colab.
#
# To use:
# 1. Upload the entire `oneiros` folder to Colab or mount Google Drive
# 2. Run each cell in order
# 3. Training will use Colab's free T4 GPU

# ============================================================
# CELL 1: Install Dependencies
# ============================================================
# !pip install transformers torch peft trl faiss-cpu sentence-transformers mutmut pandas -q

# ============================================================
# CELL 2: Mount Drive (if code is on Drive)
# ============================================================
# from google.colab import drive
# drive.mount('/content/drive')
# %cd /content/drive/MyDrive/Capstone/oneiros

# ============================================================
# CELL 3: Verify Installation
# ============================================================
"""
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
"""

# ============================================================
# CELL 4: Import Oneiros Components
# ============================================================
"""
import sys
sys.path.insert(0, '/content/oneiros')  # Or your path

from config import get_training_functions, get_testing_functions
from harness import SystemLevelDatasetLoader, ExecutionHarness
from engine import FAISSMemory, Phi3Generator, FeedbackOracle

print("Oneiros components loaded!")
"""

# ============================================================
# CELL 5: View System-Level Functions
# ============================================================
"""
training_funcs = get_training_functions()
testing_funcs = get_testing_functions()

print(f"Training functions: {len(training_funcs)}")
print(f"Testing functions: {len(testing_funcs)}")

print("\nTraining Functions:")
for f in training_funcs[:10]:
    print(f"  - {f.library}.{f.name}")

print("\nTesting Functions:")
for f in testing_funcs:
    print(f"  - {f.library}.{f.name}")
"""

# ============================================================
# CELL 6: Generate Dataset (Mutants)
# ============================================================
"""
from harness import generate_system_level_dataset

# This generates wrapper files and mutants
# Takes ~5 minutes
dataset = generate_system_level_dataset(mutants_per_function=10)

print(f"\nDataset Summary:")
print(f"  Training: {dataset.total_training_count} functions")
print(f"  Testing: {dataset.total_testing_count} functions")
print(f"  Mutants: {dataset.total_mutants}")
"""

# ============================================================
# CELL 7: Initialize FAISS Memory
# ============================================================
"""
memory = FAISSMemory()

# Add some seed inputs
memory.add("result = merge_wrapper({}, {}, on='key')", "sys_pandas_merge")
memory.add("result = json_loads_wrapper('{}')", "sys_json_loads")

print(f"Memory initialized with {memory.get_stats()['current_size']} entries")
"""

# ============================================================
# CELL 8: Load Phi-3 Model
# ============================================================
"""
# This downloads the model (~7.6 GB)
# Takes 5-10 minutes on first run

from engine import Phi3Generator

generator = Phi3Generator(load_in_4bit=True)
generator.load_model()

print("Phi-3 model loaded!")
"""

# ============================================================
# CELL 9: Generate Test Cases
# ============================================================
"""
func = training_funcs[0]  # pandas.merge

tests = generator.generate(
    function_signature=func.signature,
    docstring=func.docstring,
    function_id=func.id,
    edge_cases=func.edge_cases,
    memory_examples=memory.get_for_prompt(func.id),
    library=func.library,
    num_samples=3
)

print(f"\nGenerated {len(tests)} tests for {func.name}:")
for t in tests:
    print(f"  - {t.input_code}")
    print(f"    Valid: {t.is_valid}")
"""

# ============================================================
# CELL 10: Run Complete Learning Loop
# ============================================================
"""
from oneiros_loop import OneirosLoop, LoopConfig

config = LoopConfig(
    num_iterations=5,
    tests_per_iteration=8,
    dpo_train_every=3,
    use_mock_generator=False,  # Use real Phi-3
    verbose=True
)

loop = OneirosLoop(config)
results = loop.run()

print("\nFinal Results:")
print(f"  Bugs Found: {results['total_bugs_found']}")
print(f"  Winners: {results['total_winners']}")
print(f"  Losers: {results['total_losers']}")
"""

# ============================================================
# CELL 11: Train with DPO
# ============================================================
"""
from engine import DPOTrainer, DPODataPoint

# Create trainer
trainer = DPOTrainer()
trainer.setup_model()

# Create training pairs from loop results
pairs = []
for winner in loop.all_winners[:50]:
    for loser in loop.all_losers[:50]:
        if winner['function_id'] == loser['function_id']:
            pairs.append(DPODataPoint(
                prompt=f"Generate test for {winner['function_id']}",
                chosen=winner['input'],
                rejected=loser['input'],
                function_id=winner['function_id']
            ))

print(f"Created {len(pairs)} DPO training pairs")

# Train
results = trainer.train(pairs, num_epochs=1, batch_size=2)
print(f"Training loss: {results['loss']:.4f}")

# Save adapter
trainer.save_adapter()
"""

# ============================================================
# CELL 12: Evaluate on Test Functions
# ============================================================
"""
print("Evaluating on test functions...")

for func in testing_funcs:
    tests = generator.generate(
        function_signature=func.signature,
        docstring=func.docstring,
        function_id=func.id,
        edge_cases=func.edge_cases,
        library=func.library,
        num_samples=5
    )

    bugs_found = 0
    for test in tests:
        result = loop._execute_test({'input': test.input_code, 'id': test.id}, func)
        if result.get('found_bug'):
            bugs_found += 1

    print(f"  {func.name}: {bugs_found}/5 bugs found")
"""

# ============================================================
# CELL 13: Save Results
# ============================================================
"""
import json

results = {
    "loop_stats": loop.stats,
    "memory_stats": memory.get_stats() if memory else {},
    "generator_stats": generator.stats,
}

with open('experiment_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("Results saved to experiment_results.json")
"""

if __name__ == "__main__":
    print("Oneiros Colab Notebook")
    print("=" * 50)
    print("This file contains code for running Oneiros on Google Colab.")
    print("Copy the cells to a Colab notebook to run.")
