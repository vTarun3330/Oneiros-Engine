# Protocol v2.5 — addendum 2: training exposure and training prompt budget (additive)

The frozen v2.5 document (`c707d8ae…`) and addendum 1 (`7676aac2…`) are unchanged. This
addendum was frozen on 2026-10-01, before any v2.5 training run, any v2.5 GPU output and any
confirmation outcome existed. It replaces **only** addendum 1 §6 "Exposure" and the training
prompt budget in addendum 1 §6 "Held constant". Every other rule stands.

## 1. Why addendum 1 §6 is replaced

Addendum 1 matched arm C to the baseline adapter by **431 optimizer updates**, filling missing
updates with extra epochs. That rule silently assumed a corpus near the baseline's size.

| Quantity | Value | Source |
|---|---|---|
| Baseline unique examples | 5,735 | training-shape rebuild `fac747a1…` |
| Baseline effective examples (one epoch) | 6,889 (+7 deterministic padding) | `sft_run_config.json` |
| Baseline supervised target tokens | **570,775** (completion + EOS, effective-weighted) | exact rebuild, frozen tokenizer |
| Baseline per-example exposure | 5,511 synthetic × 1; repository 118 × 8, 96 × 4, 10 × 5 | exact rebuild |
| v2.5 module tokens | synthetic ≈ 46, repository ≈ 415 (max 1,173) per module | stage1_r2 |
| v2.5 corpus at the training-gate minimum | ≈ 300 unique (150 repository + ≤ 150 replay) | addendum 1 §5 |
| Epochs implied by 431 updates at that size | **≈ 23** | 431 × 16 / 300 |

Cycling about 300 examples about 23 times is the memorisation risk this study investigates.
It would also exceed the baseline's supervised-token budget roughly threefold.

## 2. Frozen exposure rule (arm C)

Let:
- **N** = the number of unique examples in the frozen final corpus;
- **S** = ⌈N / 16⌉, the optimizer updates per complete pass (effective batch 16, no padding
  rows);
- **T** = the supervised target tokens of one pass (completion + EOS, frozen tokenizer);
- **T_hist** = 570,775.

The plan is:

1. **Passes:** E = min(3, ⌊T_hist / T⌋). If E = 0, the corpus is refused (it would need a
   partial epoch).
2. **Optimizer updates:** U = min(431, E × S).
3. **Ceilings:** U never exceeds 431, and E × T never exceeds T_hist.
4. **Exposure:** every unique example is seen exactly E times (or, if U < E × S because of the
   431 ceiling, at most E times). No row is duplicated inside an epoch, and there are no
   padding duplicates. A small corpus is never cycled to reach 431 updates.
5. **Warmup:** min(25, U − 1) steps. The learning rate, scheduler, LoRA, base revision, seed 42
   and effective batch 16 stay as in addendum 1.
6. **Checkpoint:** the final update U is **the** checkpoint. No monitor is used, and no
   confirmation or validation data selects anything. (A train-only monitor would need its own
   addendum before training; none is planned.)
7. **Recorded:** N, S, T, E, U, E × T, the warmup, the exposure distribution and the
   pair-file hash, in the training receipt.

The plan is computed by `engine/sft_trainer_v25.py::plan_v25_exposure` and enforced by
`V25SFTTrainer.train`, which refuses a plan computed for a different dataset or one that would
cycle the data beyond E passes.

## 3. Training prompt budget

v2.5 forbids prompt compaction. Arm C must be trained on exactly the prompts it is evaluated
with, and the frozen generation contract admits prompts of **≤ 2,048** chat-templated tokens.
Addendum 1's 1,024-token training prompt limit came from the baseline, which compacted prompts.
Under v2.5 it would silently drop most repository examples: 4 of 5 verified sympy prompts are
1,140–2,255 tokens.

The training budgets are therefore:

| Budget | Value |
|---|---|
| Prompt | ≤ 2,048 tokens (equal to generation) |
| Completion | ≤ 1,024 tokens (equal to `max_new_tokens`) |
| Prompt + completion | ≤ 3,072 tokens |

A row over any budget is refused and listed; nothing is truncated. This decision uses token
lengths only, never a model outcome. It is a documented deviation from a held-constant factor.

## 4. Interpretation

- **Arm C differs from arm B** in supervision content and in exposure (arm B: 431 updates over
  6,889 effective examples). The C − B comparison is therefore reported as a **package-level
  practical comparison**, not a single-factor causal contrast.
- **A matched-compute causal contrast** would need a separately frozen and separately
  authorised matched-step control. This addendum does not require one, and arm C is never
  overtrained to imitate it.
- **The primary comparison (C − A)** is unaffected.
