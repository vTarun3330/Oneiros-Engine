# V4.1 metrics amendment: upstream dataset identity

Status: declared before any V4.1 GPU ablation result.

The former `source_metrics` field grouped records by ingestion source. That is
not equivalent to the upstream-dataset macro required by the research brief:
`oneiros_clean_mutations` contains both MBPP and HumanEval. The correction is
additive and uses only canonical source metadata:

1. `source_name` and `source_metrics` remain as ingestion-source fields for
   backward readability.
2. `dataset_name`, `dataset_metrics`, and `equal_weight_dataset_macro` use
   `source.upstream`, falling back to `source.name` when upstream is absent.
3. Unknown labels are retained as `unknown`; they make the dataset macro
   explicitly incomplete rather than disappearing from the denominator.
4. Dataset labels are experiment metadata. They are never rendered into model
   prompts and are not inferred from record ids, code, hidden tests, reference
   code, patches, or oracle outcomes.

This amendment does not change the corpus, split membership, prompt contents,
candidate count/order, sampling temperature, evaluator, reference-pass/buggy-
fail rule, Kill@k definition, 58% gate, or final-test seal. It corrects grouped
reporting and supplies the label required for a future sampling ablation. Any
comparison must use matching metrics-schema and evaluation-profile identities.

Local evidence is recorded in `results/v4_1_dataset_sampling_audit.json`.
No V4.1 model result existed when this correction was made.
