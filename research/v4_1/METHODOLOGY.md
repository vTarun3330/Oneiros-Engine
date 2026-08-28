# V4.1 research methodology

## Formal task and localization

Oneiros V4.1 studies test generation after localization: the input is an already-localized buggy Python region, public behavioral intent, and legitimate execution context. Fault localization is not evaluated. Function targets use their declared public entry point. Repository targets use public issue identifiers when available and otherwise a deterministic buggy-side definition heuristic. The benchmark-provided affected region remains an explicit experimental assumption.

Oracle-patch-derived exact target localization is reserved for diagnostic ablation E0 and cannot be a headline condition. The active V4.1 prompt constructor has no reference-code, patch, oracle-result, mutation-label, or expected-completion parameter.

## Corpus construction and leakage control

V4.1 is derived from frozen V4 without rewriting it. Repository support context comes from the buggy revision: static framework configuration, public test-module path, non-test imports/constants/fixtures/helpers, and non-target buggy source units. Every test function and any class containing test methods is excluded before context construction. Changing the gold test body cannot change the support context.

Each model-visible field records lineage. A structural audit checks forbidden lineage, verbatim reference inclusion, fixed-only changed statements, specification overlap, and independent-context overlap. Exact fixed-only implementation lines in public issue text are removed while surrounding behavior prose is retained. Remaining overlap flags require an explicit, content-addressed disposition.

## Prompt and training

All datasets use prompt schema `oneiros_unified_test_generation_v2`. The requested output is one minimal, self-contained bug-revealing test case; repository cases may contain the setup and multiple assertions needed to demonstrate one defect. SFT applies loss only to completion tokens.

Prompt compaction happens before chat rendering. System/task instructions, mode, output format, target symbols, behavioral specification, and complete target units are preserved. Support units are removed in complete AST/source units. If required sections and a complete target cannot fit, the record fails closed rather than being token-spliced.

Periodic SFT checkpoints and the actual terminal optimizer step are evaluated exactly once. Resume state reuses completed monitor results, and a terminal checkpoint may become the preserved best adapter.

## Experimental partitions

The `ablation_dev` split is hash-frozen from training groups only. Its semantic groups do not appear in remaining training, locked validation, or final test. Prompt/data/model decisions use `ablation_dev`; locked validation is used only after those choices are frozen. The final test remains sealed until the reported adapter and protocol are frozen.

## Metrics and decision rule

Eight raw candidates are generated at temperature 0.7 and top-p 0.9. Candidate order is never execution-reranked for the headline condition. Reports separate parse-valid, execution-valid, reference-valid, and killing candidates and function-level Kill@1/2/4/8. Wilson intervals apply to each completed seed’s binomial proportion, not to the arithmetic mean across seeds. Dataset and mutation-family slices accompany the overall result.

Ablation selection prioritizes leakage safety, reference validity, execution validity, Kill@8, lower-k performance, dataset balance, and family robustness. Negative or inconclusive experiments remain recorded. Locked validation retains the 58% per-completed-seed SFT gate. DPO is allowed only after SFT passes; it must be compared with frozen SFT using the same generation and evaluation configuration.

## Claim boundary

Function-level mutation discrimination is executable today. Native execution of newly generated repository tests in reconstructed buggy and fixed BugsInPy/SWE-bench environments is not yet a working production harness, so repository supervision coverage is reported separately and no real-repository generated-test kill-rate claim is made.
