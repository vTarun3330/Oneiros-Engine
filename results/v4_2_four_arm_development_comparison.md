# Four-Arm Development Selection Experiment - Final Comparison

542-function `ablation_dev` panel - successor protocol `oneiros_successor_generation_protocol_v1` - whole-output parsing - 8 candidates - seed 42.

**Decision: reject Arm B and the 15.066% O1 sidecar. Retain Arm A checkpoint 431 as the selected development SFT candidate. Retain the immutable base model as the fallback comparator.**

## Arms

| arm | what it is | adapter sha256 | artifact sha256 |
|---|---|---|---|
| base | immutable base model, no adapter | `n/a` | `ec9831b7354380c9975bc5ab493ace43af771f61c491b1c6e58d0232ecbd1826` |
| A@150 | Arm A step 150, evaluation-only container (see provenance note) | `0cd79173ecee03829f0f2059db030d9220fdb424e34b15879c550e321398c882` | `0d491f4551e877bbb6b83f108157b94d17997e228cf68041bec04679533f67ed` |
| A@431 | Arm A step 431, selected candidate | `e67dd599a37cbf2a738791c5cdf889cb4938a2ad53c991dca2fefae503b6f9e7` | `b58294e631fbe131b0031c23c7b7fe903be8294b1560cdf0bbc7a7a9a3a273ac` |
| B@150 | Arm B step 150, O1 sidecar at 15.066% | `08b6d71c239c517ca350f0de21fafec084cb8f04a7185d0dab4f0769fedc2ad8` | `e6e7e122b1ca821073fdba801b8367d7cd455d2eee3eb1b8698bd643f6eea13f` |

## Kill@k

| arm | Kill@1 | Kill@2 | Kill@4 | Kill@8 | killed/542 | Wilson 95% |
|---|---|---|---|---|---|---|
| base | 0.2343 | 0.3487 | 0.4834 | **0.6052** | 328 | 0.5634-0.6454 |
| A@150 | 0.3100 | 0.4317 | 0.5720 | **0.6900** | 374 | 0.6499-0.7275 |
| A@431 | 0.3413 | 0.4594 | 0.5793 | **0.6956** | 377 | 0.6556-0.7328 |
| B@150 | 0.2915 | 0.4262 | 0.5351 | **0.6587** | 357 | 0.6178-0.6973 |

## Validity, diversity and integrity

| arm | parse | execution | reference-valid | fns with ref-valid | redundancy | div exact | div AST | div input | outcome modes | prompt-budget fails | completion-limit |
|---|---|---|---|---|---|---|---|---|---|---|---|
| base | 0.9834 | 0.9486 | 0.5655 | 0.9151 | 0.7040 | 0.7567 | 0.3644 | 0.1525 | 2.2804 | 0 | 1/4336 = 0.0231% |
| A@150 | 0.9975 | 0.9751 | 0.4585 | 0.7989 | 0.7238 | 0.8472 | 0.3286 | 0.1664 | 2.0185 | 0 | 9/4336 = 0.2076% |
| A@431 | 0.9982 | 0.9931 | 0.4550 | 0.7934 | 0.7362 | 0.8638 | 0.3516 | 0.1523 | 1.9280 | 0 | 1/4336 = 0.0231% |
| B@150 | 0.9982 | 0.9735 | 0.4488 | 0.8100 | 0.7207 | 0.8447 | 0.3354 | 0.1645 | 2.0314 | 0 | 4/4336 = 0.0923% |

## Frozen comparisons

### PRIMARY - B@150 vs A@150 (matched training duration, isolates the O1 effect)

- Kill@8 0.690037 -> 0.658672, delta **-3.14 points**
- functions gained 44, lost 61, **net -17**
- killed by both 313, by neither 124
- McNemar exact two-sided **p = 0.118** (not significant at 0.05)
- reference-valid -0.97 points, parse +0.07, execution -0.16
- diversity exact -0.30%, input-shape -1.17%, redundancy -0.43%

**Promotion rule: DO NOT PROMOTE**
- criterion 1 (practical kill gain): FAIL
- criterion 2 (reference validity): PASS
- criterion 3 (parse and execution): PASS
- criterion 4 (diversity): PASS
- criterion 5 (integrity): PASS

### SECONDARY - B@150 vs A@431 (practical selected-checkpoint policy)

- Kill@8 0.695572 -> 0.658672, delta **-3.69 points**
- functions gained 46, lost 66, **net -20**
- killed by both 311, by neither 119
- McNemar exact two-sided **p = 0.0721264** (not significant at 0.05)
- reference-valid -0.62 points, parse +0.00, execution -1.96
- diversity exact -2.20%, input-shape +8.01%, redundancy -2.11%

**Promotion rule: DO NOT PROMOTE**
- criterion 1 (practical kill gain): FAIL
- criterion 2 (reference validity): PASS
- criterion 3 (parse and execution): FAIL
- criterion 4 (diversity): PASS
- criterion 5 (integrity): PASS

### Each arm vs the accepted base control

- **A@150**: Kill@8 +8.49 points, net +46 functions, McNemar p = 0.000175033, reference-valid -10.70 points
- **A@431**: Kill@8 +9.04 points, net +49 functions, McNemar p = 8.21285e-05, reference-valid -11.05 points
- **B@150**: Kill@8 +5.35 points, net +29 functions, McNemar p = 0.0156442, reference-valid -11.67 points

## Findings

- B@150 lost to A@150 on Kill@8 by 3.14 percentage points.
- B@150 gained 44 functions and lost 61 relative to A@150.
- McNemar exact two-sided p = 0.118, so there is no statistically reliable O1 benefit.
- O1 is rejected for this configuration and mixing ratio (15.066%). Do not increase its share and do not retrain it.
- SFT helps Kill@8 on the development panel, but causes a substantial reference-validity regression versus base: -11.05 points for A@431, -10.70 for A@150, -11.67 for B@150.
- These are development-panel results only. They are not unseen-generalization claims and not real-repository claims.
- The 38 repository targets remain without native execution evidence and are excluded from every kill rate reported here.

## Provenance limitation: the A@150 arm

The A@150 arm was NOT a separately trained checkpoint. checkpoints/local_eval_armA_ckpt150_matched/ is an evaluation-only container holding a byte-identical copy of optimizer step 150 from the completed Arm A run. No training ran in that directory. To make the evaluator accept it, three precondition files were added with explicit user authorisation: sft_complete.marker (contents state that no training ran; the code only tests existence and never reads it), dataset_manifest.sha256 and sft_metadata.json (both byte-identical copies of Arm A run files). sft_metadata.json describes Arm A complete 431-step run: its sft_loss 0.793075082335284 is the END-OF-TRAINING loss for the 431-step run and is NOT a property of step 150. For TRAINING_PHASE sft_eval the only load-bearing field is dataset_fingerprint; the six accounting fields read afterwards are assigned to locals that sft_eval never uses, because it returns before they would be written to any manifest, and none of them appear in the evaluation result artifact. Recommended permanent fix: add an explicit --adapter-dir flag so an arbitrary adapter can be evaluated without constructing a run-shaped directory.

## Interpretation limits

- ablation_dev selected the checkpoints for both arms, so these numbers are optimistically biased for the checkpoints they chose and are NOT generalization evidence.
- Locked validation is not read and the sealed final test is not opened.
- No repository-native claim: 38 repository records are held out of every kill rate.
- Interim results must not change prompts, data, thresholds or checkpoints.

## Integrity at close

- frozen evaluation receipt sha256 `062774028693cfcbf5346eafeed232027748720940af7a47b5ac1a591055c4fe`, unchanged
- source_tree_sha256 `f91416c91805929676599473ace6a6265fdebd6a0cc23f90662e0e729e110752` (matches the frozen receipt: True)
- run_contract_sha256, evaluation_scope_sha256 and evaluation_profile_sha256 are identical across all four arms
- raw-output present and hash-matching for 4,336/4,336 candidates in every arm
- 0 prompt-budget failures in every arm

## Next permitted step

Prepare, but do not launch, a locked-validation preflight for exactly two arms: the immutable base model and Arm A checkpoint 431. The sealed final test stays closed.
