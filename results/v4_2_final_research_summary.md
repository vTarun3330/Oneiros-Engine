# Oneiros — Final Research Summary

Generated from committed artifacts by `scripts/emit_final_research_summary.py`. Every figure is read from a receipt listed under *Sources*; none is transcribed.

> **POST-GENERATION AMENDMENT (2026-09-17).** This document was generated on
> 2026-09-15, before the sealed final test was attempted. The attempt was made
> on 2026-09-16 and **failed after authorization**, producing **zero sealed
> candidates and zero reportable metrics**. The one-time authorization is spent
> and **the consumed `test` split must not be rerun**.
>
> Every figure below is unchanged and remains correct — all of it is locked
> validation and development evaluation, which is now the project's **only**
> valid empirical evidence. **No final-test result exists**, so no final-test
> Oneiros claim and no Oneiros-versus-Atheris claim is supported.
>
> This amendment was written by hand and is not produced by
> `scripts/emit_final_research_summary.py`; regenerating this file would drop
> it. See [`../docs/SEALED_FINAL_INCIDENT.md`](../docs/SEALED_FINAL_INCIDENT.md)
> and [`../docs/POST_INCIDENT_RESEARCH_STATUS.md`](../docs/POST_INCIDENT_RESEARCH_STATUS.md).

## Final selection

| | |
|---|---|
| **Final candidate** | immutable base `Qwen/Qwen2.5-Coder-1.5B-Instruct` @ `2e1fd397ee46e1388853d2af2c993145b0f1098a` |
| **Arm A@431 (SFT)** | rejected for promotion at locked validation |
| **O1 sidecar / Arm B** | rejected on development |
| **Promoted SFT model** | none |

> RETAIN the immutable base model

## Stage 1 — development: four-arm experiment (ablation_dev, 542 functions)

Decision rule frozen before any result existed (`frozen_before_any_evaluation_ran`: True; `thresholds_chosen_before_seeing_results`: True).

| arm | Kill@1 | Kill@2 | Kill@4 | Kill@8 | killed/542 | reference validity |
|---|---|---|---|---|---|---|
| base | 0.2343 | 0.3487 | 0.4834 | **0.6052** | 328 | 0.5655 |
| A@150 | 0.3100 | 0.4317 | 0.5720 | **0.6900** | 374 | 0.4585 |
| A@431 | 0.3413 | 0.4594 | 0.5793 | **0.6956** | 377 | 0.4550 |
| B@150 | 0.2915 | 0.4262 | 0.5351 | **0.6587** | 357 | 0.4488 |

**Primary comparison — B@150 vs A@150 (matched training duration, isolates the O1 effect):** Kill@8 -3.14 points, 44 gained / 61 lost, net -17, McNemar exact p = 0.118 → **DO NOT PROMOTE**.

**Secondary — B@150 vs A@431:** Kill@8 -3.69 points, net -20, p = 0.07213 → **DO NOT PROMOTE**.

### O1 rejection

- B@150 lost to A@150 on Kill@8 by 3.14 percentage points.
- B@150 gained 44 functions and lost 61 relative to A@150.
- McNemar exact two-sided p = 0.118, so there is no statistically reliable O1 benefit.
- O1 is rejected for this configuration and mixing ratio (15.066%). Do not increase its share and do not retrain it.

## Stage 2 — locked validation: base vs Arm A@431 (val, 757 functions)

Decision rule frozen before either arm ran (`frozen_before_results`: True). The bar was set deliberately above the development screen because ablation_dev had selected this checkpoint and so flattered it.

| metric | base | A@431 |
|---|---|---|
| Kill@1 | 0.236460 | 0.239102 |
| Kill@2 | 0.365918 | 0.347424 |
| Kill@4 | 0.472919 | 0.479524 |
| Kill@8 | 0.594452 | 0.628798 |
| parse validity | 0.987285 | 0.988771 |
| execution validity | 0.928336 | 0.942206 |
| reference validity | 0.452114 | 0.333388 |
| redundancy | 0.677419 | 0.675749 |
| diversity exact-unique | 0.811963 | 0.864503 |
| diversity input-shape | 0.155869 | 0.141418 |
| Kill@8 functions | 450/757 | 476/757 |
| functions with a reference-valid candidate | 675 (0.8917) | 566 (0.7477) |

**Paired (n=757):** gained **114**, lost **88**, net **+26**, killed by both 362, by neither 193. **McNemar exact two-sided p = 0.0783200108.** Kill@8 delta **+3.4346 points**.

### Frozen promotion rule, as applied

| criterion | measured | result |
|---|---|---|
| 1 practical kill gain | +3.4346 pts / net +26 / p = 0.07832 | **FAIL** |
| 2 reference validity | -11.8726 pts (max drop 12.0) | **PASS** |
| 3 execution and parse | parse +0.1486, exec +1.3870 | **PASS** |
| 4 diversity | exact +6.47%, input -9.27%, redundancy -0.25% | **PASS** |
| 5 integrity | shared run contract: False | **FAIL** |

**Decision: RETAIN the immutable base model.**

Criterion 1 is the substantive failure: the gain is the right sign and too noisy to call. Criterion 5 failed for an implementation reason documented separately in `docs/LOCKED_VALIDATION_INTEGRITY_DEFECT.md`; it is preserved as recorded history and the decision does not depend on it.

## Limitations

These bound every claim made above.

1. **Reference-validity regression.** SFT cost -11.87 points of reference validity against base on the locked split (0.4521 → 0.3334), and functions with any reference-valid candidate fell from 675 to 566. The same regression appeared in every development SFT arm. It passed criterion 2 only because a 12.0-point tolerance had been declared in advance, and it passed by 0.13 points.
2. **No promoted SFT model.** The final candidate is the base model. Nothing here supports a claim of robust Oneiros SFT generalization.
3. **Repository records are excluded from every kill rate reported.** 38 were held out on the development panel, per the selection receipt. The locked panel likewise excluded its repository records; that count is recorded in the locked run artifacts rather than in the committed receipt, so it is not restated here.
4. **No real native repository result.** No repository-native execution was performed; no real-repository performance is claimed.
5. **Development-panel figures are not generalization evidence.** ablation_dev selected the checkpoints it scored. The locked val split did not, which is why the +9.04 development gain became +3.43 under locking.
6. ~~**The sealed final test has not been opened.**~~ **Superseded 2026-09-17:** it was opened once, on 2026-09-16, and the run failed after authorization with zero sealed candidates and zero metrics. The split is consumed and must not be rerun. No final-test evidence exists.

## Sources

| artifact | sha256 |
|---|---|
| `results/v4_2_development_selection_receipt.json` | `0188324fe9eb2a51eebef71bbd4cc3ed45584f12393bf0fa6539817a09e58969` |
| `results/v4_2_frozen_development_evaluation_receipt.json` | `062774028693cfcbf5346eafeed232027748720940af7a47b5ac1a591055c4fe` |
| `results/v4_2_locked_validation_preflight.json` | `97be6855d41b1186a5c8a59a8a89efa71e1f40f5e1c7f29ab387402b7598912c` |
| `results/v4_2_locked_validation_result_receipt.json` | `f7cdf84df8e65c51bd22410759c6fd54952ace3bfa80c6ab620674b53e288af9` |

Generated 2026-09-15T18:47:56Z at commit `448ac383238633dc5f0ddf9888ce50e8022730c0`.
