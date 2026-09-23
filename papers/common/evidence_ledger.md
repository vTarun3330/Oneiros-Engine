# Evidence ledger

Every numeric claim used in any of the four paper drafts, mapped to the exact
committed artifact it comes from and that artifact's SHA-256.

**Rule applied throughout:** a value appears in a paper only if it either
appears verbatim in a committed artifact listed here, **or** is derived from such
values by an arithmetic operation recorded in §I below. Nothing is averaged,
smoothed, estimated or inferred. Where a fact could not be verified it is marked
`TODO — UNVERIFIED` and is **absent from every draft**.

An earlier version of this ledger claimed “nothing is recomputed”. That was
inaccurate: the drafts state differences, a ratio and rounded displays. §I now
records every one of them with its formula, inputs, unrounded result and the
exact locations where it appears.

Repository state at the time of writing: commit
`4f465622adf273ffcae672c93dfe3cc82c7d8645`, branch
`experiment/research-eval-ablations`, clean working tree.

---

## A. Source artifacts and their digests

| Key | Path | SHA-256 |
|-----|------|---------|
| `DEV-FROZEN` | `results/v4_2_frozen_development_evaluation_receipt.json` | `062774028693cfcbf5346eafeed232027748720940af7a47b5ac1a591055c4fe` |
| `DEV-SEL` | `results/v4_2_development_selection_receipt.json` | `0188324fe9eb2a51eebef71bbd4cc3ed45584f12393bf0fa6539817a09e58969` |
| `LV-PRE` | `results/v4_2_locked_validation_preflight.json` | `97be6855d41b1186a5c8a59a8a89efa71e1f40f5e1c7f29ab387402b7598912c` |
| `LV-RES` | `results/v4_2_locked_validation_result_receipt.json` | `f7cdf84df8e65c51bd22410759c6fd54952ace3bfa80c6ab620674b53e288af9` |
| `REH-RCPT` | `results/v4_2_rehearsal_receipt.json` | `bae9510218641b67cb07f065d6f1e5b01e3340d333618f0e1f52b36641414f4f` |
| `REH-EXEC` | `results/v4_2_rehearsal_execution_receipt.json` | `06aca918d2821a1f3f161e6f14a91978643f81b30e883746d8126a967136b59f` |
| `SF-V6` | `results/v4_2_sealed_final_executable_receipt_v6.json` | `4a2bc52ae1448e12ba31fd2ad68ff0b26b54dd9507456574ea66cb3936af7c8e` |
| `STATUS` | `docs/POST_INCIDENT_RESEARCH_STATUS.md` | `58d12e2c1db263f8818d920acf3ac1bda22257ba5f6f699408a87338fa53de15` |
| `INCIDENT` | `docs/SEALED_FINAL_INCIDENT.md` | `6d2e872912b8e1562c094602d2433c3d05a516fbe91d8a4ba6cddbf8b9b4c267` |
| `SUMMARY` | `results/v4_2_final_research_summary.md` | `ab21a1ab61e101e887c0cbad7c8748da931573b805571bcd634a11bcc6492dea` |
| `FOURARM` | `results/v4_2_four_arm_development_comparison.md` | `5c248c139b9321082ec50b96326340c2aabd555161d0f7a86b7a48ae50da3b15` |
| `LV-DEFECT` | `docs/LOCKED_VALIDATION_INTEGRITY_DEFECT.md` | *(committed; digest not required — no numeric claim drawn from it)* |

The rehearsal's raw result artifact,
`results/rehearsal_ablationdev_base_qwen_s42_v1/rehearsal_result.json`
(`3d72891452c586820300079ae83bf998415975eb808bea425e1a0a4d859e5c77`), is
**retained locally and Git-ignored** because it contains raw prompts and
completions. No paper draws a value from it directly; all rehearsal figures
come from `REH-EXEC`, which is the sanitised committed receipt of that run.

---

## B. Model and protocol

| Claim | Value | Source |
|---|---|---|
| Base model | `Qwen/Qwen2.5-Coder-1.5B-Instruct` | `LV-RES`, `REH-EXEC` |
| Immutable revision | `2e1fd397ee46e1388853d2af2c993145b0f1098a` | `REH-EXEC` › `model_identity` |
| Protocol name | `oneiros_successor_generation_protocol_v1` | `DEV-SEL` › `protocol` |
| Protocol digest | `4c703a7572733475bb565c2a57aaab533387838b7cfaa9ab0ba1e9c5beab54a8` | `DEV-SEL` › `protocol.protocol_sha256` |
| Candidates per function | 8 | `DEV-SEL` › `protocol.candidates_per_function` |
| Temperature / top-p | 0.7 / 0.9 | `DEV-SEL` › `protocol` |
| Generation seed | 42 | `DEV-SEL` › `protocol.generation_seed` |
| Parse mode | `whole_output` | `DEV-SEL` › `protocol.candidate_parse_mode` |
| Raw-output retention | `true` | `DEV-SEL` › `protocol.retain_raw_output` |
| Completion / sequence limits | 1024 / 3072 tokens | `DEV-SEL` › `protocol` |
| Reranking | `none` | `DEV-SEL` › `protocol.reranking` |
| Cross-arm contract digest | `6a87294cf72e28679ab711af677151e331772d5db45c85d631eb0673f256cdef` | `DEV-SEL` › `cross_arm_contract_identity` |
| Evaluation scope digest (`ablation_dev`) | `d37f75d6f44e7f25f8ec0b38ef1be46efd77c85c31cda115e81f7a77fb91f6ca` | `DEV-SEL`, `REH-EXEC` (identical) |

---

## C. Development evaluation — `ablation_dev`, selection-only

Source for every row: **`DEV-SEL` › `arms.<arm>.metrics`**.

| Arm | Kill@1 | Kill@2 | Kill@4 | Kill@8 | K@8 fns | Parse | Exec-valid | Ref-valid |
|---|---|---|---|---|---|---|---|---|
| base | 0.234317 | 0.348708 | 0.483395 | 0.605166 | 328 | 0.983395 | 0.948570 | 0.565498 |
| A@150 | 0.309963 | 0.431734 | 0.571956 | 0.690037 | 374 | 0.997463 | 0.975092 | 0.458487 |
| A@431 | 0.341328 | 0.459410 | 0.579336 | 0.695572 | 377 | 0.998155 | 0.993081 | 0.455028 |
| B@150 | 0.291513 | 0.426199 | 0.535055 | 0.658672 | 357 | 0.998155 | 0.973478 | 0.448801 |

Wilson 95% intervals for Kill@8 (`DEV-SEL`): base `[0.563412, 0.645440]`;
A@150 `[0.649879, 0.727520]`; A@431 `[0.655567, 0.732824]`;
B@150 `[0.617762, 0.697348]`.

**Primary predeclared comparison — B@150 vs A@150** (`DEV-SEL` ›
`comparisons.primary_matched_duration_o1_effect`):

| Quantity | Value |
|---|---|
| Kill@8 delta | −3.1365 points |
| Functions gained / lost / net | +44 / −61 / −17 |
| McNemar exact two-sided p | 0.11799998150585828 |
| Outcome | O1 sidecar rejected at mixing ratio 0.15066 |

**Interpretation limit carried verbatim into every paper** (`DEV-SEL` ›
`interpretation_limits[0]`): *"ablation_dev selected the checkpoints for both
arms, so these numbers are optimistically biased for the checkpoints they chose
and are NOT generalization evidence."*

Repository exclusion on the development panel: **38 repository records held out
of every kill rate** (`DEV-SEL` › `interpretation_limits[2]`).
*Note:* `REH-EXEC` reports **52** excluded repository-mode records for the same
split under the mode-aware admission rule introduced after the incident. The two
counts come from different exclusion mechanisms and different code versions;
papers cite **52** only for the rehearsal and **38** only for the development
evaluation, and never merge them.

---

## D. Locked validation — `val`

Source for every row: **`LV-RES` › `arms.<arm>.metrics`**, n = 757.

| Arm | Kill@1 | Kill@2 | Kill@4 | Kill@8 | K@8 fns | Parse | Exec-valid | Ref-valid | Fns w/ ref-valid |
|---|---|---|---|---|---|---|---|---|---|
| base | 0.236460 | 0.365918 | 0.472919 | 0.594452 | 450 | 0.987285 | 0.928336 | 0.452114 | 675 |
| A@431 | 0.239102 | 0.347424 | 0.479524 | 0.628798 | 476 | 0.988771 | 0.942206 | 0.333388 | 566 |

Wilson 95% for Kill@8: base `[0.559083, 0.628867]`; A@431 `[0.593812, 0.662483]`.
Prompt-budget failures: 0 for both arms.

**Paired comparison** (`LV-RES` › `paired_comparison`):

| Quantity | Value |
|---|---|
| n | 757 |
| Functions gained | 114 |
| Functions lost | 88 |
| Net functions | +26 |
| Killed by both | 362 |
| Killed by neither | 193 |
| Kill@8 delta | +3.4345999999999988 points |
| McNemar exact two-sided p | 0.07832001079866063 |
| Reference-validity delta | −11.8726 points |
| Parse delta | +0.14859999999999873 points |
| Execution delta | +1.3869999999999938 points |

**Predeclared criterion 1** (`LV-RES` › `promotion_criteria.1_practical_kill_gain`):

| Sub-criterion | Observed | Required | Passed |
|---|---|---|---|
| Kill@8 gain | +3.4346 pts | `>= +3.0` | yes |
| Net functions | +26 | `>= +15` | yes |
| McNemar p | 0.0783200108 | `< 0.01` | **no** |
| Criterion overall | — | — | **failed** |

**Decision** (`LV-RES` › `decision`): `"RETAIN the immutable base model"`.

Repository exclusion on the locked panel: **24 repository records held out of
every rate** (`LV-RES` › `interpretation_limits[2]`).

`LV-RES` › `interpretation_limits[0]` states that `val` *"was never used to
select a checkpoint, tune a prompt, or set a threshold, so this is a
generalization measurement in a way the ablation_dev four-arm experiment was
not."*

---

## E. Operational rehearsal — `ablation_dev`, pipeline fidelity only

Source for every row: **`REH-EXEC`**.

| Quantity | Value | Field |
|---|---|---|
| Requested records | 594 | `scope.requested_records` |
| Function-mode targets | 542 | `scope.function_mode_targets` |
| Repository records excluded | 52 | `scope.excluded_repository_records` |
| Unknown-mode excluded | 0 | `scope.excluded_unknown_mode_records` |
| Candidates expected / observed | 4336 / 4336 | `candidates` |
| Raw outputs present | 4336 | `candidates.raw_output_present` |
| Raw-output hashes present | 4336 | `candidates.raw_output_sha256_present` |
| Missing / mismatched | 0 / 0 | `candidates.raw_output_integrity` |
| Independently recomputed mismatches | 0 | `candidates.independently_recomputed_mismatches` |
| Parse valid / failures | 4264 / 72 | `outcomes` |
| Execution valid / failures | 4113 / 223 | `outcomes` |
| Reference valid | 2452 | `outcomes.reference_valid` |
| Killed candidates | 1108 | `outcomes.killed` |
| Prompt-budget failure targets | 0 | `outcomes.prompt_budget_failure_targets` |
| Timeouts | 43 | `outcomes.completion_or_execution_timeouts` |
| Wall time | 577.753 s | `execution.wall_time_seconds` |
| Generation batches / size | 271 / 2 | `execution` |
| Exit status | 0 | `execution.exit_status` |
| Progress files | 55, monotonic 10 → 542 | `progress_evidence` |
| Weights written | `false` | `model_identity.weights_written` |
| Training performed | `false` | `model_identity.training_performed` |
| Adapter | `null` | `model_identity.adapter` |

**Kill@k** (`REH-EXEC` › `operational_kill_at_k`): Kill@1 0.234317 (127),
Kill@2 0.348708 (189), Kill@4 0.483395 (262), **Kill@8 0.605166 (328)**.

**Reproduction check** (`REH-EXEC` › `reproduction_check`):
historical `ablation_dev` base Kill@8 = 0.605166; rehearsal Kill@8 = 0.605166;
`exact_match: true`; `meaning: "pipeline fidelity, not model quality"`.

**Failure taxonomy** (`REH-EXEC` › `failure_taxonomy`): `generation_invalid` 72,
`killed_assertion_error` 761, `killed_error` 304, `killed_timeout` 43,
`reference_assertion_error` 1661, `reference_error` 151, `survived` 1344.

**Denials carried as data** (`REH-EXEC`): `final_test_measurement: false`,
`eligible_for_model_selection: false`, `supports_performance_claim: false`.

---

## F. Consumed final split

Source: **`INCIDENT`**, corroborated by `STATUS`.

| Claim | Value |
|---|---|
| Status | `failed_after_authorization` |
| Sealed candidates generated | 0 |
| Reportable final-test metrics | 0 |
| Result payload | `{}` |
| Receipt / bundle presented | `4a2bc52a…36af7c8e` / `c16b13eb…30d0a9` |
| Re-execution | permanently prohibited |

No split size, membership, identifier or record count for the consumed split is
reproduced in any paper. The failure is described in terms of the *mechanism*
(a record-admission rule applied before scope filtering), not the data.

---

## G. Unavailable — never claimed

| Fact | Status |
|---|---|
| Final-test Kill@k of any arm | **does not exist** (`INCIDENT`) |
| Native real-repository kill rate | **does not exist** — repository records excluded from every reported rate (`DEV-SEL`, `LV-RES`, `REH-EXEC`) |
| Oneiros-versus-Atheris final comparison | **does not exist**; no baseline was bundled or executed in any measurement reported here |
| Evidence that SFT generalizes | **not established**; criterion 1 failed on significance (`LV-RES`) |
| Evidence that SFT does *not* help | **not established**; a non-significant result is not a null result |

---

## H. Marked TODO — unverified, excluded from all drafts

| Item | Why excluded |
|---|---|
| `TODO — UNVERIFIED` Training hyperparameters (LoRA rank, learning rate, epochs, optimizer) | Not present in any committed receipt listed in §A. Papers describe the arms by checkpoint identity only. |
| `TODO — UNVERIFIED` Corpus construction provenance and per-source counts | Not drawn from a receipt in §A; the drafts state only split sizes that appear in §C–§E. |
| `TODO — UNVERIFIED` Hardware/GPU model for the training runs | Only the rehearsal's runtime is receipted; no training hardware claim is made. |
| `TODO — UNVERIFIED` Wall-clock cost of the four development arms | Not in `DEV-SEL`. |
| `TODO — UNVERIFIED` Reproducibility of the base measurement across seeds | Only seed 42 is receipted; no cross-seed variance claim is made. |
| `TODO — UNVERIFIED` Any comparison against an external mutation-testing tool | No such run exists in the repository. |

Each of these must be resolved from a committed artifact, or remain absent,
before any draft is submitted anywhere.

---

## I. Derived values

Every number in any draft that is **not** read verbatim from a receipt. Each row
gives the formula, the source artifacts, the inputs, the unrounded result, the
display rule, and where it appears.

All inputs come from `DEV-SEL` (`0188324fe9eb2a51eebef71bbd4cc3ed45584f12393bf0fa6539817a09e58969`)
and `LV-RES` (`f7cdf84df8e65c51bd22410759c6fd54952ace3bfa80c6ab620674b53e288af9`),
except where stated. Percentage-point differences are computed as
`(a − b) × 100` on the receipted rates, in exact decimal arithmetic.

### D1 — Development Kill@8 gain, A@431 over base

| | |
|---|---|
| Formula | `(kill_at_8[A@431] − kill_at_8[base]) × 100` |
| Source | `DEV-SEL` › `arms.A@431.metrics.kill_at_8`, `arms.base.metrics.kill_at_8` |
| Inputs | `0.695572`, `0.605166` |
| Unrounded | `9.040600` |
| Display | 2 d.p. → **9.04** points |
| Appears in | RENE §I, §Contributions, §RQ1; ICST §I, §RQ1, §Conclusion; SP&P §Intro, §Results; CAIN abstract |

### D2 — Development Kill@8 gain, A@150 over base

| | |
|---|---|
| Formula | `(0.690037 − 0.605166) × 100` |
| Unrounded | `8.487100` |
| Display | 2 d.p. → **+8.49** points |
| Appears in | RENE Table I (“vs base” column) |

### D3 — Development Kill@8 gain, B@150 over base

| | |
|---|---|
| Formula | `(0.658672 − 0.605166) × 100` |
| Unrounded | `5.350600` |
| Display | 2 d.p. → **+5.35** points |
| Appears in | RENE Table I (“vs base” column) |

### D4 — Locked Kill@8 gain

| | |
|---|---|
| Formula | `(0.628798 − 0.594452) × 100` |
| Unrounded | `3.434600` |
| Cross-check | `LV-RES` › `paired_comparison.kill_at_8_delta_points` = `3.4345999999999988` (float representation of the same quantity) |
| Display | 4 d.p. → **+3.4346** points |
| Appears in | every draft, abstract and main results table |

**Note:** D4 is both derived *and* receipted. The receipt's float value and the
exact decimal difference agree to the displayed precision; the decimal form is
used for display.

### D5 — Development-to-locked shrinkage ratio

| | |
|---|---|
| Formula | `D1 ÷ D4` |
| Inputs | `9.0406`, `3.4346` |
| Unrounded | `2.632213358178536074069760671` |
| Display | “roughly **2.6×**” — deliberately hedged, never given to more precision |
| Appears in | RENE §RQ1; ICST §RQ1, Fig. 2 caption; SP&P §Results |

**Caveat carried into the papers:** this is a ratio of two point estimates from
different populations and is reported as a single observation, not an estimate of
a general ratio. See `claims_traceability.md` I3.

### D6 — Discordant pairs in the locked McNemar test

| | |
|---|---|
| Formula | `functions_gained + functions_lost` |
| Source | `LV-RES` › `paired_comparison` |
| Inputs | `114`, `88` |
| Unrounded | `202` |
| Display | exact integer → **202** |
| Appears in | RENE §RQ1 and §Threats; ICST §RQ1 and §Threats; SP&P §Results and §Threats |

### D7 — Expected candidate count for the rehearsal

| | |
|---|---|
| Formula | `function_mode_targets × candidates_per_function` |
| Source | `REH-EXEC` (`06aca918d2821a1f3f161e6f14a91978643f81b30e883746d8126a967136b59f`) |
| Inputs | `542`, `8` |
| Unrounded | `4336` |
| Cross-check | `REH-EXEC` › `candidates.expected` = `4336` and `candidates.observed` = `4336` |
| Display | **4,336**, written in the papers as `542 × 8` |
| Appears in | RENE §RQ4; ICST §RQ4 Table III; SP&P §Results; CAIN §V |

### D8 — Change in functions with any reference-valid candidate (locked)

| | |
|---|---|
| Formula | `fnrefn[A@431] − fnrefn[base]` |
| Source | `LV-RES` › `arms.*.metrics.fnrefn` |
| Inputs | `566`, `675` |
| Unrounded | `−109` |
| Display | exact integer → **−109** |
| Appears in | ICST Table II |

### D9 — Rounded Wilson interval bounds

| | |
|---|---|
| Formula | display rounding only; no arithmetic |
| Source | `DEV-SEL`, `LV-RES` › `*.wilson_95` / `*.kill_at_8_wilson_95` |
| Rule | 6 d.p. in the receipt → **4 d.p.** in tables (e.g. `0.563412` → `0.5634`) |
| Full values | see §C and §D above, and `data/kill_at_8.csv` |
| Appears in | RENE Table I; ICST Table I; SP&P Table I |

### D10 — Rounded McNemar p-value

| | |
|---|---|
| Formula | display rounding only |
| Source | `LV-RES` › `paired_comparison.mcnemar_exact_two_sided_p` |
| Full value | `0.07832001079866063` |
| Display | 4 d.p. → **0.0783** |
| Appears in | every draft, abstract and main results table |

### D11 — Development reference-validity changes (quoted, not recomputed)

`DEV-SEL` › `explicit_findings` states these directly: “−11.05 points for A@431,
−10.70 for A@150, −11.67 for B@150”. Recomputing from the receipted rates gives
`−11.047000`, `−10.701100`, `−11.669700`, which agree with the receipt's own
rounding.

**These figures do not currently appear in any draft.** The drafts state the
development reference-validity *rates* (§C) and the locked *delta* (D12) instead.
Listed here so that if a draft later quotes them, the derivation is on record.

### D12 — Locked validity deltas (receipted, cross-checked)

| Quantity | Receipt value | Recomputed | Display |
|---|---|---|---|
| Reference validity | `−11.8726` | `(0.333388 − 0.452114) × 100 = −11.872600` | **−11.8726** |
| Parse | `0.14859999999999873` | `(0.988771 − 0.987285) × 100 = 0.148600` | **+0.1486** |
| Execution | `1.3869999999999938` | `(0.942206 − 0.928336) × 100 = 1.387000` | **+1.3870** |

All three appear in `LV-RES` › `paired_comparison` and are reproduced by exact
decimal arithmetic on the receipted rates. Appears in: RENE §RQ2 Table II; ICST
§RQ2 Table II; SP&P §Results (prose); CAIN Table I.

---

### Completeness check

`common/validate_papers.py` enforces that every numeric token appearing in the
prose or tables of any draft is either a receipted value from §B–§F or a derived
value from this section. A number that is in neither fails the validator.
