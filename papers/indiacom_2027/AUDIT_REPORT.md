# Oneiros Engine (INDIACom-2027 draft): correction and submission-readiness audit

Date: 2026-10-13.

Sources:
- `main.tex` is the shared body.
- `paper_named.tex` builds the named version and `paper_anonymous.tex` builds the anonymous version.
- Supporting files: `references.bib`, `figures/oneiros_workflow.tex`.

Preserved inputs: `archive/draft_v2_2026-10-13/` holds the previous sources and PDFs; `archive/draft_v1_2026-10-13/` holds the first draft.

Scope:
- No project code, dataset, experiment artifact, receipt, model output or Git history was modified.
- Nothing was committed or pushed.

## 1. Build and QA results

| Check | paper_named.pdf | paper_anonymous.pdf |
|---|---|---|
| Page size | 612 × 792 pt (US Letter) | 612 × 792 pt (US Letter) |
| Pages (references included) | 6 | 6 |
| LaTeX errors / undefined refs or citations | 0 / 0 | 0 / 0 |
| Overfull boxes | 0 | 1 vertical box, 1.17 pt, raised during column balancing on the last page; not visible at render |
| Fonts | All embedded; Nimbus Roman (Times-compatible), Computer Modern for maths | same |
| Placeholders (AUTHOR INPUT, TO VERIFY, TODO, FIXME, PLACEHOLDER, [EMAIL], [USN], GUIDE NAME, textcolor, color{red}) | none in source or PDF | none |
| PDF Author metadata | "Talanki Venkat Tarun, Arnav Singh, Saaj Shah, Anushka Singhal" | empty; Subject and Keywords also empty |
| Identity terms in PDF text (names, PES University, Bengaluru, Priyanka, GitHub owner, repository name) | present as intended | none |
| Citations | 24 cited = 24 in bibliography | same |
| Abstract length | 186 words | 186 words |
| Visual inspection | every page rendered and checked | every page rendered and checked |

## 2. Change log

1. **AI disclosure.** The verbose paragraph is replaced by one sentence naming the exact material.
   - The claims "subsequently revised" and "independently verified" were **not** included, because those author checks have not been completed (see §6).
   - The disclosure appears in both versions. The guide sentence ("The authors thank Dr. Priyanka H for guidance and feedback.") appears only in the named version.
2. **Page format.**
   - Now `\documentclass[10pt,conference,letterpaper]{IEEEtran}` plus geometry: Letter, 1 in top/bottom, 0.6 in left/right, per the Technical Guidelines.
   - Title set at 14 pt and author names at 12 pt bold, per the Technical Guidelines.
   - No page numbers, no custom header or footer, and no INDIACom header, ID, copyright or DOI added.
3. **Abstract.**
   - The model is named Qwen2.5-Coder-1.5B-Instruct and the p-value is given as 0.0783.
   - "Improved assertion form" is replaced by the requested wording: input selection improved; outputs shifted toward exact-equality assertions; improved expected-value prediction was not demonstrated.
   - "QLoRA" is spelled out in the abstract, because the guidelines exclude nonstandard abbreviations there.
4. **Index terms** are alphabetised as specified.
5. **Nomenclature.** A compact unnumbered section is added, as the Technical Guidelines require. Every abbreviation is also expanded at first use in the text: LLM, RQ, QLoRA, LoRA, MBPP, NF4, SHA-256, AST, BF16, CI. "machine-learning" is spelled out, and the "repos", "cx." and "mod." abbreviations were removed from Table I and Fig. 1.
6. **Title scope.** Two sentences in §I define "self-learning" (offline adaptation, not online updating) and "realistic" (dual execution plus the repository-native design, not established repository-scale effectiveness).
7. **Table I.**
   - The sealed test now reads 810 / 94; the four splits sum to 8,237.
   - The training selection is separated into 2,400 selected pairs (2,194 synthetic, 206 repository), 5,735 unique examples and 6,889 weighted exposures, explicitly "repetition, not new data".
8. **Confirmed error corrected: 6,852 → 6,889 effective examples.**
   - The old figure came from `results/v4_2_armA_clean_preflight.json`. That receipt describes an earlier run configuration: 128-token completions, first-assertion parsing and 429 planned steps.
   - The A@431 adapter was produced by the successor configuration. Its committed receipt, `results/v4_2_armA_successor_preflight_c1024.json` (`abf9934e…`), records 6,889 effective and 5,735 unique examples (5,511 synthetic plus 1,378 repository exposures) and 431 steps.
   - The checkpoint's own `sft_run_config.json` and `sft_metadata.json` agree. The text now also says A@431 is the final step (431 of 431 completed).
9. **RQ4.**
   - The full funnel is now reported (see §4) and the 114/113 difference is reconciled explicitly.
   - The text now says the five exclusions are environment exclusions, not model failures.
   - The power statement is corrected (see §4).
10. **Statistical wording.**
    - "Fine-tuning helped the first and did not help the second" is replaced by "improved discriminating-input selection but did not demonstrate an improvement in expected-value prediction". RQ3, the Discussion and the Conclusion now use matching cautious wording.
    - RQ1 now says a non-significant result is not evidence of zero effect.
    - The locked-split sentence is replaced with the requested "cannot provide a second independent confirmation …" wording.
11. **Criterion-5 defect.** The text now states:
    - a late implementation change treated the arm-specific adapter hash as a cross-arm invariant;
    - this was a harness/integrity-gate defect, and the adapter itself was not defective;
    - it did not turn the result positive; the base model was already retained because the significance gate failed.
12. **Reproducibility.** The paper now gives:
    - the full revision SHA;
    - seed 42;
    - AdamW optimizer, learning rate 1e-5, constant schedule after 25 warm-up steps, gradient clipping at 1.0;
    - batch 1 × 16 accumulation steps = 16 examples per step, 1 epoch / 431 steps;
    - NF4 quantization without double quantization, BF16 compute;
    - LoRA r16, α32, dropout 0.05 on q/k/v/o projections;
    - token budgets 1,024 / 1,024 / 3,072;
    - evaluation seed and decoding settings;
    - Python 3.12.14, PyTorch 2.5.1, Transformers 4.48.3, PEFT 0.14.0.
13. **Pretraining-contamination threat** added. It states that the leakage audit covers the project's own splits, not pretraining data.
14. **References.**
    - Atheris now uses the BibTeX `url` field, so the URL is typeset by `\url` and no longer breaks after "https:".
    - The EvalPlus web reference gains an access date, as the guidelines require for web references.
    - All 24 entries were checked earlier in this session (Crossref for DOI entries; the arXiv API for preprints). EvalPlus was copied from the project's verified list, and its URL was re-checked today (HTTP 200).
15. **Final-page balance.** The `balance` package (`\balance` before the bibliography) balances the last-page columns. No manual spacing was used.
16. **Figure 1.** Abbreviations were removed. The classification box was shortened so that no text overflows, and the receipts band was moved clear of the boxes.

## 3. AI-generated material named in the final disclosure

From the drafting history of this session, Anthropic Claude (Claude Code) drafted all of the following from the repository's committed experiment records:
- the Abstract;
- the Nomenclature;
- Sections I–VIII;
- Tables I–IV;
- Fig. 1 (TikZ);
- the reference list.

This includes the two paragraphs written from author-supplied meaning: the motivation in §I and the implementation experience in §VI.

The authors have not yet rewritten any section, so the disclosure covers everything. Final sentence:

> Anthropic Claude (Claude Code) assisted in preparing the initial drafts of the Abstract, Nomenclature, Sections I–VIII, Tables I–IV, Fig. 1 and the reference list from author-supplied experiment records.

**Outstanding author action:** the authors must revise the text and independently check every result and citation before the words "subsequently revised" and "independently verified" may be added. Paraphrasing by the assistant does not remove the need for disclosure.

## 4. Claim-by-claim evidence check (new or changed claims)

| Claim in paper | Value | Source (SHA-256 prefix) | Field | Status |
|---|---|---|---|---|
| Sealed test split | 810 records / 94 groups | `data/corpus/v4_1_research_hardened_candidate/manifest.json` (`7f68386f`) | `splits.test` | verified; split sum 6,052+594+781+810 = 8,237 = `record_count` |
| Selected pairs | 2,400 = 2,194 synthetic + 206 repository | `results/v4_2_armA_successor_preflight_c1024.json` (`abf9934e`); checkpoint `sft_metadata.json` | `selection.*`, `bounded_selection_stats.*` | verified |
| Unique / weighted examples | 5,735 / 6,889; repository share 0.200029 | same | `sampling.example_weights.effective_unique_examples`, `sampling.effective_total_examples`, `actual_real_fraction` | **corrected from 6,852** |
| Steps | 431 planned, 431 completed | same; `sft_metadata.json` | `training.optimizer_schedule.planned_optimizer_steps`; `completed_optimizer_steps` | verified |
| Seed, optimizer, clipping, accumulation | 42; adamw_torch; 1.0; 16 × batch 1 | successor preflight | `run_identity.*` | verified |
| Learning rate / schedule / warm-up / epochs | 1e-5; constant_with_warmup; 25; 1 | successor preflight | `training.*` | verified |
| LoRA, quantization | r16, α32, dropout 0.05, q/k/v/o; NF4, no double quantization | successor preflight; adapter_config (`4dda431a`) | `run_identity.lora`, `run_identity.quantization` | verified |
| BF16 compute | bf16 | checkpoint `sft_metadata.json` (local, not tracked in Git) | `compute_dtype` | verified (local file) |
| Revision SHA | 2e1fd397ee46e1388853d2af2c993145b0f1098a | successor preflight | `tokenization.model_revision` | verified. Note: `harness/research_configuration.py` carries a different, stale string, `2e1fd39751a5…`; it was not used. |
| Token budgets | prompt 1,024; completion 1,024; sequence 3,072 | successor preflight | `tokenization.*` | verified |
| Decoding | 8 candidates, T 0.7, top-p 0.9, seed 42, whole-output parsing, raw outputs retained | successor preflight `future_generation_contract`; `results/v4_2_development_selection_receipt.json` (`0188324f`) | `protocol` | verified |
| Software versions | Python 3.12.14, torch 2.5.1+cu124, transformers 4.48.3, peft 0.14.0 | checkpoint `sft_run_config.json` (local, not tracked in Git) | `reproducibility.*` | verified (local file) |
| Funnel: 70 pool → 52 scanned, 17 screen-excluded, 1 unavailable; 804 evaluated | — | `results/sft_root_cause_v27_confirmation_phase4_gate_r4.json` (`931a03ff`) | `funnel` | verified (52 + 17 + 1 = 70) |
| 114 admissions vs 113 unique functions from 31 repositories | — | Phase 4 gate `funnel.admitted = 114`; final panel `results/sft_root_cause_v27_confirmation_panel_r4.json` (`432965c0`): `acquisition_counts.within_set_duplicates_removed = 1`, `unique_functions_admitted = 113`, `unique_repositories_admitted = 31`; acquisition report (`2cb23dc5`) agrees on 113 and 31 | — | reconciled: 114 − 1 duplicate = 113 |
| 85 → 51 → 46 → 45 → 36 → 34 | — | final panel `funnel` | `with_regression_test_and_no_overlap`, `natively_qualified`, `requalified_3_of_3`, `target_reached`, `prompt_leakage_and_token_fit_passed`, `selected` | verified |
| 34 targets / 15 repositories; 29 / 14; 5 policy exclusions | — | `results/sft_root_cause_v27_common_subset_r4_v3.json` (`a803df3c`) | `targets`, `composition.native_executable` | verified |
| Power: 26.3 points at α = 0.05; none up to 26.7 at α = 0.01; discordance 0.267 | — | subset v3 `statistics.native_executable` (`mde_pp_80pct_power_alpha_0.05` = 26.3, `…_alpha_0.01` = null, `max_effect_pp_at_assumed_discordance` = 26.7); `statistics.historical_discordance` = 0.26684; `scripts/v27_common_subset_freeze_v2.py::mde` returns null when no effect up to the discordance bound reaches 80% power | — | verified |
| p-value precision | 0.0783 | `results/v4_2_locked_validation_result_receipt.json` (`f7cdf84d`) | `paired_comparison.mcnemar_exact_two_sided_p` = 0.07832… | verified |
| Criterion-5 mechanism | adapter hash placed in the run contract | locked-validation receipt | `promotion_criteria.5_integrity.why_it_failed` | verified |
| MBPP expansion | Mostly Basic Programming Problems | arXiv 2108.07732 abstract | — | verified |
| Unchanged results (Tables II–IV, RQ1–RQ3, abstract values) | unchanged | as in the earlier ledger | — | regression diff: no result value changed |

## 5. Template and conference conflicts (reported, not resolved by invention)

1. **File format.** The Technical Guidelines require MS Word (.doc/.docx). These PDFs come from LaTeX, so a Word version must be produced and checked against the official Word template before upload.
2. **Paper size.** The Technical Guidelines specify Letter (implemented here). The Registration page says "6 A4 size pages", and the downloadable template PDF is Letter.
3. **Over-length pages.** The Technical Guidelines allow 1 extra page; the Registration page says up to 2. This paper is 6 pages, so neither applies.
4. **Title size.** The Technical Guidelines say 14 pt (implemented); the IEEE-style Word template's "paper title" style is larger.
5. **Affiliations.**
   - The guidelines require superscript-letter affiliation markers, full postal addresses, and emails for every author.
   - They also say the first-page footer carries affiliations and emails "as the template shows", which conflicts with an affiliation block under the names.
   - None of this can be completed without the information in §6.
6. **Index-terms format.** The guidelines show "(Index Terms - …)". IEEEtran prints "Index Terms—…".
7. **Subsections.** The guidelines describe lettered, italic subsection heads. The paper uses bold run-in paragraph labels rather than formal subsections; confirm this is acceptable.
8. **Stale pages.** The guidelines page text still refers to "INDIACom - 2016", and the template header reads INDIACom-2026.
9. **Anonymisation.** Review is "double blind", but no anonymisation instructions are published. Confirm the upload procedure with Track 4 before submitting either PDF.

## 6. Unresolved author metadata (needed before the named version is final)

1. Verified email address for each of the four authors.
2. Verified department or programme for each author. Not inferred from USNs; the USN codes differ.
3. Full PES University campus postal address.
4. The one corresponding author, with email, full postal address and phone number including country and area code. The INDIACom guidelines require this.
5. Whether Dr. Priyanka H meets the IEEE authorship criteria and approves coauthorship. Until then she appears only in the Acknowledgment.
6. Confirmation that the authors have revised the text and independently verified all results and citations, which is needed to extend the AI-disclosure sentence.
7. INDIACom's confirmation of the AI-disclosure expectation. No INDIACom policy is published; IEEE requires disclosure in the acknowledgments.

## 7. Anonymity risks to review (not changed silently)

- The repository remote is `github.com/vTarun3330/Oneiros-Engine`. If that repository is public, searching the title "Oneiros Engine" may identify the authors. The paper contains no repository URL.
- The anonymous PDF keeps the AI-disclosure acknowledgment, which identifies no one.

## 8. Other files in the folder

- `main.pdf` and `main.docx` are superseded first-draft outputs; copies are archived in `archive/draft_v1_2026-10-13/`.
- Two stray text files from an earlier scan remain at `C:\Program Files\Git\tmp_paper_named.txt` and `C:\Program Files\Git\tmp_paper_anonymous.txt`. A safety check prevented removing them, so please delete them manually.
