# Next-direction decision memo

Written 2026-09-25. This is a design only: no split has been created, nothing
has been mined, no model has been run, and no protected data has been opened. The
machine-readable version is `results/v4_3_next_direction_design.json`.

## Where the project stands

Two lines are closed by their own frozen gates:

| line | result |
|---|---|
| execution-supervision SFT (12% examples; then a 25%-example / 58%-token composite) | mechanism gates failed; the composite had retention inconclusive |
| execution-feedback repair at inference (C vs sham-control B) | Kill@8 +0.38 pp, 90% CI [−1.07, +1.77]: **FAIL** |

Every number the project has produced since the sealed-final incident is either
train-derived or measured on a previously inspected panel. The only locked test
split was consumed. **The project currently cannot make any held-out claim.**

The one sizeable effect in the last pilot was secondary and unmatched. The
16-sample arm with text-only de-duplicating selection beat the canonical
8-sample protocol by +6.44 pp Kill@8 (B − A, CI [+2.46, +10.59]). It says
nothing about execution feedback, but it is a cheap, separable question.

## Recommendation

1. **Primary: an independently constructed repository-native evaluation.** New
   bugs from repositories that have never touched any Oneiros split, reproduced
   natively on buggy and fixed revisions, and evaluated once after a full dress
   rehearsal on a separate pool. This is the evaluation the research plan asks
   for, and it has never been done for generated tests: 0 generated tests have
   been run natively, and official-test reproduction was 3 of 5.
2. **Secondary (optional): a controlled sampling-budget study.** It asks
   whether 16 samples with text-only selection of 8 beat the first 8, at twice
   the generation cost. It is exploratory and train-derived, about 20 GPU
   minutes, and independent of the primary line.

**Not recommended:**
- any further execution-supervision or execution-feedback variant;
- reusing the consumed test split;
- opening the reserved confirmation lineages for a different question.

## Disjointness is proven, not asserted

`harness/repository_isolation.py` checks every future candidate against the
complete upstream copy of every source that has ever fed an Oneiros split. The
corpus manifest lists exactly MBPP, HumanEval, BugsInPy, SWE-bench Verified and
curated examples, and the checker confirms that coverage in code.

The reference universe:
- **35 excluded repositories:** every BugsInPy project, every SWE-bench Verified repository, and the legacy real-bug repositories (aiohttp, boto3, celery, click, django, flask, numpy, pytest, redis, requests, sqlalchemy);
- **1,484 known commits;**
- **998 known patches;**
- **33,172 reference functions**, from upstream MBPP, HumanEval, BugsInPy, the SWE-bench Verified patches and the train shard.

It reuses the frozen near-duplicate audit: AST-normalised code, exact Jaccard
over 5-token shingles, threshold 0.80.

Its self-checks behave correctly:
- a real BugsInPy patch is rejected, both as the same repository and as an identical patch;
- a renamed MBPP function is rejected as a near-duplicate;
- a fork of an excluded repository is rejected;
- a novel function is admitted.

Because every split is a subset of these upstream sources, disjointness from the
consumed test split follows without ever reading it. This proof method still
needs your acceptance (decision D7).

## Estimates (explicit assumptions in the JSON)

| work | GPU | CPU wall | storage |
|---|---|---|---|
| sampling-budget study (optional) | ~20 min | ~6 min | ~15 MB |
| mining + isolation records | — | ~1 h (+1–2 h network) | 4–8 GB |
| environment build + qualification | — | ~4 h | ~45 GB kept, ~150 GB peak |
| dress rehearsal | ~5 min | ~40 min | ~1 GB |
| one-time evaluation | ~30 min | ~8.5 h (native ~3.75 h + Atheris ~4.7 h) | ~1 GB |

WSL has 32 CPUs, 62 GB RAM and 952 GB free. The GPU is one local RTX 4500 Ada.

## Blockers

| id | blocker |
|---|---|
| B1 | No candidate data exists locally; mining needs network access and a licence review. |
| B2 | WSL lacks Python 3.10 and 3.13 and any environment manager; Docker is absent. |
| B3 | Native execution of generated tests has never been demonstrated; official reproduction was 3 of 5, with 40% infrastructure failure. |
| B4 | The Atheris adapter only drives primitives in fixed ranges; repository functions take objects and state. |
| B5 | Repository prompts are capped at 2,048 tokens, so compaction may drop context. |
| B6 | The consumed test split can never be read, so disjointness from it rests on the source-universe proof. |
| B7 | No untouched train-derived panel remains for the sampling study. |

## Decisions needed from you

| id | decision |
|---|---|
| D1 | Approve network mining and the temporal rule (fixes on or after 2025-01-01). |
| D2 | Approve the licence list, and decide whether same-organisation repositories are excluded. |
| D3 | Set the target size (100 minimum, 150 planned) and the 8% per-repository cap. |
| D4 | Decide who seals the one-time set, and whether it becomes the project's new locked evaluation. |
| D5 | Set the Atheris budget (600 CPU-s × 3 seeds per target) and the jointly-eligible comparison rule. |
| D6 | Permit installing `uv`, plus Python 3.10 and 3.13, in WSL. |
| D7 | Accept the source-universe proof of disjointness from the consumed test split. |
| D8 | Run, defer or skip the sampling-budget study, and choose its panel. |
| D9 | Choose the models for the one-time evaluation (base and frozen control proposed). |

## Proposed execution order

1. Approve this design and decisions D1–D9.
2. Optionally, run the sampling-budget study: about 20 GPU minutes, independent of everything else.
3. Tooling: install `uv` and the interpreters in WSL, extend the Atheris adapter, and test on synthetic targets.
4. Mining and licence screening, with a per-candidate isolation record. Split the result into a rehearsal pool and a final pool; no model ever sees the final pool before the one-time run.
5. Environment builds and official-test qualification for both pools.
6. A full dress rehearsal on the rehearsal pool, including a crash-and-resume test.
7. Freeze the final set by hash, freeze the protocol, analysis and budgets, and request one-time authorization.
8. Run the one-time evaluation, only after explicit authorization.

Detailed drafts: [REPOSITORY_NATIVE_EVALUATION_PROTOCOL.md](REPOSITORY_NATIVE_EVALUATION_PROTOCOL.md)
and [SAMPLING_BUDGET_STUDY_PROTOCOL.md](SAMPLING_BUDGET_STUDY_PROTOCOL.md).
