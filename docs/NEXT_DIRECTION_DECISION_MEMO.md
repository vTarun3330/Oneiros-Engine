# Next-direction decision memo

Written 2026-09-25 and revised the same day, after the isolation design was sent
back for repair. This is a design only: no split exists, nothing has been mined,
no tool has been installed, no model has run, and no protected data has been
opened.

The machine-readable versions are:
- `results/v4_3_next_direction_design.json`;
- `results/v4_3_reference_universe_receipt.json`;
- `results/v4_3_repository_native_power_analysis.json`.

## Where the project stands

Two lines are closed by their own frozen gates:

| line | result |
|---|---|
| execution-supervision SFT (12% examples; then a 25%-example / 58%-token composite) | mechanism gates failed; the composite had retention inconclusive |
| execution-feedback repair at inference (C vs sham-control B) | Kill@8 +0.38 pp, 90% CI [−1.07, +1.77]: **FAIL** |

Every number the project has produced since the sealed-final incident is either
train-derived or measured on a previously inspected panel. The only locked test
split was consumed. **The project currently cannot make a held-out claim.**

## Recommendation

1. **Primary: an independently constructed repository-native evaluation.** New
   bugs from repositories outside every indexed source, reproduced natively on
   buggy and fixed revisions. A dress rehearsal on a separate pool runs before
   any one-time evaluation.
2. **Secondary (optional): a sampling-budget study.** It asks whether 16
   samples with frozen text-only selection of 8 beat the nested first 8.
   Exploratory, about 20 GPU minutes.

## The isolation claim, stated narrowly

> Disjointness is enforced against the complete indexed source universe under
> the recorded repository, fork, commit, patch, issue, and function-similarity
> checks.

This does **not** claim that no model has seen a repository. The 2025 cutoff for
fix commits is contamination-risk mitigation, not proof.

### What the repaired checker does (`harness/repository_isolation.py`)

**It fails closed.** A candidate is admitted only with complete, structurally
valid evidence:
- canonical owner/name;
- a verified URL and numeric repository identity;
- explicit fork status, with a verified parent if it is a fork;
- full 40-hex buggy and fixed commit SHAs;
- a non-empty normalised patch;
- a parseable target function with at least 12 comparison shingles;
- an issue or PR identity in the same repository;
- licence evidence and the licence-file hash;
- the target file and module.

Anything missing, unknown or malformed is refused as
`insufficient_isolation_evidence`. There are 27 refusal cases under test.

**Evidence-based coverage.** For every corpus source, and for the train view and
the legacy real-bug files, the audit requires concrete indexed counts and
hash-bound input files:

| source | indexed |
|---|---|
| BugsInPy | 17 projects, 501 bugs (every `project.info`, `bug.info` and `bug_patch.txt` bound: 1,019 files) |
| SWE-bench Verified | 500 instances |
| MBPP | 974 tasks |
| HumanEval | 164 tasks |
| legacy real bugs | 493 metadata entries, 532 Python files (533 files bound) |
| train shard | 6,052 records (the view manifest and shard file bound) |
| curated seeds | all 10 `CURATED_BUGSINPY_BUGS` definitions, reconstructed from code |

The curated definition is the only generator of curated records, so every
curated seed in any split is covered: the 7 in train, and the corpus's eighth,
which must be one of `black_1`, `cookiecutter_1` or `scrapy_1`. No non-train
record was opened.

Tests show that each of the following fails:
- a declared but unindexed source;
- an unknown source;
- a zero-count source;
- the removal of any required input.

**A hash-bound universe.** `results/v4_3_reference_universe_receipt.json`
(`receipt_sha256` `2e7883a7…`) records:
- 28 full and 35 bare repository names;
- 1,484 commits;
- 998 normalised patch hashes;
- 1,001 instance IDs;
- 33,192 function fingerprints;
- the path and SHA-256 of all 1,559 input files;
- a SHA-256 for each collection;
- the source hashes of the builder, checker and normaliser.

The receipt is byte-identical across rebuilds. I removed one non-deterministic
input to make it so: re-running the historical curated execution filter. Every
isolation record carries the receipt hash, and a record made against any other
universe is invalid.

## Power (prospective, from permitted evidence only)

Measured on the retention and tool-assisted panels, the paired Kill@8
discordance between independently sampled arms is 0.17–0.21. The lineage
clustering of paired differences is about 0. Repository clustering in new data
is unknown, so it is swept up to ρ = 0.20.

Under the **frozen, unchanged +5 pp gate**, which requires the estimate to be at
least 5 pp *and* the 90% lower bound to be above 0:

| N | power at true +8 pp | power at +10 pp | 80%-power MDE (d 0.21, ρ 0.05, 25 repos) |
|---|---|---|---|
| 100 | 0.50 | 0.67 | 11.8 pp |
| 150 | 0.62 | 0.79 | 10.2 pp |
| 200 | 0.70 | 0.86 | 9.2 pp |
| 300 | 0.79 | 0.93 | 8.1 pp |

- **Recommendation:** N = 300, with a minimum of 200.
- **Mining pool needed:** roughly 2,000–4,300 candidate fix commits, at a 7–15% yield.
- **Cross-check:** a Monte Carlo simulation matches the analytic power within 0.02.

## Estimates (N = 300)

| work | GPU | CPU wall | storage |
|---|---|---|---|
| sampling-budget study (optional) | ~20 min | ~6 min | ~15 MB |
| mining + isolation records | — | ~2 h (+2–4 h network) | 8–16 GB |
| environment build + qualification | — | ~8 h | ~90 GB kept, ~290 GB peak |
| dress rehearsal | ~5 min | ~1 h | ~1 GB |
| one-time evaluation | ~1 h | ~26 h (native ~7.5 h + Atheris ~19 h) | ~2 GB |

WSL has 32 CPUs, 62 GB RAM and 952 GB free.

## Blockers

| id | blocker |
|---|---|
| B1 | No candidate data exists; mining needs network access and a licence review. |
| B2 | WSL lacks Python 3.10 and 3.13 and any hash-locking tool (`uv` or pip-tools). |
| B3 | Native execution of generated tests has never been demonstrated. |
| B4 | The Atheris adapter covers primitives only. |
| B5 | Repository prompts are capped at 2,048 tokens. |
| B6 | The consumed test split is covered only through enforced upstream indexing. |
| B7 | No untouched train panel exists for the sampling study. |
| B8 | N = 300 needs a large mining pool and about 35–40 CPU-hours. |

## Decisions needed

| id | decision |
|---|---|
| D1 | Approve mining and the 2025 cutoff (as mitigation). |
| D2 | Approve the licence list and the same-organisation rule. |
| D3 | Set N (300 recommended, 200 minimum); the gate is unchanged. |
| D4 | Decide who seals the final set and its status. |
| D5 | Set the Atheris modes and budgets. |
| D6 | Permit installing `uv` and Python 3.10/3.13 in WSL. |
| D7 | Accept enforcement against the complete indexed universe. |
| D8 | Run, defer or skip the sampling study. |
| D9 | Choose the models for the one-time evaluation. |

## Can D7 now be accepted?

**Recommended: yes, as the narrowed claim.**

- **What is now true:**
  - the checker fails closed;
  - coverage is evidence-based, with concrete counts and bound inputs;
  - the universe is hash-bound, deterministic, and verified from disk in tests;
  - every decision is tied to the exact universe it was made against.
- **What remains limited:** the claim covers enforcement against the indexed
  universe only. It cannot rule out that the base model saw a candidate
  repository during pretraining.

## Proposed execution order

1. Approve the design and decisions D1–D9.
2. Optionally, run the sampling study.
3. Install tooling (`uv`, interpreters); extend the Atheris adapter.
4. Mine candidates, split them into a rehearsal pool and a final pool, and write isolation records bound to the receipt hash.
5. Build hash-locked shared environments and run official-test qualification.
6. Run the dress rehearsal, including crash-and-resume.
7. Freeze by hash and request one-time authorization.
8. Run the one-time evaluation, only after explicit authorization.

Detailed drafts: [REPOSITORY_NATIVE_EVALUATION_PROTOCOL.md](REPOSITORY_NATIVE_EVALUATION_PROTOCOL.md)
and [SAMPLING_BUDGET_STUDY_PROTOCOL.md](SAMPLING_BUDGET_STUDY_PROTOCOL.md).
