# Repository-native evaluation — protocol draft (revised)

**Status: draft.** Nothing has been mined or installed, no split exists and no
model has run. Everything here is frozen and hash-bound before any model sees
the final set. Structured content lives in
`results/v4_3_next_direction_design.json` (`repository_native_protocol`).

## Claim scope

> Disjointness is enforced against the complete indexed source universe under
> the recorded repository, fork, commit, patch, issue, and function-similarity
> checks.

That is the whole claim. It does not state that no model has seen a repository.

## 1. Candidate sources

We mine bug-fix commits from permissively licensed, pure-Python, pytest-tested
GitHub repositories outside the indexed universe. The fix must have been merged
**on or after 2025-01-01**. That cutoff is **contamination-risk mitigation
only**: the buggy code, the repository or a similar fix may still have been seen
in pretraining.

**Not used wholesale:**
- BugsInPy and SWE-bench / SWE-bench Verified, whose repositories are all indexed;
- HumanEvalFix, HumanEvalPack and EvalPlus, which derive from HumanEval and MBPP.

**Screened repository by repository, as a candidate index only:**
- aggregates such as Defectors or PyTraceBugs;
- SWE-rebench / SWE-Gym style collections.

**Screening candidates (names only, nothing verified yet):**
- more-itertools, toolz, boltons, attrs, cattrs, marshmallow, python-dateutil;
- arrow, pendulum, humanize, sortedcontainers, networkx, sqlglot, pyparsing;
- tomlkit, rich, typer, cachetools, glom, parse, inflect, babel, isort;
- jsonschema, tabulate, textdistance, dateparser, schedule, python-slugify, validators.

Repositories in the same organisation as an excluded one are flagged for
decision D2.

## 2. Isolation (fail-closed, receipt-bound)

`harness/repository_isolation.check_candidate(candidate, universe, receipt_sha256)`
writes a record for every candidate. Admission requires `admissible=true`.

**Mandatory evidence.** Anything missing, unknown or malformed is refused as
`insufficient_isolation_evidence`:
- canonical lower-case owner/name;
- a verified `https://github.com/<owner>/<name>` URL and numeric repository identity;
- fork status of `not_fork`, or `fork` with a verified parent owner/name and numeric parent identity (unknown is refused);
- full 40-hex buggy and fixed commit SHAs, which must differ;
- a non-empty normalised patch;
- a parseable target function with at least 12 comparison shingles;
- an issue or PR identity `owner/name#n` in the same repository;
- an SPDX licence from the admitted list, plus the licence-file SHA-256;
- the target file path and module.

**Disjointness checks:**

| check | rule |
|---|---|
| repository | owner/name, bare name and verified fork parent are all outside the universe |
| issue | not a known benchmark instance |
| patch lineage | commits are not known commits; the normalised patch is not identical to any indexed patch, and its Jaccard with each is below 0.80 |
| function lineage | Jaccard against every indexed reference function is below 0.80; the nearest match, score and fingerprint are recorded |
| within the new set | one target per fix commit and per function lineage; a pairwise near-duplicate check |

**Coverage.** `audit_source_coverage` requires concrete indexed counts and bound
input files for every corpus source, and for the train view and the legacy
real-bug files. An unknown, unindexed or zero-count source fails, as does a
missing input.

**Receipt.** `results/v4_3_reference_universe_receipt.json` hash-binds the
entire universe:
- repository names;
- commits;
- patch hashes;
- instance IDs;
- function fingerprints;
- all 1,559 input files;
- a SHA-256 for each collection;
- the source hashes of the builder, checker and normaliser.

Every isolation record carries the receipt hash
(`isolation_record_is_current`). If the universe changes, old decisions become
invalid and must be recomputed.

## 3. Licensing and reproducibility

- **Licences:** MIT, BSD-2-Clause, BSD-3-Clause, Apache-2.0, ISC or PSF-2.0. The SPDX identifier comes from the licence file at the buggy commit, and the file is hashed.
- **Checkouts:** full clones only. The blob-less ingestion cache cannot check out files. Buggy and fixed SHAs and tree hashes are recorded.
- **Python:** the versions available now are 3.8, 3.9, 3.11 and 3.12; 3.10 and 3.13 need approval (D6). Each target uses the lowest version its project supports.
- **One shared, hash-locked environment per target:**
  - **Lock:** dependencies are resolved once, from the declared requirements of *both* revisions, into a lock with hashes (`uv pip compile --generate-hashes`, or `pip-compile --generate-hashes`). The lock is installed with hash checking (`uv pip sync`, or `pip install --require-hashes`). `pip freeze` is not a lock: it records versions without hashes.
  - **Project code:** the project itself is not installed. Each revision's checkout is put on the path, and rebuilt in place if it has compiled parts, when its tests run.
  - **No shared lock:** if no single lock satisfies both revisions, or either revision cannot import or build against it, the target is **disqualified** before evaluation. Per-revision environments are used only under a separately predeclared and approved rule.
- **Network:** allowed only while building an environment, never while tests run.
- **Evidence recorded:** interpreter, lock file and its hash, resolver and its version, install log hash, WSL distribution, kernel and CPU count.

## 4. Native-execution qualification

1. Build the shared environment and check out both revisions.
2. Run the official regression test three times on each revision. An independently written and reviewed test may substitute, and is labelled as such.
3. **Qualify** a bug only if the test fails 3 of 3 times on the buggy revision and passes 3 of 3 on the fixed revision.
4. Inject generated tests into both revisions under an isolated file name, with per-test timeouts. Any discordant result is re-run once as a flakiness check.

**Outcome classes:**

| class | meaning | counted as |
|---|---|---|
| semantic kill | assertion fails on buggy; passes on fixed | kill |
| crash kill | project-code exception on buggy; passes on fixed | kill (reported separately) |
| timeout on buggy | times out on buggy; passes on fixed | reported separately |
| passes both / fails both / inverted | no discrimination / reference-invalid / pathological | not a kill |
| harness failure | collection, injection or pytest internal error | **excluded; never a kill or a model failure** |
| dependency failure | a package is missing or incompatible | excluded |
| environment failure | interpreter, build or filesystem failure | excluded; the target is disqualified beforehand |

## 5. Target and context policy

**Permitted in the prompt:**
- the buggy-side target function;
- its signature, type hints and docstring;
- the buggy-side imports, constants and class skeletons needed to call it;
- public docstrings present at the buggy commit.

**Forbidden in the prompt:**
- the fixed code, and the patch or diff;
- hidden, official or gold tests;
- expected outputs;
- issue or PR discussion of the fix;
- the fix commit message.

**Enforcement.** Prompts are built from a permitted view, and every prompt is
scanned for lines of the fixed function, the patch or the official test.

**Recorded per target:**
- complexity tier and bug family;
- LOC, nesting, outgoing calls, and parameter count and kinds;
- statefulness;
- Python version and repository.

## 6. Composition

- **Size: N = 300 recommended, 200 minimum**, from the power analysis (§9). The +5 pp gate is not changed to fit the sample.
- **Uniqueness:** unique bugs, function lineages and fix commits.
- **Caps:**
  - no repository above 8%;
  - no bug family above 25%;
  - no candidate index above 50%.
- **Tiers:** simple, moderate and complex, with at least 20% each where supply allows. A shortfall is reported and never filled by repetition. All exclusions are reported with their reasons.
- **Rehearsal pool:** 20–30 qualified bugs from at least 5 repositories that are never admitted to the final set.

## 7. Oneiros versus Atheris

**Tool.** Actual `atheris` 2.3.0 (libFuzzer) on WSL Python 3.11. It is never the
simulated `baseline.coverage_fuzzer`.

**Two separately labelled modes:**
- **Ordinary crash/contract Atheris** (the primary comparison) fuzzes the buggy function alone. A finding is an uncaught exception or a violated declared contract. It counts as a kill only if the same input, replayed on the fixed revision, does not fail.
- **Differential-oracle Atheris** runs the fixed and buggy revisions side by side. It is a generous **upper bound**, reported separately, because Oneiros never receives the fixed implementation.

**Applicability.**
- **Eligible targets:** a target is Atheris-applicable if the adapter can build well-typed arguments that reach it.
- **Reporting:** applicability and ineligibility counts, with reasons, are reported separately.
- **No silent zeros:** an ineligible target is **never** recorded as a tested zero.

**Primary paired comparison.** Oneiros against ordinary Atheris, on the
**jointly eligible** set only.

**Adapter.**
- type-hint and docstring driven;
- full-range numbers and bytes;
- nested containers;
- dataclass and NamedTuple construction;
- literal seeds taken from the buggy function;
- reachability reported per target.

**Budgets.** Atheris gets 600 CPU-seconds per target, per mode, per seed, with
seeds 42, 43 and 44. Oneiros gets 8 candidates, reported as model calls, tokens,
GPU seconds and CPU execution seconds. The budgets differ in kind, and no
"equal budget" claim is made. Crash kills and semantic kills are reported
separately.

## 8. Dress rehearsal (permitted data only)

The rehearsal runs every stage on the rehearsal pool:
1. isolation records bound to the receipt hash;
2. environment locks;
3. qualification;
4. prompt construction and leakage scan;
5. generation with the base model and the frozen control;
6. native execution on both revisions;
7. both Atheris modes;
8. outcome classes and receipts;
9. the frozen analysis;
10. a deliberate crash-and-resume.

**Gate:** every stage has receipts, there are no unexplained failures, and at
least 90% of qualified targets execute on both revisions.

## 9. Metrics, power and decision rules

**Primary:** Kill@1, Kill@4 and Kill@8 (a native semantic or crash kill per
target), and unique defects killed.

**Secondary:**
- semantic and crash kills reported separately;
- reference validity per requested candidate;
- parse and execution success;
- slices by complexity tier, bug family and repository;
- **repository-stratified performance and repository-cluster sensitivity**, meaning the paired estimate recomputed with each repository dropped. This is not leave-one-repository-out generalisation, which would require retraining;
- ordinary and differential Atheris kills, applicability counts, and kills per CPU-second.

**Intervals:** Wilson 95% for single proportions. Paired comparisons use exact
McNemar and a repository-cluster bootstrap 90% interval (10,000 replicates,
fixed seed).

**Decision rule (frozen):**
- **Pass:** a comparison passes only with a Kill@8 gain of at least +5 pp, a lower bound above 0, and reference validity within −3 pp.
- **Otherwise:** everything else is descriptive.
- **No selection on the final set:** no configuration is chosen using it.

**Prospective power** (`results/v4_3_repository_native_power_analysis.json`):
- **Evidence:** permitted train-derived data only (discordance 0.17–0.21; lineage ICC about 0).
- **Clustering:** repository ρ is swept from 0 to 0.20.
- **80%-power MDE** (discordance 0.21, ρ 0.05, 25 repositories): 11.8 pp at N=100, 10.2 at 150, 9.2 at 200 and 8.1 at 300.
- **Lower bound binds:** the lower-bound requirement binds above 5 pp at every N (about 8.1 pp at N=100 and 5.4 pp at N=300).
- **Mining pool:** 2,000–4,300 candidates are needed for N=300.

## 10. Receipts and rollback

**Per target:**
- repository, licence and the licence-file hash;
- commit and tree hashes;
- the isolation record, including the receipt hash;
- the hash-locked environment;
- the qualification runs;
- the prompt hash and leakage scan.

**Per run:**
- LF-canonical source hashes;
- model, revision and adapter hash;
- seeds and settings;
- raw outputs with their hashes;
- logs, return codes, timeouts and durations for both revisions;
- Atheris logs, corpora, coverage and seeds per mode;
- the WSL environment details.

**Rollback:**
- nothing is overwritten;
- failed runs are preserved and marked invalid;
- the one-time set is sealed by hash before any model sees it, and is never re-evaluated after a protocol change.

## Estimates (N = 300)

| stage | GPU | CPU wall | storage |
|---|---|---|---|
| mining + isolation | — | ~2 h (+2–4 h network) | 8–16 GB |
| environment build | — | ~4 h | ~90 GB kept, ~270 GB peak |
| qualification | — | ~4 h | — |
| dress rehearsal | ~5 min | ~1 h | ~1 GB |
| one-time evaluation | ~1 h | ~26 h | ~2 GB |

The assumptions:
- a 7–15% yield from mined candidates to targets;
- 45 s per native test run;
- 4 minutes per environment build;
- 16 parallel jobs;
- 2 models × 8 candidates;
- 2 Atheris modes × 3 seeds × 600 CPU-s.
