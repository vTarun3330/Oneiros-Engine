# Repository-native evaluation — protocol draft (revised)

**Status: draft.** Nothing has been mined or installed, no split exists and no
model has run. Everything here is frozen and hash-bound before any model sees
the final set. Structured content is in the design artifact
(`next_direction_design.json`, under `repository_native_protocol`). That
artifact lives in the current generation of `results/next_direction_bundle`.

## Claim scope

> Disjointness is enforced against the complete indexed source universe under
> the recorded repository, fork, commit, patch, issue, and function-similarity
> checks.

That is the whole claim. It does not state that no model has seen a repository.

## 1. Candidate sources

We mine bug-fix commits from permissively licensed, pure-Python, pytest-tested
GitHub repositories outside the indexed universe.

**Temporal rule (frozen in the receipt, enforced at admission).** The
authenticated fixed-commit **committer timestamp must be on or after
2025-01-01T00:00:00Z**.
- **Refused timestamps:** missing or unparseable ones, and contradictory ones: an author time later than the committer time, a buggy commit later than the fix, or a commit later than the evidence was retrieved.
- **Why not "authored and merged":** the earlier wording is withdrawn, because PR merge evidence cannot be made reliable for every candidate.
- **Status:** the cutoff is **contamination-risk mitigation only**. The buggy code, the repository or a similar fix may still have been seen in pretraining.

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

## 2. Isolation (fail-closed, bound to a verified frozen universe)

**Status: D7 is pending.** The universe-to-receipt binding below is new and
awaits acceptance.

### Three separate stages

`harness/repository_isolation.check_candidate(candidate, frozen)` writes one
record per candidate. It shows each stage's result separately, and admission
requires every stage to be empty.

| stage | function | what it decides |
|---|---|---|
| 1. evidence schema | `evidence_problems` | every mandatory field is present and well formed |
| 2. evidence authentication | `authentication_problems` | the stored evidence is internally consistent and agrees with the declared identity (offline) |
| 3. source-universe overlap | `overlap_problems` | repository, fork parent, commit, issue, patch and function similarity against the frozen universe |

Network **acquisition** of the evidence is a later, separately approved step
(D1). What it must store is listed in `ACQUISITION_REQUIREMENTS`. Nothing in the
checker makes a network call.

### Mandatory evidence (stage 1)

Anything missing, unknown or malformed is refused as
`insufficient_isolation_evidence`:
- canonical lower-case owner/name, the `https://github.com/<owner>/<name>` URL, numeric repository ID and node ID;
- the repository API response: URL, raw bytes, SHA-256, UTC retrieval timestamp, and ETag where available;
- explicit fork status; for a fork, the parent's full_name, numeric ID and node ID (unknown is refused);
- full 40-hex buggy and fixed commit SHAs, which must differ, with their commit objects;
- the tree objects on the paths to the licence file and the target file at both revisions;
- an admitted SPDX licence, the licence-file path at the buggy commit, its blob ID and its SHA-256;
- the target file path and module, and the target-file blob IDs at both revisions;
- a non-empty normalised patch;
- a parseable target function with at least 12 comparison shingles;
- the issue or PR identity `owner/name#n`, with its API response (URL, bytes, SHA-256, timestamp).

### Authentication (stage 2, offline cross-checks)

**Repository response.** It must hash to its recorded SHA-256. Its `full_name`
must equal both the queried name (a rename or redirect is refused) and the
declared repository. Its `id`, `node_id`, `html_url`, `fork`,
`parent{full_name, id, node_id}` and `license.spdx_id` must each match the
declared values.

**Issue response.** It must hash to its recorded SHA-256. Its `number`, `url`
and `repository_url` (or, for a PR, its base repository) must match.

**Git objects.** Every object is re-hashed to its git object ID. The licence
path must resolve through the buggy commit's trees to the recorded licence
blob, and that blob's SHA-256 must match.

**Exact diff (policy A, frozen in the receipt).** The submitted patch is
**never authoritative**. The rules:
- **Direct parent:** the fixed commit has exactly one parent, the buggy commit. Merge commits are refused.
- **Single changed file:** comparing the authenticated buggy and fixed trees must show exactly one changed path, the target file. A changed subtree without its tree objects is refused (partial evidence is never accepted), and so is a multi-file fix. Policy B, multi-file fixes, is not selected.
- **Derived diff:** the canonical diff is recomputed from the two authenticated target blobs (difflib unified diff, 3 context lines, `a/`/`b/` headers).
- **Submitted patch:** it must carry exactly the derived removed and added lines, per file. Its hunks are parsed by their header counts, so a line that looks like a header cannot hide.
- **Patch lineage:** patch hashes, shingles, and identical and near-duplicate checks use only the derived diff.
- **Target function changed:** the declared target function must occur exactly once in the buggy file. At least one derived hunk must overlap its AST span, the same qualified function must exist in the fixed file, and the normalised bodies must differ.
- **Recorded:** the derived diff's SHA-256, the changed-file list, the hunks, the target-function spans (buggy and fixed) and the authentication result are stored in the isolation record.

**Temporal rule.** This is proved from the authenticated commit timestamps (§1).

Tests use offline synthetic fixtures (`harness/isolation_evidence_fixtures.py`).
They show that each of the following is refused:
- an edited response whose hash was recomputed but whose content no longer matches;
- a wrong parent, a merge commit, an undeclared fork, a licence mismatch, a missing commit object;
- a partial patch carrying only an unrelated valid change;
- a patch that omits one of two real changes, invents lines, or comes from another file;
- a fix in an unrelated file, or a changed target file whose declared function is unchanged;
- a known benchmark change disguised by submitting only a novel subset;
- a multi-file fix;
- a fix before the cutoff, and each missing or contradictory timestamp.

The exact, complete, single-file patch is admitted.

### Overlap (stage 3)

| check | rule |
|---|---|
| repository | owner/name, bare name and verified fork parent are all outside the universe |
| issue | not a known benchmark instance |
| patch lineage | commits are not known commits; the **derived** normalised diff is not identical to any indexed patch, and its Jaccard with each is below 0.80 |
| function lineage | Jaccard against every indexed reference function is below 0.80; the nearest match, score and fingerprint are recorded |
| within the new set | one target per fix commit and per function lineage; a pairwise near-duplicate check |

### Frozen universe and receipt

**Loading refuses on any mismatch.** Loading rebuilds the universe from disk and
freezes it against the receipt in the current bundle generation
(`reference_universe_receipt.json`, which also freezes the diff policy and the
temporal rule). It refuses unless all of the following hold:
- the recomputed collections equal the receipt's, and so do their hashes;
- the internal receipt hash verifies;
- all 1,558 data inputs, 6 code inputs and 10 canonical sources match their recorded hashes;
- the recomputed coverage equals the receipt's, and passes.

**Callers cannot supply a hash.** `check_candidate` accepts only the
`FrozenReferenceUniverse` and reads the receipt hash from it. There is no
receipt-hash parameter, and the object cannot be built directly.
`isolation_record_is_current(record, frozen)` accepts only the same verified
universe and an unmodified record digest.

**Integrity is not a signature.** `record_sha256` only detects accidental
modification: anyone who edits a record can recompute it. Downstream dataset
construction must:
1. keep each candidate's authenticated evidence sidecar (`candidate_to_sidecar`);
2. reload the same frozen universe;
3. rerun `check_candidate` and compare the record byte for byte (`revalidate_isolation_record`).

An integration test shows that edited `admissible` or `reasons` fields are
rejected even with a recomputed `record_sha256`.

**Coverage.** `audit_source_coverage` takes the sources that need indexing from
the corpus manifest bound in the universe. It records the manifest's SHA-256
and re-verifies it on disk. A changed manifest invalidates the receipt. The
receipt also binds, by canonical hash:
- `utils/dataset_identity.py`;
- the curated definition;
- every module that affects classification, normalisation, curated-seed identification or train-view loading.

**Curated seeds.** All ten `CURATED_BUGSINPY_BUGS` definitions are indexed. That
is a **conservative superset** of the eight corpus seeds. No non-train record was
opened.

## 3. Licensing and reproducibility

- **Licences:** MIT, BSD-2-Clause, BSD-3-Clause, Apache-2.0, ISC or PSF-2.0. The SPDX identifier from the repository API must agree with the declaration. The licence file is resolved through the buggy commit's trees to its blob, and that blob is hashed (§2).
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

- **Size: N = 400 recommended**, powered for a true +8 pp gain at ρ = 0.05 with at most 12 targets per repository (§9). **N = 200 is a feasibility option only:** a lower-power compromise that needs explicit approval. The +5 pp gate is not changed to fit the sample.
- **Uniqueness:** unique bugs, function lineages and fix commits.
- **Caps:**
  - no repository above 12 targets. The earlier 8% share would allow 32 per repository at N = 400, too many for the planned power;
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
- **Portable, verified evidence:** paired Kill@8 comes from the retention and tool-assisted panels, and every path is repository-relative. Before the artifact is written, and again when the design accepts it:
  - each evaluation envelope and raw result hash is checked against the pilots' decision receipts;
  - paired arms must cover identical record IDs;
  - panel lineages must cover exactly the evaluated records;
  - every record must be a train-view record.

  The design builder recomputes the whole artifact and refuses any difference.
- **Measured inputs:** discordance 0.17–0.21; lineage ICC about 0.
- **Effect at the gate:** a true gain of exactly +5 pp passes with probability **at most 50% at every N**, so no N gives 80% power there.
- **Planning scenario** (analytic / clustered Monte Carlo; true +8 pp, discordance 0.21, ρ 0.05, ≤ 12 targets per repository):

| N | power at +8 pp | 80%-power MDE |
|---|---|---|
| 100 | 0.41 / 0.44 | 13.6 pp |
| 150 | 0.54 / 0.54 | 11.2 pp |
| 200 | 0.64 / 0.63 | 9.8 pp |
| 300 | 0.79 / 0.77 | 8.1 pp |
| **400** | **0.86 / 0.82** | **7.4 pp** |
| 500 | 0.88 / 0.85 | 7.1 pp |

- **Sensitivity:**
  - with a fixed 25 repositories the design effect grows with N, and N = 500 still needs about 7.4 pp;
  - with only 10 repositories at N = 300 the MDE is 10.1 pp (ρ 0.05);
  - at ρ = 0.10 the planning effect needs N = 500, and at ρ = 0.20 no N up to 500 reaches it;
  - clustered Monte Carlo covers ρ = 0, 0.05, 0.10 and 0.20.
- **Recommendation:** N = 400 from at least 34 repositories. N = 200 is a feasibility compromise (power about 0.64).

## 10. Receipts and rollback

**Per target:**
- repository and issue API responses, with their hashes and retrieval timestamps;
- the licence file's blob and hash, and the commit and tree objects;
- the derived canonical diff's SHA-256, the changed files and the target-function spans;
- the isolation record (the frozen universe's receipt hash and the record digest) and its evidence sidecar, so the record can be revalidated;
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

**Publication:**
- design artifacts are published as one immutable generation under `results/next_direction_bundle`: the receipt, the design, a verified copy of the power artifact, and a manifest of file hashes;
- one small `CURRENT` pointer selects the generation. Readers accept only a complete, manifest-verified generation, so a crash at any point leaves either the complete old generation or the complete new one;
- accepted generations are never overwritten;
- the standalone power artifact is replaced with a single atomic file replacement, which is per-file atomic only.

**Rollback:**
- nothing is overwritten;
- failed runs are preserved and marked invalid;
- the one-time set is sealed by hash before any model sees it, and is never re-evaluated after a protocol change.

## Estimates (N = 400; every N in the power artifact)

| stage | GPU | CPU wall | storage |
|---|---|---|---|
| mining + acquisition + isolation | — | ~2.7 h (+2.7–5.3 h network) | 11–21 GB |
| environment build | — | ~5 h | ~120 GB kept, ~360 GB peak |
| qualification | — | ~5.6 h | — |
| dress rehearsal | ~5 min | ~1 h | ~1 GB |
| one-time evaluation | ~80 min | ~35 h (native 10 h + Atheris 25 h) | ~2.7 GB |

At N = 200 the total is about 25 CPU wall-hours and 194 GB peak. At N = 300 it
is about 37 hours and 289 GB, and at N = 500 about 61 hours and 480 GB. WSL
has 952 GB free.

The assumptions:
- a 7–15% yield from mined candidates to targets (N = 400 needs 2,700–5,700 candidates);
- 45 s per native test run;
- 4 minutes per environment build;
- 16 parallel jobs;
- 2 models × 8 candidates;
- 2 Atheris modes × 3 seeds × 600 CPU-s.
