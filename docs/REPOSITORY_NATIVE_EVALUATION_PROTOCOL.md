# Repository-native evaluation — protocol draft

**Status: draft.** Nothing has been mined, no split exists, and no model has run.
Everything here is to be frozen, and hash-bound in a receipt, before any model
sees the final set. Structured content lives in
`results/v4_3_next_direction_design.json`, under `repository_native_protocol`.

## 1. Candidate sources

**What is mined.** Bug-fix commits from permissively licensed, pure-Python,
pytest-tested GitHub repositories outside the 35-repository exclusion universe.
The fix must have been merged **on or after 2025-01-01**, after the Qwen2.5-Coder
release. The buggy code may predate that date; this is unavoidable and reported.

**Not used:**

| source | why |
|---|---|
| BugsInPy | all 17 projects are excluded |
| SWE-bench and SWE-bench Verified | same 12 repositories |
| HumanEvalFix, HumanEvalPack, EvalPlus | derived from HumanEval or MBPP |
| aggregates such as Defectors or PyTraceBugs | project overlap; screened one repository at a time, never used wholesale |
| SWE-rebench / SWE-Gym style collections | a candidate *index* only; each repository is screened like any other |

**Screening candidates (names only, nothing verified yet):**
- more-itertools, toolz, boltons, attrs, cattrs, marshmallow;
- python-dateutil, arrow, pendulum, humanize;
- sortedcontainers, networkx, sqlglot, pyparsing, tomlkit;
- rich, typer, cachetools, glom, parse, inflect, babel, isort;
- jsonschema, tabulate, textdistance, dateparser, schedule, python-slugify, validators.

Licence, activity, test suite, Python support and the supply of bug fixes from
2025 onward are all verified during mining. **Same-organisation repositories are
flagged for a decision**: pallets, pylint-dev, pytest-dev and psf each own an
excluded repository.

## 2. Isolation, proven per candidate

`harness/repository_isolation.check_candidate` writes a record for every
candidate. A candidate is admitted only if `admissible=true`, which requires all
of the following:

| check | rule |
|---|---|
| repository | canonical owner/name, the bare name and the fork parent are all outside the universe |
| issue | not a known benchmark instance |
| patch lineage | buggy and fixed commits are not known commits; the normalised patch is not identical to any of the 998 known patches, and its Jaccard with each is below 0.80 |
| function lineage | the target function's Jaccard against all 33,172 reference functions is below 0.80; the nearest match and its score are recorded |
| within the new set | one target per fix commit and per function lineage; a pairwise near-duplicate check |

**Why the consumed test split need not be read.** Every Oneiros split is built
only from MBPP, HumanEval, BugsInPy, SWE-bench Verified and curated examples. The
reference universe is the full upstream copy of each, a superset of every split.
This argument requires your acceptance (decision D7).

## 3. Licensing and reproducibility

- **Licences:** MIT, BSD-2-Clause, BSD-3-Clause, Apache-2.0, ISC or PSF-2.0. The SPDX identifier comes from the LICENSE file at the buggy commit and is recorded with its hash.
- **Checkouts:** full clones only. The existing blob-less ingestion cache cannot check out files. Buggy and fixed SHAs and tree hashes are recorded.
- **Python versions:**
  - available in WSL today: 3.8, 3.9, 3.11 and 3.12;
  - 3.10 and 3.13 need an approved installer (decision D6);
  - each bug uses the lowest version its project supports at the buggy commit.
- **Environments:** a separate virtual environment per revision, installed from the project's own metadata at that commit. The installed packages are frozen to a lock with hashes. Network access is allowed only while building an environment, never while tests run.

## 4. Native-execution qualification

1. Build environments for both the buggy and the fixed revision.
2. Run the official regression test three times on each revision. An independently written and reviewed test may substitute, and is recorded as such.
3. **Qualify** a bug only if the test fails 3 of 3 times on the buggy revision and passes 3 of 3 on the fixed revision, with identical locks.
4. Inject each generated test into both revisions under an isolated file name, with a per-test timeout. Any discordant result is re-run once to detect flakiness.

**Outcome classes:**

| class | meaning | counted as |
|---|---|---|
| semantic kill | assertion fails on buggy; passes on fixed | kill |
| crash kill | uncaught exception from project code on buggy; passes on fixed | kill (reported separately) |
| timeout on buggy | times out on buggy; passes on fixed | reported separately, not a semantic kill |
| passes both / fails both / inverted | no discrimination / reference-invalid / pathological | not a kill |
| harness failure | collection, injection or pytest internal error | **excluded, never a kill or a model failure** |
| dependency failure | a package is missing or incompatible | excluded |
| environment failure | interpreter, build or filesystem failure | excluded; the target is disqualified before evaluation |

## 5. Target function and context policy

**Permitted in the prompt:**
- the buggy-side target function;
- its signature, type hints and docstring;
- the buggy-side module context needed to call it: imports, constants and class skeletons;
- public docstrings present at the buggy commit.

**Forbidden in the prompt:**
- the fixed code, or the patch or diff;
- hidden, official or gold tests;
- expected outputs;
- issue or PR discussion of the fix;
- the fix commit message.

**Enforcement.** Prompts are built from a permitted view, exactly as in the
tool-assisted loop. Every prompt is scanned for any line of the fixed function,
the patch or the official test. The check that the view-built prompt equals the
full-record prompt is repeated.

**Recorded per target:**
- complexity tier (existing AST policy) and bug family (existing repository defect taxonomy);
- LOC, nesting depth, outgoing calls, and parameter count and kinds;
- statefulness: method, global or IO use;
- Python version and repository.

## 6. Composition

- **Size:** 100 targets minimum, 150 planned.
- **Uniqueness:** one target per unique bug; unique function lineages and fix commits.
- **Caps:**
  - no repository above 8%;
  - no bug family above 25%;
  - no candidate index above 50%.
- **Tiers:** simple, moderate and complex, with at least 20% each where supply allows. Any shortfall is reported and never filled by repetition. Exact unique counts and every exclusion, with its reason, are reported.
- **Rehearsal pool:** 20–30 qualified bugs from at least 5 repositories that are never admitted to the final set.

## 7. Oneiros versus Atheris

**Tool.** Actual `atheris` 2.3.0 (libFuzzer) on WSL Python 3.11. It is never the
simulated `baseline.coverage_fuzzer`, which would be reported under its own name
if used at all.

**Targets and oracle.**
- **Same targets:** both systems run on an identical target set. Targets Atheris cannot drive are reported as Atheris-ineligible.
- **Two comparisons:** one on the jointly eligible subset, and one on all targets with Atheris scoring zero on the ineligible ones.
- **Oracle:** identical buggy and fixed revisions. Atheris gets a differential fixed-versus-buggy oracle, which Oneiros never sees; this is stated as being generous to Atheris.

**Adapter.** The strongest reasonable structured-input adapter:
- driven by type hints and docstrings;
- full-range numbers and bytes;
- nested containers, including dict, set and tuple;
- dataclass and NamedTuple construction;
- seeds taken from literals in the buggy function.

Reachability of the defect is reported per target.

**Budgets.** Atheris gets 600 CPU-seconds per target with seeds 42, 43 and 44.
Oneiros gets 8 candidates, reported as model calls, tokens, GPU seconds and CPU
execution seconds. The two budgets differ in kind, and no "equal budget" claim
is made.

**Reporting.** Crash kills and semantic kills are reported separately, alongside
kills per CPU-second, per-seed results and seed-aggregated results.

## 8. Dress rehearsal (permitted data only)

The rehearsal runs every stage end to end on the rehearsal pool, never the final
set:
1. mining and isolation records;
2. environment locks;
3. official-test qualification;
4. prompt construction and leakage scan;
5. generation with both the base model and the frozen control;
6. native execution on both revisions;
7. Atheris;
8. scoring, outcome classes and receipts;
9. the frozen analysis;
10. a deliberate crash-and-resume.

**Gate:** every stage completes with receipts, with no unexplained failure, and
at least 90% of qualified targets execute on both revisions. Only then is
one-time evaluation requested.

## 9. Metrics and decision rules

**Primary:** Kill@1, Kill@4 and Kill@8 (a native semantic or crash kill per
target), and unique defects killed.

**Secondary:**
- semantic and crash kills reported separately;
- reference validity (passes on fixed) per requested candidate;
- parse and execution success;
- slices by complexity tier, bug family and repository;
- a leave-one-repository-out generalisation summary;
- Atheris kills and kills per CPU-second.

**Intervals:** Wilson 95% intervals for single proportions. Paired comparisons
use exact McNemar and a repository-cluster bootstrap with a 90% interval (10,000
replicates, fixed seed).

**Decision rule** (predeclared before the one-time run):
- **Pass:** a model comparison passes only with a Kill@8 gain of at least +5 pp, a cluster-bootstrap lower bound above 0, and reference validity within −3 pp.
- **Otherwise:** everything else is descriptive.
- **No selection on the final set:** no configuration is chosen using it.

## 10. Receipts and rollback

**Per target:**
- repository and licence;
- buggy and fixed commit and tree hashes;
- the isolation record;
- environment lock and interpreter;
- the qualification runs;
- the prompt hash and leakage scan.

**Per run:**
- LF-canonical source hashes;
- model, revision and adapter hash;
- seeds and generation settings;
- raw outputs and their hashes;
- return codes, timeouts and durations for both revisions;
- Atheris logs, corpus sizes, coverage and seeds;
- the WSL distribution, kernel, CPUs and memory.

**Rollback:**
- nothing is ever overwritten;
- failed runs are preserved and marked invalid;
- the one-time set is sealed by hash before any model sees it, and is never re-evaluated after a protocol change.

## Estimates

| stage | GPU | CPU wall | storage |
|---|---|---|---|
| mining + isolation | — | ~1 h (+1–2 h network) | 4–8 GB |
| environment build | — | ~2 h | ~45 GB kept, ~135 GB before pruning |
| qualification | — | ~2 h | — |
| dress rehearsal | ~5 min | ~40 min | ~1 GB |
| one-time evaluation | ~30 min | ~8.5 h | ~1 GB |

The funnel assumption: 1,500 candidate fix commits → 450 heuristic → about 430
isolation-admissible → about 260 environments built → about 180 deterministic
reproductions → about 150 extractable single-function targets. Timing assumes
45 s per native test run, 4 minutes per environment build and 16 parallel jobs.
