# Next-direction decision memo

Written 2026-09-25 and revised three times the same day. The revisions:
- made isolation fail closed;
- bound every isolation decision to a verified universe;
- derived patch lineage from the authenticated commit diff, made the temporal rule executable, and made publication crash-safe.

This is a design only: no split exists, nothing has been mined, no tool has
been installed, no model has run, no network call was made, and no protected
data has been opened.

The machine-readable versions are one crash-safe bundle generation under
`results/next_direction_bundle`, selected by its `CURRENT` pointer (§ Publication
is crash-safe):
- `next_direction_design.json`;
- `reference_universe_receipt.json`;
- `repository_native_power_analysis.json`, a verified copy of `results/v4_3_repository_native_power_analysis.json`.

The previous top-level receipt and design files are superseded. They remain in
git history at `9d29a4c`.

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

**D7 is accepted (2026-09-25) under exactly this narrowed claim, and no
broader one.** It is not a claim that the model could not have seen a
repository during pretraining.

The acceptance followed three fixes:
- the universe-to-receipt binding;
- patch lineage, which could be spoofed by submitting a partial patch;
- the crash-safety of publication.

### Every decision is bound to a verified frozen universe

`FrozenReferenceUniverse` holds:
- the sealed, immutable universe;
- a private copy of its receipt;
- the internally computed receipt SHA-256;
- the verification report.

It can be created only by `freeze_reference_universe` or
`load_frozen_reference_universe`. Those functions refuse unless all of the
following hold:
- the recomputed collections equal the receipt's, and so do their hashes;
- the internal receipt hash verifies;
- every data input, code input and canonical source matches on disk;
- the recomputed coverage equals the receipt's, and passes.

`check_candidate(candidate, frozen)` has **no receipt-hash parameter**; it reads
the hash from the verified object. `isolation_record_is_current(record, frozen)`
accepts only the same verified universe and an unmodified record digest.

**Negative test (tracked universe):**

| attempt | result |
|---|---|
| `check_candidate(candidate, "a"*64)` | `TypeError`: an isolation decision requires a verified FrozenReferenceUniverse; a receipt hash cannot be supplied separately |
| `check_candidate(candidate, frozen, "a"*64)` | `TypeError`: takes 2 positional arguments |
| `FrozenReferenceUniverse(receipt_sha256="a"*64)` | `TypeError`: created only by freeze/load |
| record edited to carry `"a"*64`, digest recomputed | `isolation_record_is_current` → `False` |

Further tests show that each of the following invalidates a frozen universe:
- pairing universe A with receipt B;
- a changed commit, patch, function fingerprint or collection count;
- a changed input file;
- a changed source file (`utils/dataset_identity.py`, the curated definition, the normaliser, the train-view loader);
- a receipt re-sealed around a changed collection;
- a receipt missing a canonical source.

### Coverage binds every input that decides it

The receipt (`receipt_sha256` `40ce55b0…`, file `01077dc1…`) also freezes the
diff policy and the temporal rule. It binds:
- the corpus manifest (`7f68386f…`), from which coverage decides which sources need indexing. Coverage records that manifest's SHA-256 and re-checks it on disk; changing its sources or counts refuses the old receipt;
- 1,558 data inputs: BugsInPy 1,019 (every `project.info`, `bug.info` and `bug_patch.txt`), SWE-bench Verified 1, HumanEval 1, MBPP 1, legacy real bugs 533, the train view 2, and the corpus manifest 1;
- 6 code inputs and 10 canonical sources, by LF-canonical hash: the checker, `harness/source_identity.py`, the normaliser, `harness/bugsinpy_loader.py`, `scripts/build_corpus_v1.py`, `utils/dataset_identity.py`, `harness/corpus_view.py`, `harness/corpus.py`, `harness/function_complexity.py` and the builder;
- 28 full and 35 bare repository names, 1,484 commits, 998 patch hashes, 1,001 instance IDs and 33,192 function fingerprints.

The receipt is byte-identical across rebuilds: generations `0c2914bf…` and `9703c7b4…` carry the same receipt bytes.

**Curated seeds.** All ten `CURATED_BUGSINPY_BUGS` definitions are indexed. That
is a **conservative source superset** of the eight corpus seeds: the 7 in train,
plus an eighth that must be one of `black_1`, `cookiecutter_1` or `scrapy_1`. No
non-train record was opened.

### Candidate evidence is authenticated, not self-declared

Three stages are kept separate:
1. **Schema:** fields are present and well formed.
2. **Authentication:** offline cross-checks of the acquired evidence.
3. **Overlap:** checks against the frozen universe.

Network acquisition is a **later, separately approved step**. The checker never
calls the network.

**Required evidence:**
- **Repository and issue API responses:** URL, raw bytes, SHA-256, retrieval timestamp and ETag. The repository's numeric ID and node ID, the queried and returned full_name, fork status, parent identity (full_name, ID, node ID) and licence must all match the declaration. The issue or PR must name the same repository.
- **Git objects:** the buggy and fixed commit objects, re-hashed to their IDs. The fixed commit's parent must be the buggy commit. Tree objects must resolve the licence file at the buggy commit to its blob and hash, and the target file to its blobs at both revisions. Those blobs must contain the target function and the patch's removed and added lines.

Offline synthetic fixtures show that tampered evidence is refused even when its
hashes have been recomputed.

### Patch lineage comes from the authenticated commit diff (policy A)

**The defect.** Authentication used to check only that the submitted patch's
removed lines occurred in the buggy file and its added lines in the fixed file.
A reproduction: the buggy and fixed blobs carry two changes, one in the target
function and one to an unrelated `TIMEOUT` setting, but the submitted patch
carries only the setting change. That passed both the schema and authentication
stages, returning `[]` from each.

**It is now refused** with `submitted_patch_does_not_match_derived_diff`. The
frozen policy A works as follows:
- **Direct parent:** the fixed commit's only parent is the buggy commit.
- **One changed path:** the authenticated trees must show exactly one changed path, the target file. Multi-file fixes, and changed subtrees without their tree objects, are refused.
- **Derived diff:** the canonical diff is recomputed from the authenticated blobs, and only it feeds patch hashes and the identical and near-duplicate checks. The submitted patch is not authoritative; the authenticated canonical diff is. The submitted patch is only required to carry the same derived added and removed lines per file. Byte-identical patch text and identical hunk placement are not required.
- **Target function changed:** a derived hunk must overlap the declared target function's AST span, and that function's normalised body must change between revisions.
- **Recorded:** each isolation record stores the derived diff's SHA-256, the changed files, the hunks and the target-function spans.

**Tests** cover each case: a partial unrelated patch, an omitted real change,
invented lines, a patch from another file, an unrelated-file fix, a target file
changed with the declared function unchanged, a known benchmark change hidden
behind a novel subset, a merge commit, missing tree evidence and a multi-file
fix. The exact single-file patch is admitted.

### The temporal rule is executable

**The rule:** the authenticated fixed-commit committer timestamp is on or after
2025-01-01T00:00:00Z. It is enforced at admission and frozen in the receipt.
The earlier "authored and merged" wording is withdrawn, because PR merge
evidence cannot be made reliable for every candidate.

**Refused timestamps:** missing and contradictory ones, meaning an author time
after the committer time, a buggy commit after the fix, or a commit after the
evidence was retrieved.

**Boundary tests:** one second before the cutoff is refused; exactly at it and
after it are admitted. The rule is contamination-risk mitigation only.

### A record's self-hash is not a signature

`record_sha256` only detects accidental edits. Downstream dataset construction
must keep each candidate's evidence sidecar, reload the same frozen universe,
rerun `check_candidate`, and compare byte for byte
(`revalidate_isolation_record`). An integration test shows that an edited
`admissible` or `reasons` field is rejected even when its `record_sha256` was
recomputed. The self-hash alone would still call it current.

## Power (prospective, verified, portable)

**Inputs verified before any calculation.** Every evidence path is
repository-relative:
- each evaluation's envelope hash **and** raw-result hash match the closed pilots' decision receipts;
- paired arms cover identical record IDs;
- panel lineages cover exactly the evaluated records;
- every record is in the train view.

The design builder does not trust the power file. It recomputes the whole
artifact from the verified evidence and refuses any difference. Measured paired
discordance is 0.17–0.21, and lineage ICC is about 0.

Under the **frozen, unchanged +5 pp gate**, a pass needs an estimate of at least
5 pp *and* a 90% lower bound above 0:

- **A true gain of exactly +5 pp** passes with probability **at most 50% at every N**. No N gives 80% power there.
- **Planning scenario:** true +8 pp, discordance 0.21, ρ 0.05, at most 12 targets per repository. Power is shown as analytic / clustered Monte Carlo:

| N | power at +8 pp | power at +10 pp (analytic) | 80%-power MDE |
|---|---|---|---|
| 100 | 0.41 / 0.44 | 0.56 | 13.6 pp |
| 150 | 0.54 / 0.54 | 0.71 | 11.2 pp |
| 200 | 0.64 / 0.63 | 0.82 | 9.8 pp |
| 300 | 0.79 / 0.77 | 0.93 | 8.1 pp |
| **400** | **0.86 / 0.82** | 0.96 | **7.4 pp** |
| 500 | 0.88 / 0.85 | 0.98 | 7.1 pp |

- **Sensitivity** (clustered Monte Carlo for ρ = 0, 0.05, 0.10 and 0.20):
  - at ρ = 0.10, +8 pp needs N = 500; at ρ = 0.20, no N up to 500 reaches 80%;
  - with a fixed 25 repositories, larger N buys little (MDE 7.4 pp at N = 500);
  - with 10 repositories at N = 300, the MDE is 10.1 pp;
  - an 8% per-repository share cap allows 32 targets per repository at N = 400 and costs power, so the composition cap becomes **12 targets per repository**.
- **Recommendation: N = 400**, from at least 34 repositories, powered for +8 pp at ρ 0.05.
- **N = 200 is a feasibility option only.** It is a lower-power compromise, with power about 0.64 at +8 pp, and is not adequately powered. It is usable only with explicit approval.

## Estimates

| N | mined candidates (7–15% yield) | CPU wall | peak storage | GPU (one-time) |
|---|---|---|---|---|
| 200 | 1,334–2,858 | ~25 h | ~194 GB | ~40 min |
| 300 | 2,000–4,286 | ~37 h | ~289 GB | ~60 min |
| **400** | 2,667–5,715 | ~49 h | ~384 GB | ~80 min |
| 500 | 3,334–7,143 | ~61 h | ~480 GB | ~100 min |

Network time for mining is extra: about 2.7–5.3 h at N = 400. WSL has 32 CPUs,
62 GB RAM and 952 GB free. The optional sampling study needs about 20 GPU
minutes.

## Publication is crash-safe

The earlier multi-file `os.replace` sequence was per-file atomic only. A process
kill between two replacements could have left a mixed generation. It is
replaced by an immutable-generation bundle:
1. Files and a manifest of their hashes are written into a private staging directory, fsynced, re-read and verified.
2. The staging directory is renamed to its generation directory, whose ID is derived from the manifest hash.
3. One small `CURRENT` pointer is replaced last.

Readers accept only a complete, manifest-verified generation, and reject any
modified, extra or mispointed file. Accepted generations are never overwritten.

**Crash tests.** A child process is killed with `os._exit` at each of the 8
publication points. Every time, readers see the complete old generation, or,
only after the pointer replacement, the complete new one. They never see a
mixture, and a later publication still succeeds.

**Durability caveat.** Windows has no directory fsync. There, a crash can at
worst lose the newest pointer update, never mix generations. The standalone
power artifact uses a single-file replacement, which is per-file atomic only.

The closed-pilot check verifies both the evaluation envelope and the raw
rehearsal-result hash for all six evaluations.

## Blockers

| id | blocker |
|---|---|
| B1 | No candidate data exists; mining needs network access and a licence review. |
| B2 | WSL lacks Python 3.10 and 3.13 and any hash-locking tool (`uv` or pip-tools). |
| B3 | Native execution of generated tests has never been demonstrated. |
| B4 | The Atheris adapter covers primitives only. |
| B5 | Repository prompts are capped at 2,048 tokens. |
| B6 | The consumed test split is covered only through enforced upstream indexing (D7). |
| B7 | No untouched train panel exists for the sampling study. |
| B8 | N = 400 needs roughly 2,700–5,700 mined candidates from 34 or more repositories, and about 50 CPU-hours. |
| B9 | Candidate GitHub metadata and git objects must be acquired and hash-bound. The offline authenticator exists; the acquisition step does not. |

## Decisions (recorded 2026-09-25)

| id | status | decision |
|---|---|---|
| D1 | approved | Read-only GitHub mining and hash-bound acquisition of repository metadata, issue/PR responses, commit, tree and blob objects and licence evidence. The fixed commit's authenticated committer timestamp must be on or after 2025-01-01T00:00:00Z (mitigation only). |
| D2 | approved conservatively | Only MIT, BSD-2-Clause, BSD-3-Clause, Apache-2.0, ISC and PSF-2.0. Same-organisation repositories are excluded by default and listed separately for explicit review, never silently admitted. |
| D3 | approved | N = 400 from at least 34 repositories, with at most 12 targets per repository. The +5 pp gate is unchanged, and shortages are never filled by repetition. A bounded acquisition pilot comes first. |
| D4 | decided | The final set is sealed by hash after qualification and before any model sees it. The user is the one-time final-evaluation authorizer. No development tool, model call or prompt inspection may touch the sealed set before that authorization. |
| D5 | approved | Actual Atheris 2.3.0 in two labelled modes: ordinary crash/contract (the primary baseline) and differential-oracle (an upper bound). Seeds 42, 43 and 44; 600 CPU-seconds per target, per mode, per seed. Applicability is reported separately, and an ineligible target is never counted as a zero. |
| D6 | approved | `uv` and Python 3.10/3.13 inside WSL only, pinned, with hash-locked environments. Windows Python is untouched. |
| D7 | **accepted** | Accepted only under the narrowed claim quoted above. |
| D8 | deferred | The sampling-budget GPU study is not run; the repository-native critical path comes first. |
| D9 | provisionally approved | Exactly two arms: the pinned Qwen2.5-Coder-1.5B base model and the existing frozen control adapter, with identities resolved from the frozen receipts. Proceeding is refused if either identity is missing or ambiguous. |

## D7 acceptance

**Accepted on 2026-09-25, under the narrowed indexed-universe claim only.**

- **What is now true:**
  - a decision can be made only against a universe verified against its receipt, and no caller can supply the receipt hash;
  - every input that decides coverage is hash-bound;
  - candidate identity, fork, issue and licence evidence is authenticated;
  - patch lineage comes only from the diff derived from authenticated git objects, and must modify the declared target function, so partial or unrelated patches are refused;
  - the temporal rule is enforced from the authenticated commit timestamps;
  - records are trusted only after byte-for-byte revalidation;
  - the published artifacts cannot be observed in a mixed state.
- **What remains limited:**
  - the claim covers enforcement against the indexed universe only. It cannot rule out pretraining exposure, and the temporal rule only mitigates that risk;
  - under policy A, only single-file fixes are admissible, which narrows the candidate supply;
  - no real candidate can pass until the approved acquisition step (D1) is built (B9);
  - on Windows, publication durability relies on NTFS metadata journaling.

## Proposed execution order

1. Decisions D1–D9 are recorded, and D8 defers the sampling study.
2. Build the acquisition pipeline, then run a bounded acquisition pilot (D3).
3. Install tooling (`uv`, interpreters); extend the Atheris adapter.
4. Mine candidates and acquire their metadata and git objects, hash-bound. Split them into a rehearsal pool and a final pool, with isolation records made against the frozen universe.
5. Build hash-locked shared environments and run official-test qualification.
6. Run the dress rehearsal, including crash-and-resume.
7. Freeze by hash and request one-time authorization.
8. Run the one-time evaluation, only after explicit authorization.

Detailed drafts: [REPOSITORY_NATIVE_EVALUATION_PROTOCOL.md](REPOSITORY_NATIVE_EVALUATION_PROTOCOL.md)
and [SAMPLING_BUDGET_STUDY_PROTOCOL.md](SAMPLING_BUDGET_STUDY_PROTOCOL.md).
