# Real-repository data expansion plan

**Goal.** Raise real-world coverage with *unique, executable* repository
examples validated by each project's own buggy/fixed behaviour. Not by
repeating MBPP, and not by relaxing the 0.70 source cap.

## Why this is the binding constraint

The O1 dataset makes the gap concrete: of 1,316 selected positives, **12** are
real-repository rows — 0.91%. The other 99.09% are synthetic mutants of
HumanEval and MBPP functions. Nothing about repository-level behaviour can be
claimed from that, at any sample size, because the sample barely exists.

The frozen balanced view says the same thing from the other side: 346 unique
repository targets against 666 synthetic, and the 50/50 split it reports is
reached **only after repeating 320 repository rows** — `target_met: false`,
`repetitions_do_not_count_as_unique_targets: true`. Repetition is what we have
instead of data.

Concentration compounds it. Across 346 unique repository targets, **django
holds 154 (44.5%)** against a 0.35 project cap, which is therefore recorded as
`cap_applied: false, audit_only_until_more_unique_repositories_are_added`. The
shortfall is **94 more non-django unique targets** before the cap can bind at
all.

## The actual bottleneck is environment reproduction, not sourcing

This is the finding that should drive the effort, and it is already measured.
Of 501 BugsInPy tasks attempted in the v3 expansion staging run:

| outcome | n | share |
|---|---:|---:|
| accepted | 197 | 39.3% |
| **excluded: `official_f2p_failed`** | **297** | **59.3%** |
| excluded: no self-contained assertion pair or fragment | 7 | 1.4% |

`official_f2p_failed` means the project's own fail-to-pass test did not
reproduce the documented failure in our environment. It is an infrastructure
outcome, not a judgement about the bug. Six in ten candidate repository
examples are lost to it, and the earlier v3 ingestion run was worse: 144
accepted of 501 (28.7%), with 354 `official_f2p_failed`.

**So the highest-yield work is recovering those 297, not finding new
repositories.** At the staging run's own acceptance rate, sourcing 300 fresh
tasks yields ~118 examples; recovering half of the already-sourced failures
yields ~148, from projects already cloned and pinned.

## Stage R-0 — bank what is already validated (no new sourcing)

49 of the 197 staged-accepted records are **not yet in the development view**.
They are already executable and already validated against native buggy/fixed
behaviour. Promoting them is pure gain and costs no ingestion.

- Promote the 49 through the normal corpus-view build, re-hash the shards.
- Split assignment follows the existing rule. Records landing in val or the
  sealed test are *assigned and not looked at*.
- Report per-project counts so the effect on the django share is visible.

Expected: repository unique targets 346 → up to ~395, django share 44.5% → ~39%
depending on split assignment. Still short of the 0.35 cap, so R-1 follows.

## Stage R-1 — diagnose `official_f2p_failed`, do not retry it blindly

A blanket re-run reproduces the same 297 failures. Classify first, from the
evidence already recorded in `ingestion_report.json` (each excluded task
carries its commands and output):

1. **Dependency resolution** — a pin that no longer resolves, or a wheel with
   no build for the pinned Python. Fix: pin transitively and vendor the
   resolved set.
2. **Python version** — the project needs an interpreter we did not provision.
3. **Platform** — the test assumes POSIX paths, permissions, or signals.
   `data/bugsinpy_v3_linux_ingestion/` exists precisely because of this; route
   these there rather than forcing them on Windows.
4. **Genuinely flaky or network-dependent** — reject permanently and record the
   reason, so nobody re-attempts them a third time.

Publish the classification counts before fixing anything. If class 4 dominates,
the recovery ceiling is low and the effort should move to sourcing instead —
that is a real possible outcome and the plan must be able to return it.

## Stage R-2 — source new projects, chosen against the concentration

New tasks are worth more when they come from projects we are short of. Priority
order is set by the project cap arithmetic, not by repository popularity:
anything **not** django, and preferably not in the current top three. The
staging run already surfaced luigi (31) and scrapy (30) as productive
non-django sources.

Per-project intake cap: no project may contribute more than 0.35 of repository
unique targets. A project that would breach it stops contributing; its surplus
is not accepted "to be trimmed later", because that is how django reached 44.5%.

## Admission rule — unchanged, and non-negotiable

Every example must carry, from the project's own test suite:

- the fixed implementation and the buggy implementation,
- the project's own test that **fails on buggy and passes on fixed**, executed
  in both states in our environment and recorded,
- a self-contained assertion pair or a repository fragment,
- a unique semantic target. No repetition counts toward any balance figure.

An example failing any of these is excluded with its reason recorded. We do not
synthesise a mutant to replace a repository bug we could not reproduce — that
converts a real-world example into another synthetic one and would defeat the
purpose of the expansion.

## What this plan explicitly does not do

- **Does not relax the 0.70 source cap.** The cap is what keeps MBPP from
  dominating; raising it to grow a dataset is the failure it exists to prevent.
- **Does not repeat MBPP or any other row.** Repetition has already been tried
  and is the reason the frozen view's 50/50 is provisional.
- **Does not claim repository performance before the data exists.** Until
  repository unique targets are numerous and un-concentrated enough to report a
  per-project breakdown, repository results stay descriptive.

## Sequencing and cost

| stage | work | GPU | expected unique repository targets |
|---|---|---|---|
| R-0 | promote 49 already-validated records | none | 346 → ~395 |
| R-1 | classify and recover `official_f2p_failed` | none | ~395 → 450-550 if recovery is 30-50% |
| R-2 | source non-django projects to the cap | none | to 0.35 django share |

All three stages are CPU/IO only. No stage requires GPU, and none of them is a
training run.
