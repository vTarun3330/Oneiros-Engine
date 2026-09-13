# `v4_2_armA_frozen_preflight.json` is DIAGNOSTIC ONLY

**This receipt must not be used to authorize training.**

## Why

The full pytest suite was run concurrently with this preflight, while it was in
its tokenization phase. That violates the no-concurrent-CPU-work condition the
preflight is supposed to be measured under.

This is not a hypothetical concern in this project. The successor generation run
overlapped a 768-test suite and a 44k-token pass, and 23,538 of 44,760
candidates (52.6%) came back `worker_error` purely from contention - transient,
not semantic, and it invalidated the artifact until every one of them was
re-scored. Timing and execution-derived figures measured under contention are
not trustworthy here, so this receipt is quarantined rather than trusted.

## What it does and does not contaminate

Selection is deterministic and does not depend on load, so `selection_sha256`
`598eef61c18e3e394d73a4fc2b72aa91309c7bf5a97a06eddb20be5f372eef23` is expected
to reproduce exactly in the clean rerun - and the clean rerun is what proves it,
not this file.

`elapsed_seconds` (1049.7) is meaningless. Any figure derived from subprocess
execution under contention is suspect.

## Its one recorded gate failure is real and separate

`all_automated_tests_pass` is **false**, and not because tests failed: the
cached `results/v4_1_local_test_status.json` records `passed: 452, failed: 0`
from 2026-09-07 against `source_tree_sha256` `c4b7d7bb...`, while the tree is
now `61237858...`. The gate correctly refuses a test record that describes a
different source tree. It must be re-recorded by actually running the suite, not
bypassed.

## Superseded by

The clean, isolated Arm A preflight run with no concurrent pytest, dataset
build, or other CPU-heavy task. Until that exists, there is no accepted Arm A
baseline.
