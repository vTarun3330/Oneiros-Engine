# Future-maintenance backlog

Known issues that are **not** being fixed now, with the reason. Nothing here is
a correctness defect in a recorded result; anything that were would be repaired
rather than listed.

An item lands here when fixing it would change a frozen artifact for a cosmetic
gain. Freezing has a cost, and paying it to improve a console message is a bad
trade — but so is forgetting the issue exists.

---

## 1. The rehearsal completion banner does not print Kill@k

**Status:** open, cosmetic, console only. **Do not fix in place.**

`scripts/run_rehearsal_evaluation.py` ends with a summary block that loops over
`k` and looks for a top-level `kill_at_{k}` key:

```python
for k in (1, 2, 4, 8):
    key = f"kill_at_{k}"
    if key in artifact:
        print(f"  Kill@{k:<2}: {artifact[key]:.6f}")
```

The scorer emits a single nested `kill_at_k` mapping instead, keyed by the
stringified k with `functions`, `rate` and `wilson_95` under each. So the lookup
never matches and the banner silently prints no rates. The
2026-09-17 rehearsal printed only `targets`, `runtime` and `raw ok`.

**Impact: display only.** Every value is present and correct in
`rehearsal_result.json` and in
`results/v4_2_rehearsal_execution_receipt.json`. No number was wrong, lost or
misreported — the console was quiet where it should have been informative.

**Why it is not being fixed now.** The runner is bound by hash into
`results/v4_2_rehearsal_receipt.json` (`bae9510218641b67…`). Editing it changes
its canonical hash, which invalidates that receipt, which would require
regenerating the receipt — and the completed rehearsal is frozen against the
receipt as it stands. Rewriting a frozen artifact to improve a print statement
inverts the priority that the whole sealed-final incident was about.

**When to fix it.** Whenever the rehearsal runner is next changed for a
substantive reason, and a new receipt is generated anyway. Then:

- read the nested shape (`artifact["kill_at_k"][str(k)]["rate"]`);
- add a test asserting the banner prints a rate for each k, so a shape change
  cannot silence it again — the existing tests checked the artifact, which is
  why this survived.

**Do not** rerun the 2026-09-17 rehearsal to get a prettier log. It completed,
its artifact is intact, and its numbers are recorded.

---

## 2. `verify_corpus` loads every split before scoping

**Status:** open, noted during the sealed-final post-mortem.

`harness/corpus.py:verify_corpus` hashes and deserializes the combined
`records.json` and then checks all splits; `load_corpus_split` likewise loads
the combined records before selecting ids. A train/ablation_dev flag therefore
does not, by itself, prevent a held-out split's records being resident.

This is the same load-everything-then-scope shape that
`harness/evaluation_admission.py` was written to replace on the evaluation side.
It is lower-risk now that the consumed split is refused unconditionally at the
loader, but it remains the wrong default for any future held-out set.

Detail in item 4 of [`LOCAL_WINDOWS_GPU.md`](LOCAL_WINDOWS_GPU.md).
