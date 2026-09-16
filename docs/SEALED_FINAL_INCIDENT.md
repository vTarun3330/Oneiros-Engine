# Sealed final test — incident report (redacted)

**Status: `failed_after_authorization`. The sealed final test is consumed. It
produced zero sealed candidates and zero reportable metrics.**

This report is redacted. It contains no authorization token, no sealed record
id, no sealed source, test code or payload, and no sealed split membership.

---

## 1. Sequence

All times UTC, 2026-09-16.

| Time | Event |
|---|---|
| 18:11:36 | `authorization_issued` — one-time token minted against bundle `c16b13eb…`, registered with the guard |
| 18:11:37 | Entrypoint launched with the exact command recorded in the v6 receipt |
| ~18:11:49 | Base model loaded (`Qwen/Qwen2.5-Coder-1.5B-Instruct` @ `2e1fd397…`) |
| 18:11:50 | Pre-authorization smoke **passed** — 2 synthetic records, 16 candidate slots, `parse_mode=whole_output`, seed 42 applied, `sealed_data_touched=False` |
| 18:11:52 | `sealed_access_granted` — **token spent** |
| 18:11:52 | `SplitSchemaError` raised while loading the split. Run aborted in the same second |
| 18:11:52 | `results/sealed_final_run_state.json` written, `status: failed_after_authorization`, `may_never_run_again: true` |

Total wall clock from launch to exit: approximately 95 seconds, nearly all of it
model loading. The failure occurred within one second of the token being spent.

## 2. Identity

| | |
|---|---|
| Executable receipt | `results/v4_2_sealed_final_executable_receipt_v6.json` |
| Receipt SHA-256 | `4a2bc52ae1448e12ba31fd2ad68ff0b26b54dd9507456574ea66cb3936af7c8e` |
| Bundle SHA-256 | `c16b13eb6abc2bdab40cd2004c78902092faaf10fed82baa76506af78430d0a9` |
| Schema | `oneiros_sealed_final_readiness_v6` |
| Repository commit | `e66b6e2339ab36c728d04266f5647750070fd95c`, clean tree, equal to origin |
| Evaluator | `harness/sealed_final_evaluator.py`, `oneiros_sealed_final_evaluator_v1`, canonical `56869a08…` |
| Measurement logic | `776aab410ebd989fda7ef042b7dd98cd875c8e170f4e94b5ae9154dee14d6213` |

Receipt and bundle hashes were verified correct at run time. Nothing about the
frozen configuration was wrong, and nothing was edited after the freeze.

## 3. What did and did not happen on sealed targets

**No model generation occurred on any sealed target.** No prompt was rendered
from a sealed record, no sequence was generated, no candidate was parsed, and
no candidate was executed or scored. `run_state.result` is `{}`.

The only generation in the entire run was the pre-authorization smoke, which ran
on two **synthetic** records (`add_two`, `scale_by`) before the token was
presented. Its 16 retained raw outputs are synthetic and carry no sealed
content.

**The evaluator nevertheless loaded the corpus and traversed sealed records
before failing.** After the guard granted access, `sealed_records()` read the
corpus `records.json` and `splits.json` from disk, resolved every id in the
sealed split to its record, and iterated those records field by field to
validate them. That traversal is what raised. So sealed record *fields were
examined in memory*, even though no sealed content ever reached a prompt, a
model, or an artifact on disk.

This distinction is recorded precisely because it matters to any later judgment
about contamination, and it should not be softened in either direction: the
split was opened and traversed; it was not measured.

## 4. Root cause

**Category: the sealed loader's record-admission rule is stricter than the
evaluation path it exists to mirror.**

`harness/sealed_final_loader.py:92-99` requires all seven fields of
`REQUIRED_RECORD_FIELDS` — including `entry_point` and `specification` — to be
**non-empty** on *every* record in the split, and raises before returning any
record.

Repository-style records carry blank `entry_point` and, usually, blank
`specification`. Those fields are present as keys and empty by design; they are
not missing. The historical evaluator has always handled them by loading every
record in the split and *then* scoping the kill rate to
`execution_mode == FUNCTION_EXECUTION_MODE`
(`scripts/train_on_dataset.py:2102-2105`). Repository records are excluded from
every kill rate at that filter, which is exactly what the v6 receipt's own
`interpretation_limits` already stated.

The sealed loader inverted the order: validate everything, then load. So the
records the pipeline was always going to discard caused a refusal before the
discard could happen.

**The same rejection reproduces on permitted splits** — see
§6. It was therefore never a property of the sealed data. It was a property of
the loader, observable at any time, on data that was never sealed.

### Why it was not caught

Every test exercising `select_split_records` used a synthetic fixture or a
hand-built record list. None ran it against real corpus data containing
repository-style records. The pre-authorization smoke proved the *generator*
worked on two synthetic records; it did not prove the *loader* worked on a real
split. Six receipt generations, three independent audits and 1,516 passing tests
all examined the receipt and its bindings. The defect was one layer below, in
the component deliberately designed to be unreachable before the token — and
that design is precisely what kept it untested.

## 5. Policy consequence

**No retry on the existing sealed split.** The single authorization is spent,
`may_never_run_again` is `true`, and the split is recorded as opened. Re-running
the sealed evaluator against this split is prohibited regardless of the fact
that no measurement was produced.

Also prohibited, and not done: issuing a second token, deleting or editing the
guard state, modifying or regenerating any receipt, retraining, prompt or
threshold changes, re-scoring, and any further inspection of the sealed corpus.

The repository is unchanged at `e66b6e2` apart from this report. No source,
receipt, prompt or setting was altered in response to the failure.

## 6. Permitted-split reproduction (read-only, non-sealed)

Run against `val` and `ablation_dev` only. No sealed split was named, counted,
or loaded.

| | `val` | `ablation_dev` |
|---|---|---|
| ids in split | 781 | 594 |
| function-mode records | 757 | 542 |
| repository-mode records | 24 | 52 |
| records failing the loader check | **27** | **52** |
| blank `entry_point` | 24 | 52 |
| blank `specification` | 26 | 35 |
| keys genuinely absent | **0** | **0** |
| `select_split_records` | **REJECTED** | **REJECTED** |
| historical `_record_to_pair` | accepted all 781 | accepted all 594 |
| historical kill-rate scope | 757 functions (24 repository excluded) | 542 functions (52 repository excluded) |

The historical scope figures — **757** and **542** — are exactly the counts
reported by locked validation and by the four-arm development evaluation. The
permitted splits have always had this shape, and the established evaluator has
always handled it.

### A second finding, beyond repository records

In `val`, 3 of the 27 failing records are **function-mode** records with a blank
`specification`. They are inside the kill-rate scope and were measured by locked
validation.

Breakdown of failing records by kind and blank field:

| split | kind | blank fields | count |
|---|---|---|---|
| `val` | function | `specification` | **3** |
| `val` | repository | `entry_point`, `specification` | 23 |
| `val` | repository | `entry_point` | 1 |
| `ablation_dev` | repository | `entry_point`, `specification` | 35 |
| `ablation_dev` | repository | `entry_point` | 17 |

This matters: a fix that merely exempted repository-style records would still be
wrong. A blank `specification` is legitimate on scored function records too.

## 7. Preserved evidence

The guard's original files remain exactly where it wrote them, unmodified.
`results/**` is covered by `.gitignore`, so none of them is committed. They
contain the one-time token and the raw sealed-run state and **must not be
committed**.

Token-redacted, id-redacted copies were produced for quotation and handoff:

| File | SHA-256 |
|---|---|
| `sealed_final_state.REDACTED.json` | `f23402b9ed6f41ca22cc5b1b157aa5c91fe53630fa84a4e907ac87087f9caeb8` |
| `sealed_final_run_state.REDACTED.json` | `7520680143956904974966a3f8176812f5d5094044c9c6d743f2db212b2fc962` |
| `sealed_final_audit.REDACTED.log` | `b40113cf390e7fb289f2e205057de71b0bd1e225f6f80d70529cc971ab817fcc` |
| `sealed_final_run_console.REDACTED.log` | `03d17bc75dff5295e7f3cc4c5a18eec421a5edba2db894f6f20a65dc95085b0e` |
| `permitted_split_reproduction.log` | `0f44e9ce85d8a88d809e6f8677dcebfd5951e12dfefdb1b4658cbcb38ee6baf3` |

**Durable location: `docs/evidence/sealed_final_incident/`**, committed. Each
file was verified before staging to contain no authorization token, no sealed
record identifier, no sealed source, test code, payload or split membership, and
to hash to the value recorded above. The three `.log` files are covered by the
`*.log` rule in `.gitignore` and were force-added deliberately. A
`.gitattributes` in that directory sets `* -text`, so Git stores these files
byte-for-byte: without it, line-ending normalisation would make a fresh clone
fail the hash check this report asks a reader to perform.

They were produced in the session scratchpad at
`…/531cd266-1819-47da-8691-f9ba27ee2a33/scratchpad/sealed_final_evidence/redacted/`,
which is session-scoped; the committed copies are the retained record.

One residual disclosure, recorded rather than removed: the failure message
preserved in `sealed_final_run_console.REDACTED.log` and in the `error` field of
`sealed_final_run_state.REDACTED.json` states how many records failed the
admission check. That is a count, not an identifier and not membership, and it
is kept because these files are the verbatim evidence of the run and their
hashes are cited above. It is noted here so the disclosure is deliberate and
visible rather than overlooked.

Unredacted originals, for reference only, at their guard-written paths:

| File | SHA-256 |
|---|---|
| `results/sealed_final_state.json` | `5b6def43edc0b5933b5cc8a9e2a3770fdf0c8c575de20bfbd60fb9f65b3bfcf7` |
| `results/sealed_final_run_state.json` | `85bc6f1489e8861b98e86af6eaf9e45f483ae58e2b156d068c8e65dbca640a5d` |
| `results/sealed_final_audit.log` | `b40113cf390e7fb289f2e205057de71b0bd1e225f6f80d70529cc971ab817fcc` |

## 8. What this contributes to the research result

**Nothing.**

The sealed final test contributes **no performance result**. There is no
Kill@1, Kill@2, Kill@4 or Kill@8, no validity metric, no reference-validity
figure, and no artifact hash of any measurement, because no measurement exists.

It supports **no Oneiros claim** and **no Atheris comparison**. The run was
scoped to the immutable base model only, with no adapter and no bundled
baseline, so even a successful run could not have supported a comparative claim
against Atheris or any other baseline. A failed run supports less than that.

The project's standing empirical result is unchanged and rests where it already
rested: locked validation on `val`, which retained the base model.

## 9. Standing instruction

The sealed split is consumed. Do not present any receipt to
`scripts/run_sealed_final_test.py` for this split, do not issue another token
against bundle `c16b13eb…`, and do not delete or edit the guard state files —
they are the record that this happened.

Any future final measurement requires a **new, independently constructed final
set** and an explicit decision by the project owner. No such set has been
created, proposed for creation, or evaluated.
