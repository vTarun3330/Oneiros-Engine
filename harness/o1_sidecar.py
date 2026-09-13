"""Append verified O1 completions as additional SFT examples. Nothing else.

This is deliberately a THIRD mechanism rather than a reuse of the two that
already exist, because both of those silently do something other than what an
oracle sidecar needs:

* ``apply_relearning_round`` oversamples the records a correction names and
  never reads the correction text. It appends the corpus's own training pair,
  so the model is trained on the GOLDEN completion while the run reports
  itself as having trained on corrections.
* ``MULTI_MUTANT_COMPLETIONS`` does override a completion, but only for records
  the bounded selection already chose, so it adds nothing and silently reaches
  a fraction of the batch - 76 of 800 pairs, the one time it was measured.

Either would produce an arm B that differs from arm A in ways nobody asked for
while under-delivering the thing being tested. So this module does one thing:
it builds additional SFT data points whose completion is the O1 text verbatim,
appends them to the baseline examples, and leaves every baseline example
untouched.

THE CONTRACT, in full:

1. Baseline examples are returned unchanged and in their original order. The
   O1 rows are appended. Nothing is dropped, reordered, reweighted or
   substituted.
2. The completion is the O1 ``completion`` string, byte for byte. A corpus
   golden is never substituted for it.
3. repeats is 1. A row whose completion already appears - in the baseline or
   in another O1 row - is dropped rather than repeated.
4. Every row must be train-only, present in the supplied pairs, and fit the
   token budget. A row failing any of these is refused, not trimmed.
5. An empty or absent sidecar returns the baseline unchanged, so arm B with no
   sidecar is arm A exactly.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

SCHEMA = "oneiros_o1_sft_sidecar_v1"


def _sha_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_o1_sidecar(directory: str | Path) -> tuple[list[dict[str, Any]],
                                                    dict[str, Any]]:
    """Read and validate a sidecar directory, refusing anything unverified."""
    directory = Path(directory)
    sidecar_path = directory / "train.sidecar.json"
    manifest_path = directory / "manifest.json"
    if not sidecar_path.exists() or not manifest_path.exists():
        raise FileNotFoundError(
            f"an O1 sidecar requires train.sidecar.json and manifest.json "
            f"under {directory}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA:
        raise ValueError(
            f"sidecar schema is {manifest.get('schema_version')!r}, "
            f"expected {SCHEMA!r}")

    actual = hashlib.sha256(sidecar_path.read_bytes()).hexdigest()
    if actual != manifest.get("sidecar_sha256"):
        raise ValueError(
            f"sidecar file hashes to {actual[:16]}... but its manifest records "
            f"{str(manifest.get('sidecar_sha256'))[:16]}...; the rows are not "
            "the rows that were verified")
    if manifest.get("evaluation_split") != "train":
        raise ValueError(
            f"sidecar split is {manifest.get('evaluation_split')!r}; only "
            "train-split supervision may be trained on")
    if manifest.get("sealed_final_test_accessed") is not False:
        raise ValueError("sidecar manifest does not deny sealed-test access")
    if manifest.get("all_records_in_train_shard") is not True:
        raise ValueError(
            "sidecar manifest does not assert that every record is in the "
            "train shard")

    rows = json.loads(sidecar_path.read_text(encoding="utf-8"))
    for row in rows:
        if not row.get("verified"):
            raise ValueError(
                f"{row.get('record_id')} is unverified; a failed generation is "
                "never a label")
        if not str(row.get("completion") or "").strip():
            raise ValueError(f"{row.get('record_id')} has an empty completion")
        if int(row.get("repeats", 1)) != 1:
            raise ValueError(
                f"{row.get('record_id')} asks for {row.get('repeats')} repeats; "
                "the O1 sidecar never repeats a row")
    return rows, manifest


def append_o1_examples(
    baseline_examples: Sequence[Any],
    rows: Iterable[dict[str, Any]],
    pairs_by_id: dict[str, dict[str, Any]],
    *,
    make_data_point: Callable[[dict[str, Any], str, str], Any],
    build_prompt: Callable[[dict[str, Any]], str],
    completion_fits: Callable[[dict[str, Any], str, str], bool] | None = None,
) -> tuple[list[Any], dict[str, Any]]:
    """Append O1 rows to the baseline. The baseline is never modified.

    Returns the combined examples and a report of exactly what happened, so a
    run cannot claim supervision it did not deliver.
    """
    baseline = list(baseline_examples)
    seen = {str(getattr(example, "completion", "")) for example in baseline}
    baseline_completion_count = len(seen)

    appended: list[Any] = []
    dropped: Counter = Counter()
    unreachable: list[str] = []
    rejected: list[dict[str, str]] = []
    offered = 0

    for row in rows:
        offered += 1
        record_id = str(row["record_id"])
        pair = pairs_by_id.get(record_id)
        if pair is None:
            unreachable.append(record_id)
            dropped["record_not_in_training_pairs"] += 1
            continue
        completion = str(row["completion"])
        if completion in seen:
            dropped["duplicate_completion"] += 1
            continue
        prompt = build_prompt(pair)
        if completion_fits is not None and not completion_fits(
                pair, prompt, completion):
            rejected.append({"record_id": record_id,
                             "reason": "exceeds the token budget"})
            dropped["token_budget"] += 1
            continue
        seen.add(completion)
        appended.append(make_data_point(pair, prompt, completion))

    combined = baseline + appended
    return combined, {
        "schema_version": SCHEMA,
        "baseline_examples": len(baseline),
        "baseline_unchanged": True,
        "sidecar_rows_offered": offered,
        "appended_examples": len(appended),
        "combined_examples": len(combined),
        "repeats_per_row": 1,
        "duplicated_rows": 0,
        "dropped": dict(dropped.most_common()),
        "records_not_in_training_pairs": unreachable[:20],
        "records_not_in_training_pairs_count": len(unreachable),
        "token_budget_rejections": rejected[:20],
        "baseline_distinct_completions": baseline_completion_count,
        "completion_source": "O1 verified completion text, verbatim",
        "golden_substituted": False,
        "appended_completion_sha256": [
            _sha_text(str(getattr(example, "completion", "")))
            for example in appended[:5]
        ],
    }
