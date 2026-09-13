"""Emit the O1 positives as a bounded, no-repeat SFT sidecar. CPU only.

WHY THIS IS NOT JUST A FILE COPY.

The trainer already has two injection paths and NEITHER does what an oracle
sidecar needs.

* ``--relearning-dataset`` oversamples the *records* a correction names and
  throws the correction text away: ``apply_relearning_round`` looks the record
  up in the ordinary train pairs and appends that pair. The completion is never
  read. Routed through it, O1's verified corrected assertions would contribute
  nothing but extra weight on the corpus's own goldens.

* ``--multi-mutant-dataset`` does override the completion, but only for records
  the bounded selection *already chose*. It adds nothing. That exact failure is
  recorded in the trainer: keying by displayed record covered 663 train records
  instead of 5,588, and only 76 of 800 selected pairs received multi-mutant
  supervision - "a multi-mutant run that was 96% ordinary supervision".

O1 is keyed by displayed record and cannot be keyed any other way: each
positive was executed against the reference and its own displayed mutant, not
against siblings. Mapping it onto siblings would be an unverified label. So the
sidecar carries both halves - the records to force in, and the completion to
use for each - and this script reports the coverage arithmetic instead of
letting a run discover it afterwards.

NOTHING IS REPEATED. ``repeats`` is 1 on every row. The bound is enforced by
subsampling proportionally within each source stratum, so the O1 share caps
survive the subsample rather than being re-derived from a truncated head.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness.corpus_view import load_development_split

SCHEMA = "oneiros_o1_sft_sidecar_v1"

#: A cap the sidecar may never exceed, whatever ratio is requested. O1 is an
#: auxiliary component; a sidecar past this share is a different experiment.
MAX_SIDECAR_SHARE = 0.35


def _sha_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _shares(rows: list[dict[str, Any]], key: str) -> dict[str, float]:
    total = len(rows) or 1
    return {value: round(count / total, 4)
            for value, count in Counter(r[key] for r in rows).most_common()}


def proportional_subsample(rows: list[dict[str, Any]], keep: int, key: str
                           ) -> list[dict[str, Any]]:
    """Keep ``keep`` rows while holding each ``key`` stratum's share.

    Largest-remainder allocation, then a deterministic pick inside each
    stratum. Truncating a sorted list instead would silently rewrite the source
    shares the O1 build was capped to produce.
    """
    if keep >= len(rows):
        return list(rows)
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[row[key]].append(row)
    for bucket in buckets.values():
        bucket.sort(key=lambda r: _sha_text(r["sft_target"]))

    exact = {value: len(bucket) * keep / len(rows)
             for value, bucket in buckets.items()}
    quota = {value: int(share) for value, share in exact.items()}
    # Largest remainder, ties broken by name so the result is reproducible.
    remaining = keep - sum(quota.values())
    for value, _ in sorted(exact.items(),
                           key=lambda kv: (-(kv[1] - int(kv[1])), kv[0])):
        if remaining <= 0:
            break
        if quota[value] < len(buckets[value]):
            quota[value] += 1
            remaining -= 1

    out: list[dict[str, Any]] = []
    for value in sorted(buckets):
        out.extend(buckets[value][:quota[value]])
    out.sort(key=lambda r: (r["record_id"], r["candidate_position"]))
    return out


def build(dataset_dir: Path, corpus_dir: Path, baseline_pairs: int,
          ratio: float,
          completion_tokens: Callable[[str], int] | None = None,
          max_completion_tokens: int | None = None) -> dict[str, Any]:
    manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    positives_path = dataset_dir / "positives.json"

    problems: list[str] = []
    actual = _sha_file(positives_path)
    if actual != manifest.get("positives_sha256"):
        problems.append(
            f"positives.json hashes to {actual[:16]}... but its manifest "
            f"records {str(manifest.get('positives_sha256'))[:16]}...")
    if manifest.get("evaluation_split") != "train":
        problems.append(f"dataset split is {manifest.get('evaluation_split')!r}")
    if manifest.get("sealed_final_test_accessed") is not False:
        problems.append("dataset manifest does not deny sealed-test access")
    if manifest.get("canonical_records_json_opened") is not False:
        problems.append("dataset manifest does not deny canonical records.json")
    if ratio > MAX_SIDECAR_SHARE:
        problems.append(
            f"requested sidecar share {ratio} exceeds the auxiliary ceiling "
            f"{MAX_SIDECAR_SHARE}")

    rows = json.loads(positives_path.read_text(encoding="utf-8"))
    for row in rows:
        if not str(row.get("sft_target") or "").strip():
            problems.append(f"{row.get('record_id')} has an empty sft_target")
        if row.get("supervision_role") not in (
                "positive_original", "positive_verified_correction"):
            problems.append(
                f"{row.get('record_id')} is not a positive role: "
                f"{row.get('supervision_role')!r}")

    # Train-only. The canonical records.json is never opened.
    train_ids = {str(r["id"]) for r in load_development_split(
        corpus_dir, "train", include_excluded=True)}
    outside = sorted({r["record_id"] for r in rows} - train_ids)
    if outside:
        problems.append(
            f"{len(outside)} sidecar records are outside the train shard, "
            f"first: {outside[:3]}")

    # Token budget FIRST, then subsample. Filtering afterwards would make the
    # achieved ratio depend on whether the draw happened to include an
    # over-budget row, so the same requested ratio could yield a different
    # count on a different day. Nothing is truncated to fit: a truncated
    # assertion is not the assertion that was verified.
    eligible = rows
    over_budget: list[dict[str, Any]] = []
    if completion_tokens is not None and max_completion_tokens:
        measured = [(row, completion_tokens(row["sft_target"])) for row in rows]
        over_budget = [{"record_id": row["record_id"], "tokens": count}
                       for row, count in measured
                       if count > max_completion_tokens]
        eligible = [row for row, count in measured
                    if count <= max_completion_tokens]

    # keep / (baseline + keep) = ratio
    keep = min(len(eligible), int(round(ratio * baseline_pairs / (1 - ratio))))
    selected = proportional_subsample(eligible, keep, "source_dataset")

    by_record: dict[str, list[str]] = defaultdict(list)
    for row in selected:
        by_record[row["record_id"]].append(row["sft_target"])

    sidecar = [{
        "record_id": row["record_id"],
        "completion": row["sft_target"],
        "completion_shape": row["candidate_shape"],
        "candidate_position": row["candidate_position"],
        "supervision_role": row["supervision_role"],
        "label": row["label"],
        "source_dataset": row["source_dataset"],
        "bug_family": row["bug_family"],
        "complexity_tier": row["complexity_tier"],
        "origin": row["origin"],
        "function_lineage": row["function_lineage"],
        "repeats": 1,
        "verified": True,
        "kills_displayed_target": True,
        "verification": (
            "executed against the reference and the displayed mutant at O1 "
            "build time; never extended to sibling mutants, which this "
            "candidate was not executed against"),
    } for row in selected]

    return {
        "problems": problems,
        "sidecar": sidecar,
        "report": {
            "schema_version": SCHEMA,
            "source_dataset_dir": dataset_dir.name,
            "source_positives_sha256": actual,
            "source_candidates_sha256": manifest.get("candidates_sha256"),
            "source_derived_artifact_sha256": manifest.get("source_derived_sha256"),
            "evaluation_split": "train",
            "sealed_final_test_accessed": False,
            "canonical_records_json_opened": False,
            "all_records_in_train_shard": not outside,
            "available_positives": len(rows),
            "token_budget_eligible_positives": len(eligible),
            "max_completion_tokens": max_completion_tokens,
            "dropped_over_token_budget": len(over_budget),
            "dropped_over_token_budget_examples": over_budget[:5],
            "completions_truncated": 0,
            "unused_verified_positives": len(rows) - len(selected),
            "requested_sidecar_share": ratio,
            "baseline_pairs": baseline_pairs,
            "sidecar_rows": len(sidecar),
            "achieved_sidecar_share": round(
                len(sidecar) / (baseline_pairs + len(sidecar)), 4)
            if baseline_pairs + len(sidecar) else 0.0,
            "auxiliary_ceiling": MAX_SIDECAR_SHARE,
            "repeats_per_row": 1,
            "duplicated_rows": 0,
            "unique_completions": len({r["completion"] for r in sidecar}),
            "distinct_records": len(by_record),
            "distinct_lineages": len({r["function_lineage"] for r in sidecar}),
            "max_completions_for_one_record": max(
                (len(v) for v in by_record.values()), default=0),
            "shares": {
                dimension: _shares(sidecar, dimension) for dimension in
                ("source_dataset", "bug_family", "complexity_tier",
                 "origin", "supervision_role")
            },
            "real_repository_rows": sum(
                1 for r in sidecar if r["origin"] == "real_repository"),
            "subsample_method": (
                "largest-remainder proportional allocation within each "
                "source_dataset stratum, deterministic by completion hash. "
                "Shares are preserved rather than re-derived from a truncated "
                "head."),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path,
                        default=ROOT / "results" / "v4_2_oracle_dataset_full")
    parser.add_argument("--corpus", type=Path,
                        default=ROOT / "data" / "corpus"
                        / "v4_1_research_hardened_candidate")
    parser.add_argument("--baseline-pairs", type=int, required=True,
                        help="arm A's selected training pair count")
    parser.add_argument("--ratio", type=float, default=0.20,
                        help="sidecar share of the combined arm-B mixture")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-completion-tokens", type=int, default=128,
                        help="the arm's function-mode completion budget")
    parser.add_argument("--model-name",
                        default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    parser.add_argument("--model-revision", default=None,
                        help="defaults to the pinned immutable snapshot")
    arguments = parser.parse_args()

    from config import immutable_revision_for
    revision = arguments.model_revision or immutable_revision_for(
        arguments.model_name)
    if not revision or revision == "main":
        print(f"REFUSED: {arguments.model_name} has no immutable revision; a "
              "branch pointer cannot define which tokens were counted.")
        return 1
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        arguments.model_name, revision=revision, trust_remote_code=True,
        local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def _tokens(text: str) -> int:
        return len(tokenizer(text.strip() + tokenizer.eos_token,
                             add_special_tokens=False)["input_ids"])

    result = build(arguments.dataset_dir, arguments.corpus,
                   arguments.baseline_pairs, arguments.ratio,
                   completion_tokens=_tokens,
                   max_completion_tokens=arguments.max_completion_tokens)
    if result["problems"]:
        print("REFUSED: the O1 sidecar cannot be emitted.")
        for problem in result["problems"]:
            print("  - " + problem)
        return 1

    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    sidecar_path = arguments.output_dir / "train.sidecar.json"
    sidecar_path.write_text(
        json.dumps(result["sidecar"], indent=2) + "\n", encoding="utf-8")
    report = dict(result["report"])
    report["tokenizer_model_name"] = arguments.model_name
    report["tokenizer_revision"] = revision
    report["sidecar_sha256"] = _sha_file(sidecar_path)
    (arguments.output_dir / "manifest.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
