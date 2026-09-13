"""Choose a deterministic, stratified probe before spending an hour re-scoring.

Two populations are selected, and the control is the point:

* AFFECTED - candidates whose original status was an execution-harness
  failure. If the transient hypothesis holds, these re-score cleanly.
* CONTROL - candidates that already scored successfully. Re-scoring them must
  return the SAME semantic outcome. If controls disagree, the executor is not
  merely flaky under load; it is non-deterministic, and repairing the affected
  set would launder noise into the dataset.

Without controls a probe can only show that re-scoring produces answers. It
cannot show the answers are right.

Stratification spans source dataset, bug family, complexity tier, and - because
the failures were bimodal per function - whether the function lost all of its
candidates or only some. Selection is by sorted hash of the candidate identity,
so the same probe is chosen every time and cannot be nudged.

Reads only the hash-verified train development shard.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness.corpus import write_json
from harness.corpus_view import load_development_split
from scripts.rescore_harness_failures import HARNESS_STATUSES

CONTROL_STATUSES = ("pass", "assertion_error")


def _key_hash(record_id: str, rank: int, position: int) -> str:
    return hashlib.sha256(
        f"{record_id}|{rank}|{position}".encode("utf-8")).hexdigest()


def _tier(record: dict[str, Any]) -> str:
    quality = record.get("quality")
    if isinstance(quality, dict):
        tier = quality.get("complexity_tier") or quality.get("tier")
        if tier:
            return str(tier)
    reference = str(record.get("reference_code") or "")
    lines = len([line for line in reference.splitlines() if line.strip()])
    return "simple" if lines <= 6 else "moderate" if lines <= 14 else "complex"


def build(artifact: Path, corpus_dir: Path, affected_size: int,
          control_size: int) -> dict[str, Any]:
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    if payload.get("evaluation_split") != "train":
        raise SystemExit("probe may only be drawn from the train split")

    records = {str(r["id"]): r for r in load_development_split(
        corpus_dir, "train", include_excluded=True)}

    affected: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    control: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    strata_seen: Counter = Counter()

    for result in payload.get("function_results") or []:
        record_id = str(result.get("record_id"))
        record = records.get(record_id)
        if record is None:
            continue
        outcomes = result.get("candidate_outcomes") or []
        failures = sum(1 for o in outcomes
                       if str(o.get("reference_status")) in HARNESS_STATUSES)
        # The failures were bimodal per function; both modes must be probed.
        mode = ("all_eight_failed" if failures == len(outcomes) and outcomes
                else "partially_failed" if failures else "clean")
        dataset = str(result.get("dataset_name") or "unknown")
        family = str(result.get("bug_family") or "unknown")
        tier = _tier(record)

        for position, outcome in enumerate(outcomes):
            status = str(outcome.get("reference_status") or "")
            entry = {
                "record_id": record_id,
                "rank": int(outcome.get("rank") or 0),
                "position": position,
                "dataset": dataset, "bug_family": family,
                "complexity_tier": tier, "function_mode": mode,
                "original_status": status,
                "original_killed": bool(outcome.get("killed")),
                "original_reference_valid": bool(outcome.get("reference_valid")),
            }
            stratum = (dataset, family, tier, mode)
            if status in HARNESS_STATUSES:
                affected[stratum].append(entry)
                strata_seen[stratum] += 1
            elif status in CONTROL_STATUSES:
                control[(dataset, family, tier)].append(entry)

    def draw(buckets: dict[tuple, list[dict[str, Any]]], total: int):
        """Round-robin across strata, deterministic within each."""
        for entries in buckets.values():
            entries.sort(key=lambda e: _key_hash(
                e["record_id"], e["rank"], e["position"]))
        ordered = [buckets[k] for k in sorted(buckets)]
        chosen: list[dict[str, Any]] = []
        depth = 0
        while len(chosen) < total and ordered:
            progressed = False
            for entries in ordered:
                if depth < len(entries) and len(chosen) < total:
                    chosen.append(entries[depth])
                    progressed = True
            if not progressed:
                break
            depth += 1
        return chosen

    affected_probe = draw(affected, affected_size)
    control_probe = draw(control, control_size)

    selection_hash = hashlib.sha256(json.dumps(
        [[e["record_id"], e["rank"], e["position"]]
         for e in affected_probe + control_probe],
        sort_keys=True).encode("utf-8")).hexdigest()

    def spread(entries, key):
        return dict(Counter(e[key] for e in entries).most_common())

    return {
        "schema_version": "oneiros_rescore_probe_v1",
        "artifact": artifact.as_posix().split("results/", 1)[-1],
        "sealed_final_test_accessed": False,
        "record_source": "hash-verified development view train shard",
        "selection_hash": selection_hash,
        "selection_rule": (
            "round-robin across strata, ordered within a stratum by sha256 of "
            "record_id|rank|position - deterministic and not nudgeable"),
        "affected_probe_size": len(affected_probe),
        "control_probe_size": len(control_probe),
        "affected_strata_available": len(affected),
        "affected_spread": {
            "dataset": spread(affected_probe, "dataset"),
            "complexity_tier": spread(affected_probe, "complexity_tier"),
            "function_mode": spread(affected_probe, "function_mode"),
            "bug_family": spread(affected_probe, "bug_family"),
        },
        "control_spread": {
            "dataset": spread(control_probe, "dataset"),
            "complexity_tier": spread(control_probe, "complexity_tier"),
            "original_status": spread(control_probe, "original_status"),
        },
        "candidate_keys": [[e["record_id"], e["rank"], e["position"]]
                           for e in affected_probe],
        "control_keys": [[e["record_id"], e["rank"], e["position"]]
                         for e in control_probe],
        "control_expected": {
            f'{e["record_id"]}|{e["rank"]}|{e["position"]}': {
                "reference_status": e["original_status"],
                "killed": e["original_killed"],
                "reference_valid": e["original_reference_valid"],
            } for e in control_probe},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--corpus", type=Path,
                        default=ROOT / "data" / "corpus" / "v4_1_research_hardened_candidate")
    parser.add_argument("--affected", type=int, default=600)
    parser.add_argument("--control", type=int, default=400)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    probe = build(arguments.artifact, arguments.corpus, arguments.affected,
                  arguments.control)
    write_json(arguments.output, probe)
    print(json.dumps({k: v for k, v in probe.items()
                      if k not in ("candidate_keys", "control_keys",
                                   "control_expected")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
