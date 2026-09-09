"""Emit the single-assertion view in the schema the trainer already loads.

The trainer keys verified supervision by every sibling mutant a completion
actually kills, not by the record it is displayed against. That distinction
matters: keying by the displayed record alone covered 663 train records
instead of 5588, and turned a "multi-mutant" run into 96% ordinary
supervision.

A single assertion kills FEWER siblings than the three-assertion test function
it was chosen from, so inheriting the original sibling list would claim kills
this completion does not make. Every sibling is therefore re-executed against
the chosen assertion, and the emitted view records only what was observed.

This is where the single-assertion decision costs something, and the cost is
measured here rather than discovered during training.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness.corpus import write_json
from harness.parallel_execution import progress_map
from harness.safe_execution import classify_assertions


def _verify_siblings(job: dict[str, Any]) -> dict[str, Any]:
    """Which of this lineage's siblings the single assertion actually kills."""
    assertion = job["completion"]
    reference = job["reference"]
    killed: list[str] = []
    survived: list[str] = []
    for sibling_id, mutant in job["siblings"]:
        rows = classify_assertions([assertion], reference, mutant, job["timeout"])
        row = rows[0] if rows else {}
        if not row.get("valid"):
            # Invalid on the reference would have been caught earlier; if it
            # shows up here the completion is not usable at all.
            return {**job, "killed": [], "survived": [], "invalid": True}
        (killed if row.get("killed") else survived).append(sibling_id)
    return {**job, "killed": killed, "survived": survived, "invalid": False}


def build(view_path: Path, source_view: Path, corpus_dir: Path,
          timeout: float) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    view = json.loads(view_path.read_text(encoding="utf-8"))
    payload = json.loads((source_view / "train.examples.json").read_text(encoding="utf-8"))
    examples = payload["examples"] if isinstance(payload, dict) and "examples" in payload else payload
    by_lineage = {str(e["lineage"]): e for e in examples}
    records = {
        str(r["id"]): r for r in
        json.loads((corpus_dir / "records.json").read_text(encoding="utf-8"))
    }

    jobs = []
    for item in view["items"]:
        original = by_lineage.get(str(item["lineage"]))
        if original is None:
            continue
        displayed = records.get(str(item["displayed_record_id"])) or {}
        support = str(displayed.get("support_context") or "")
        reference = support + "\n" + str(displayed.get("reference_code") or "")
        siblings = []
        for sibling_id in original.get("sibling_mutant_ids") or []:
            sibling = records.get(str(sibling_id))
            if sibling:
                siblings.append(
                    (str(sibling_id), support + "\n" + str(sibling.get("code_under_test") or "")))
        jobs.append({
            "lineage": str(item["lineage"]),
            "displayed_record_id": str(item["displayed_record_id"]),
            "entry_point": item["entry_point"],
            "source_dataset": item["source_dataset"],
            "tier": item["tier"],
            "block": item["block"],
            "completion": item["completion"],
            "reference": reference,
            "siblings": siblings,
            "original_killed": int(original.get("mutants_killed") or 0),
            "timeout": timeout,
        })

    verified = progress_map(_verify_siblings, jobs, "sibling verification", every=50)
    usable = [v for v in verified if not v["invalid"]]

    emitted = [{
        "lineage": v["lineage"],
        "entry_point": v["entry_point"],
        "displayed_record_id": v["displayed_record_id"],
        "source_dataset": v["source_dataset"],
        "completion": v["completion"],
        "assertion_count": 1,
        "assertions": [v["completion"]],
        "primary_mutation_family": by_lineage[v["lineage"]].get("primary_mutation_family"),
        "sibling_mutant_ids": list(v["killed"]) + list(v["survived"]),
        "mutants_evaluated": len(v["killed"]) + len(v["survived"]),
        "mutants_killed": len(v["killed"]),
        "surviving_mutant_ids": v["survived"],
        "killed_mutant_ids": v["killed"],
        "kills_displayed_target": v["displayed_record_id"] in v["killed"],
        "verified": True,
        "verification_status": "verified",
        "curriculum_block": v["block"],
        "curriculum_tier": v["tier"],
    } for v in usable]

    kept = [e for e in emitted if e["kills_displayed_target"]]
    before = sum(v["original_killed"] for v in usable)
    after = sum(len(v["killed"]) for v in usable)

    report = {
        "schema_version": "oneiros_single_assertion_training_view_v1",
        "sealed_final_test_accessed": False,
        "lineages_in": len(jobs),
        "lineages_usable": len(usable),
        "lineages_emitted": len(kept),
        "dropped_not_killing_displayed_target": len(emitted) - len(kept),
        "sibling_kills_multi_assertion": before,
        "sibling_kills_single_assertion": after,
        "sibling_kill_retention": round(after / before, 4) if before else None,
        "mean_siblings_killed": round(after / max(len(usable), 1), 2),
        "tier_counts": dict(Counter(e["curriculum_tier"] for e in kept)),
        "cost_of_the_decision": (
            "a single assertion distinguishes fewer siblings than the test "
            "function it was chosen from. The retention figure is what the "
            "fewer-surer-assertions decision costs in supervision breadth, "
            "measured rather than assumed."
        ),
    }
    return kept, report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--view", type=Path,
                        default=ROOT / "data" / "training_views" / "regularised_v1" / "view.json")
    parser.add_argument("--source-view", type=Path,
                        default=ROOT / "data" / "training_views" / "multi_mutant_v1")
    parser.add_argument("--corpus", type=Path,
                        default=ROOT / "data" / "corpus" / "v4_1_research_hardened_candidate")
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "data" / "training_views" / "single_assertion_curriculum_v1")
    arguments = parser.parse_args()

    examples, report = build(arguments.view, arguments.source_view,
                             arguments.corpus, arguments.timeout)
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(arguments.output_dir / "train.examples.json", examples)
    write_json(arguments.output_dir / "train.manifest.json", report)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
