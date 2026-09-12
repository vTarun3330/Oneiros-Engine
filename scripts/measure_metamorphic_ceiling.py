"""How many unkilled targets could be killed WITHOUT predicting a value?

This is the decisive measurement for what to build next. The bottleneck is
oracle prediction, and mbpp's one-sentence specification makes the correct
value unknowable. A metamorphic assertion relates two calls instead of naming
a value, so it sidesteps the problem rather than solving it - but only if such
assertions actually exist for these targets, and in quantity.

The procedure, for every target the arm FAILED:

1. take the argument tuples the model itself already passed to the function -
   73% of them probe an input where reference and mutant differ, so the inputs
   are the part of the candidate that is mostly right;
2. propose value-free relations blind, from argument shape alone;
3. require each proposal to pass the FROZEN candidate policy, so the ceiling is
   one the existing pipeline could actually reach;
4. execute against reference and mutant: reference-valid AND kills, which is
   the same bar every other candidate in this project is held to.

A relation is never chosen by looking at the reference. Proposals are made
blind and then verified, so this measures existence, not hindsight.

The number is a CEILING for a value-free strategy: it assumes the model would
emit the right relation, which no model does for free. Its use is the other
direction - if the ceiling is small, this direction is dead and the project
should stop chasing kill rate.
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

from harness.candidate_policy import validate_generated_test
from harness.corpus import write_json
from harness.metamorphic_relations import (
    RELATION_NAMES, extract_argument_tuples, propose,
)
from harness.parallel_execution import progress_map
from harness.safe_execution import classify_assertions

#: Cap per target so one function with many candidates cannot dominate the
#: runtime. Proposals are ordered by the relation list, not by hope.
MAX_PROPOSALS_PER_TARGET = 60

#: Relations that state something about the function's SEMANTICS. The two left
#: out are deliberately excluded from the strict tier:
#:
#: * ``self_consistency_on_copy`` (``f(x) == f(x)``) asserts nothing. It kills
#:   only when the mutant crashes or is non-deterministic on that input, so it
#:   is a crash-finder wearing a relation's clothes.
#: * ``type_preservation`` (``type(f(x)) == type(x)``) is a hypothesis a
#:   developer would rarely write and holds or fails largely by coincidence.
#:
#: Both are still counted in the headline, because they are reference-valid
#: killing tests under the same rule as everything else in this project. The
#: strict tier exists so the headline cannot rest on them.
SEMANTIC_RELATIONS = frozenset({
    "argument_permutation", "idempotence", "involution",
    "list_order_invariance", "list_reversal_invariance", "length_preservation",
})


def _evaluate(job: dict[str, Any]) -> dict[str, Any]:
    """Which proposals for this target are reference-valid and kill."""
    killing: list[dict[str, Any]] = []
    valid_count = 0
    for proposal in job["proposals"]:
        rows = classify_assertions([proposal["code"]], job["reference"],
                                   job["mutant"], job["timeout"])
        row = rows[0] if rows else {}
        if not row.get("valid"):
            continue
        valid_count += 1
        if row.get("killed"):
            # How the mutant failed matters. A relation that comes out FALSE on
            # the mutant is evidence about its behaviour; one that makes the
            # mutant crash is weaker - the same kill could often be had from a
            # malformed call. Reported separately rather than pooled.
            error = str((row.get("mutant") or {}).get("error") or "")
            exception = error.split(":", 1)[0].strip()
            killing.append({
                **proposal,
                "kill_mode": ("relation_false" if exception == "AssertionError"
                              else "mutant_raised"),
                "mutant_exception": exception,
            })
    return {
        "record_id": job["record_id"],
        "dataset": job["dataset"],
        "entry_point": job["entry_point"],
        "proposals": len(job["proposals"]),
        "reference_valid": valid_count,
        "killing": killing,
    }


def build(artifact: Path, corpus_dir: Path, benchmark: str | None,
          timeout: float, limit: int | None) -> dict[str, Any]:
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    if payload.get("final_test_measurement") or payload.get("evaluation_split") == "test":
        raise SystemExit(str(artifact) + " is sealed final-test data; refusing")

    records = json.loads((corpus_dir / "records.json").read_text(encoding="utf-8"))
    by_id = {str(record["id"]): record for record in records}

    jobs: list[dict[str, Any]] = []
    panel: Counter = Counter()
    unkilled: Counter = Counter()
    rejected_by_policy = 0
    targets_without_arguments = 0

    for result in payload.get("function_results") or []:
        dataset = str(result.get("dataset_name") or "unknown")
        if benchmark and dataset != benchmark:
            continue
        panel[dataset] += 1
        if result.get("killed"):
            continue
        unkilled[dataset] += 1
        record = by_id.get(str(result.get("record_id")))
        if record is None:
            continue
        entry = str(record.get("entry_point") or "")
        if not entry:
            continue
        support = str(record.get("support_context") or "")
        reference = support + "\n" + str(record.get("reference_code") or "")
        mutant = support + "\n" + str(record.get("code_under_test") or "")

        tuples: list[list[str]] = []
        seen: set[tuple[str, ...]] = set()
        for outcome in result.get("candidate_outcomes") or []:
            for arguments in extract_argument_tuples(
                    str(outcome.get("code") or ""), entry):
                key = tuple(arguments)
                if key not in seen:
                    seen.add(key)
                    tuples.append(arguments)
        if not tuples:
            targets_without_arguments += 1
            continue

        proposals: list[dict[str, Any]] = []
        for arguments in tuples:
            for proposal in propose(entry, arguments):
                # The ceiling must be reachable by the existing pipeline, so a
                # proposal the frozen policy would reject does not count.
                if not validate_generated_test(proposal["code"], entry, True).valid:
                    rejected_by_policy += 1
                    continue
                proposals.append(proposal)
                if len(proposals) >= MAX_PROPOSALS_PER_TARGET:
                    break
            if len(proposals) >= MAX_PROPOSALS_PER_TARGET:
                break
        if not proposals:
            continue

        jobs.append({
            "record_id": str(result.get("record_id")),
            "dataset": dataset,
            "entry_point": entry,
            "reference": reference,
            "mutant": mutant,
            "proposals": proposals,
            "timeout": timeout,
        })
        if limit is not None and len(jobs) >= limit:
            break

    evaluated = progress_map(_evaluate, jobs, "metamorphic proposals", every=25)

    per_dataset: dict[str, Any] = {}
    for dataset in sorted(panel):
        rows = [row for row in evaluated if row["dataset"] == dataset]
        recovered = [row for row in rows if row["killing"]]
        families: Counter = Counter()
        for row in recovered:
            # Count each target once per family, so one target contributing
            # twenty permutations does not look like twenty successes.
            for family in {item["relation"] for item in row["killing"]}:
                families[family] += 1
        killed = panel[dataset] - unkilled[dataset]
        # A target counts as recovered "strongly" when at least one relation
        # came out FALSE on the mutant, rather than only making it crash.
        strong = [row for row in recovered
                  if any(item["kill_mode"] == "relation_false"
                         for item in row["killing"])]
        # Strictest tier: a semantic relation that came out false. This is the
        # number the direction should be judged on.
        strict = [row for row in recovered
                  if any(item["kill_mode"] == "relation_false"
                         and item["relation"] in SEMANTIC_RELATIONS
                         for item in row["killing"])]
        cross: Counter = Counter()
        for row in recovered:
            for item in row["killing"]:
                cross[(item["relation"], item["kill_mode"])] += 1
        per_dataset[dataset] = {
            "panel": panel[dataset],
            "killed_by_the_arm": killed,
            "kill_rate": round(killed / panel[dataset], 6) if panel[dataset] else None,
            "unkilled": unkilled[dataset],
            "unkilled_targets_examined": len(rows),
            "targets_recovered_by_a_value_free_assertion": len(recovered),
            "recovery_share_of_unkilled": round(
                len(recovered) / len(rows), 6) if rows else None,
            "ceiling_kill_rate": round(
                (killed + len(recovered)) / panel[dataset], 6)
            if panel[dataset] else None,
            "targets_recovered_by_a_false_relation": len(strong),
            "targets_recovered_only_by_making_the_mutant_crash":
                len(recovered) - len(strong),
            "conservative_ceiling_kill_rate": round(
                (killed + len(strong)) / panel[dataset], 6)
            if panel[dataset] else None,
            "targets_recovered_by_a_false_SEMANTIC_relation": len(strict),
            "strict_ceiling_kill_rate": round(
                (killed + len(strict)) / panel[dataset], 6)
            if panel[dataset] else None,
            "strict_gain_over_arm": round(len(strict) / panel[dataset], 6)
            if panel[dataset] else None,
            "relation_by_kill_mode": {
                relation + " / " + mode: count
                for (relation, mode), count in sorted(cross.items())
            },
            "relation_families_that_worked": dict(families.most_common()),
            "examples": [
                {"record_id": row["record_id"],
                 "entry_point": row["entry_point"],
                 "assertion": row["killing"][0]["code"],
                 "relation": row["killing"][0]["relation"]}
                for row in recovered[:12]
            ],
        }

    total_panel = sum(panel.values())
    total_killed = total_panel - sum(unkilled.values())
    total_recovered = sum(1 for row in evaluated if row["killing"])

    return {
        "schema_version": "oneiros_metamorphic_ceiling_v1",
        "artifact": artifact.as_posix().split("results/", 1)[-1],
        "evaluation_split": payload.get("evaluation_split"),
        "sealed_final_test_accessed": False,
        "method": {
            "inputs": "argument tuples the model itself already used",
            "relations": list(RELATION_NAMES),
            "proposed": "blind, from argument shape; the reference is never consulted",
            "accepted": "must pass the frozen candidate policy, be reference-valid, and kill",
            "cap_per_target": MAX_PROPOSALS_PER_TARGET,
        },
        "proposals_rejected_by_frozen_policy": rejected_by_policy,
        "unkilled_targets_with_no_usable_argument_tuple": targets_without_arguments,
        "panel": total_panel,
        "kill_rate": round(total_killed / total_panel, 6) if total_panel else None,
        "targets_recovered": total_recovered,
        "ceiling_kill_rate": round(
            (total_killed + total_recovered) / total_panel, 6) if total_panel else None,
        "interpretation": (
            "a CEILING for a value-free strategy. It assumes the model emits "
            "the right relation, which no model does for free. If the ceiling "
            "is close to the current kill rate, this direction is dead."
        ),
        "per_benchmark": per_dataset,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--corpus", type=Path,
                        default=ROOT / "data" / "corpus" / "v4_1_research_hardened_candidate")
    parser.add_argument("--benchmark", default=None)
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    report = build(arguments.artifact, arguments.corpus, arguments.benchmark,
                   arguments.timeout, arguments.limit)
    write_json(arguments.output, report)
    print(json.dumps({k: v for k, v in report.items()
                      if k not in ("per_benchmark", "method")}, indent=2))
    for dataset, block in report["per_benchmark"].items():
        print("")
        print(dataset + ":")
        print("  arm kill rate        = " + str(block["kill_rate"]))
        print("  unkilled examined    = " + str(block["unkilled_targets_examined"]))
        print("  recovered value-free = "
              + str(block["targets_recovered_by_a_value_free_assertion"])
              + "  (" + str(block["recovery_share_of_unkilled"]) + " of unkilled)")
        print("  of which relation-false = "
              + str(block["targets_recovered_by_a_false_relation"])
              + ", crash-only = "
              + str(block["targets_recovered_only_by_making_the_mutant_crash"]))
        print("  CEILING kill rate    = " + str(block["ceiling_kill_rate"])
              + "   (conservative " + str(block["conservative_ceiling_kill_rate"])
              + ", STRICT " + str(block["strict_ceiling_kill_rate"]) + ")")
        print("  strict recovered     = "
              + str(block["targets_recovered_by_a_false_SEMANTIC_relation"])
              + "   gain " + str(block["strict_gain_over_arm"]))
        print("  families             = " + json.dumps(
            block["relation_families_that_worked"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
