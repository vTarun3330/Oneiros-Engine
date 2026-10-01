"""v2.5 CPU program 2, Phase 8: partition NEW repository universes before any acquisition outcome.

Universe = union of the project's frozen candidate repository lists (docs/repository_native_*
repositories*.json) minus the frozen exclusion ledger of the confirmation-panel specification
(every training repository, every BugsInPy / SWE-bench repository, the v2.4 diagnostic panel and
every repository of the three earlier acquisition pilots). Repositories are keyed by the
owner-independent name (forks and renames collide).

Partition (deterministic, outcome-free): order the eligible keys by sha256("oneiros-v25-
partition:" + key); every third key (positions 0, 3, 6, ...) goes to the TRAINING-EXPANSION
pool, the other two thirds to the CONFIRMATION-ONLY pool, so the confirmation universe (which
must yield >= 80 targets from >= 10 repositories) is never consumed by training. Candidate-level
disjointness (commit, issue, function fingerprint, bug lineage) is enforced at acquisition time
against the frozen training fingerprints and against the other pool's ledger rows. An
append-only usage ledger records every later use.

    python scripts/v25_universe_partition.py
"""
from __future__ import annotations

import glob
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RECEIPT = "results/sft_root_cause_v25_universe_partition.json"
LEDGER = "results/sft_root_cause_v25_repository_usage_ledger.jsonl"
PANEL_SPEC = "results/sft_root_cause_v25_confirmation_panel_spec.json"
SALT = "oneiros-v25-partition:"


def main() -> int:
    from scripts.native_rehearsal_rebuild_v22 import publish_once
    from scripts.v25_confirmation_panel_spec import repo_key
    spec = json.loads((ROOT / PANEL_SPEC).read_text(encoding="utf-8"))
    excluded = set(spec["exclusion_ledger"]["excluded_repository_keys"])
    sources, names = {}, {}
    for path in sorted(glob.glob(str(ROOT / "docs" / "repository_native_*repositories*.json"))):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        repos = data.get("repositories", data) if isinstance(data, dict) else data
        rel = Path(path).relative_to(ROOT).as_posix()
        sources[rel] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        for full in repos:
            names.setdefault(repo_key(full), set()).add(full)
    eligible = sorted(k for k in names if k not in excluded)
    ordered = sorted(eligible, key=lambda k: hashlib.sha256((SALT + k).encode()).hexdigest())
    training = sorted(k for i, k in enumerate(ordered) if i % 3 == 0)
    confirmation = sorted(k for i, k in enumerate(ordered) if i % 3 != 0)
    assert not set(training) & set(confirmation)
    receipt = {
        "schema_version": "oneiros_v25_universe_partition_v1",
        "frozen_before_outcomes": True,
        "source_lists_sha256": sources,
        "exclusion_ledger": {"spec": PANEL_SPEC, "excluded_keys": len(excluded)},
        "universe_keys": len(names), "excluded_from_universe": sorted(set(names) & excluded),
        "eligible_keys": len(eligible),
        "rule": "order by sha256(salt + key); positions 0 mod 3 -> training expansion; others "
                "-> confirmation only",
        "training_expansion_pool": {k: sorted(names[k]) for k in training},
        "confirmation_only_pool": {k: sorted(names[k]) for k in confirmation},
        "disjointness": {"repository_key_overlap": 0,
                         "candidate_level": "commit, issue, function fingerprint and bug "
                                            "lineage checked at acquisition against the frozen "
                                            "training fingerprints and the other pool"},
        "protected_material": "no validation, reserved-confirmation or sealed record is read; "
                              "every corpus source repository is excluded by the ledger",
        "licence_and_provenance": "recorded per repository by the acquisition mechanism "
                                  "(licence validation is part of its admission pipeline)",
        "usage_ledger": {"path": LEDGER, "rule": "append-only; one row per use: utc, pool, "
                                                 "repository, purpose, artifact sha256"}}
    status = publish_once(RECEIPT, receipt)
    ledger = ROOT / LEDGER
    if not ledger.exists():
        ledger.write_text("", encoding="utf-8")
    print(json.dumps({"status": status, "universe": len(names), "eligible": len(eligible),
                      "training_expansion": len(training), "confirmation_only":
                      len(confirmation),
                      "sha256": hashlib.sha256((ROOT / RECEIPT).read_bytes()).hexdigest()},
                     indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
