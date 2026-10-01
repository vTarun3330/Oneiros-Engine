"""v2.6 decision semantics (stdlib only): separate, machine-readable readiness states.

Corrects the single ambiguous v2.5 readiness result. Rules:
- only OBSERVED, deduplicated counts can pass the frozen 150/8/60 training gate; a projection
  never passes or fails an observed-data gate;
- zero or missing denominators fail closed;
- duplicates never inflate canonical tests, repositories or lineages;
- a confirmation panel is an actual frozen list of targets; a projection is never a panel;
- an actual valid panel below 80 targets / 10 repositories (or missing a stratum / family
  requirement) is BUILT_UNDERPOWERED (exploratory, cannot support promotion), not INVALID;
- an invalid panel (missing identities/hashes, duplicates, training overlap, > 8 per
  repository) is INVALID.
"""
from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

TRAINING_GATE = {"repository_tests": 150, "repositories": 8, "lineages": 60}
PANEL_MINIMUM = {"targets": 80, "repositories": 10, "max_per_repository": 8,
                 "complexity_strata": ("simple", "moderate", "complex"), "defect_families": 4}
PANEL_REQUIRED = ("target_id", "repository", "lineage", "function_fingerprint", "buggy_commit",
                  "fixed_commit", "environment_lock_sha256", "complexity", "defect_family")
STATES = ("artifact_integrity", "training_corpus_gate", "training_acquisition_status",
          "confirmation_panel_status", "confirmation_power_status",
          "atheris_comparability_status", "current_source_test_status",
          "gpu_evaluation_authorized", "gpu_training_authorized")


class ProjectionNotEvidence(ValueError):
    """A projected / expected quantity was offered where observed evidence is required."""


def observed_training_gate(rows: Optional[Iterable[Mapping[str, Any]]]) -> Dict[str, Any]:
    """``rows``: OBSERVED verified repository tests with canonical_test, repository, lineage."""
    if rows is None:
        return {"status": "FAIL", "reason": "no observed rows (fail closed)"}
    rows = list(rows)
    if any(r.get("kind") == "projection" or r.get("projected") for r in rows):
        raise ProjectionNotEvidence("projected rows cannot enter an observed-data gate")
    if any(not r.get("canonical_test") or not r.get("repository") or not r.get("lineage")
           for r in rows):
        return {"status": "FAIL", "reason": "a row lacks canonical_test/repository/lineage"}
    unique = {}
    for r in rows:
        unique.setdefault(r["canonical_test"], r)       # duplicates never inflate counts
    counts = {"repository_tests": len(unique),
              "repositories": len({r["repository"] for r in unique.values()}),
              "lineages": len({r["lineage"] for r in unique.values()})}
    passed = all(counts[k] >= TRAINING_GATE[k] for k in TRAINING_GATE)
    return {"status": "PASS" if passed else "FAIL", "observed": counts, "gate": TRAINING_GATE,
            "duplicates_ignored": len(rows) - len(unique),
            "shortfall": {k: max(0, TRAINING_GATE[k] - counts[k]) for k in TRAINING_GATE}}


def panel_status(panel: Optional[Mapping[str, Any]],
                 training_fingerprints: Sequence[str] = (),
                 training_repositories: Sequence[str] = ()) -> Dict[str, Any]:
    if panel is None:
        return {"status": "NOT_BUILT"}
    if panel.get("kind") == "projection" or "targets" not in panel:
        raise ProjectionNotEvidence("a projection is never a confirmation panel")
    targets = list(panel["targets"])
    problems = []
    if not panel.get("frozen") or not panel.get("targets_sha256"):
        problems.append("panel not frozen with a hash")
    for t in targets:
        missing = [k for k in PANEL_REQUIRED if not t.get(k)]
        if missing:
            problems.append(f"{t.get('target_id')}: missing {missing}")
    for key in ("target_id", "lineage", "function_fingerprint"):
        dup = [v for v, n in Counter(t.get(key) for t in targets).items() if n > 1]
        if dup:
            problems.append(f"duplicate {key}: {dup[:3]}")
    per_repo = Counter(t.get("repository") for t in targets)
    if per_repo and max(per_repo.values()) > PANEL_MINIMUM["max_per_repository"]:
        problems.append("more than 8 targets in a repository")
    overlap = {t.get("function_fingerprint") for t in targets} & set(training_fingerprints)
    repo_overlap = set(per_repo) & set(training_repositories)
    if overlap or repo_overlap:
        problems.append(f"training overlap: {len(overlap)} functions, {sorted(repo_overlap)}")
    if not targets:
        problems.append("empty panel")
    counts = {"targets": len(targets), "repositories": len(per_repo),
              "complexity_strata": sorted({t.get("complexity") for t in targets}),
              "defect_families": len({t.get("defect_family") for t in targets})}
    if problems:
        return {"status": "INVALID", "problems": problems, "counts": counts}
    powered = (counts["targets"] >= PANEL_MINIMUM["targets"]
               and counts["repositories"] >= PANEL_MINIMUM["repositories"]
               and set(PANEL_MINIMUM["complexity_strata"]) <= set(counts["complexity_strata"])
               and counts["defect_families"] >= PANEL_MINIMUM["defect_families"])
    return {"status": "CONFIRMATION_READY" if powered else "BUILT_UNDERPOWERED",
            "counts": counts, "minimum": {k: v for k, v in PANEL_MINIMUM.items()},
            "claim_scope": "confirmatory" if powered else
            "exploratory only: cannot support promotion or equivalence"}


def readiness(states: Mapping[str, Any]) -> Dict[str, Any]:
    """All nine states must be present explicitly (missing -> fail closed as UNKNOWN)."""
    out = {k: states.get(k, "UNKNOWN") for k in STATES}
    out["gpu_evaluation_authorized"] = states.get("gpu_evaluation_authorized") is True
    out["gpu_training_authorized"] = states.get("gpu_training_authorized") is True
    return out
