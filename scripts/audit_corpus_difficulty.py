"""Audit whether the corpus and the evaluation panel span all difficulty tiers.

A model trained only on easy functions will not learn hard behaviour, and a
panel made only of hard functions cannot show where an approach starts to
fail. Both are measured here, per split and for the exact monitor panel that
produces every kill-rate number we report.

Complexity is read from the buggy-side AST index only, so this audit can never
be influenced by reference code, gold tests, or oracle outcomes.

    python scripts/audit_corpus_difficulty.py
"""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import CANONICAL_CORPUS_VERSION  # noqa: E402
from harness.corpus_view import load_complexity_index, load_development_split  # noqa: E402

RESULTS = ROOT / "results"
SPLITS = ("train", "ablation_dev", "val")
TIERS = ("simple", "moderate", "complex", "repository_no_function_ast")


def _tier(record_id: str, index: dict, execution_mode: str) -> str:
    if str(execution_mode or "").startswith("repository_"):
        return "repository_no_function_ast"
    entry = index.get(record_id) or {}
    return entry.get("tier") or "unindexed"


def _distribution(records, index) -> dict:
    tiers = collections.Counter()
    datasets = collections.Counter()
    families = collections.Counter()
    for record in records:
        tiers[_tier(record["id"], index, record.get("execution_mode", ""))] += 1
        datasets[str(record.get("source_name") or record.get("dataset") or "unknown")] += 1
        families[str(record.get("bug_family") or "unknown")] += 1
    total = max(1, sum(tiers.values()))
    return {
        "records": sum(tiers.values()),
        "tier_counts": dict(sorted(tiers.items())),
        "tier_fractions": {k: round(v / total, 4) for k, v in sorted(tiers.items())},
        "dataset_counts": dict(sorted(datasets.items())),
        "mutation_family_counts": dict(sorted(families.items())),
    }


def _panel_ids() -> tuple[list[str], str] | tuple[None, None]:
    """The exact 100-function panel every reported kill rate is measured on."""
    for candidate in sorted(RESULTS.glob("local_*/sft_monitor_selection.json")):
        payload = json.loads(candidate.read_text(encoding="utf-8"))
        if payload.get("function_count") == 100:
            return payload["record_ids"], str(candidate)
    return None, None


def main() -> int:
    corpus_dir = ROOT / "data" / "corpus" / CANONICAL_CORPUS_VERSION
    index = load_complexity_index(corpus_dir)

    report: dict = {
        "schema_version": "oneiros_corpus_difficulty_audit_v1",
        "corpus_version": CANONICAL_CORPUS_VERSION,
        "complexity_lineage": "buggy_revision_ast_only",
        "final_test_measurement": False,
        "splits": {},
    }

    by_split_records = {}
    for split in SPLITS:
        records = load_development_split(corpus_dir, split)
        by_split_records[split] = records
        report["splits"][split] = _distribution(records, index)

    panel_ids, panel_source = _panel_ids()
    if panel_ids:
        by_id = {r["id"]: r for r in by_split_records["ablation_dev"]}
        panel_records = [by_id[i] for i in panel_ids if i in by_id]
        report["evaluation_panel"] = {
            "source": panel_source,
            "declared_records": len(panel_ids),
            "resolved_records": len(panel_records),
            **_distribution(panel_records, index),
        }

    findings: list[str] = []
    train = report["splits"]["train"]["tier_fractions"]
    for tier in ("simple", "moderate", "complex"):
        share = train.get(tier, 0.0)
        if share < 0.05:
            findings.append(
                f"train tier '{tier}' is only {share:.1%} of records - too thin to "
                "teach or to detect a regression on that tier"
            )
    panel = report.get("evaluation_panel", {}).get("tier_fractions", {})
    for tier in ("simple", "moderate", "complex"):
        if panel and panel.get(tier, 0.0) < 0.05:
            findings.append(
                f"evaluation panel tier '{tier}' is only {panel.get(tier, 0):.1%} - "
                "kill rates cannot show a crossover point on that tier"
            )
    if panel and panel.get("repository_no_function_ast", 0.0) > 0.0:
        findings.append(
            "evaluation panel contains repository records without function-level "
            "AST complexity; they are excluded from tier fractions above"
        )
    report["findings"] = findings or ["all tiers present at >=5% in train and panel"]

    out = RESULTS / "v4_1_corpus_difficulty_audit.json"
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(f"{'split':<16}{'records':>9}{'simple':>9}{'moderate':>10}{'complex':>9}{'repo':>7}")
    for split in SPLITS:
        counts = report["splits"][split]["tier_counts"]
        print(f"{split:<16}{report['splits'][split]['records']:>9}"
              f"{counts.get('simple', 0):>9}{counts.get('moderate', 0):>10}"
              f"{counts.get('complex', 0):>9}{counts.get('repository_no_function_ast', 0):>7}")
    if "evaluation_panel" in report:
        counts = report["evaluation_panel"]["tier_counts"]
        print(f"{'PANEL(100fn)':<16}{report['evaluation_panel']['resolved_records']:>9}"
              f"{counts.get('simple', 0):>9}{counts.get('moderate', 0):>10}"
              f"{counts.get('complex', 0):>9}{counts.get('repository_no_function_ast', 0):>7}")
    print()
    for finding in report["findings"]:
        print(f"  - {finding}")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
