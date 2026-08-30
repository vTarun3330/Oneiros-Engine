"""Build the V4.1 failure taxonomy (research specification §43).

The taxonomy is a diagnostic over evaluation artifacts that already exist.  It
never generates, never re-executes, and never opens the sealed final-test split:
an artifact whose ``final_test_measurement`` is true, or whose evaluation split
is ``test``, is refused outright.

Usage::

    py -3.12 scripts/failure_taxonomy.py results/<run>/sft_validation_*_seed_42.json \
        --output results/<run>/failure_taxonomy_seed_42.json \
        --markdown results/<run>/FAILURE_TAXONOMY.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from metrics.research_evaluation import (
    FAILURE_TAXONOMY_CATEGORIES,
    summarise_failure_taxonomy,
)
from utils.reproducibility import source_tree_sha256


SEALED_SPLIT = "test"


def load_evaluation_artifact(path: Path) -> Dict[str, Any]:
    """Load one evaluation artifact, refusing any final-test measurement."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("final_test_measurement"):
        raise SystemExit(
            f"Refusing {path}: artifact is a sealed final-test measurement"
        )
    if str(payload.get("evaluation_split", "")) == SEALED_SPLIT:
        raise SystemExit(f"Refusing {path}: artifact evaluates the sealed test split")
    if not isinstance(payload.get("function_results"), list):
        raise SystemExit(f"Refusing {path}: artifact has no function_results list")
    return payload


def build_report(paths: Sequence[Path]) -> Dict[str, Any]:
    """Summarise one or more legitimate evaluation artifacts."""
    if not paths:
        raise ValueError("At least one evaluation artifact is required")
    sources: List[Dict[str, Any]] = []
    pooled: List[Mapping[str, Any]] = []
    for path in paths:
        payload = load_evaluation_artifact(path)
        results = payload["function_results"]
        pooled.extend(results)
        sources.append({
            "artifact": str(path).replace("\\", "/"),
            "evaluation_split": payload.get("evaluation_split"),
            "seed": payload.get("seed"),
            "adapter": payload.get("adapter"),
            "adapter_sha256": payload.get("adapter_sha256"),
            "function_results": len(results),
            "taxonomy": summarise_failure_taxonomy(results),
        })
    return {
        "report_schema_version": 1,
        "source_tree_sha256": source_tree_sha256(ROOT),
        "sealed_final_test_accessed": False,
        "artifacts": sources,
        "pooled": summarise_failure_taxonomy(pooled),
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    """Render the human-readable summary required alongside the JSON ledger."""
    pooled = report["pooled"]
    lines = [
        "# Oneiros V4.1 failure taxonomy",
        "",
        "Diagnostic over legitimately produced evaluation artifacts only. The "
        "sealed final-test split is never read.",
        "",
        f"- Classified functions: {pooled['classified_functions']}",
        f"- Functions without recorded candidate outcomes: "
        f"{pooled['unclassifiable_functions']}",
        f"- Classified candidates: {pooled['classified_candidates']}",
        "",
        "## Overall candidate outcomes",
        "",
        "| Category | Candidates | Share |",
        "| --- | ---: | ---: |",
    ]
    counts = pooled["overall"]["counts"]
    rates = pooled["overall"]["rates"]
    for category in FAILURE_TAXONOMY_CATEGORIES:
        if category not in counts:
            continue
        lines.append(
            f"| {category} | {counts[category]} | {rates[category] * 100:.2f}% |"
        )

    for title, key in (
        ("## By dataset", "by_source"),
        ("## By mutation family", "by_mutation_family"),
    ):
        lines.extend(["", title, ""])
        for name, table in report["pooled"][key].items():
            top = sorted(
                table["counts"].items(), key=lambda item: item[1], reverse=True
            )[:4]
            summary = ", ".join(f"{cat} {num}" for cat, num in top) or "no candidates"
            lines.append(f"- **{name}** ({table['candidates']} candidates): {summary}")
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "artifacts", nargs="+", type=Path,
        help="Evaluation result JSON files from ablation_dev or locked validation",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_report(args.artifacts)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps(report["pooled"], indent=2))


if __name__ == "__main__":
    main()
