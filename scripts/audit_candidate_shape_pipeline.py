"""Show that the evaluation path can only ever score a single assertion.

The review asked for a model that writes ONE test covering four or five
defects. The dataset does exactly that - 663 verified multi-mutant completions
killing a mean of 8.43 sibling mutants each - and the model is trained on them.
This audit is about what happens next, at generation time, and the answer is
uncomfortable:

``engine/generator.py`` parses each model output by scanning for the FIRST line
beginning with ``assert `` and discarding everything else. A model that emits

    def test_clamp_boundaries():
        assert clamp(-1, 0, 10) == 0
        assert clamp(11, 0, 10) == 10

is recorded as ``assert clamp(-1, 0, 10) == 0``. The remaining assertions -
the ones that make it a multi-mutant test - never reach the validator, the
executor, or the artifact.

Three consequences follow, and all three are checkable from stored artifacts:

1. Every evaluated candidate is a single assertion, in every arm, by
   construction rather than by measurement.
2. ``--allow-test-function-candidates`` is inert in this path. It widens the
   validator, but the parser upstream has already collapsed the output, so no
   test-function-shaped candidate ever reaches the rule. The earlier finding
   that widening admitted zero candidates for the base model was real but
   incomplete: it admits zero for every arm, for this reason.
3. Kill@8 measures eight single assertions. The multi-mutant capability is
   demonstrated in the DATASET and has never been measured in the MODEL.

What this audit deliberately does NOT claim: that the model fails to emit
multi-assertion tests. That is unknown and unknowable from these artifacts,
because only ``raw_output_sha256`` is retained - the raw text is hashed, not
stored. The honest statement is that the pipeline cannot express the
capability, not that the model lacks it.
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness.corpus import write_json
from utils.reproducibility import source_tree_sha256

GENERATOR = ROOT / "engine" / "generator.py"


def parser_collapses_to_one_assertion() -> dict[str, Any]:
    """Read the generator and confirm the collapse is in the code, not inferred."""
    source = GENERATOR.read_text(encoding="utf-8")
    scans_for_assert = bool(
        re.search(r"if line\.startswith\(\s*['\"]assert ['\"]\s*\)", source)
    )
    breaks_after_first = bool(
        re.search(r"test_code = line\s*\n\s*break", source)
    )
    return {
        "file": "engine/generator.py",
        "scans_lines_for_a_leading_assert": scans_for_assert,
        "stops_at_the_first_match": breaks_after_first,
        "collapses_output_to_one_assertion": scans_for_assert and breaks_after_first,
        "effect": (
            "a multi-assertion test function is reduced to its first assertion "
            "before validation, execution and recording"
        ),
    }


def observed_shapes(artifact: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(artifact.read_text(encoding="utf-8"))
    except Exception:
        return None
    rows = payload.get("function_results") or []
    if not rows:
        return None

    shapes: collections.Counter[str] = collections.Counter()
    text_shapes: collections.Counter[str] = collections.Counter()
    raw_output_retained = 0
    for row in rows:
        for outcome in row.get("candidate_outcomes") or []:
            shapes[str(outcome.get("candidate_shape"))] += 1
            if outcome.get("raw_output_sha256"):
                raw_output_retained += 1
            code = str(outcome.get("code") or "")
            stripped = code.lstrip()
            if not stripped:
                text_shapes["empty"] += 1
            elif stripped.startswith("def test"):
                text_shapes["test_function"] += 1
            elif stripped.startswith("assert"):
                text_shapes["bare_assertion"] += 1
            else:
                text_shapes["other"] += 1

    return {
        "artifact": artifact.as_posix().split("results/", 1)[-1],
        "split": payload.get("evaluation_split"),
        "seed": payload.get("seed"),
        "recorded_candidate_shape": dict(shapes),
        "recorded_code_shape": dict(text_shapes),
        "test_function_candidates": text_shapes.get("test_function", 0),
        "raw_output_text_retained": 0,
        "raw_output_hash_retained": raw_output_retained,
    }


def audit() -> dict[str, Any]:
    parser_evidence = parser_collapses_to_one_assertion()
    panels: list[dict[str, Any]] = []
    for pattern in ("results/*/sft_validation_standard_*.json",
                    "results/*/base_validation_standard_*.json"):
        for path in sorted(glob.glob(str(ROOT / pattern))):
            if ".progress." in path:
                continue
            row = observed_shapes(Path(path))
            if row:
                panels.append(row)

    total_test_functions = sum(row["test_function_candidates"] for row in panels)
    return {
        "schema_version": "oneiros_candidate_shape_pipeline_audit_v1",
        "source_tree_sha256": source_tree_sha256(ROOT),
        "sealed_final_test_accessed": False,
        "question": (
            "can the evaluation path score the multi-assertion test the model "
            "was trained to write?"
        ),
        "answer": (
            "No. The generator collapses each output to its first assertion "
            "before anything else sees it, so every scored candidate is a "
            "single assertion by construction."
            if parser_evidence["collapses_output_to_one_assertion"] else
            "The generator no longer collapses output; re-examine this audit."
        ),
        "parser_evidence": parser_evidence,
        "artifacts_audited": len(panels),
        "test_function_candidates_across_all_artifacts": total_test_functions,
        "panels": panels,
        "consequences": [
            "every evaluated candidate is a single assertion, in every arm",
            "--allow-test-function-candidates is inert in this generation "
            "path: the validator is widened but the parser upstream has "
            "already collapsed the output, so no test-function-shaped "
            "candidate ever reaches the rule",
            "Kill@8 measures eight single assertions, never one broad test",
            "the multi-mutant capability is demonstrated in the dataset "
            "(8.43 mutants killed per verified completion) and has never "
            "been measured in the trained model",
        ],
        "explicitly_not_claimed": (
            "This does NOT show the model fails to emit multi-assertion tests. "
            "Only raw_output_sha256 is retained - the raw text is hashed, not "
            "stored - so what the model emitted cannot be recovered from these "
            "artifacts. The pipeline cannot express the capability; whether the "
            "model has it is unmeasured."
        ),
        "to_measure_it": (
            "retain raw model output alongside its hash, and add a generation "
            "path that validates the whole output under the widened shape "
            "policy instead of scanning for the first assertion. Both are "
            "changes to the frozen evaluation protocol and belong in a "
            "labelled successor run, not in this one."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "results" / "v4_2_candidate_shape_pipeline_audit.json",
    )
    arguments = parser.parse_args()
    report = audit()
    write_json(arguments.output, report)
    print(json.dumps({
        "answer": report["answer"],
        "parser_evidence": report["parser_evidence"],
        "artifacts_audited": report["artifacts_audited"],
        "test_function_candidates_across_all_artifacts":
            report["test_function_candidates_across_all_artifacts"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
