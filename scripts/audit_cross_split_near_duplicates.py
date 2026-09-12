"""Are the evaluation splits actually disjoint, or only disjoint by identifier?

The existing leakage audit (``leakage_audit.json``) asks whether a prompt
contains the answer. This asks a different question that nothing in the project
had asked: whether a validation function is a near-copy of a training function.

Identifier-level and lineage-level disjointness are already enforced at build
time, and they do not settle this. Two MBPP tasks can carry different ids,
different ``group_id`` lineages and near-identical bodies - "find the maximum of
three numbers" and "find the largest among three given numbers" are separate
upstream tasks and the same function. When that happens across a split
boundary, a memorising model scores as a generalising one, and the train/val
gap this project has been trying to explain is partly an artefact.

Method, chosen so the result is checkable rather than trusted:

* the reference is normalised through the AST with docstrings removed, so
  formatting, comments and docstring wording cannot manufacture a difference
  or hide a similarity;
* similarity is exact Jaccard over 5-token shingles, not an embedding, so any
  reported pair can be re-derived by hand;
* candidates come from an inverted shingle index, so the exact score is
  computed only for pairs that share something, and nothing is approximated.

THE SEALED TEST SPLIT IS NEVER READ. Auditing it would require loading it, and
it stays closed until everything is frozen.
"""
from __future__ import annotations

import argparse
import ast
import io
import json
import sys
import tokenize
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness.corpus import write_json

SEALED = "test"
SHINGLE = 5
#: Above this, two functions are the same function written twice. Chosen so a
#: pair at the threshold is still obviously a duplicate when read by hand; the
#: report also carries the whole distribution so the cut can be re-examined
#: without re-running the scan.
NEAR_DUPLICATE_JACCARD = 0.80
#: Pairs below this are not scored at all - they cannot reach the threshold.
CANDIDATE_MIN_SHARED = 3


class _StripDocstrings(ast.NodeTransformer):
    """Docstring wording is specification, not implementation."""

    def _strip(self, node):
        self.generic_visit(node)
        body = node.body
        if body and isinstance(body[0], ast.Expr) \
                and isinstance(body[0].value, ast.Constant) \
                and isinstance(body[0].value.value, str):
            node.body = body[1:] or [ast.Pass()]
        return node

    visit_FunctionDef = _strip
    visit_AsyncFunctionDef = _strip
    visit_ClassDef = _strip
    visit_Module = _strip


def normalise(code: str) -> str | None:
    """Structure only: no comments, no docstrings, no formatting."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    try:
        return ast.unparse(_StripDocstrings().visit(tree))
    except Exception:
        return None


def _tokens(code: str) -> list[str]:
    out: list[str] = []
    try:
        for token in tokenize.generate_tokens(io.StringIO(code).readline):
            if token.type in (tokenize.NL, tokenize.NEWLINE, tokenize.INDENT,
                              tokenize.DEDENT, tokenize.COMMENT,
                              tokenize.ENDMARKER, tokenize.ENCODING):
                continue
            out.append(token.string)
    except (tokenize.TokenError, IndentationError):
        return code.split()
    return out


def shingles(code: str, size: int = SHINGLE) -> frozenset[str]:
    tokens = _tokens(code)
    if len(tokens) < size:
        # Short functions still need a signature, or every one-liner collides
        # with every other one-liner at Jaccard 1.0 and the report is noise.
        return frozenset([" ".join(tokens)]) if tokens else frozenset()
    return frozenset(
        " ".join(tokens[i:i + size]) for i in range(len(tokens) - size + 1))


def _spec_shingles(text: str, size: int = 4) -> frozenset[str]:
    words = str(text or "").lower().split()
    if len(words) < size:
        return frozenset([" ".join(words)]) if words else frozenset()
    return frozenset(
        " ".join(words[i:i + size]) for i in range(len(words) - size + 1))


def jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    if not left or not right:
        return 0.0
    intersection = len(left & right)
    if not intersection:
        return 0.0
    return intersection / len(left | right)


def _index(entries: Iterable[tuple[str, frozenset[str]]]):
    postings: dict[str, list[str]] = defaultdict(list)
    for key, items in entries:
        for item in items:
            postings[item].append(key)
    return postings


#: Reported alongside the headline cut. A single threshold count is the kind of
#: headline this project has repeatedly had to retract: the first run of this
#: audit returned "0 near-duplicates in val" while the closest pair sat at
#: 0.7971, three thousandths under the cut. The sweep makes that visible.
THRESHOLD_SWEEP = (0.70, 0.75, 0.80, 0.85, 0.90, 0.95)


def compare(left: dict[str, frozenset[str]],
            right: dict[str, frozenset[str]]) -> list[dict[str, Any]]:
    """Every ``right`` entry paired with its nearest ``left`` entry.

    Returns every entry, scored, rather than only those above a threshold, so
    the caller can sweep the cut instead of baking one in here.
    """
    postings = _index(left.items())
    scored: list[dict[str, Any]] = []

    for right_id, right_shingles in right.items():
        shared: Counter = Counter()
        for shingle in right_shingles:
            for left_id in postings.get(shingle, ()):
                shared[left_id] += 1
        best_id, best_score = None, 0.0
        for left_id, count in shared.items():
            if count < CANDIDATE_MIN_SHARED:
                continue
            score = jaccard(left[left_id], right_shingles)
            if score > best_score:
                best_id, best_score = left_id, score
        scored.append({
            "evaluation_record": right_id,
            "nearest_training_record": best_id,
            "jaccard": round(best_score, 4),
        })
    return scored


def summarise(scored: list[dict[str, Any]], lineage_of: dict[str, str],
              threshold: float) -> dict[str, Any]:
    """Threshold sweep, record counts and FUNCTION counts.

    Record counts overstate the problem: one duplicated function contributes
    every mutant built from it, so 45 records can be one function. Both are
    reported, because the record count is what an evaluation panel actually
    scores and the function count is what was actually duplicated.
    """
    sweep: dict[str, Any] = {}
    for cut in THRESHOLD_SWEEP:
        hits = [row for row in scored if row["jaccard"] >= cut]
        sweep[f"{cut:.2f}"] = {
            "records": len(hits),
            "distinct_evaluation_functions": len(
                {lineage_of.get(row["evaluation_record"], row["evaluation_record"])
                 for row in hits}),
        }

    ranked = sorted(scored, key=lambda row: -row["jaccard"])
    above = [row for row in ranked if row["jaccard"] >= threshold]
    return {
        "evaluation_records_scored": len(scored),
        "max_similarity": ranked[0]["jaccard"] if ranked else None,
        "threshold_sweep": sweep,
        "near_duplicate_records_at_threshold": len(above),
        "near_duplicate_functions_at_threshold": len(
            {lineage_of.get(row["evaluation_record"], row["evaluation_record"])
             for row in above}),
        "closest_pairs": ranked[:10],
    }


def build(corpus_dir: Path, threshold: float) -> dict[str, Any]:
    records = json.loads(
        (corpus_dir / "records.json").read_text(encoding="utf-8"))
    splits = json.loads(
        (corpus_dir / "splits.json").read_text(encoding="utf-8"))
    by_id = {str(record["id"]): record for record in records}

    if SEALED in splits:
        # Named explicitly so the report can state it was skipped on purpose
        # rather than leaving a reader to infer it from an absence.
        pass

    prepared: dict[str, dict[str, Any]] = {}
    unparsable: Counter = Counter()
    for split, ids in splits.items():
        if split == SEALED:
            continue
        code_shingles: dict[str, frozenset[str]] = {}
        spec_shingles: dict[str, frozenset[str]] = {}
        lineages: set[str] = set()
        lineage_of: dict[str, str] = {}
        exact: dict[str, list[str]] = defaultdict(list)
        for record_id in ids:
            record = by_id.get(str(record_id))
            if record is None:
                continue
            normalised = normalise(str(record.get("reference_code") or ""))
            if normalised is None:
                unparsable[split] += 1
                continue
            code_shingles[str(record_id)] = shingles(normalised)
            spec_shingles[str(record_id)] = _spec_shingles(
                record.get("specification"))
            lineage = str(record.get("group_id") or record_id)
            lineages.add(lineage)
            lineage_of[str(record_id)] = lineage
            exact[normalised].append(str(record_id))
        prepared[split] = {
            "code": code_shingles, "spec": spec_shingles,
            "lineages": lineages, "lineage_of": lineage_of, "exact": exact,
        }

    train = prepared.get("train", {"code": {}, "spec": {}, "lineages": set(),
                                   "lineage_of": {}, "exact": {}})
    comparisons: dict[str, Any] = {}
    for panel in ("ablation_dev", "val"):
        if panel not in prepared:
            continue
        lineage_of = prepared[panel]["lineage_of"]
        reference = summarise(
            compare(train["code"], prepared[panel]["code"]), lineage_of, threshold)
        specification = summarise(
            compare(train["spec"], prepared[panel]["spec"]), lineage_of, threshold)
        exact_overlap = sorted(
            set(train["exact"]) & set(prepared[panel]["exact"]))
        shared_lineages = sorted(train["lineages"] & prepared[panel]["lineages"])
        comparisons[f"train_vs_{panel}"] = {
            "exact_normalised_reference_collisions": len(exact_overlap),
            "shared_group_id_lineages": len(shared_lineages),
            "shared_lineage_examples": shared_lineages[:10],
            "reference_code": reference,
            "specification_text": specification,
        }

    return {
        "schema_version": "oneiros_cross_split_near_duplicate_audit_v1",
        "corpus": corpus_dir.name,
        "sealed_final_test_accessed": False,
        "sealed_split_excluded": SEALED,
        "method": {
            "normalisation": "ast.unparse with docstrings removed",
            "similarity": f"exact Jaccard over {SHINGLE}-token shingles",
            "near_duplicate_threshold": threshold,
            "candidate_generation": "inverted shingle index, exact scoring",
        },
        "records_unparsable_by_split": dict(unparsable),
        "comparisons": comparisons,
        "reading": (
            "a near-duplicate pair means a function in an EVALUATION panel has "
            "a near-copy in train. Those targets measure recall, not "
            "generalisation, and any kill rate should be reported with them "
            "held out as well as included."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path,
                        default=ROOT / "data" / "corpus" / "v4_1_research_hardened_candidate")
    parser.add_argument("--threshold", type=float, default=NEAR_DUPLICATE_JACCARD)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    report = build(arguments.corpus, arguments.threshold)
    write_json(arguments.output, report)
    printable = {k: v for k, v in report.items() if k != "comparisons"}
    print(json.dumps(printable, indent=2))
    for name, block in report["comparisons"].items():
        reference = block["reference_code"]
        print("")
        print(name + ":")
        print("  records scored          = "
              + str(reference["evaluation_records_scored"]))
        print("  exact collisions        = "
              + str(block["exact_normalised_reference_collisions"]))
        print("  shared lineages         = "
              + str(block["shared_group_id_lineages"]))
        print("  max similarity          = " + str(reference["max_similarity"]))
        print("  at threshold (records)  = "
              + str(reference["near_duplicate_records_at_threshold"]))
        print("  at threshold (functions)= "
              + str(reference["near_duplicate_functions_at_threshold"]))
        print("  sweep = " + json.dumps(reference["threshold_sweep"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
