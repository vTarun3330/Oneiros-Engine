"""Dataset normalization and validation helpers for Phase 3 training."""
from __future__ import annotations

import ast
import hashlib
import json
import re
import warnings
from pathlib import Path
from typing import Iterable, List


def extract_dataset_assertions(test_cases: Iterable[str], entry_point: str) -> List[str]:
    """Return unique, standalone assertions from dataset test blocks.

    Test suites from HumanEval and MBPP often format one assertion over several
    lines.  Line-based extraction silently turns those into syntactically
    invalid fragments, so parse the complete block and round-trip each
    ``assert`` AST node instead.  ``candidate(...)`` is normalized to the
    concrete entry point used by the paired implementation.
    """
    assertions: List[str] = []
    seen = set()
    for block in test_cases:
        if not isinstance(block, str) or not block.strip():
            continue
        try:
            tree = ast.parse(block)
        except SyntaxError:
            # Keep the fallback deliberately conservative: malformed source is
            # reported by the dataset-preparation gate and cannot enter data.
            continue
        nodes = sorted(
            (node for node in ast.walk(tree) if isinstance(node, ast.Assert)),
            key=lambda node: (node.lineno, node.col_offset),
        )
        for node in nodes:
            # The BugsInPy compatibility runtime is Python 3.8, which does
            # not provide ast.unparse.  The parsed source segment preserves
            # the validated assertion without changing its semantics.
            assertion = ast.get_source_segment(block, node)
            if not assertion:
                continue
            assertion = re.sub(
                r"\bcandidate\s*\(", f"{entry_point}(" , assertion
            )
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", SyntaxWarning)
                    compile(assertion, "<dataset-assertion>", "exec")
            except SyntaxError:
                continue
            if assertion not in seen:
                assertions.append(assertion)
                seen.add(assertion)
    return assertions


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _function_key(pair: dict) -> str:
    code = pair["golden_code"].replace("\r\n", "\n").replace("\r", "\n").strip()
    return hashlib.sha256(code.encode("utf-8")).hexdigest()


def _parse_python(source: str) -> ast.AST:
    """Parse benchmark code without surfacing harmless regex-literal warnings."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        return ast.parse(source)


def verify_prepared_dataset(data_dir: Path) -> dict:
    """Fail closed unless the exact prepared, group-disjoint dataset is present."""
    manifest_path = data_dir / "dataset_manifest.json"
    if not manifest_path.exists():
        raise RuntimeError(
            "Missing data/dataset_manifest.json. Run scripts/prepare_training_dataset.py before training."
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("Dataset manifest is unreadable; re-run dataset preparation.") from exc
    if manifest.get("schema_version") != 1 or not manifest.get("quality_gate", {}).get(
        "all_retained_pairs_behaviorally_verified"
    ):
        raise RuntimeError("Dataset manifest does not certify behavioral validation.")

    splits = {}
    for name in ("train", "val", "test"):
        metadata = manifest.get("splits", {}).get(name, {})
        path = data_dir / metadata.get("path", "")
        if not path.exists() or _sha256_file(path) != metadata.get("sha256"):
            raise RuntimeError(f"{name} split does not match the prepared dataset manifest.")
        try:
            records = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"{name} split is unreadable.") from exc
        if len(records) != metadata.get("pair_count") or not isinstance(records, list):
            raise RuntimeError(f"{name} split count differs from the prepared manifest.")
        splits[name] = records

    groups = {name: {_function_key(pair) for pair in records} for name, records in splits.items()}
    if groups["train"] & groups["val"] or groups["train"] & groups["test"] or groups["val"] & groups["test"]:
        raise RuntimeError("Function leakage detected between training splits.")

    seen_mutations = set()
    for name, records in splits.items():
        for pair in records:
            required = {"id", "golden_code", "mutant_code", "entry_point", "test_cases"}
            if not isinstance(pair, dict) or required - pair.keys():
                raise RuntimeError(f"Malformed pair in {name} split.")
            if pair["golden_code"] == pair["mutant_code"]:
                raise RuntimeError(f"Identical golden/mutant pair in {name} split.")
            try:
                golden_tree = _parse_python(pair["golden_code"])
                _parse_python(pair["mutant_code"])
                if pair["entry_point"] not in {
                    node.name for node in ast.walk(golden_tree)
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                }:
                    raise RuntimeError(f"Missing entry point in {name} split.")
            except SyntaxError as exc:
                raise RuntimeError(f"Invalid Python in {name} split.") from exc
            tests = pair["test_cases"]
            if not isinstance(tests, list) or not tests:
                raise RuntimeError(f"Pair without normalized assertions in {name} split.")
            for assertion in tests:
                if not isinstance(assertion, str) or not assertion.startswith("assert "):
                    raise RuntimeError(f"Non-assertion fixture in {name} split.")
                with warnings.catch_warnings():
                    # Valid benchmark fixtures sometimes intentionally include
                    # regular-expression escapes in string literals.
                    warnings.simplefilter("ignore", SyntaxWarning)
                    compile(assertion, "<prepared-assertion>", "exec")
            mutation_key = (_function_key(pair), pair["entry_point"], pair["mutant_code"])
            if mutation_key in seen_mutations:
                raise RuntimeError("Duplicate mutant pair detected across prepared splits.")
            seen_mutations.add(mutation_key)
    return manifest
