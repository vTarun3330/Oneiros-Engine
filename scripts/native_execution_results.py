"""The single authoritative execution-results loader (amendment v2.4 section D).

Validates the execution contract and its results TOGETHER against the frozen artifacts:
  * the contract file exists; its canonical hash equals every row's ``contract_sha256``;
  * its design version, executor/inner/IO hashes, job file and job_sha256, manifest,
    preparation, condition, telemetry schema, limits, cohort and the exact generation files,
    generation contracts and identities equal the current frozen artifacts;
  * every execution key maps to exactly one arm x target x seed x slot generation candidate
    (no missing, duplicate, extra, stale, partial, malformed or cross-arm row);
  * every cell: module hash and generation telemetry equal the candidate; the class, the
    static admission and fixed validity are recomputed from the actual generated module.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Mapping


class ExecutionRefused(SystemExit):
    """The execution artifacts cannot be used."""


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_execution(results_path: Path, contract_path: Path, *, cohort: Mapping[str, Any],
                   prepared: Mapping[str, Any], generations: Mapping[str, Any]) -> Dict[str, Any]:
    from scripts import native_generated_tests_execute_wsl as ex
    from scripts.native_generation_io import TELEMETRY_SCHEMA
    contract_path, results_path = Path(contract_path), Path(results_path)
    if not contract_path.is_file():
        raise ExecutionRefused("REFUSED: execution contract missing")
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    chash = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    scripts = Path(ex.__file__).resolve().parent
    expected = {
        "design_version": ex.DESIGN_VERSION,
        "executor_sha256": _sha(Path(ex.__file__)), "inner_sha256": _sha(scripts / "native_sandbox_inner.sh"),
        "io_sha256": _sha(scripts / "native_generation_io.py"),
        "prep": {"path": prepared["path"], "sha256": prepared["sha256"]},
        "manifest_sha256": cohort["manifest_sha256"],
        "job_file_sha256": cohort["job_file_sha256"], "job_sha256": cohort["job_sha256"],
        "cohort": {"qualified": len(cohort["qualified"]),
                   "generation_targets": cohort["generation"],
                   "pre_generation_exclusions": cohort["pre_generation_exclusions"],
                   "expected": cohort["expected"]},
        "generations_sha256": generations["files_sha256"],
        "generation_contracts_sha256": generations["contracts_sha256"],
        "generation_identity_sha256": generations["identity_sha256"],
        "telemetry_schema": TELEMETRY_SCHEMA, "condition": cohort["condition"],
        "limits": ex.LIMITS, "module_limits": ex.MODULE_LIMITS}
    wrong = sorted(k for k, v in expected.items() if contract.get(k) != v)
    extra = sorted(set(contract) - set(expected))
    if wrong or extra:
        raise ExecutionRefused(f"REFUSED: execution contract differs from the frozen artifacts "
                               f"in {wrong + extra}")
    if not results_path.is_file():
        raise ExecutionRefused("REFUSED: execution results missing")
    data = results_path.read_bytes()
    if data and not data.endswith(b"\n"):
        raise ExecutionRefused("REFUSED: execution results end with a partial line")
    cells = {f"{arm}::{t}::{s}::{k}" for arm in ex.ARMS for t in cohort["generation"]
             for s in ex.SEEDS for k in range(ex.SLOTS)}
    names = {k: prepared["rows"][k]["qualname"].split(".")[-1] for k in cohort["generation"]}
    rows: Dict[str, Dict[str, Any]] = {}
    for number, line in enumerate(data.decode("utf-8").splitlines(), 1):
        try:
            row = json.loads(line)
        except ValueError:
            raise ExecutionRefused(f"REFUSED: execution line {number} malformed") from None
        key = row.get("key")
        if row.get("contract_sha256") != chash:
            raise ExecutionRefused(f"REFUSED: execution line {number} is stale (other contract)")
        if key not in cells:
            raise ExecutionRefused(f"REFUSED: execution line {number}: unexpected cell {key!r}")
        if key in rows:
            raise ExecutionRefused(f"REFUSED: execution line {number}: duplicate cell {key}")
        arm, target, seed, slot = key.split("::")
        if (row.get("arm"), row.get("target_key"), row.get("seed"), row.get("slot")) != \
                (arm, target, int(seed), int(slot)):
            raise ExecutionRefused(f"REFUSED: execution line {number}: key/fields disagree "
                                   "(cross-arm or cross-cell row)")
        grow = generations["rows"][(arm, target, int(seed))]
        problems = ex.cell_problems(row, grow["candidates"][int(slot)], grow, names[target])
        if problems:
            raise ExecutionRefused(f"REFUSED: execution cell {key} fails verification against "
                                   f"its generated candidate: {problems[:3]}")
        rows[key] = row
    if set(rows) != cells:
        raise ExecutionRefused(f"REFUSED: execution grid incomplete: "
                               f"{len(cells - set(rows))} of {len(cells)} cells missing")
    return {"rows": [rows[k] for k in sorted(rows)], "contract_sha256": chash,
            "contract_file_sha256": _sha(contract_path), "results_sha256": _sha(results_path),
            "cells": len(rows)}
