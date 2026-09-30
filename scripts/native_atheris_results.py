"""Atheris kill judgement and the single authoritative results loader (amendment v2.4 E).
Standard library only: imported by the Atheris runner (WSL, CPython 3.11) and the analysis.

``judge`` is the one kill rule, applied to the confirmation replays of one witness:
  ordinary      buggy ``raise`` AND fixed ``ok``, both stable across every confirmation, no
                replay error (two different exceptions are NOT a kill);
  posthoc /
  differential  stable, both comparable (``ok``/``raise``) and different (unchanged; the
                differential mode stays an oracle-assisted upper bound).

``classify_probe`` separates applicability from infrastructure: a legitimately unsupported
signature (no_signature, instance_method_receiver, variadic_signature, unsupported_parameter)
is ``atheris_ineligible``; import, runtime, probe, environment, interpreter or view failures -
anything else - are ``infrastructure_failure``.

``load_results`` validates an Atheris contract and its results TOGETHER: exact design version,
script/inner/preparation/manifest hashes, budget, tolerance and corpus cap; exactly
qualified x modes x seeds unique cells with no missing, duplicate, extra, malformed or stale
row; every eligible row usable (reached, views unchanged, within budget, cleanups, allowed end
reason, no replay error, kill recomputed from its confirmations). A target is Atheris-usable
only if all its rows are; an eligible target with any unusable row is an explicit
infrastructure exclusion, never a non-kill.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

DESIGN_VERSION = "oneiros_native_generated_tests_atheris_v4"
MODES = ("ordinary", "posthoc", "differential")
SEEDS = (42, 43, 44)
CONFIRMATIONS = 2
BUDGET = 600
CORPUS_CAP = 2000
USABLE_END = ("completed", "cpu_budget_exhausted")
APPLICABILITY_REASONS = ("no_signature", "instance_method_receiver", "variadic_signature")
APPLICABILITY_PREFIXES = ("unsupported_parameter:",)
STATUSES = ("eligible", "atheris_ineligible", "infrastructure_failure")


class AtherisRefused(SystemExit):
    """The Atheris contract or results cannot be used."""


def canonical_sha(value: Any) -> str:
    """Identical to native_generation_io.contract_sha (kept local: stdlib, import-free)."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"))
                          .encode("utf-8")).hexdigest()


def classify_probe(info: Mapping[str, Any]) -> str:
    if info.get("eligible") is True:
        return "eligible"
    reason = info.get("reason")
    if isinstance(reason, str) and (reason in APPLICABILITY_REASONS
                                    or reason.startswith(APPLICABILITY_PREFIXES)):
        return "atheris_ineligible"
    return "infrastructure_failure"


def tolerance(budget: float) -> float:
    return 1.0 + 0.02 * budget


def judge(mode: str, buggy: Sequence[list], fixed: Sequence[list],
          replay_error: Optional[str] = None) -> Dict[str, Any]:
    if replay_error:
        return {"kill": False, "replay_error": replay_error}
    buggy, fixed = list(buggy), list(fixed)
    if len(buggy) != CONFIRMATIONS or len(fixed) != CONFIRMATIONS:
        return {"kill": False, "stable": False, "buggy_all": buggy, "fixed_all": fixed}
    stable = all(x == buggy[0] for x in buggy) and all(x == fixed[0] for x in fixed)
    b, f = buggy[0], fixed[0]
    if mode == "ordinary":
        kill = stable and b[0] == "raise" and f[0] == "ok"
    else:
        kill = stable and b[0] in ("ok", "raise") and f[0] in ("ok", "raise") and b != f
    return {"kill": bool(kill), "stable": stable, "buggy": b, "fixed": f,
            "buggy_all": buggy, "fixed_all": fixed}


def contract_hash(contract: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()


def _row_problems(row: Mapping[str, Any], budget: float) -> List[str]:
    problems = []
    for field, want in (("reached", True), ("views_unchanged", True), ("within_budget", True),
                        ("cleanup_ok", True), ("replay_cleanup_ok", True)):
        if row.get(field) is not want:
            problems.append(f"{field} is not true")
    if row.get("end_reason") not in USABLE_END:
        problems.append(f"end_reason {row.get('end_reason')!r}")
    agg = row.get("aggregate_cpu_seconds")
    if not isinstance(agg, (int, float)) or agg > budget + tolerance(budget):
        problems.append("aggregate CPU over the budget")
    if row.get("replay_errors") != 0:
        problems.append("replay errors")
    checks = row.get("confirmations")
    if not isinstance(checks, list):
        return problems + ["no confirmation evidence"]
    recomputed = [judge(row["mode"], c.get("buggy_all") or [], c.get("fixed_all") or [],
                        c.get("replay_error")) for c in checks]
    if any(r["kill"] != c.get("kill") for r, c in zip(recomputed, checks)):
        problems.append("a witness verdict disagrees with its confirmations")
    kills = sum(r["kill"] for r in recomputed)
    if row.get("confirmed") != kills or row.get("kill") is not (kills > 0) or \
            row.get("witnesses") != len(checks):
        problems.append("kill/confirmed/witnesses disagree with the confirmations")
    return problems


def _eligibility_problems(contract: Mapping[str, Any], qualified: Sequence[str],
                          prep_rows: Mapping[str, Mapping[str, Any]]) -> List[str]:
    problems = []
    live = contract.get("live_views")
    expected_live = {k: dict(prep_rows[k]["view_manifest_sha256"]) for k in qualified}
    if live != expected_live or contract.get("live_views_sha256") != canonical_sha(expected_live):
        problems.append("contract live views differ from the preparation manifests")
    eligibility = contract.get("eligibility")
    if not isinstance(eligibility, dict) or set(eligibility) != set(qualified) or \
            contract.get("eligibility_sha256") != canonical_sha(eligibility):
        return problems + ["contract eligibility map is incomplete or unbound"]
    for key, e in eligibility.items():
        row = prep_rows[key]
        if e.get("target_key") != key or e.get("module") != row["module"] or \
                e.get("qualname") != row["qualname"]:
            problems.append(f"eligibility identity differs for {key}")
        if e.get("views") != expected_live[key] or \
                e.get("prep_record_sha256") != canonical_sha(row):
            problems.append(f"eligibility views/preparation record differ for {key}")
        status = e.get("status")
        if status not in STATUSES or classify_probe(
                {"eligible": status == "eligible", "reason": e.get("reason")}) != status:
            problems.append(f"eligibility status {status!r} inconsistent with reason "
                            f"{e.get('reason')!r} for {key}")
        if status == "eligible" and not isinstance(e.get("plan"), list):
            problems.append(f"eligible target without an argument plan: {key}")
    return problems


def load_results(results_path: Path, contract_path: Path, *, qualified: Sequence[str],
                 manifest_sha256: str, prep: Mapping[str, str],
                 prep_rows: Mapping[str, Mapping[str, Any]], script_sha256: str,
                 inner_sha256: str, verdicts_sha256: str, seeds: Sequence[int] = SEEDS, budget: int = BUDGET
                 ) -> Dict[str, Any]:
    contract = json.loads(Path(contract_path).read_text(encoding="utf-8"))
    expected = {"design_version": DESIGN_VERSION, "script_sha256": script_sha256,
                "inner_sha256": inner_sha256, "verdicts_sha256": verdicts_sha256,
                "prep": dict(prep),
                "manifest_sha256": manifest_sha256, "budget_cpu_seconds": budget,
                "tolerance": tolerance(budget), "seeds": list(seeds), "modes": list(MODES),
                "corpus_cap": CORPUS_CAP}
    wrong = sorted(k for k, v in expected.items() if contract.get(k) != v)
    if wrong:
        raise AtherisRefused(f"REFUSED: Atheris contract differs in {wrong}")
    problems = _eligibility_problems(contract, qualified, prep_rows)
    if problems:
        raise AtherisRefused(f"REFUSED: Atheris contract live views / eligibility: {problems[:3]}")
    eligibility = contract["eligibility"]
    chash = contract_hash(contract)
    cells = {f"{t}::{m}::{s}" for t in qualified for m in MODES for s in seeds}
    rows: Dict[str, Dict[str, Any]] = {}
    data = Path(results_path).read_bytes()
    if data and not data.endswith(b"\n"):
        raise AtherisRefused("REFUSED: Atheris results end with a partial line")
    for number, line in enumerate(data.decode("utf-8").splitlines(), 1):
        try:
            row = json.loads(line)
        except ValueError:
            raise AtherisRefused(f"REFUSED: Atheris line {number} malformed") from None
        key = row.get("key")
        if row.get("contract_sha256") != chash:
            raise AtherisRefused(f"REFUSED: Atheris line {number} is stale (other contract)")
        if key not in cells:
            raise AtherisRefused(f"REFUSED: Atheris line {number}: unexpected cell {key!r}")
        if key in rows:
            raise AtherisRefused(f"REFUSED: Atheris line {number}: duplicate cell {key}")
        if key != f"{row.get('target_key')}::{row.get('mode')}::{row.get('seed')}":
            raise AtherisRefused(f"REFUSED: Atheris line {number}: key/fields disagree")
        e = eligibility[row["target_key"]]
        if row.get("eligible") is not (e["status"] == "eligible") or \
                row.get("status") != e["status"] or row.get("reason") != e["reason"]:
            raise AtherisRefused(f"REFUSED: Atheris line {number} disagrees with the frozen "
                                 f"eligibility map for {row['target_key']}")
        rows[key] = row
    if set(rows) != cells:
        raise AtherisRefused(f"REFUSED: Atheris grid incomplete: {len(cells - set(rows))} of "
                             f"{len(cells)} cells missing")
    targets = {}
    for t in qualified:
        trow = [rows[f"{t}::{m}::{s}"] for m in MODES for s in seeds]
        status = eligibility[t]["status"]
        if status == "atheris_ineligible":
            targets[t] = {"status": "atheris_ineligible", "reason": eligibility[t]["reason"]}
            continue
        if status == "infrastructure_failure":
            targets[t] = {"status": "infrastructure_excluded",
                          "problems": {"probe": [eligibility[t]["reason"]]}}
            continue
        problems = {f"{r['mode']}::{r['seed']}": p for r in trow if (p := _row_problems(r, budget))}
        if problems:
            targets[t] = {"status": "infrastructure_excluded", "problems": problems}
            continue
        targets[t] = {"status": "usable",
                      "kill_by_mode": {m: any(rows[f"{t}::{m}::{s}"]["kill"] for s in seeds)
                                       for m in MODES}}
    return {"contract_sha256": chash, "cells": len(rows), "targets": targets,
            "live_views_sha256": contract["live_views_sha256"],
            "eligibility_sha256": contract["eligibility_sha256"],
            "counts": {s: sum(v["status"] == s for v in targets.values())
                       for s in ("usable", "atheris_ineligible", "infrastructure_excluded")}}
