"""v2.5 CPU program 2, Phase 6: candidate overlay and native targets for ALL train repositories
with eligible fragments (strict train-only loader, fail-closed access audit).

1. ``converted_candidates_r2r.jsonl``: the stage1_r2 candidates, with every Phase 6A
   recoverable row replaced by its recovered module (rule, before/after hashes and removed
   aliases recorded). Nothing else changes.
2. ``targets.jsonl``: one target per official task that has an eligible fragment; interpreters
   from the frozen attempt spec (BugsInPy ``bug.info``; SWE-bench version maps; the canary
   spec maps for django and sympy).

    python scripts/v25_native_targets_r2.py --out results/sft_root_cause/v25_native_r2
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

R2 = "results/sft_root_cause/v25_corpus_stage1_r2/converted_candidates.jsonl"
RECOVERY = "results/sft_root_cause/v25_policy_recovery/policy_recovery_rows.jsonl"
CANARY_SPEC = "results/sft_root_cause_v25_native_env_canary_spec.json"


def sha_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def overlay(cands: list, recovery: dict) -> list:
    from scripts.v25_converted_corpus import canonical_ids
    from scripts.v25_policy_recovery import RULE
    out = []
    for c in cands:
        r = recovery.get(c["index"])
        if r is not None and r["category"].startswith("recoverable"):
            module = r["after_module"]
            c = {**c, "conversion": {"accepted": True, "module": module,
                                     "module_sha256": hashlib.sha256(module.encode()).hexdigest(),
                                     "identities": canonical_ids(module),
                                     "recovery": {"rule": RULE,
                                                  "before_sha256": r["before_sha256"],
                                                  "after_sha256": r["after_sha256"],
                                                  "removed_aliases": r["removed_aliases"]}}}
        out.append(c)
    return out


def unittest_selectors(project: str, bug: str) -> list:
    """BugsInPy ``python -m unittest [-q] a.b.Class.test_x`` -> ``a/b.py::Class::test_x``
    (attempt-2 infrastructure fix: these ids produced no selector in attempt 1)."""
    text = (ROOT / "data" / "BugsInPy_repo" / "projects" / project / "bugs" / bug /
            "run_test.sh").read_text(encoding="utf-8")
    out = []
    for line in text.splitlines():
        parts = line.split()
        if parts[:3] != ["python", "-m", "unittest"]:
            continue
        for tok in parts[3:]:
            if tok.startswith("-"):
                continue
            comps = tok.split(".")
            cls = next((i for i, c in enumerate(comps) if c[:1].isupper()), None)
            if cls is None or cls == 0 or cls != len(comps) - 2:
                out.append(f"unresolved:{tok}")
                continue
            out.append("/".join(comps[:cls]) + ".py::" + "::".join(comps[cls:]))
    return out


def _test_patch(patch_dir: str, task: str) -> str | None:
    path = ROOT / patch_dir / f"{task}.json"
    return json.loads(path.read_text(encoding="utf-8"))["test_patch"] if path.is_file() else None


def normalise_selectors(selectors: list, test_patch: str | None) -> list:
    """v2.6 generic selector fixes (attempt a3):
    R1 - Django 4.x reports ``test_x (module.Class.test_x)``: a selector ending in a doubled
    method name is reduced to ``module.Class.test_x``;
    R2 - an ``unresolved:test_x`` selector resolves to ``path::test_x`` when exactly one file of
    the hash-verified official test patch adds ``def test_x(``."""
    import re
    files = {}
    if test_patch:
        for part in re.split(r"(?=^diff --git )", test_patch, flags=re.M):
            m = re.match(r"diff --git a/(\S+)", part)
            if m:
                files[m.group(1)] = part
    out = []
    for sel in selectors:
        if sel.startswith("django:"):
            comps = sel.split(":", 1)[1].split(".")
            if len(comps) >= 2 and comps[-1] == comps[-2]:
                sel = "django:" + ".".join(comps[:-1])
        elif sel.startswith("unresolved:"):
            name = sel.split(":", 1)[1]
            hits = [p for p, text in files.items()
                    if re.search(rf"^\+\s*def {re.escape(name)}\(", text, re.M)]
            if len(hits) == 1:
                sel = f"{hits[0]}::{name}"
        out.append(sel)
    return out


def interpreter_for(project: str, dataset: str, prov: dict, canary: dict) -> str | None:
    from scripts.v25_native_attempt_spec import SWEBENCH_INTERPRETERS, bugsinpy_python
    if dataset == "BugsInPy":
        return bugsinpy_python(project, str(prov["bug_id"]))
    maps = dict(SWEBENCH_INTERPRETERS)
    maps.update({p: r["interpreter_map"] for p, r in canary["repositories"].items()})
    m = maps.get(project, {})
    return m.get(str(prov.get("version"))) or m.get("*")


def main(argv=None) -> int:
    from scripts.v25_converted_corpus import access_receipt, install_audit, load_train_records
    from scripts.v25_native_targets import _bugsinpy_tests, _swebench_selectors
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True)
    parser.add_argument("--normalise-selectors", action="store_true",
                        help="v2.6 a3: apply normalise_selectors (R1, R2)")
    parser.add_argument("--patches", default="results/sft_root_cause/v25_native_r2/patches")
    parser.add_argument("--targets-name", default="targets.jsonl",
                        help="successor targets file; an existing overlay must reproduce")
    args = parser.parse_args(argv)
    install_audit()
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    if (out / args.targets_name).exists():
        raise SystemExit(f"REFUSED: {args.targets_name} exists")
    existing = out / "converted_candidates_r2r.jsonl"
    previous = existing.read_bytes() if existing.exists() else None
    records = {r["id"]: r for r in load_train_records()}
    cands = [json.loads(l) for l in (ROOT / R2).read_text(encoding="utf-8").splitlines()]
    recovery = {r["index"]: r for r in map(json.loads, (ROOT / RECOVERY).read_text(
        encoding="utf-8").splitlines())}
    merged = overlay(cands, recovery)
    data = ("\n".join(json.dumps(c, sort_keys=True) for c in merged) + "\n").encode("utf-8")
    if previous is not None and previous != data:
        raise SystemExit("REFUSED: the candidate overlay does not reproduce")
    existing.write_bytes(data)
    canary = json.loads((ROOT / CANARY_SPEC).read_text(encoding="utf-8"))
    by_task = {}
    for c in merged:
        if c["execution_mode"] == "function_assertion" or not c["conversion"]["accepted"]:
            continue
        by_task.setdefault(records[c["id"]]["provenance"]["official_task_id"], []).append(c)
    rows = []
    for task in sorted(by_task):
        frags = sorted(by_task[task], key=lambda c: c["index"])
        rec = records[frags[0]["id"]]
        prov, c0 = rec["provenance"], frags[0]
        swe = c0["dataset"] != "BugsInPy"
        row = {"task": task, "project": c0["project"], "canary": False,
               "dataset": "SWE-bench Verified" if swe else "BugsInPy",
               "repository_url": prov.get("repository_url") or
               f"https://github.com/{prov['repository']}",
               "interpreter": interpreter_for(c0["project"], c0["dataset"], prov, canary),
               "version": prov.get("version"),
               "target_file": (prov.get("patched_source_paths") or [None])[0],
               "target_symbols": rec.get("target_symbols") or [],
               "fragments": [{"index": x["index"], "id": x["id"],
                              "module_sha256": x["conversion"]["module_sha256"],
                              "recovered": "recovery" in x["conversion"]} for x in frags]}
        if swe:
            row.update(buggy_commit=prov["base_commit"],
                       gold_patch_sha256=prov["gold_patch_sha256"],
                       test_patch_sha256=prov["test_patch_sha256"],
                       selectors=(normalise_selectors(_swebench_selectors(prov),
                                                      _test_patch(args.patches, task))
                                  if args.normalise_selectors else _swebench_selectors(prov)),
                       test_paths=prov.get("test_paths"))
        else:
            row.update(buggy_commit=prov["buggy_commit"], fixed_commit=prov["fixed_commit"],
                       bugsinpy_bug=prov["bug_id"],
                       selectors=_bugsinpy_tests(c0["project"], str(prov["bug_id"]))
                       or unittest_selectors(c0["project"], str(prov["bug_id"])))
        rows.append(row)
    (out / args.targets_name).write_bytes(("\n".join(json.dumps(r, sort_keys=True) for r in rows)
                                         + "\n").encode("utf-8"))
    access = access_receipt()
    print(json.dumps({"targets": len(rows), "by_project": {p: sum(r["project"] == p
                                                                  for r in rows)
                                                           for p in sorted({r["project"]
                                                                            for r in rows})},
                      "fragments": sum(len(r["fragments"]) for r in rows),
                      "recovered_fragments": sum(f["recovered"] for r in rows
                                                 for f in r["fragments"]),
                      "unmapped_interpreter": [r["task"] for r in rows if not r["interpreter"]],
                      "candidates_sha256": sha_file(out / "converted_candidates_r2r.jsonl"),
                      "targets_sha256": sha_file(out / args.targets_name),
                      "access_passed": access["passed"]}, indent=1))
    return 0 if access["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
