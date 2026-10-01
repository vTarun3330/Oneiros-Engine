"""v2.5 stage-1 corpus (r2): convert the adapter's exact train-only SFT examples to
``pytest_module_v1`` (``prepare``), then - after two WSL repeatability runs - write the
corrected receipt (``receipt``).

r2 corrections over the historical stage-1 receipt (bf3f9828, preserved):
- an access audit is installed BEFORE any corpus load; it records every corpus / protected path
  opened and FAILS on non-train shards, the combined records/splits, the combined complexity
  manifest (it carries non-train metadata), external indexes, reserved-confirmation ids or sealed
  results. Complexity is recomputed from each TRAIN record's buggy-side code with the frozen
  ``oneiros_buggy_ast_complexity_v1`` policy;
- every converted module must pass the frozen v2.4 candidate policy (static_check);
- full provenance; repeatability runs (not independent validation); duplicate accounting by
  raw text, canonical AST, (function, canonical test) and near-duplicate clusters; an exclusion
  audit; a row-level comparison with the historical stage 1.

    python scripts/v25_converted_corpus.py prepare --examples <rebuilt.jsonl> --out <dir>
    wsl -u root -- bash scripts/wsl_native_python.sh scripts/v25_verify_converted_wsl.py ...
    python scripts/v25_converted_corpus.py receipt --dir <dir>
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CORPUS_REL = "data/corpus/v4_1_research_hardened_candidate"
CORPUS = ROOT / CORPUS_REL
EXAMPLES_SHA = "30e9436f7aa16709b8710ba410d653666c3eb8e11d4bdf9de55979a6f4c02167"
REBUILD_RECEIPT = "results/sft_root_cause_native_v24_sft_training_shape_audit.json"
HISTORICAL = {"receipt": "results/sft_root_cause_v25_converted_corpus_stage1.json",
              "dir": "results/sft_root_cause/v25_corpus_stage1"}
RECEIPT = "results/sft_root_cause_v25_converted_corpus_stage1_r2.json"
V24_PANEL_PROJECTS = ("marshmallow", "pyparsing", "attrs", "cattrs", "humanize", "tomlkit",
                      "cachetools", "sqlglot")
ALLOWED_CORPUS_FILES = {f"{CORPUS_REL}/manifest.json",
                        f"{CORPUS_REL}/development_view/manifest.json",
                        f"{CORPUS_REL}/development_view/train.records.json",
                        f"{CORPUS_REL}/development_view/training_exclusions.json"}
OPENED: list = []
VIOLATIONS: list = []


def _rel(path) -> str | None:
    try:
        return Path(os.fsdecode(path)).resolve().relative_to(ROOT).as_posix()
    except (ValueError, OSError, TypeError):
        return None


def forbidden_reason(rel: str) -> str | None:
    """Fail-closed rule for this stage (stricter than the harness's protected list)."""
    name = rel.rsplit("/", 1)[-1]
    if rel.startswith("data/corpus/"):
        return None if rel in ALLOWED_CORPUS_FILES else "non_train_or_combined_corpus_file"
    if rel.startswith("results/sealed_final"):
        return "sealed_final"
    if name == "unopened_confirmation.ids.json":
        return "reserved_confirmation"
    return None


def _audit(event, args):
    if event != "open" or not args or not isinstance(args[0], (str, bytes, os.PathLike)):
        return
    rel = _rel(args[0])
    if rel is None or not (rel.startswith(("data/", "results/sealed"))
                           or rel.endswith("unopened_confirmation.ids.json")):
        return
    OPENED.append(rel)
    reason = forbidden_reason(rel)
    if reason:
        VIOLATIONS.append({"path": rel, "reason": reason})
        raise PermissionError(f"BLOCKED ({reason}): {rel}")


_INSTALLED = []


def install_audit() -> None:
    """Process-wide and irreversible (audit hooks cannot be removed): CLI use only."""
    if not _INSTALLED:
        sys.addaudithook(_audit)
        _INSTALLED.append(True)


def access_receipt() -> dict:
    return {"rule": "only " + ", ".join(sorted(ALLOWED_CORPUS_FILES)) + " may be opened under "
                    "data/corpus; sealed_final and reserved-confirmation ids never",
            "opened": sorted(set(OPENED)), "violations": VIOLATIONS,
            "combined_complexity_manifest_opened": any(
                p.endswith("complexity_manifest.json") for p in OPENED),
            "passed": not VIOLATIONS}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def policy_status(module: str, target_name: str) -> str:
    """The frozen v2.4 candidate policy (static_check) on a complete converted module."""
    from scripts.native_generated_tests_execute_wsl import static_check
    return static_check(module, target_name.split(".")[-1] or "target")["status"]


def canonical_ids(module: str) -> dict:
    """Raw-text, canonical-AST and near-duplicate (constants abstracted) identities."""
    tree = ast.parse(module)
    canon = hashlib.sha256(ast.dump(tree, annotate_fields=False).encode()).hexdigest()

    class Abstract(ast.NodeTransformer):
        def visit_Constant(self, node):
            return ast.copy_location(ast.Constant(value=type(node.value).__name__), node)
    near = hashlib.sha256(ast.dump(Abstract().visit(ast.parse(module)),
                                   annotate_fields=False).encode()).hexdigest()
    return {"raw": hashlib.sha256(module.encode()).hexdigest(), "canonical_ast": canon,
            "near_duplicate_cluster": near}


def load_train_records() -> list:
    """Train shard only, with every train check of ``verify_development_view`` EXCEPT hashing
    the combined complexity sidecar (it carries non-train metadata and is not used here):
    view schema, parent-corpus binding, sealed split declared excluded, train shard hash and
    membership, per-record content hashes and schema, group disjointness, exclusions hash."""
    from harness import corpus_view as cv
    parent = json.loads((CORPUS / "manifest.json").read_text(encoding="utf-8"))
    view_dir = CORPUS / "development_view"
    view = json.loads((view_dir / "manifest.json").read_text(encoding="utf-8"))
    files = parent.get("files", {})
    if view.get("schema_version") != cv.VIEW_SCHEMA_VERSION or \
            view.get("source_corpus_id") != parent.get("corpus_id") or \
            view.get("source_records_sha256") != files.get("records.json", {}).get("sha256") or \
            view.get("source_splits_sha256") != files.get("splits.json", {}).get("sha256") or \
            set(view.get("sealed_splits_excluded", [])) != set(cv.SEALED_SPLITS) or \
            "train" not in view.get("included_splits", []):
        raise SystemExit("REFUSED: development view manifest fails its train gates")
    descriptor = view["splits"]["train"]
    shard = view_dir / descriptor["filename"]
    if sha(shard) != descriptor["sha256"]:
        raise SystemExit("REFUSED: train shard hash")
    records = json.loads(shard.read_text(encoding="utf-8"))
    ids = [r["id"] for r in records]
    if len(records) != descriptor["record_count"] or \
            hashlib.sha256(json.dumps(ids, sort_keys=True, separators=(",", ":"))
                           .encode()).hexdigest() != descriptor["record_ids_sha256"] or \
            parent["splits"]["train"]["record_ids_sha256"] != descriptor["record_ids_sha256"]:
        raise SystemExit("REFUSED: train shard membership")
    groups = set()
    for r in records:
        if cv.record_content_hash(r) != r.get("content_hash") or \
                r.get("schema_version") != parent.get("schema_version"):
            raise SystemExit("REFUSED: modified train record")
        groups.add(r.get("group_id"))
    exclusions = view["training_exclusions"]
    epath = view_dir / exclusions["filename"]
    if sha(epath) != exclusions["sha256"]:
        raise SystemExit("REFUSED: training exclusions hash")
    excluded = {e["record_id"] for e in json.loads(epath.read_text(encoding="utf-8"))}
    return [r for r in records if r["id"] not in excluded]


def train_complexity(code: str, entry_point: str) -> str:
    from harness.function_complexity import analyze_function_complexity
    try:
        return analyze_function_complexity(code, entry_point).tier
    except ValueError:
        return "unscorable"


def prepare(examples: Path, out: Path) -> int:
    install_audit()                                  # before ANY corpus/config load
    from scripts import v25_module_conversion as mc
    if sha(examples) != EXAMPLES_SHA:
        raise SystemExit("REFUSED: examples differ from the audited exact rebuild")
    records = {r["id"]: r for r in load_train_records()}
    rows = [json.loads(l) for l in examples.read_text(encoding="utf-8").splitlines()]
    out.mkdir(parents=True, exist_ok=True)
    if (out / "converted_candidates.jsonl").exists():
        raise SystemExit("REFUSED: candidates exist; earlier attempts stay quarantined")
    lines = []
    for i, ex in enumerate(rows):
        rec = records[ex["id"]]
        key = hashlib.sha256((ex["prompt_sha256"] + "\x00" + ex["completion"]).encode())
        synthetic = ex["execution_mode"] == "function_assertion"
        base = {"index": i, "example_key": key.hexdigest(), "id": ex["id"],
                "dataset": ex["dataset"], "project": ex["project"],
                "bug_family": ex["bug_family"], "execution_mode": ex["execution_mode"],
                "entry_point": rec["entry_point"], "effective_repeats": ex["effective_repeats"],
                "original_completion_sha256": mc.sha256(ex["completion"]),
                "completion_chars": len(ex["completion"]), "group_id": rec.get("group_id"),
                "provenance_project": (rec.get("provenance") or {}).get("project"),
                "complexity": (train_complexity(rec["code_under_test"], rec["entry_point"])
                               if synthetic else "repository_unscored")}
        if synthetic:
            conv = mc.convert_synthetic(ex["completion"], rec["entry_point"],
                                        rec["reference_code"], rec["code_under_test"])
        else:
            conv = mc.convert_repository(ex["completion"], rec.get("support_context") or "")
        if conv["accepted"]:
            target = rec["entry_point"] or ((rec.get("target_symbols") or ["target"])[0])
            status = policy_status(conv["module"], target)
            if status != "ok":
                conv = {"accepted": False, "reason": f"candidate_policy:{status}"}
            else:
                conv["identities"] = canonical_ids(conv["module"])
                if synthetic:
                    base.update(reference_code=rec["reference_code"],
                                code_under_test=rec["code_under_test"])
        lines.append(json.dumps({**base, "conversion": conv}, sort_keys=True))
    target_file = out / "converted_candidates.jsonl"
    target_file.write_bytes(("\n".join(lines) + "\n").encode("utf-8"))
    access = access_receipt()
    (out / "prepare_access_receipt.json").write_bytes(
        (json.dumps(access, indent=1, sort_keys=True) + "\n").encode("utf-8"))
    summary = Counter((json.loads(l)["execution_mode"], json.loads(l)["conversion"]["accepted"],
                       json.loads(l)["conversion"].get("reason")) for l in lines)
    print(json.dumps({"rows": len(lines), "sha256": sha(target_file),
                      "access_passed": access["passed"], "opened": access["opened"],
                      "by_mode_accepted_reason": {" | ".join(map(str, k)): v
                                                  for k, v in sorted(summary.items(),
                                                                     key=str)}}, indent=1))
    return 0 if access["passed"] else 2


def _dist(values) -> dict:
    values = sorted(values)
    if not values:
        return {}
    return {"n": len(values), "p50": statistics.median(values),
            "p90": values[int(0.9 * (len(values) - 1))], "max": values[-1]}


def _share(rows, field) -> dict:
    c = Counter(str(r[field]) for r in rows)
    return {k: round(v / len(rows), 4) for k, v in sorted(c.items())} if rows else {}


def receipt(directory: Path) -> int:
    cand = directory / "converted_candidates.jsonl"
    runs = sorted(directory.glob("verification_run[0-9].jsonl"))
    if len(runs) != 2 or sha(runs[0]) != sha(runs[1]):
        raise SystemExit("REFUSED: two byte-identical repeatability runs are required")
    install_audit()
    from harness import native_launch_gate as gate
    from scripts.native_rehearsal_rebuild_v22 import publish_once
    env = json.loads((directory / "synthetic_env_manifest.json").read_text(encoding="utf-8"))
    prep_access = json.loads((directory / "prepare_access_receipt.json")
                             .read_text(encoding="utf-8"))
    candidates = [json.loads(l) for l in cand.read_text(encoding="utf-8").splitlines()]
    verdicts = {v["index"]: v for v in map(json.loads, runs[0].read_text(encoding="utf-8")
                                           .splitlines())}
    timing = {}
    for t in directory.glob("verification_run1.timing.jsonl"):
        timing = {x["index"]: x for x in map(json.loads, t.read_text(encoding="utf-8")
                                             .splitlines())}
    rows = []
    for c in candidates:
        v = verdicts.get(c["index"], {})
        ids = c["conversion"].get("identities") or {}
        rows.append({**{k: c[k] for k in ("index", "id", "dataset", "project", "bug_family",
                                           "complexity", "execution_mode", "effective_repeats",
                                           "group_id", "provenance_project",
                                           "completion_chars", "original_completion_sha256")},
                     "converted": c["conversion"]["accepted"],
                     "reject_reason": c["conversion"].get("reason"),
                     "raw": ids.get("raw"), "canonical": ids.get("canonical_ast"),
                     "near": ids.get("near_duplicate_cluster"),
                     "module_chars": len(c["conversion"].get("module") or ""),
                     "verification": v.get("status", "not_run"), "class": v.get("class"),
                     "verified": v.get("accepted") is True,
                     "seconds": (timing.get(c["index"]) or {}).get("seconds")})
    ok = [r for r in rows if r["verified"]]
    occ = Counter(r["raw"] for r in ok)
    near_clusters = defaultdict(set)
    for r in ok:
        near_clusters[r["near"]].add(r["canonical"])
    duplicates = {
        "rows_verified": len(ok),
        "unique_record_ids": len({r["id"] for r in ok}),
        "unique_function_group_ids": len({r["group_id"] for r in ok}),
        "unique_raw_module_texts": len(occ),
        "unique_canonical_ast_modules": len({r["canonical"] for r in ok}),
        "unique_function_canonical_test_pairs": len({(r["group_id"], r["canonical"])
                                                     for r in ok}),
        "repeated_module_text_occurrences": sum(n - 1 for n in occ.values()),
        "exact_duplicate_example_occurrences": sum(
            n - 1 for n in Counter((r["id"], r["raw"]) for r in ok).values()),
        "near_duplicate_clusters": len(near_clusters),
        "near_duplicate_clusters_with_several_canonical_tests": sum(
            len(v) > 1 for v in near_clusters.values()),
        "effective_training_weight_verified": sum(r["effective_repeats"] for r in ok),
        "note": "the verified rows are NOT 5k distinct tests: identical module texts recur "
                "across sibling mutants of the same function"}
    excluded = [r for r in rows if r["execution_mode"] == "function_assertion"
                and not r["verified"]]
    synthetic = [r for r in rows if r["execution_mode"] == "function_assertion"]
    by_reason = defaultdict(list)
    for r in excluded:
        by_reason[r["reject_reason"] or f"{r['verification']}:{r['class']}"].append(r)
    exclusion_audit = {
        reason: {"rows": len(rs), "dataset": _share(rs, "dataset"),
                 "complexity": _share(rs, "complexity"), "bug_family": _share(rs, "bug_family"),
                 "functions": len({r["group_id"] for r in rs}),
                 "top_functions": Counter(r["group_id"] for r in rs).most_common(8),
                 "completion_chars": _dist([r["completion_chars"] for r in rs]),
                 "seconds": _dist([r["seconds"] for r in rs if r["seconds"] is not None])}
        for reason, rs in sorted(by_reason.items())}
    exclusion_audit["_reference_all_synthetic"] = {
        "rows": len(synthetic), "dataset": _share(synthetic, "dataset"),
        "complexity": _share(synthetic, "complexity"),
        "completion_chars": _dist([r["completion_chars"] for r in synthetic]),
        "seconds_verified": _dist([r["seconds"] for r in ok if r["seconds"] is not None])}
    lost_functions = {r["group_id"] for r in excluded} - {r["group_id"] for r in ok}
    exclusion_audit["_functions_lost_entirely"] = {
        "count": len(lost_functions),
        "complexity": _share([r for r in synthetic if r["group_id"] in lost_functions],
                             "complexity")}
    hist = {}
    hdir = ROOT / HISTORICAL["dir"]
    if (hdir / "verification_run1.jsonl").is_file():
        old = {v["index"]: v for v in map(json.loads, (hdir / "verification_run1.jsonl")
                                          .read_text(encoding="utf-8").splitlines())}
        old_conv = {c["index"]: c["conversion"] for c in map(
            json.loads, (hdir / "converted_candidates.jsonl").read_text(encoding="utf-8")
            .splitlines())}

        def state(conv, verdict):
            return ["converted" if conv.get("accepted") else
                    f"rejected:{conv.get('reason')}", verdict.get("accepted") is True]
        changed = []
        for r in rows:
            before = state(old_conv.get(r["index"], {}), old.get(r["index"], {}))
            after = ["converted" if r["converted"] else f"rejected:{r['reject_reason']}",
                     r["verified"]]
            if before != after:
                changed.append({"index": r["index"], "execution_mode": r["execution_mode"],
                                "project": r["project"], "old": before, "new": after,
                                "reason": r["reject_reason"]})
        hist = {"historical_receipt_sha256": sha(ROOT / HISTORICAL["receipt"]),
                "historical_verified": sum(v.get("accepted") is True for v in old.values()),
                "r2_verified": len(ok), "changed_rows": len(changed),
                "changed_by_transition": dict(Counter(
                    f"{c['old']}->{c['new']}:{c['reason']}" for c in changed)),
                "changed_rows_all": changed,
                "explanation": "r2 repository conversion rejects fragments that define no "
                               "top-level test and modules refused by the frozen candidate "
                               "policy; the historical converter accepted every fragment"}
    leak = sorted({r["provenance_project"] for r in rows
                   if (r["provenance_project"] or "").lower() in V24_PANEL_PROJECTS})
    by = lambda f: {str(k): {"rows": sum(1 for r in rows if r[f] == k),  # noqa: E731
                             "verified": sum(1 for r in ok if r[f] == k),
                             "verified_functions": len({r["group_id"] for r in ok
                                                        if r[f] == k}),
                             "verified_canonical_tests": len({r["canonical"] for r in ok
                                                              if r[f] == k})}
                    for k in sorted({r[f] for r in rows}, key=str)}
    ident = gate.source_identity(ROOT)
    import subprocess
    commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True,
                            text=True).stdout.strip()
    access = access_receipt()
    out = {
        "schema_version": "oneiros_v25_converted_corpus_stage1_r2_v1",
        "supersedes": {"receipt": HISTORICAL["receipt"],
                       "sha256": sha(ROOT / HISTORICAL["receipt"]),
                       "why": "its validation_or_sealed_access=false claim was incorrect at the "
                              "metadata level (the combined complexity manifest was opened); "
                              "duplicate accounting and provenance were incomplete"},
        "output_type": "pytest_module_v1",
        "provenance": {
            "git_commit": commit,
            "executable_tree_sha256": ident["executable_tree_sha256"],
            "protocol_v2_5_sha256": ident["protocol_sha256"].get(
                "docs/SFT_ROOT_CAUSE_NATIVE_GENERATED_TEST_PROTOCOL_V2_5.md"),
            "source_examples_sha256": EXAMPLES_SHA,
            "source_examples_rebuild_receipt_sha256": sha(ROOT / REBUILD_RECEIPT),
            "source_examples_rebuild_note": "rebuilt with the training-commit loader, which "
                                            "itself opened the combined complexity manifest "
                                            "(historical; the original training did the same)",
            "scripts_sha256": {p: sha(ROOT / p) for p in (
                "scripts/v25_module_conversion.py", "scripts/v25_converted_corpus.py",
                "scripts/v25_verify_converted_wsl.py",
                "scripts/native_generated_tests_execute_wsl.py",
                "scripts/native_sandbox_inner.sh", "harness/function_complexity.py")},
            "candidate_policy": "static_check in native_generated_tests_execute_wsl.py "
                                "(hash above)",
            "synthetic_environment": env,
            "converted_candidates_sha256": sha(cand),
            "repeatability_runs_sha256": [sha(p) for p in runs],
            "repeatability_note": "two runs of the same deterministic pipeline; repeatability, "
                                  "NOT independent validation"},
        "access_audit": {"prepare": prep_access, "receipt": access,
                         "passed": prep_access["passed"] and access["passed"]},
        "counts": {"examples": len(rows), "converted": sum(r["converted"] for r in rows),
                   "verified_positive_rows": len(ok)},
        "duplicates": duplicates,
        "rejections": {"conversion": dict(Counter(r["reject_reason"] for r in rows
                                                  if not r["converted"])),
                       "verification": dict(Counter(f"{r['verification']}:{r['class']}"
                                                    for r in rows if r["converted"]
                                                    and not r["verified"]))},
        "exclusion_audit": exclusion_audit,
        "slices": {"execution_mode": by("execution_mode"), "dataset": by("dataset"),
                   "complexity": by("complexity"), "bug_family": by("bug_family")},
        "complexity_note": "synthetic tiers are small-function AST complexity "
                           "(oneiros_buggy_ast_complexity_v1 on train buggy code), not "
                           "repository-scale complexity",
        "comparison_with_historical_stage1": hist,
        "isolation": {"split": "train only", "v24_panel_projects_present": leak},
        "repository_examples": "converted with their public import header; verification "
                               "pending native buggy/fixed environments",
        "claims": {"training_ready": False, "sft_improvement": False}}
    if not out["access_audit"]["passed"]:
        raise SystemExit(f"REFUSED: access audit failed: {VIOLATIONS}")
    status = publish_once(RECEIPT, out)
    print(json.dumps({"status": status, "counts": out["counts"], "duplicates": duplicates,
                      "rejections": out["rejections"], "comparison": {
                          k: v for k, v in hist.items() if k != "changed_rows_all"},
                      "access": out["access_audit"]["passed"]}, indent=1, default=str))
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--examples", required=True)
    p.add_argument("--out", required=True)
    r = sub.add_parser("receipt")
    r.add_argument("--dir", required=True)
    args = parser.parse_args(argv)
    if args.cmd == "prepare":
        return prepare(Path(args.examples), ROOT / args.out)
    return receipt(ROOT / args.dir)


if __name__ == "__main__":
    raise SystemExit(main())
