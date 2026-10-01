"""v2.5 stage-1 corpus: convert the adapter's exact train-only SFT examples to pytest_module_v1
(``prepare``), then - after WSL verification - summarise integrity and balance (``receipt``).

Inputs are the exact rebuilt examples (sha256 30e9436f..., receipt fac747a1...) joined with
their TRAIN-split records only; an audit hook hard-blocks every non-train corpus file.

    python scripts/v25_converted_corpus.py prepare --examples <rebuilt.jsonl> --out <dir>
    wsl -u root -- bash scripts/wsl_native_python.sh scripts/v25_verify_converted_wsl.py ...
    python scripts/v25_converted_corpus.py receipt --dir <dir>
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import v25_module_conversion as mc

CORPUS = ROOT / "data" / "corpus" / "v4_1_research_hardened_candidate"
EXAMPLES_SHA = "30e9436f7aa16709b8710ba410d653666c3eb8e11d4bdf9de55979a6f4c02167"
RECEIPT = "results/sft_root_cause_v25_converted_corpus_stage1.json"
BLOCKED = ("val.records.json", "ablation_dev.records.json", "external_eval_index.json")
V24_PANEL_PROJECTS = ("marshmallow", "pyparsing", "attrs", "cattrs", "humanize", "tomlkit",
                      "cachetools", "sqlglot")


def _guard(event, args):
    if event == "open" and args and isinstance(args[0], (str, bytes, os.PathLike)):
        p = os.fsdecode(args[0]).replace("\\", "/")
        name = p.rsplit("/", 1)[-1]
        if "/data/corpus/" in p and (name in BLOCKED or (
                name in ("records.json", "splits.json") and "/development_view/" not in p)):
            raise PermissionError(f"BLOCKED non-train corpus file: {p}")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(examples: Path, out: Path) -> int:
    sys.addaudithook(_guard)
    from harness.corpus_view import load_development_split
    if sha(examples) != EXAMPLES_SHA:
        raise SystemExit("REFUSED: examples differ from the audited exact rebuild")
    records = {r["id"]: r for r in load_development_split(CORPUS, "train")}
    rows = [json.loads(l) for l in examples.read_text(encoding="utf-8").splitlines()]
    out.mkdir(parents=True, exist_ok=False)
    lines = []
    for i, ex in enumerate(rows):
        rec = records[ex["id"]]
        key = hashlib.sha256((ex["prompt_sha256"] + "\x00" + ex["completion"]).encode())
        base = {"index": i, "example_key": key.hexdigest(), "id": ex["id"],
                "dataset": ex["dataset"], "project": ex["project"],
                "bug_family": ex["bug_family"], "complexity": ex["complexity"],
                "execution_mode": ex["execution_mode"], "entry_point": rec["entry_point"],
                "effective_repeats": ex["effective_repeats"],
                "original_completion_sha256": mc.sha256(ex["completion"]),
                "group_id": rec.get("group_id"), "source": rec.get("source"),
                "provenance_project": (rec.get("provenance") or {}).get("project")}
        if ex["execution_mode"] == "function_assertion":
            conv = mc.convert_synthetic(ex["completion"], rec["entry_point"],
                                        rec["reference_code"], rec["code_under_test"])
            if conv["accepted"]:
                base.update(reference_code=rec["reference_code"],
                            code_under_test=rec["code_under_test"])
        else:
            conv = mc.convert_repository(ex["completion"], rec.get("support_context") or "")
        lines.append(json.dumps({**base, "conversion": conv}, sort_keys=True))
    target = out / "converted_candidates.jsonl"
    target.write_bytes(("\n".join(lines) + "\n").encode("utf-8"))
    summary = Counter((json.loads(l)["execution_mode"], json.loads(l)["conversion"]["accepted"],
                       json.loads(l)["conversion"].get("reason")) for l in lines)
    print(json.dumps({"rows": len(lines), "sha256": sha(target),
                      "by_mode_accepted_reason": {" | ".join(map(str, k)): v
                                                  for k, v in sorted(summary.items(),
                                                                     key=str)}}, indent=1))
    return 0


def _slice(rows):
    return {"rows": len(rows), "verified": sum(r["verified"] for r in rows),
            "effective_weight_verified": sum(r["effective_repeats"] for r in rows
                                             if r["verified"])}


def receipt(directory: Path) -> int:
    from scripts.native_rehearsal_rebuild_v22 import publish_once
    cand = directory / "converted_candidates.jsonl"
    runs = sorted(directory.glob("verification_run*.jsonl"))
    if len(runs) < 2:
        raise SystemExit("REFUSED: protocol v2.5 B.2 needs two independent verification runs")
    if len({sha(p) for p in runs}) != 1:
        raise SystemExit("REFUSED: the verification runs are not byte-identical")
    sys.addaudithook(_guard)
    from harness.corpus_view import load_complexity_index
    tiers = {k: v.get("tier") for k, v in load_complexity_index(CORPUS).items()}
    candidates = {r["index"]: {**r, "complexity": tiers.get(r["id"], "unindexed")}
                  for r in map(json.loads, cand.read_text(encoding="utf-8").splitlines())}
    verdicts = {v["index"]: v for v in map(json.loads, runs[0].read_text(encoding="utf-8")
                                           .splitlines())}
    rows = []
    for i, c in sorted(candidates.items()):
        v = verdicts.get(i, {})
        rows.append({**{k: c[k] for k in ("index", "id", "dataset", "project", "bug_family",
                                           "complexity", "execution_mode", "effective_repeats",
                                           "group_id", "provenance_project")},
                     "converted": c["conversion"]["accepted"],
                     "reject_reason": c["conversion"].get("reason"),
                     "module_sha256": c["conversion"].get("module_sha256"),
                     "verification": v.get("status", "not_run"),
                     "class": v.get("class"), "verified": v.get("accepted") is True})
    verified = [r for r in rows if r["verified"]]
    modules = Counter(r["module_sha256"] for r in verified)
    pairs = Counter((r["module_sha256"], r["id"]) for r in verified)
    leak = sorted({r["provenance_project"] for r in rows
                   if (r["provenance_project"] or "").lower() in V24_PANEL_PROJECTS})
    by = lambda f: {str(k): _slice([r for r in rows if r[f] == k])  # noqa: E731
                    for k in sorted({r[f] for r in rows}, key=str)}
    out = {"schema_version": "oneiros_v25_converted_corpus_stage1_v1",
           "output_type": mc.OUTPUT_TYPE, "conversion_version": mc.VERSION,
           "inputs": {"examples_sha256": EXAMPLES_SHA,
                      "converted_candidates_sha256": sha(cand),
                      "verification_runs_sha256": [sha(p) for p in runs],
                      "verification_runs_identical": True},
           "scripts_sha256": {p: sha(ROOT / p) for p in (
               "scripts/v25_module_conversion.py", "scripts/v25_converted_corpus.py",
               "scripts/v25_verify_converted_wsl.py")},
           "counts": {"examples": len(rows),
                      "converted": sum(r["converted"] for r in rows),
                      "verified_positive": len(verified),
                      "verified_unique_modules": len(modules),
                      "verified_unique_records": len({r["id"] for r in verified}),
                      "verified_unique_functions": len({r["group_id"] for r in verified})},
           "rejections": {"conversion": dict(Counter(r["reject_reason"] for r in rows
                                                     if not r["converted"])),
                          "verification": dict(Counter(f"{r['verification']}:{r['class']}"
                                                       for r in rows if r["converted"]
                                                       and not r["verified"]))},
           "duplicates": {"exact_duplicate_example_pairs": sum(v - 1 for v in pairs.values()),
                          "same_module_text_across_records": sum(v - 1 for v in
                                                                 modules.values())},
           "slices": {"execution_mode": by("execution_mode"), "dataset": by("dataset"),
                      "complexity": by("complexity"), "bug_family": by("bug_family")},
           "isolation": {"split": "train only (non-train corpus files hard-blocked)",
                         "v24_panel_projects_present": leak,
                         "validation_or_sealed_access": False},
           "repository_examples": "converted with their public import header; verification "
                                  "pending native buggy/fixed environments (not prepared)"}
    status = publish_once(RECEIPT, out)
    print(json.dumps({"status": status, "counts": out["counts"],
                      "rejections": out["rejections"], "duplicates": out["duplicates"],
                      "v24_leak": leak}, indent=1))
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
