"""Execute the post-hoc exploratory mechanical-repair matrix (R1-R4) on the retained v2.4
candidates, with the FROZEN v2.4 executor functions (static policy, sandboxed buggy/fixed runs,
kill reruns, classification) on the same live-verified preparation views.

Never regenerates, reranks or drops a candidate and never touches the v2.4 primary output:
it reads the exact v2.4 generations (hash-checked) and writes only to a NEW directory. R0 is
the v2.4 primary execution itself (hash-checked). A candidate a condition leaves byte-identical
keeps its v2.4 row; an identical repaired module of the same target is executed once.

    wsl -u root -- bash scripts/wsl_native_python.sh scripts/repair_ceiling_execute_wsl.py \
        --out results/sft_root_cause/native_v24_repair_ceiling
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
import native_generated_tests_execute_wsl as ex  # noqa: E402
import native_generation_io as gio  # noqa: E402
import repair_ceiling_transforms as rt  # noqa: E402

COND = "primary_whole_module"
PREP = "results/sft_root_cause/native_v21_rehearsal/records.jsonl"
MANIFEST = "results/sft_root_cause_native_v24_rehearsal_manifest_v7.json"
JOB = "results/sft_root_cause_native_v24_rehearsal_job_v5.json"
GENERATIONS = "results/sft_root_cause/native_v24_generations"
V24_RESULTS = "results/sft_root_cause/native_v24_execution/results_primary_whole_module.jsonl"
EXPECTED_SHA = {
    f"{GENERATIONS}/base/generations_{COND}_base.jsonl":
        "e71367acefdbad5a422aba3790376454f04831d80c4b5dec6ac1429e31eb9f44",
    f"{GENERATIONS}/sft/generations_{COND}_sft.jsonl":
        "5af755e9f947175a1996037865e1c96023faf36e78cd87678be066367d6e59d9",
    V24_RESULTS: "e40c912d23eccb3fb8a6b0cd06c11e27a48052582f4e60d1f1507ce7471a0d55",
}
REPAIRS = ("R1", "R2", "R3", "R4")


def sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True)
    parser.add_argument("--definition", required=True,
                        help="the committed repair-ceiling definition receipt")
    args = parser.parse_args(argv)
    out = REPO / args.out
    if out.exists():
        raise SystemExit(f"REFUSED: {out} exists; the diagnostic writes a new directory once")
    definition = json.loads((REPO / args.definition).read_text(encoding="utf-8"))
    if definition.get("transforms") != rt.definition() or \
            definition.get("transforms_sha256") != sha_file(HERE / "repair_ceiling_transforms.py"):
        raise SystemExit("REFUSED: transforms differ from the committed definition")
    for rel, want in EXPECTED_SHA.items():
        if sha_file(REPO / rel) != want:
            raise SystemExit(f"REFUSED: {rel} differs from the completed v2.4 artifact")
    cohort = gio.resolve_cohort(REPO / JOB, REPO / MANIFEST, COND)
    prepared = gio.resolve_prep(REPO / PREP, REPO / MANIFEST, REPO)
    prep = prepared["rows"]
    gio.verify_live_views(prep, cohort["generation"])
    arms = gio.arm_paths(REPO / GENERATIONS, None, None, COND)
    gens = gio.load_arm_generations(arms, cohort)
    prompts = {i["target_key"]: i["prompt"] for i in
               json.loads((REPO / JOB).read_text(encoding="utf-8"))[COND]["items"]}
    v24 = {r["key"]: r for r in (json.loads(l) for l in
                                 (REPO / V24_RESULTS).read_text(encoding="utf-8").splitlines())}
    out.mkdir(parents=True)
    handles = {c: (out / f"results_{c}.jsonl").open("w", encoding="utf-8", newline="\n")
               for c in REPAIRS}
    scratch = Path(tempfile.mkdtemp(prefix="oneiros_repair_"))
    executed = reused_v24 = reused_same = 0
    try:
        for key in cohort["generation"]:
            row = prep[key]
            target = {"target_key": key, "module": row["module"], "qualname": row["qualname"],
                      "python": row["python_path"], "env_dir": row["env_dir"],
                      "views": row["views"], "module_sha256": row["module_sha256"]}
            canary = ex.canary_check(target, scratch / "canary")
            cache = {}
            for arm in ("base", "sft"):
                for seed in (42, 43, 44):
                    grow = gens["rows"][(arm, key, seed)]
                    for slot, cand in enumerate(grow["candidates"]):
                        ckey = f"{arm}::{key}::{seed}::{slot}"
                        base_row = v24[ckey]
                        if base_row["module_sha256"] != rt.sha256(cand["module"]):
                            raise SystemExit(f"REFUSED: v2.4 row {ckey} is not this candidate")
                        for c in REPAIRS:
                            rep = rt.apply(c, cand["module"], prompts[key])
                            record = {"key": ckey, "condition": c, "arm": arm, "seed": seed,
                                      "slot": slot, "target_key": key,
                                      **{k: rep[k] for k in ("changed", "applied", "not_applied",
                                                             "original_module_sha256",
                                                             "repaired_module_sha256",
                                                             "target_import")},
                                      "generation": base_row["generation"]}
                            if not rep["changed"]:
                                reused_v24 += 1
                                record.update(source="v2.4_primary_row_same_module",
                                              **{k: base_row[k] for k in
                                                 ("class", "classification", "fixed_valid",
                                                  "static")})
                            elif rep["repaired_module_sha256"] in cache:
                                reused_same += 1
                                record.update(source="same_repaired_module_executed_once",
                                              **cache[rep["repaired_module_sha256"]])
                            else:
                                executed += 1
                                if canary["ok"]:
                                    outcome = ex.execute_candidate(target, rep["module"],
                                                                   scratch / "c")
                                    cls = outcome["classification"]
                                else:
                                    outcome, cls = {"static": None}, {"class":
                                                                      "environment_failure"}
                                result = {"class": cls["class"], "classification": cls,
                                          "fixed_valid": ex.fixed_valid_of(cls),
                                          "static": outcome.get("static"),
                                          "canary_failed": not canary["ok"]}
                                cache[rep["repaired_module_sha256"]] = result
                                record.update(source="executed", **result)
                            handles[c].write(json.dumps(record, sort_keys=True,
                                                        default=str) + "\n")
                            handles[c].flush()
                            os.fsync(handles[c].fileno())
            print(f"{key}: executed={executed} reused_v24={reused_v24} "
                  f"reused_same={reused_same}", flush=True)
    finally:
        for h in handles.values():
            h.close()
        shutil.rmtree(scratch, ignore_errors=True)
    summary = {"executed": executed, "reused_v24_rows": reused_v24,
               "reused_same_module": reused_same,
               "files": {c: sha_file(out / f"results_{c}.jsonl") for c in REPAIRS},
               "rows": {c: len((out / f"results_{c}.jsonl").read_bytes().splitlines())
                        for c in REPAIRS}}
    (out / "run_summary.json").write_text(json.dumps(summary, indent=1) + "\n",
                                          encoding="utf-8")
    print(json.dumps(summary, indent=1))
    return 0 if all(n == 1104 for n in summary["rows"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
