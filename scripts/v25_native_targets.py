"""v2.5 Phase 6/7 target specifications for native train-repository environments.

Builds ``targets.jsonl`` for the frozen canary spec from TRAIN records only (strict train-only
loader with the fail-closed access audit). For SWE-bench instances it can also fetch the gold
and official test patches ONE TRAIN INSTANCE AT A TIME (row-level dataset API; the local
multi-instance parquet is never opened) and accept them only if their SHA-256 equals the train
record's ``gold_patch_sha256`` / ``test_patch_sha256``.

    python scripts/v25_native_targets.py targets --out results/sft_root_cause/v25_native
    python scripts/v25_native_targets.py patches --out results/sft_root_cause/v25_native
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import time
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SPEC = "results/sft_root_cause_v25_native_env_canary_spec.json"
CANDIDATES = "results/sft_root_cause/v25_corpus_stage1_r2/converted_candidates.jsonl"
API = ("https://datasets-server.huggingface.co/filter?dataset=princeton-nlp/SWE-bench_Verified"
       "&config=default&split=test&length=1&where=")


def sha_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _bugsinpy_tests(project: str, bug: str) -> list:
    text = (ROOT / "data" / "BugsInPy_repo" / "projects" / project / "bugs" / bug /
            "run_test.sh").read_text(encoding="utf-8")
    return [t for line in text.splitlines() if line.strip().startswith(("pytest", "python"))
            for t in line.split() if "::" in t or t.endswith(".py")]


def _swebench_selectors(prov: dict) -> list:
    paths = prov.get("test_paths") or []
    out = []
    for name in prov.get("fail_to_pass") or []:
        m = re.fullmatch(r"(\w+) \(([\w.]+)\)", name)
        if m:                                   # Django: "test_x (pkg.module.Class)"
            out.append(f"django:{m.group(2)}.{m.group(1)}")
        elif len(paths) == 1:
            out.append(f"{paths[0]}::{name}")
        else:
            out.append(f"unresolved:{name}")
    return out


def build_targets(out: Path) -> int:
    from scripts.v25_converted_corpus import install_audit, load_train_records
    install_audit()
    spec = json.loads((ROOT / SPEC).read_text(encoding="utf-8"))
    records = {r["id"]: r for r in load_train_records()}
    cands = [json.loads(l) for l in (ROOT / CANDIDATES).read_text(encoding="utf-8").splitlines()]
    by_task = {}
    for c in cands:
        if c["execution_mode"] == "function_assertion" or not c["conversion"]["accepted"]:
            continue
        prov = records[c["id"]]["provenance"]
        by_task.setdefault(prov["official_task_id"], []).append((c, records[c["id"]]))
    rows = []
    for project, repo in spec["repositories"].items():
        for task in repo["all_instances"]:
            c, rec = by_task[task][0]
            prov = rec["provenance"]
            swe = not task.startswith("bugsinpy::")
            version = prov.get("version")
            interp = repo["interpreter_map"].get(version) or repo["interpreter_map"].get("*")
            row = {"task": task, "project": project, "canary": task in repo["canary_instances"],
                   "dataset": "SWE-bench Verified" if swe else "BugsInPy",
                   "repository_url": prov.get("repository_url") or
                   f"https://github.com/{prov['repository']}",
                   "interpreter": interp, "version": version,
                   "target_file": (prov.get("patched_source_paths") or [None])[0],
                   "target_symbols": rec.get("target_symbols") or [],
                   "fragments": [{"index": x["index"], "id": x["id"],
                                  "module_sha256": x["conversion"]["module_sha256"]}
                                 for x, _ in by_task[task]]}
            if swe:
                row.update(buggy_commit=prov["base_commit"],
                           gold_patch_sha256=prov["gold_patch_sha256"],
                           test_patch_sha256=prov["test_patch_sha256"],
                           selectors=_swebench_selectors(prov))
            else:
                bug = prov["bug_id"]
                row.update(buggy_commit=prov["buggy_commit"], fixed_commit=prov["fixed_commit"],
                           bugsinpy_bug=bug, selectors=_bugsinpy_tests(project, bug))
            rows.append(row)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "targets.jsonl"
    path.write_bytes(("\n".join(json.dumps(r, sort_keys=True) for r in rows) + "\n")
                     .encode("utf-8"))
    print(json.dumps({"targets": len(rows), "canary": sum(r["canary"] for r in rows),
                      "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                      "unresolved_selectors": sum(s.startswith("unresolved:") for r in rows
                                                  for s in r["selectors"])}, indent=1))
    return 0


def split_pr_diff(diff: str) -> tuple:
    """SWE-bench's split of a pull-request diff: non-test files (gold patch) and test files
    (test patch), without git's ``index`` lines. Used only as a fallback source whose result
    is accepted when BOTH hashes equal the train record's."""
    diff = re.sub(r"^index [0-9a-f]+\.\.[0-9a-f]+( \d+)?\n", "", diff, flags=re.M)
    parts = [p for p in re.split(r"(?=^diff --git )", diff, flags=re.M)
             if p.startswith("diff --git")]

    def is_test(part: str) -> bool:
        return re.search(r"(^|/)(tests?|testing)/|test_[^/ ]*\.py|_test\.py",
                         part.split("\n", 1)[0]) is not None
    return ("".join(p for p in parts if not is_test(p)),
            "".join(p for p in parts if is_test(p)))


def pr_diff_patches(task: str) -> tuple | None:
    repo, number = task.rsplit("-", 1)
    owner, name = repo.split("__")
    try:
        with urllib.request.urlopen(f"https://github.com/{owner}/{name}/pull/{number}.diff",
                                    timeout=60) as resp:
            return split_pr_diff(resp.read().decode("utf-8"))
    except Exception:                                      # noqa: BLE001 - recorded upstream
        return None


def fetch_patches(out: Path, only_canary: bool) -> int:
    rows = [json.loads(l) for l in (out / "targets.jsonl").read_text(encoding="utf-8")
            .splitlines()]
    cache = out / "patches"
    cache.mkdir(parents=True, exist_ok=True)
    status = {}
    for r in rows:
        if r["dataset"] != "SWE-bench Verified" or (only_canary and not r["canary"]):
            continue
        dest = cache / f"{r['task']}.json"
        if dest.exists():
            status[r["task"]] = "cached"
            continue
        pr = pr_diff_patches(r["task"])
        if pr is not None and sha_text(pr[0]) == r["gold_patch_sha256"] and \
                sha_text(pr[1]) == r["test_patch_sha256"]:
            dest.write_bytes(json.dumps({"task": r["task"], "patch": pr[0], "test_patch": pr[1],
                                         "source": "github_pull_request_diff"},
                                        sort_keys=True).encode("utf-8"))
            status[r["task"]] = "fetched_and_hash_verified"
            continue
        where = urllib.parse.quote(f"\"instance_id\"='{r['task']}'")
        try:
            with urllib.request.urlopen(API + where, timeout=60) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except Exception as exc:                         # noqa: BLE001 - recorded
            status[r["task"]] = f"patch_unavailable:{type(exc).__name__}"
            continue
        found = [x["row"] for x in data.get("rows", []) if x["row"].get("instance_id") == r["task"]]
        if len(found) != 1:
            status[r["task"]] = f"patch_unavailable:{data.get('error', 'not_found')[:60]}"
            continue
        gold, test = found[0].get("patch", ""), found[0].get("test_patch", "")
        if sha_text(gold) != r["gold_patch_sha256"] or sha_text(test) != r["test_patch_sha256"]:
            status[r["task"]] = "patch_hash_mismatch"
            continue
        dest.write_bytes(json.dumps({"task": r["task"], "patch": gold, "test_patch": test},
                                    sort_keys=True).encode("utf-8"))
        status[r["task"]] = "fetched_and_hash_verified"
        time.sleep(0.5)
    print(json.dumps(status, indent=1))
    return 0 if all(v in ("cached", "fetched_and_hash_verified") for v in status.values()) else 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("targets")
    t.add_argument("--out", required=True)
    p = sub.add_parser("patches")
    p.add_argument("--out", required=True)
    p.add_argument("--all", action="store_true")
    args = parser.parse_args(argv)
    if args.cmd == "targets":
        return build_targets(ROOT / args.out)
    return fetch_patches(ROOT / args.out, only_canary=not args.all)


if __name__ == "__main__":
    raise SystemExit(main())
