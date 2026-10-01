"""v2.5 Phase 4/10: export the structurally selected TRAIN-only rejection-sampling pilot targets
(WSL, stdlib only). Writes, per target, the prompt-builder DTO and buggy source plus the native
environment pointers the executor needs. No model, no generation, no execution.

Sample rule (frozen in addendum 1 section 7; no outcome used): qualified, reproducible train
environments of pytest-runnable repositories whose target localises to a function; ordered by
sha256(task id); at most PER_REPO per repository, then filled to SAMPLE in the same order.

    wsl -u root -- bash scripts/wsl_native_python.sh scripts/v25_pilot_export_wsl.py \
        --envs <env rows>... --out results/sft_root_cause/v25_pilot/export.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
from v25_verify_repository_wsl import SANDBOX_RUNNERS, target_for  # noqa: E402
import native_generated_tests_execute_wsl as ex  # noqa: E402

SAMPLE, PER_REPO = 30, 15
TARGETS = "results/sft_root_cause/v25_native/targets.jsonl"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--envs", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    targets = {t["task"]: t for t in map(json.loads, (REPO / TARGETS).read_text(
        encoding="utf-8").splitlines())}
    envs = {}
    for rel in args.envs:
        for row in map(json.loads, (REPO / rel).read_text(encoding="utf-8").splitlines()):
            envs[row["task"]] = row
    pool, excluded = [], {}
    for task, env in sorted(envs.items(), key=lambda kv: hashlib.sha256(kv[0].encode())
                            .hexdigest()):
        if env["project"] not in SANDBOX_RUNNERS:
            excluded[task] = "runner_incompatible"
        elif env["category"] != "qualified" or not env.get("environment_reproducible"):
            excluded[task] = f"environment:{env['category']}"
        else:
            target, why = target_for(env)
            if target is None:
                excluded[task] = why
            else:
                pool.append((task, env, target))
    chosen, per = [], {}
    for task, env, target in pool:                      # pass 1: at most PER_REPO each
        if per.get(env["project"], 0) < PER_REPO and len(chosen) < SAMPLE:
            chosen.append((task, env, target))
            per[env["project"]] = per.get(env["project"], 0) + 1
    for item in pool:                                   # pass 2: fill in the same order
        if len(chosen) >= SAMPLE:
            break
        if item not in chosen:
            chosen.append(item)
    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        raise SystemExit(f"REFUSED: {out} exists")
    lines = []
    for task, env, target in chosen:
        t = targets[task]
        rel = t["target_file"]
        root = ex.import_root(rel)
        buggy = (Path(env["views"]["buggy"]) / (rel[len(root) + 1:] if root else rel)) \
            .read_text(encoding="utf-8")
        repo = t["repository_url"].split("github.com/")[-1].rstrip("/")
        dto = {"target_key": task, "repository": repo, "buggy_commit": t["buggy_commit"],
               "target_file": rel, "qualname": target["qualname"]}
        lines.append(json.dumps({"task": task, "project": env["project"], "dto": dto,
                                 "buggy_source": buggy, "executor_target": target},
                                sort_keys=True))
    out.write_bytes(("\n".join(lines) + "\n").encode("utf-8"))
    print(json.dumps({"pool": len(pool), "chosen": len(chosen),
                      "per_project": dict(sorted(per.items())),
                      "excluded": {k: v for k, v in sorted(excluded.items())},
                      "sha256": hashlib.sha256(out.read_bytes()).hexdigest()}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
