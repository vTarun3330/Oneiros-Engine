"""v2.5 (WSL, stdlib only): model-visible prompt sources of VERIFIED repository fragments.

For every fragment accepted in run 1 of a repository verification directory it writes the
prompt-builder DTO (target from verifier-side localisation) and the buggy-side view source of
the target file. Nothing fixed-side, no patch, no official test. Never overwrites.

    wsl -u root -- bash scripts/wsl_native_python.sh scripts/v25_repository_prompt_sources_wsl.py \
        --envs <env rows>... --verify-dir DIR --out FILE
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
import native_generated_tests_execute_wsl as ex  # noqa: E402

TARGETS = "results/sft_root_cause/v25_native/targets.jsonl"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--envs", nargs="+", required=True)
    parser.add_argument("--verify-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--targets", default=TARGETS)
    args = parser.parse_args(argv)
    out = REPO / args.out
    if out.exists():
        raise SystemExit(f"REFUSED: {out} exists")
    targets = {t["task"]: t for t in map(json.loads, (REPO / args.targets).read_text(
        encoding="utf-8").splitlines())}
    envs = {}
    for rel in args.envs:
        for row in map(json.loads, (REPO / rel).read_text(encoding="utf-8").splitlines()):
            envs[row["task"]] = row
    run1 = REPO / args.verify_dir / "repository_verification_run1.jsonl"
    rows = []
    for v in map(json.loads, run1.read_text(encoding="utf-8").splitlines()):
        if not v.get("accepted"):
            continue
        t, env = targets[v["task"]], envs[v["task"]]
        rel = t["target_file"]
        root = ex.import_root(rel)
        buggy = (Path(env["views"]["buggy"]) / (rel[len(root) + 1:] if root else rel)) \
            .read_text(encoding="utf-8")
        repo = t["repository_url"].split("github.com/")[-1].rstrip("/")
        rows.append({"index": v["index"], "task": v["task"], "project": v["project"],
                     "lineage": v["task"],
                     "dto": {"target_key": f"{v['task']}#{v['index']}", "repository": repo,
                             "buggy_commit": t["buggy_commit"], "target_file": rel,
                             "qualname": v["qualname"]},
                     "buggy_source": buggy})
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))
                    .encode("utf-8"))
    print(json.dumps({"rows": len(rows), "sha256": hashlib.sha256(out.read_bytes())
                      .hexdigest()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
