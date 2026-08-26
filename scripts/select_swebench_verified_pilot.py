"""Select a deterministic repository-balanced SWE-bench Verified pilot."""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness.corpus import sha256_file, write_json
from harness.swebench_verified import patch_paths, selector_target


DIFFICULTY_RANK = {
    "<15 min fix": 0,
    "15 min - 1 hour": 1,
    "1-4 hours": 2,
    ">4 hours": 3,
}


def _json_list(value: Any) -> List[str]:
    return json.loads(value) if isinstance(value, str) else list(value or [])


def select(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_repo: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        f2p = _json_list(row.get("FAIL_TO_PASS"))
        source_paths = patch_paths(row.get("patch", ""))
        test_paths = patch_paths(row.get("test_patch", ""))
        if not f2p or not source_paths or not test_paths:
            continue
        pathlike = sum(selector_target(item) is not None for item in f2p)
        candidate = {
            "instance_id": row["instance_id"],
            "repo": row["repo"],
            "version": row.get("version", ""),
            "difficulty": row.get("difficulty", ""),
            "source_paths": source_paths,
            "test_paths": test_paths,
            "fail_to_pass_count": len(f2p),
            "pass_to_pass_count": len(_json_list(row.get("PASS_TO_PASS"))),
            "pathlike_fail_to_pass": pathlike,
            "gold_patch_chars": len(row.get("patch", "")),
            "test_patch_chars": len(row.get("test_patch", "")),
        }
        by_repo[row["repo"]].append(candidate)

    selected = []
    for repo in sorted(by_repo):
        candidates = sorted(by_repo[repo], key=lambda item: (
            0 if item["pathlike_fail_to_pass"] else 1,
            DIFFICULTY_RANK.get(item["difficulty"], 99),
            item["gold_patch_chars"] + item["test_patch_chars"],
            item["instance_id"],
        ))
        selected.append(candidates[0])
    return selected


def main() -> None:
    source = ROOT / "data" / "swebench_verified_source" / "SWE-bench_Verified.test.parquet"
    output = ROOT / "data" / "swebench_verified_ingestion" / "pilot_selection.json"
    rows = pq.read_table(source).to_pylist()
    selected = select(rows)
    repos = {row["repo"] for row in rows}
    if len(selected) != len(repos):
        raise RuntimeError(
            f"Pilot must select one usable instance from every repository: "
            f"selected={len(selected)}, repos={len(repos)}"
        )
    payload = {
        "schema_version": 1,
        "source_file": str(source.relative_to(ROOT)),
        "source_sha256": sha256_file(source),
        "source_rows": len(rows),
        "selection_strategy": (
            "one_per_repository; prefer path-addressable F2P, lower difficulty, "
            "then smaller gold+test patch"
        ),
        "selected_count": len(selected),
        "selected": selected,
    }
    write_json(output, payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
