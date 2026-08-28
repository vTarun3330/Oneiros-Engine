"""Run the complete local V4.1 test suite and persist a source-bound status."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness.corpus import write_json
from utils.reproducibility import source_tree_sha256


def main() -> int:
    command = [sys.executable, "-m", "pytest", "-q"]
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    output = f"{result.stdout}\n{result.stderr}".strip()
    passed_matches = re.findall(r"(\d+) passed", output)
    failed_matches = re.findall(r"(\d+) failed", output)
    passed = int(passed_matches[-1]) if passed_matches else 0
    failed = int(failed_matches[-1]) if failed_matches else (0 if result.returncode == 0 else 1)
    payload = {
        "schema_version": "oneiros_v4_1_local_test_status_v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "command": command,
        "returncode": result.returncode,
        "passed": passed,
        "failed": failed,
        "source_tree_sha256": source_tree_sha256(ROOT),
        "completion_only_masking_verified": (
            result.returncode == 0
            and "test_completion_only_collator_masks_prompt_tokens" in
            ROOT.joinpath("tests/test_v4_1_research_hardening.py").read_text(encoding="utf-8")
        ),
        "sealed_final_test_accessed": False,
        "output_tail": output[-4000:],
    }
    output_path = ROOT / "results" / "v4_1_local_test_status.json"
    write_json(output_path, payload)
    print(output)
    print(json.dumps(payload, indent=2))
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
