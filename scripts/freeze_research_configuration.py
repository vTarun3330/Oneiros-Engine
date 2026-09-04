"""Write the machine-readable selected-research-configuration receipt.

The receipt is the single place a reader can look to see what was actually run.
It is content-hashed, so a later claim can be checked against the configuration
that produced it rather than against a reconstruction from log files.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness.corpus import write_json
from harness.research_configuration import (
    build_selected_configuration,
    configuration_sha256,
)
from utils.reproducibility import build_reproducibility_manifest


DEFAULT_OUTPUT = ROOT / "results" / "v4_2_selected_research_configuration.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--corpus-version", default="v4_1_research_hardened_candidate",
    )
    parser.add_argument(
        "--training-view", default="data/training_views/balanced_sft_v1",
    )
    arguments = parser.parse_args()

    configuration = build_selected_configuration(
        ROOT, arguments.corpus_version, arguments.training_view,
    )
    configuration["reproducibility"] = build_reproducibility_manifest(
        ROOT,
        configuration["model"]["base_model_name"],
        configuration["model"]["base_model_revision"],
    )
    configuration["configuration_sha256"] = configuration_sha256({
        key: value for key, value in configuration.items()
        if key not in {"reproducibility", "configuration_sha256"}
    })
    write_json(arguments.output, configuration)
    print(json.dumps({
        "output": str(arguments.output.relative_to(ROOT)).replace("\\", "/"),
        "configuration_sha256": configuration["configuration_sha256"],
        "phase": configuration["phase"],
        "dpo": configuration["scope"]["out_of_scope"][0][:60],
        "sft_research_target_kill_at_8": configuration[
            "targets_and_gates"
        ]["sft_research_target_kill_at_8"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
