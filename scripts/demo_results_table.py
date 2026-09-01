"""Panel demo step 5: print the four reported result rows straight from the saved JSON."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

RUNS = [
    ("V2 SFT", "results/v2_full_sft/sft_validation_results.json"),
    ("V3 Full SFT", "results/v3_full_sft_monitored_20260819_1/sft_validation_results.json"),
    ("V3 SFT + DPO", "results/v3_dpo_smoke_20260820_1/dpo_validation_checkpoint_100.json"),
    ("Latest aligned SFT smoke",
     "results/v3_repo1024_aligned_constant_lr_smoke_800_20260821_1/sft_monitor_checkpoint_143.json"),
]

HEADER = "%-26s %10s %7s %11s" % ("Experiment", "Evaluated", "Killed", "Kill rate")


def main():
    print(HEADER)
    print("-" * len(HEADER))
    for name, rel in RUNS:
        path = ROOT / rel
        if not path.is_file():
            print("%-26s %10s %7s %11s" % (name, "MISSING", "-", "-"))
            continue
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
        print("%-26s %10s %7s %10.2f%%" % (
            name,
            data.get("function_validation_records"),
            data.get("function_validation_killed"),
            float(data.get("function_kill_rate", 0)) * 100,
        ))
    return 0


if __name__ == "__main__":
    sys.exit(main())
