"""Panel demo step 0: confirm every artifact the demonstration depends on is present."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

REQUIRED = [
    "data/corpus/v3_final_candidate/records.json",
    "checkpoints/v3_full_sft_monitored_20260819_1/sft_adapter/adapter_model.safetensors",
    "results/v3_full_sft_monitored_20260819_1/sft_validation_results.json",
    "results/v3_dpo_smoke_20260820_1/dpo_validation_checkpoint_100.json",
    "results/v3_repo1024_aligned_constant_lr_smoke_800_20260821_1/sft_monitor_checkpoint_143.json",
]


def main():
    missing = 0
    for rel in REQUIRED:
        path = ROOT / rel
        present = path.is_file()
        if not present:
            missing += 1
        size = "%10.1f MB" % (path.stat().st_size / 1048576) if present else " " * 13
        print("%-5s %s  %s" % (present, size, rel))
    print()
    if missing:
        print("MISSING %d of %d required artifacts" % (missing, len(REQUIRED)))
        return 1
    print("All %d required artifacts present." % len(REQUIRED))
    return 0


if __name__ == "__main__":
    sys.exit(main())
