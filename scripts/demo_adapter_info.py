"""Panel demo step 4: show the trained LoRA adapter and the checkpoint metadata that selected it."""
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUN = "v3_full_sft_monitored_20260819_1"
ADAPTER = ROOT / "checkpoints" / RUN / "sft_adapter" / "adapter_model.safetensors"
META = ROOT / "checkpoints" / RUN / "sft_metadata.json"


def main():
    if not ADAPTER.is_file():
        print("Adapter not found:", ADAPTER)
        return 1
    stat = ADAPTER.stat()
    print("Adapter file   :", ADAPTER.name)
    print("Size           : %d bytes (%.1f MB)" % (stat.st_size, stat.st_size / 1048576))
    print("Last modified  :", datetime.fromtimestamp(stat.st_mtime).strftime("%d-%m-%Y %H:%M:%S"))
    print()

    with open(META, encoding="utf-8") as handle:
        meta = json.load(handle)
    best = meta.get("monitor_best_metrics") or {}

    print("Status             :", meta.get("status"))
    print("Optimizer steps    :", meta.get("completed_optimizer_steps"))
    print("Monitor gate passed:", meta.get("monitor_gate_passed"))
    print("Best checkpoint    :", meta.get("monitor_best_adapter"))
    print("Best panel         : %s/%s" % (best.get("function_validation_killed"),
                                          best.get("function_validation_records")))
    rate = best.get("function_kill_rate")
    if rate is not None:
        print("Best panel rate    : %.2f%%" % (float(rate) * 100))
    return 0


if __name__ == "__main__":
    sys.exit(main())
