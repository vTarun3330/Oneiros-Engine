"""Panel demo: replay the execution oracle on one verified corpus record.

Loads baseline.benchmark_runner directly by file path so the demo does not pull
in baseline/__init__.py -> engine -> torch. This lets the replay run on the
project venv as well as on the full ML interpreter.
"""
import importlib.util
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

RECORD_ID = sys.argv[1] if len(sys.argv) > 1 else "curated::bugsinpy_black_1"


def load_safe_exec():
    spec = importlib.util.spec_from_file_location(
        "oneiros_benchmark_runner", ROOT / "baseline" / "benchmark_runner.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.safe_exec


SUGGESTIONS = (
    "curated::bugsinpy_black_1",
    "mutation::humaneval_HumanEval_121_mut_01039",
    "curated::bugsinpy_pandas_1",
)


def load_record(record_id):
    path = ROOT / "data" / "corpus" / "v3_final_candidate" / "records.json"
    with open(path, encoding="utf-8") as handle:
        records = json.load(handle)
    for item in records:
        if item["id"] == record_id:
            return item
    return None


def main():
    started = time.time()
    safe_exec = load_safe_exec()
    record = load_record(RECORD_ID)
    if record is None:
        # Never show a traceback during the panel demonstration.
        print("No corpus record with id %r." % RECORD_ID)
        print("Verified ids you can use:")
        for known in SUGGESTIONS:
            print("   ", known)
        return 1
    test = record["tests"][0]["code"]

    reference = safe_exec(record["reference_code"], test)
    defective = safe_exec(record["code_under_test"], test)

    print("Record:", record["id"])
    print("Source:", record.get("source", "n/a"))
    print("Specification:", record["specification"])
    print("Verified test:", test)
    print("Reference execution:", "PASS" if reference[0] else "FAIL", reference[2])
    print("Defective execution:", "PASS" if defective[0] else "FAIL", defective[2])
    print(
        "Oracle verdict:",
        "MUTATION KILLED" if reference[0] and not defective[0] else "NOT KILLED",
    )
    print("Elapsed: %.1fs" % (time.time() - started))


if __name__ == "__main__":
    main()
