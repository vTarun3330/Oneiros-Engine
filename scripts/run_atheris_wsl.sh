#!/usr/bin/env bash
# Drive the actual-Atheris differential harness, one target per process.
#
# atheris.Setup() may be called only once per process and libFuzzer terminates
# the process itself when -runs is exhausted, so a loop inside Python cannot
# work.  Each target therefore gets its own interpreter, and each writes its own
# result file which the caller merges.
#
# Usage: run_atheris_wsl.sh TASKS_JSON OUTPUT_DIR COUNT [MAX_RUNS] [TIME_BUDGET]
set -u

TASKS="$1"
OUTDIR="$2"
COUNT="$3"
MAX_RUNS="${4:-20000}"
TIME_BUDGET="${5:-10}"
PYTHON="${ATHERIS_PYTHON:-/opt/atheris311/bin/python}"
HARNESS="${ATHERIS_HARNESS:-/mnt/c/Users/Student2/Desktop/Capstone/oneiros/baseline/atheris_harness.py}"

mkdir -p "$OUTDIR"
# libFuzzer drops crash-* artifacts into the working directory; keep them out
# of the repository.
WORKDIR="$(mktemp -d)"
cd "$WORKDIR" || exit 1

for ((i = 0; i < COUNT; i++)); do
  "$PYTHON" "$HARNESS" \
    --tasks "$TASKS" \
    --output "$OUTDIR/task_$(printf '%05d' "$i").json" \
    --task-index "$i" \
    --max-runs "$MAX_RUNS" \
    --time-budget "$TIME_BUDGET" >/dev/null 2>&1
  if (( (i + 1) % 25 == 0 )); then
    echo "  atheris: $((i + 1))/$COUNT targets" >&2
  fi
done

cd / && rm -rf "$WORKDIR"
echo "atheris driver complete: $COUNT targets -> $OUTDIR" >&2
