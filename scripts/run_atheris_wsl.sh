#!/usr/bin/env bash
# Drive the actual-Atheris differential harness, one target per process.
#
# atheris.Setup() may be called only once per process and libFuzzer terminates
# the process itself when -runs is exhausted, so a loop inside Python cannot
# work.  Each target therefore gets its own interpreter, and each writes its own
# result file which the caller merges.
#
# Usage: run_atheris_wsl.sh TASKS_JSON OUTPUT_DIR COUNT [MAX_RUNS] \
#            [TIME_BUDGET] [UNIT_TIMEOUT] [WALL_LIMIT] [SEED]
set -u

TASKS="$1"
OUTDIR="$2"
COUNT="$3"
MAX_RUNS="${4:-20000}"
TIME_BUDGET="${5:-10}"
UNIT_TIMEOUT="${6:-5}"
WALL_LIMIT="${7:-90s}"
SEED="${8:-42}"
PYTHON="${ATHERIS_PYTHON:-/opt/atheris311/bin/python}"
HARNESS="${ATHERIS_HARNESS:-/mnt/c/Users/Student2/Desktop/Capstone/oneiros/baseline/atheris_harness.py}"

mkdir -p "$OUTDIR"
# libFuzzer drops crash-* artifacts into the working directory; keep them out
# of the repository.
WORKDIR="$(mktemp -d)"
trap 'cd /; rm -rf -- "$WORKDIR"' EXIT
cd "$WORKDIR" || exit 1

for ((i = 0; i < COUNT; i++)); do
  RESULT="$OUTDIR/task_$(printf '%05d' "$i").json"
  LOG="$OUTDIR/task_$(printf '%05d' "$i").log"
  STARTED_MS="$(date +%s%3N)"
  timeout --signal=TERM --kill-after=5s "$WALL_LIMIT" \
    "$PYTHON" "$HARNESS" \
    --tasks "$TASKS" \
    --output "$RESULT" \
    --task-index "$i" \
    --max-runs "$MAX_RUNS" \
    --time-budget "$TIME_BUDGET" \
    --unit-timeout "$UNIT_TIMEOUT" \
    --seed "$SEED" >"$LOG" 2>&1
  STATUS=$?
  ELAPSED_MS=$(( $(date +%s%3N) - STARTED_MS ))
  if (( ELAPSED_MS < 0 )); then
    ELAPSED_MS=0
  fi
  if [[ -f "$RESULT" ]]; then
    "$PYTHON" "$HARNESS" \
      --finalize-output "$RESULT" \
      --process-returncode "$STATUS" \
      --wall-limit "$WALL_LIMIT" \
      --runner-elapsed-seconds="${ELAPSED_MS}e-3"
  else
    echo "atheris target $i produced no checkpoint (return code $STATUS)" >&2
  fi
  if (( (i + 1) % 25 == 0 )); then
    echo "  atheris: $((i + 1))/$COUNT targets" >&2
  fi
done

echo "atheris driver complete: $COUNT targets -> $OUTDIR" >&2
