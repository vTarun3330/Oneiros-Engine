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
# Targets are fully independent: separate result file, separate log,
# separate process. The loop was serial for no reason other than that it
# was written as a loop, and 4542 runs at a few seconds each is hours.
JOBS="${9:-$(( $(nproc) - 2 ))}"
if (( JOBS < 1 )); then JOBS=1; fi
PYTHON="${ATHERIS_PYTHON:-/opt/atheris311/bin/python}"
HARNESS="${ATHERIS_HARNESS:-/mnt/c/Users/Student2/Desktop/Capstone/oneiros/baseline/atheris_harness.py}"

mkdir -p "$OUTDIR"
# Resolve to absolute paths BEFORE any cd. Each target now runs inside its own
# working directory, so a relative OUTDIR or TASKS silently stops resolving and
# every target fails with "no checkpoint" while the driver reports success.
OUTDIR="$(cd "$OUTDIR" && pwd)"
TASKS="$(cd "$(dirname "$TASKS")" && pwd)/$(basename "$TASKS")"
# libFuzzer drops crash-* artifacts into the working directory; keep them out
# of the repository.
WORKDIR="$(mktemp -d)"
trap 'cd /; rm -rf -- "$WORKDIR"' EXIT
cd "$WORKDIR" || exit 1

run_one_target() {
  local i="$1"
  local RESULT="$OUTDIR/task_$(printf '%05d' "$i").json"
  local LOG="$OUTDIR/task_$(printf '%05d' "$i").log"
  # Each target gets its own directory so concurrent libFuzzer processes cannot
  # overwrite one another's crash-* artifacts.
  local TASKDIR
  TASKDIR="$(mktemp -d "$WORKDIR/task_XXXXXX")"
  local STARTED_MS
  STARTED_MS="$(date +%s%3N)"
  (
    cd "$TASKDIR" || exit 1
    timeout --signal=TERM --kill-after=5s "$WALL_LIMIT"       "$PYTHON" "$HARNESS"       --tasks "$TASKS"       --output "$RESULT"       --task-index "$i"       --max-runs "$MAX_RUNS"       --time-budget "$TIME_BUDGET"       --unit-timeout "$UNIT_TIMEOUT"       --seed "$SEED" >"$LOG" 2>&1
  )
  local STATUS=$?
  local ELAPSED_MS=$(( $(date +%s%3N) - STARTED_MS ))
  if (( ELAPSED_MS < 0 )); then
    ELAPSED_MS=0
  fi
  if [[ -f "$RESULT" ]]; then
    "$PYTHON" "$HARNESS"       --finalize-output "$RESULT"       --process-returncode "$STATUS"       --wall-limit "$WALL_LIMIT"       --runner-elapsed-seconds="${ELAPSED_MS}e-3"
  else
    echo "atheris target $i produced no checkpoint (return code $STATUS)" >&2
  fi
  rm -rf -- "$TASKDIR"
}
export -f run_one_target
export OUTDIR TASKS PYTHON HARNESS MAX_RUNS TIME_BUDGET UNIT_TIMEOUT WALL_LIMIT SEED WORKDIR

echo "atheris driver: $COUNT targets across $JOBS workers" >&2
seq 0 $(( COUNT - 1 )) | xargs -P "$JOBS" -I{} bash -c 'run_one_target {}'

echo "atheris driver complete: $COUNT targets -> $OUTDIR" >&2
