#!/usr/bin/env bash
# Queue the successor-protocol train-split regeneration behind the legacy run,
# behind two gates that must both pass BEFORE any model is loaded.
#
# Ordering is enforced here rather than by watching a terminal: the two jobs
# would contend for the same GPU, and the legacy run must finish so its receipt
# can be finalised against a completed artifact rather than a checkpoint.
#
# GATE 1 - sequence fit. Every prompt in the generation panel must fit beside
#          the FULL 1024-token completion budget. Failing here costs seconds;
#          discovering it at record 4,000 costs hours, and silently shortening
#          the output would change what is being measured.
# GATE 2 - run configuration. Refuses a command that leaves the model, the
#          attention backend or the prompt budget implicit.
#
# Either gate failing aborts without launching. The output length is never
# reduced to make a gate pass.
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1
PY="./.venv-gpu/Scripts/python.exe"

# The legacy run has already completed and its receipt is finalised, so this
# revision launches directly rather than waiting again.
LEGACY_ARTIFACT="results/local_base_qwen_train_full_s42/base_validation_train_seed_42.json"
SUCCESSOR_LOG="results/queue_train_successor.log"
GATE_REPORT="results/v4_2_sequence_fit_gate_train_successor.json"

echo "[queue] waiting for the legacy run to finish"
until [ -f "$LEGACY_ARTIFACT" ]; do
  sleep 30
done
echo "[queue] legacy artifact present"

echo "[queue] finalising the legacy protocol receipt against the COMPLETED artifact"
"$PY" scripts/build_legacy_generation_receipt.py "$LEGACY_ARTIFACT" > /dev/null || {
  echo "[queue] ABORT: could not finalise the legacy receipt"; exit 1; }

echo "[queue] GATE 1: sequence fit (tokenizer only, no model load)"
"$PY" scripts/sequence_fit_gate.py \
  --split train --prompt-budget 1024 --completion-budget 1024 \
  --sequence-limit 3072 --output "$GATE_REPORT" > results/queue_gate1.log 2>&1 || {
  echo "[queue] ABORT: sequence-fit gate failed; successor NOT launched"
  tail -20 results/queue_gate1.log
  exit 1; }
echo "[queue] GATE 1 passed"

# ONE definition of the run, used for BOTH the preflight and the launch.
# They were previously written out twice and drifted: --max-sequence-tokens was
# added to the preflight string and not to the launch, so the gate would have
# validated a command that was never the one executed.
RUN_ARGS=(
  --run-name local_base_qwen_train_successor_s42
  --phase base_eval
  --evaluation-split train
  --retain-raw-output
  --candidate-parse-mode whole_output
  --allow-test-function-candidates
  --generation-completion-token-limit 1024
  --max-sequence-tokens 3072
  --base-model-name Qwen/Qwen2.5-Coder-1.5B-Instruct
  --attention-implementation sdpa
  --sft-prompt-token-limit 1024
  --seed 42
)
CMD="python scripts/train_on_dataset.py ${RUN_ARGS[*]}"

echo "[queue] GATE 2: run configuration preflight"
echo "[queue] command: $CMD"
"$PY" scripts/gpu_run_preflight.py "$CMD" || {
  echo "[queue] ABORT: preflight failed; successor NOT launched"; exit 1; }
echo "[queue] GATE 2 passed"

echo "[queue] launching successor-protocol train generation"
"$PY" scripts/train_on_dataset.py "${RUN_ARGS[@]}" > "$SUCCESSOR_LOG" 2>&1

STATUS=$?
echo "[queue] successor generation exited with $STATUS"
exit $STATUS
