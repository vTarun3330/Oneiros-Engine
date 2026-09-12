#!/usr/bin/env bash
# Queue the successor-protocol train-split regeneration behind the legacy run.
#
# Ordering matters and is enforced here rather than by watching a terminal:
# the two jobs would contend for the same GPU, and the legacy run must finish
# so its receipt can be finalised against a completed artifact rather than a
# progress checkpoint.
#
# The chain refuses to launch if preflight fails, so a misconfigured run cannot
# start unattended overnight.
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1
PY="./.venv-gpu/Scripts/python.exe"

LEGACY_ARTIFACT="results/local_base_qwen_train_full_s42/base_validation_train_seed_42.json"
SUCCESSOR_LOG="results/queue_train_successor.log"

echo "[queue] waiting for the legacy run to finish"
until [ -f "$LEGACY_ARTIFACT" ]; do
  sleep 30
done
echo "[queue] legacy artifact present"

# Finalise the receipt against the COMPLETED artifact, so it records
# run_completed=true and the real function count rather than a mid-run one.
echo "[queue] finalising the legacy protocol receipt"
"$PY" scripts/build_legacy_generation_receipt.py "$LEGACY_ARTIFACT" > /dev/null || {
  echo "[queue] ABORT: could not finalise the legacy receipt"; exit 1; }

CMD="python scripts/train_on_dataset.py --run-name local_base_qwen_train_successor_s42 --phase base_eval --evaluation-split train --retain-raw-output --candidate-parse-mode whole_output --allow-test-function-candidates --generation-completion-token-limit 1024 --base-model-name Qwen/Qwen2.5-Coder-1.5B-Instruct --attention-implementation sdpa --sft-prompt-token-limit 1024 --seed 42"

echo "[queue] preflight"
"$PY" scripts/gpu_run_preflight.py "$CMD" || {
  echo "[queue] ABORT: preflight failed; successor generation NOT launched"; exit 1; }

echo "[queue] launching successor-protocol train generation"
"$PY" scripts/train_on_dataset.py \
  --run-name local_base_qwen_train_successor_s42 \
  --phase base_eval \
  --evaluation-split train \
  --retain-raw-output \
  --candidate-parse-mode whole_output \
  --allow-test-function-candidates \
  --generation-completion-token-limit 1024 \
  --base-model-name Qwen/Qwen2.5-Coder-1.5B-Instruct \
  --attention-implementation sdpa \
  --sft-prompt-token-limit 1024 \
  --seed 42 > "$SUCCESSOR_LOG" 2>&1

STATUS=$?
echo "[queue] successor generation exited with $STATUS"
exit $STATUS
