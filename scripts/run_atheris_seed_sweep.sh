#!/usr/bin/env bash
# Run actual Atheris on the val panel at seeds 43 and 44, both budgets.
#
# Every Atheris number reported so far is seed 42 only. The simulated coverage
# fuzzer spans 0.378-0.514 across its three seeds - a 13.6-point spread wider
# than the entire SFT effect under study - so a single-seed number for a
# stochastic baseline is not a measurement, it is one draw.
#
# Matched-8 runs first for both seeds because that is the headline comparison;
# the 20,000-run arm is the expensive one and can finish later.
set -u
REPO=/mnt/c/Users/Student2/Desktop/Capstone/oneiros
TASKS="$REPO/results/v4_2_atheris_tasks_val.json"
RUNNER="$REPO/scripts/run_atheris_wsl.sh"
COUNT=757

for SEED in 43 44; do
  echo "[SWEEP] matched8 seed $SEED $(date -Is)"
  bash "$RUNNER" "$TASKS" "$REPO/results/atheris_val_matched8_seed$SEED" \
    "$COUNT" 8 10 5 30s "$SEED"
done

for SEED in 43 44; do
  echo "[SWEEP] 20000 seed $SEED $(date -Is)"
  bash "$RUNNER" "$TASKS" "$REPO/results/atheris_val_20000_seed$SEED" \
    "$COUNT" 20000 10 5 90s "$SEED"
done
echo "[SWEEP] done $(date -Is)"
