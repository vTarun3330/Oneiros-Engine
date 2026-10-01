#!/usr/bin/env bash
# v2.5 CPU program 2, Phase 6D: two repeatability runs of repository-candidate verification over
# the attempt-2 environments (every project in the pytest sandbox; django through the v2.5
# Django settings layer). Run detached; outputs are never overwritten (the verifier refuses).
set -u
REPO=/mnt/c/Users/Student2/Desktop/Capstone/oneiros
cd "$REPO"
D=results/sft_root_cause/v25_native_r2
for RUN in 1 2; do
  echo "=== run $RUN $(date -Is)"
  bash scripts/wsl_native_python.sh scripts/v25_verify_repository_wsl.py \
    --envs "$D/envs_a2_final.jsonl" --run "$RUN" --out "$D/verify_a2" \
    --candidates "$D/converted_candidates_r2r.jsonl" --targets "$D/targets_a2.jsonl" \
    --runners all --django-layer | grep -v '^{"task"'
done
echo "=== done $(date -Is)"
