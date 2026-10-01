#!/usr/bin/env bash
# v2.5 CPU program 2, Phase 6B-6D: build native environments for every project of the frozen
# attempt order (results/sft_root_cause_v25_native_attempt_spec.json), then re-attempt django and
# sympy with view rule v2. Fresh work root and one output directory per project; an existing
# output is never overwritten (the builder refuses). Run detached:
#   wsl -u root -- bash -c 'nohup bash scripts/v25_native_r2_build.sh RUN > /root/v25_r2_build_RUN.log 2>&1 &'
set -u
RUN="${1:?run number}"
ATTEMPT="${2:-a2}"
VIEW="${3:-v2}"
REPO=/mnt/c/Users/Student2/Desktop/Capstone/oneiros
cd "$REPO"
OUT=results/sft_root_cause/v25_native_r2
ORDER="youtube-dl ansible fastapi flask httpie sanic tornado matplotlib astropy pylint django sympy"
for p in $ORDER; do
  mkdir -p "$OUT/$p"
  if [ -e "$OUT/$p/envs_${ATTEMPT}_run$RUN.jsonl" ]; then echo "skip $p (exists)"; continue; fi
  echo "=== $p $(date -Is)"
  bash scripts/wsl_native_python.sh scripts/v25_native_env_wsl.py \
    --targets "$OUT/targets_${ATTEMPT}.jsonl" --projects "$p" --view-rule "$VIEW" \
    --work /root/oneiros_v25_native_r2 --patches "$OUT/patches" \
    --out "$OUT/$p/envs_${ATTEMPT}_run$RUN.jsonl"
done
echo "=== done $(date -Is)"
