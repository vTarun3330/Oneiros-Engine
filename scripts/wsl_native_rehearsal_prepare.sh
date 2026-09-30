#!/usr/bin/env bash
# Formal v2.1 re-qualification of the 24 rehearsal targets inside WSL (root).
# Environments and views stay under /root/oneiros_native_v21_prep; exports are copied to the
# repository by wsl_native_rehearsal_export.sh.
set -euo pipefail
REPO=/mnt/c/Users/Student2/Desktop/Capstone/oneiros
PY="$(uv python find 3.13)"
PREP=/root/oneiros_native_v21_prep
mkdir -p "$PREP"
"$PY" "$REPO/scripts/native_rehearsal_prepare_wsl.py" \
  "$REPO/results/sft_root_cause_phase4_receiver_capture_manifest_v2.json" "$PREP" \
  "$REPO/docs/SFT_ROOT_CAUSE_NATIVE_GENERATED_TEST_PROTOCOL_V2.md" \
  "$REPO/docs/SFT_ROOT_CAUSE_NATIVE_GENERATED_TEST_PROTOCOL_V2_1.md" 2>&1 | tee -a "$PREP/console.log"
