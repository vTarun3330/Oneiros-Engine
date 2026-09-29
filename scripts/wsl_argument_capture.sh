#!/usr/bin/env bash
# Run the Phase 8 argument-capture pilot inside WSL (root; uv-managed CPython 3.13).
# Usage (from Windows):  MSYS_NO_PATHCONV=1 wsl.exe -u root -- bash /mnt/c/<repo>/scripts/wsl_argument_capture.sh /mnt/c/<repo>
set -euo pipefail
REPO="$1"
PY="$(uv python find 3.13)"
OUT="$REPO/results/sft_root_cause/phase4_argument_capture"
mkdir -p "$OUT"
"$PY" "$REPO/scripts/native_argument_capture_wsl.py" \
  "$REPO/results/sft_root_cause_phase4_argument_capture_manifest_v1.json" "$OUT" 2>&1 | tee "$OUT/console.log"
