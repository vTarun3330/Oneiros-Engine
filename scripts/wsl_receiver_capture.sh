#!/usr/bin/env bash
# Phase 8 receiver-aware pilot inside WSL (root; uv CPython 3.13).
set -euo pipefail
REPO="$1"
PY="$(uv python find 3.13)"
OUT="$REPO/results/sft_root_cause/phase4_receiver_capture"
mkdir -p "$OUT"
"$PY" "$REPO/scripts/native_receiver_capture_wsl.py" "$REPO/results/sft_root_cause_phase4_receiver_capture_manifest_v2.json" "$OUT" 2>&1 | tee -a "$OUT/console.log"
