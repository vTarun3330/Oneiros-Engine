#!/usr/bin/env bash
# v2.7 formal 3/3 requalification of confirmation targets (run INSIDE scripts/wsl_isolated.sh).
#   bash scripts/wsl_isolated.sh bash scripts/v27_prepare_wsl.sh <manifest.json> <name>
# Environments, checkouts and views stay under /root/oneiros_v27_prep/<name> (absolute path, as
# the v2.1 preparation requires); records and the SEPARATED exports (buggy_view = model-visible,
# verifier = hidden) are then copied to results/sft_root_cause/v27_confirmation/<name>_prep.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MANIFEST="$1"; NAME="$2"
PY="$(uv python find 3.13)"
PREP="/root/oneiros_v27_prep/$NAME"
DST="$REPO/results/sft_root_cause/v27_confirmation/${NAME}_prep"
[ -e "$DST" ] && { echo "REFUSED: $DST exists"; exit 2; }
mkdir -p "$PREP"
"$PY" "$REPO/scripts/native_rehearsal_prepare_wsl.py" "$REPO/$MANIFEST" "$PREP" \
  "$REPO/docs/repository_native_v27_confirmation_panel_gate.json" 2>&1 | tee -a "$PREP/console.log"
mkdir -p "$DST"
cp -r "$PREP/exports/buggy_view" "$PREP/exports/verifier" "$DST/" 2>/dev/null || true
cp "$PREP/records.jsonl" "$PREP/contract.json" "$PREP/run_summary.json" "$PREP/console.log" "$DST/"
echo "=== prepared $(date -Is)"
