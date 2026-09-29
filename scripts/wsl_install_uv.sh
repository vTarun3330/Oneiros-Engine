#!/usr/bin/env bash
# Install the pinned uv release (same version that built .venv-gpu) inside WSL, verified
# against the release's published SHA-256, then the two approved interpreters (D6).
set -euo pipefail
VERSION="0.12.7"
DEST="/opt/oneiros-uv"
BASE="https://github.com/astral-sh/uv/releases/download/${VERSION}"
ARCHIVE="uv-x86_64-unknown-linux-gnu.tar.gz"
mkdir -p "$DEST"
cd "$DEST"
curl -sSfL -o "$ARCHIVE" "$BASE/$ARCHIVE"
curl -sSfL -o "$ARCHIVE.sha256" "$BASE/$ARCHIVE.sha256"
sha256sum -c "$ARCHIVE.sha256"
tar xzf "$ARCHIVE"
install -m 0755 uv-x86_64-unknown-linux-gnu/uv /usr/local/bin/uv
install -m 0755 uv-x86_64-unknown-linux-gnu/uvx /usr/local/bin/uvx
uv --version
uv python install 3.10 3.13
uv python find 3.10
uv python find 3.13
