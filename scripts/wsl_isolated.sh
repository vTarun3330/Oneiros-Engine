#!/usr/bin/env bash
# v2.7 credential isolation for every WSL job that runs UNTRUSTED repository code (dependency
# installs, setup scripts, official tests, native environments, Atheris).
#
#   wsl -u root -- bash scripts/wsl_isolated.sh <command> [args...]
#
# The command runs in a private mount namespace in which /mnt is an empty tmpfs: the Windows
# drives (and with them the user profile, the GitHub CLI hosts.yml credential store, SSH keys
# and every other Windows file) do not exist. Only THIS repository checkout is bound back at
# its usual path, so scripts, manifests and results keep working. Credential-shaped
# environment variables are unset. Nothing is changed outside the namespace.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
case "$REPO" in /mnt/*) ;; *) echo "wsl_isolated: unexpected repository path $REPO" >&2; exit 2;; esac
exec unshare --mount --fork -- /bin/bash -c '
  set -euo pipefail
  REPO="$1"; shift
  mount --make-rprivate /
  HOLD=/run/oneiros_repo_hold
  WSLHOLD=/run/oneiros_wsl_hold
  mkdir -p "$HOLD" "$WSLHOLD"
  mount --bind "$REPO" "$HOLD"
  # /mnt/wsl holds WSL runtime files only (resolv.conf for DNS), no Windows data
  if [ -d /mnt/wsl ]; then mount --bind /mnt/wsl "$WSLHOLD"; fi
  mount -t tmpfs -o size=16m,mode=0755 tmpfs /mnt
  mkdir -p "$REPO" /mnt/wsl
  mount --bind "$HOLD" "$REPO"
  mountpoint -q "$WSLHOLD" && mount --bind "$WSLHOLD" /mnt/wsl
  umount -l "$HOLD"
  mountpoint -q "$WSLHOLD" && umount -l "$WSLHOLD"
  for v in $(env | cut -d= -f1); do
    case "$v" in
      GITHUB_*|GH_*|SSH_*|GIT_ASKPASS|GIT_CREDENTIAL*|GCM_*|*TOKEN*|*SECRET*|*PASSWORD*|*API_KEY*) unset "$v";;
    esac
  done
  cd "$REPO"
  exec "$@"
' oneiros-isolated "$REPO" "$@"
