#!/usr/bin/env bash
# Inner half of the generated-code sandbox (protocol v2 section 5, amendment v2.1 section B).
# Runs as root INSIDE  unshare --mount --pid --net --fork --mount-proc  and execs the command
# as nobody with no_new_privs, no capabilities and prlimit limits.
#
# Environment (set by scripts/native_generated_tests_execute_wsl.py):
#   SB_RO        newline-separated absolute paths kept visible read-only at their own path
#   SB_TARGET    host directory (a sanitised revision view) mounted read-only at /target
#   SB_OUT       host directory bound writable at /sandbox_out (outputs only)
#   SB_WORK_SRC  host directory copied into the private /tmp/work
#   SB_PYTHONPATH  PYTHONPATH for the command (identical for both revisions)
#   SB_CPU SB_AS SB_NPROC SB_NOFILE SB_FSIZE SB_WALL   limits
set -euo pipefail
mount --make-rprivate /
STAGE="$(mktemp -d /run/oneiros_sb.XXXXXX)"
mount -t tmpfs -o size=64m,mode=0700 tmpfs "$STAGE"
declare -a RO=()
i=0
while IFS= read -r p; do
  [ -z "$p" ] && continue
  RO+=("$p"); mkdir -p "$STAGE/ro$i"; mount --bind "$p" "$STAGE/ro$i"; i=$((i + 1))
done <<< "${SB_RO:-}"
mkdir -p "$STAGE/out" "$STAGE/work" "$STAGE/target"
mount --bind "$SB_OUT" "$STAGE/out"
mount --bind "$SB_TARGET" "$STAGE/target"
cp -a "$SB_WORK_SRC/." "$STAGE/work/"
# Hide everything user- or host-specific: homes, Windows drives, worktrees, caches.
for d in /root /home /mnt /tmp /srv /opt /var/lib /var/tmp /var/cache /media; do
  [ -d "$d" ] && mount -t tmpfs -o size=256m,mode=0755 tmpfs "$d"
done
i=0
for p in "${RO[@]}"; do
  mkdir -p "$p"
  mount --bind "$STAGE/ro$i" "$p"
  mount -o remount,bind,ro,nosuid,nodev "$p"
  i=$((i + 1))
done
# One canonical path for either revision: nothing in it names buggy, fixed or a commit.
mkdir -p /target
mount --bind "$STAGE/target" /target
mount -o remount,bind,ro,nosuid,nodev /target
mkdir -p /sandbox_out /tmp/work
mount --bind "$STAGE/out" /sandbox_out
cp -a "$STAGE/work/." /tmp/work/
chown -R 65534:65534 /tmp/work /sandbox_out
chmod 0700 /tmp/work
chmod 1777 /tmp
umount -l "$STAGE"
rmdir "$STAGE" 2>/dev/null || true
cd /tmp/work
exec setpriv --reuid=65534 --regid=65534 --clear-groups --no-new-privs \
  --inh-caps=-all --bounding-set=-all \
  prlimit --cpu="$SB_CPU" --as="$SB_AS" --nproc="$SB_NPROC" --nofile="$SB_NOFILE" \
          --fsize="$SB_FSIZE" --core=0 -- \
  timeout -k 5 "$SB_WALL" \
  env -i PATH=/usr/bin:/bin HOME=/tmp/work TMPDIR=/tmp/work LANG=C.UTF-8 \
      PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONHASHSEED=0 \
      PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH="$SB_PYTHONPATH" \
      "$@"
