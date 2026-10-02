#!/usr/bin/env bash
# Canary for scripts/wsl_isolated.sh: run INSIDE the isolation; prints one JSON line.
# PASS requires: no credential store reachable, no Windows drive contents beyond the repository
# path, the repository readable and writable, credential variables absent, DNS working (for
# dependency installs and clones).
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
creds=$(ls /mnt/c/Users/*/AppData/Roaming/"GitHub CLI"/hosts.yml 2>/dev/null | wc -l)
appdata=$(ls -d /mnt/c/Users/*/AppData 2>/dev/null | wc -l)
other=$(ls /mnt 2>/dev/null | grep -v -E '^(c|wsl)$' | wc -l)
repo_ok=0; [ -r "$REPO/scripts/wsl_isolated.sh" ] && repo_ok=1
probe="$REPO/results/.isolation_probe_$$"
write_ok=0; (echo x > "$probe" && rm -f "$probe") 2>/dev/null && write_ok=1
envvars=$(env | grep -cE '^(GITHUB_|GH_|SSH_)' || true)
dns_ok=0; getent hosts github.com >/dev/null 2>&1 && dns_ok=1
printf '{"credential_store_visible": %s, "appdata_visible": %s, "other_mnt_entries": %s, "repository_readable": %s, "repository_writable": %s, "credential_env_vars": %s, "dns_ok": %s}\n' \
  "$creds" "$appdata" "$other" "$repo_ok" "$write_ok" "$envvars" "$dns_ok"
