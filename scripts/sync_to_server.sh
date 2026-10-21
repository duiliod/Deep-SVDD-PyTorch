#!/usr/bin/env bash
# Copy this repository (code, configs, splits, labels) and/or the NIfTI masks to
# the cluster with rsync. Run it from the machine that has the data (not on the
# cluster). It does not need GitHub access from the cluster.
#
#   scripts/sync_to_server.sh code               # code, configs, splits, labels (a few MB)
#   scripts/sync_to_server.sh data               # masks of the 5 structures (~14 GB each)
#   scripts/sync_to_server.sh data HIP THA       # masks of some structures only
#   scripts/sync_to_server.sh all                # code + data
#   DRY_RUN=1 scripts/sync_to_server.sh data     # show what would be copied
#
# Interrupted copies can simply be repeated (rsync only transfers what is missing).
#
# Environment variables (optional):
#   SERVER      ssh target                 (default: ddeangeli@192.168.7.244)
#   REMOTE_DIR  folder on the server       (default: deepsvdd_oasis, relative to the remote home)

set -eu
SERVER="${SERVER:-ddeangeli@192.168.7.244}"
REMOTE_DIR="${REMOTE_DIR:-deepsvdd_oasis}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ALL=(THA PAL PUT HIP AMY)

WHAT="${1:-}"; [ "$#" -gt 0 ] && shift
case "$WHAT" in code|data|all) ;; *) echo "Usage: $0 {code|data|all} [STRUCTURES...]" >&2; exit 2 ;; esac
if [ "$#" -gt 0 ]; then STRUCTS=("$@"); else STRUCTS=("${ALL[@]}"); fi
for s in "${STRUCTS[@]}"; do
  case " ${ALL[*]} " in *" $s "*) ;; *) echo "ERROR: unknown structure '$s' (valid: ${ALL[*]})" >&2; exit 1 ;; esac
done

RSYNC=(rsync -a --partial --human-readable --info=progress2)
[ -n "${DRY_RUN:-}" ] && RSYNC+=(-n -v)

if [ -z "${DRY_RUN:-}" ]; then
  ssh "$SERVER" "mkdir -p '$REMOTE_DIR'"
fi

if [ "$WHAT" = code ] || [ "$WHAT" = all ]; then
  echo "== code -> $SERVER:$REMOTE_DIR"
  "${RSYNC[@]}" \
    --exclude '.git/' --exclude 'data/*/hippocampi_images/' --exclude 'runs/' --exclude 'logs/' \
    --exclude 'slurm_logs/' --exclude 'src/results/' --exclude '__pycache__/' --exclude '*.ipynb' \
    --exclude '*.ckpt' --exclude '*.out' \
    "$ROOT/" "$SERVER:$REMOTE_DIR/"
fi

if [ "$WHAT" = data ] || [ "$WHAT" = all ]; then
  need_gb=$(( ${#STRUCTS[@]} * 14 ))
  free_gb=$(ssh "$SERVER" "df -BG --output=avail \"\$HOME\" | tail -1 | tr -dc 0-9")
  echo "== data: ${STRUCTS[*]} (~${need_gb} GB needed, ${free_gb} GB free on the server's home)"
  if [ "$free_gb" -lt "$need_gb" ] && [ -z "${DRY_RUN:-}" ]; then
    echo "ERROR: not enough free space on the server." >&2; exit 1
  fi
  for S in "${STRUCTS[@]}"; do
    echo "-- $S"
    if [ -z "${DRY_RUN:-}" ]; then
      ssh "$SERVER" "mkdir -p '$REMOTE_DIR/data/$S/hippocampi_images'"
    fi
    "${RSYNC[@]}" "$ROOT/data/$S/hippocampi_images/" "$SERVER:$REMOTE_DIR/data/$S/hippocampi_images/"
  done
fi
echo "done."
