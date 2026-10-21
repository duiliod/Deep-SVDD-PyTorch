#!/usr/bin/env bash
# Train several structures one after another on the CURRENT machine (no Slurm),
# using scripts/train_one.sh for each of them.
#
#   scripts/run_sequential.sh                 # THA PAL PUT HIP AMY
#   scripts/run_sequential.sh HIP THA         # only those
#   nohup scripts/run_sequential.sh > run_all.out 2>&1 &
#
# One log per structure in logs/<S>.log. A structure that fails does not stop the
# next ones; the exit code is the number of failed structures.
# Same environment variables as scripts/train_one.sh (PYTHON_BIN, CONDA_ENV, ...).

set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ALL=(THA PAL PUT HIP AMY)
if [ "$#" -gt 0 ]; then STRUCTS=("$@"); else STRUCTS=("${ALL[@]}"); fi
for s in "${STRUCTS[@]}"; do
  case " ${ALL[*]} " in *" $s "*) ;; *) echo "ERROR: unknown structure '$s' (valid: ${ALL[*]})" >&2; exit 1 ;; esac
done

# keep the machine awake while training (desktop machines)
if [ -z "${DEEPSVDD_INHIBITED:-}" ] && command -v systemd-inhibit >/dev/null 2>&1; then
  export DEEPSVDD_INHIBITED=1
  exec systemd-inhibit --what=sleep:idle --who=deepsvdd --why="Training Deep SVDD" "${BASH_SOURCE[0]}" "$@"
fi

LOGS_DIR="${LOGS_DIR:-$ROOT/logs}"
mkdir -p "$LOGS_DIR"
failed=0
for S in "${STRUCTS[@]}"; do
  echo "[$(date '+%F %T')] $S: starting (log: $LOGS_DIR/$S.log)"
  bash "$ROOT/scripts/train_one.sh" "$S" > "$LOGS_DIR/$S.log" 2>&1
  rc=$?
  if [ "$rc" -eq 0 ]; then
    echo "[$(date '+%F %T')] $S: OK"
  else
    failed=$(( failed + 1 )); echo "[$(date '+%F %T')] $S: FAILED rc=$rc (see $LOGS_DIR/$S.log)"
  fi
done
exit "$failed"
