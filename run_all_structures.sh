#!/usr/bin/env bash
# Train Deep SVDD (autoencoder pretraining + SVDD) for several brain
# structures, one after another.
#
# Usage:
#   ./run_all_structures.sh                  # THA PAL PUT HIP AMY, in that order
#   ./run_all_structures.sh HIP THA          # only those, in that order
#   nohup ./run_all_structures.sh > run_all.out 2>&1 &
#
# Re-running is safe: structures that finished (logs/<S>.done) are skipped, and
# the partial results of an interrupted structure are moved to
# src/results/_incomplete/ before it is trained again from scratch. (This is
# needed because lightning.py loads the pretrained AE from version_0 only.)
#
# Optional environment variables:
#   CONDA_ENV    conda env to use                    (default: PRIMO)
#   CONDA_SH     path to conda.sh                    (default: $HOME/anaconda3/etc/profile.d/conda.sh)
#   REMOTE_ROOT  path the configs/splits expect      (default: /home/ddeangeli/deepsvdd)
#   LOGS_DIR     where logs and .done markers go     (default: <repo>/logs)
#   EXTRA_ARGS   extra args for train_SVDD.py        (e.g. "--fast-dev-run --no-pretrained")
#   DRY_RUN=1    only print what would be done

set -u

ALL_STRUCTURES=(THA PAL PUT HIP AMY)
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Keep the machine awake while training: re-exec once under systemd-inhibit.
if [ -z "${DEEPSVDD_INHIBITED:-}" ] && [ -z "${DRY_RUN:-}" ] && command -v systemd-inhibit >/dev/null 2>&1; then
  export DEEPSVDD_INHIBITED=1
  exec systemd-inhibit --what=sleep:idle --who=deepsvdd --why="Training Deep SVDD models" "${BASH_SOURCE[0]}" "$@"
fi

CONDA_ENV="${CONDA_ENV:-PRIMO}"
CONDA_SH="${CONDA_SH:-$HOME/anaconda3/etc/profile.d/conda.sh}"
REMOTE_ROOT="${REMOTE_ROOT:-/home/ddeangeli/deepsvdd}"
LOGS_DIR="${LOGS_DIR:-$ROOT/logs}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
TRAINER_DIR="$ROOT/src/trainer"
RESULTS_DIR="$ROOT/src/results"
CFG_AE="../../configs/AE_LeNet_bothHippocampiFlip_lr_001_rd_32_MSE_16filters.yml"
CFG_SVDD="../../configs/AE_LeNet_bothHippocampiFlip_lr_001_rd_32_MSE_16filtersSVDD.yml"
VISDOM_PORT=8097

if [ "$#" -gt 0 ]; then
  STRUCTURES=("$@")
else
  STRUCTURES=("${ALL_STRUCTURES[@]}")
fi
for s in "${STRUCTURES[@]}"; do
  case " ${ALL_STRUCTURES[*]} " in
    *" $s "*) ;;
    *) echo "ERROR: unknown structure '$s' (valid: ${ALL_STRUCTURES[*]})" >&2; exit 1 ;;
  esac
done

ts() { date '+%F %T'; }
fmt_elapsed() { printf '%dh%02dm' $(( $1 / 3600 )) $(( ($1 % 3600) / 60 )); }

# --- dry run: show the plan and stop ------------------------------------------
if [ -n "${DRY_RUN:-}" ]; then
  echo "[dry-run] conda env: $CONDA_ENV   logs: $LOGS_DIR   remote root: $REMOTE_ROOT"
  for S in "${STRUCTURES[@]}"; do
    if [ -f "$LOGS_DIR/$S.done" ]; then
      echo "[dry-run] $S: already done -> would skip"
    else
      echo "[dry-run] $S: cd $TRAINER_DIR && python -u train_SVDD.py --configAE $CFG_AE --configSVDD $CFG_SVDD --structure $S $EXTRA_ARGS"
    fi
  done
  exit 0
fi

mkdir -p "$LOGS_DIR"

# --- the configs/splits refer to $REMOTE_ROOT/{data,split,configs} -------------
mkdir -p "$REMOTE_ROOT" || exit 1
for d in data split configs; do
  target="$REMOTE_ROOT/$d"
  if [ -e "$target" ] && [ ! -L "$target" ]; then
    echo "ERROR: $target exists and is not a symlink; refusing to overwrite it." >&2
    exit 1
  fi
  ln -sfn "$ROOT/$d" "$target"
done

# --- environment ---------------------------------------------------------------
if [ ! -f "$CONDA_SH" ]; then
  echo "ERROR: conda.sh not found at $CONDA_SH (set CONDA_SH)." >&2
  exit 1
fi
# shellcheck disable=SC1090
source "$CONDA_SH"
conda activate "$CONDA_ENV" || { echo "ERROR: cannot activate conda env '$CONDA_ENV'." >&2; exit 1; }

# --- visdom (the loggers in configs/visdom.yml need a server) ------------------
visdom_up() { (exec 3<>"/dev/tcp/localhost/$VISDOM_PORT") 2>/dev/null; }
if ! visdom_up; then
  echo "[$(ts)] starting a local Visdom server on port $VISDOM_PORT (log: $LOGS_DIR/visdom_server.log)"
  nohup python -m visdom.server -port "$VISDOM_PORT" > "$LOGS_DIR/visdom_server.log" 2>&1 &
  for _ in $(seq 1 30); do visdom_up && break; sleep 1; done
  if ! visdom_up; then
    echo "ERROR: Visdom server did not come up; see $LOGS_DIR/visdom_server.log" >&2
    exit 1
  fi
fi

# --- train ---------------------------------------------------------------------
summary=()
failed=0
for S in "${STRUCTURES[@]}"; do
  DONE="$LOGS_DIR/$S.done"
  LOG="$LOGS_DIR/$S.log"

  if [ -f "$DONE" ]; then
    echo "[$(ts)] $S: already done, skipping"
    summary+=("$S: skipped (already done)")
    continue
  fi

  # leftovers of an interrupted run would be picked up as version_0 by lightning.py
  for d in "$RESULTS_DIR/${S}_"*; do
    [ -e "$d" ] || continue
    mkdir -p "$RESULTS_DIR/_incomplete"
    mv "$d" "$RESULTS_DIR/_incomplete/$(basename "$d")_$(date +%Y%m%d_%H%M%S)"
    echo "[$(ts)] $S: moved previous partial results to $RESULTS_DIR/_incomplete/"
  done

  echo "[$(ts)] $S: starting (log: $LOG)"
  start=$(date +%s)
  # EXTRA_ARGS is intentionally unquoted so that it splits into several args
  ( cd "$TRAINER_DIR" && python -u train_SVDD.py \
      --configAE "$CFG_AE" --configSVDD "$CFG_SVDD" --structure "$S" $EXTRA_ARGS ) > "$LOG" 2>&1
  rc=$?
  elapsed=$(( $(date +%s) - start ))

  if [ "$rc" -eq 0 ]; then
    ts > "$DONE"
    echo "[$(ts)] $S: finished OK in $(fmt_elapsed "$elapsed")"
    summary+=("$S: OK ($(fmt_elapsed "$elapsed"))")
  else
    failed=$(( failed + 1 ))
    echo "[$(ts)] $S: FAILED with exit code $rc after $(fmt_elapsed "$elapsed") - see $LOG"
    summary+=("$S: FAILED rc=$rc ($(fmt_elapsed "$elapsed"))")
  fi
done

echo
echo "===== summary ====="
printf '%s\n' "${summary[@]}"
exit "$failed"
