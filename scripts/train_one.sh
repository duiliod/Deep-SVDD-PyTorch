#!/usr/bin/env bash
# Train Deep SVDD (autoencoder pretraining + SVDD) for ONE brain structure.
# Used by run_all_structures.sh (local) and slurm/train_structure.sbatch (cluster).
#
# Usage:  scripts/train_one.sh {THA|PAL|PUT|HIP|AMY}
#
# Every structure runs in its own folder, so several of them can train at the
# same time without touching each other's files:
#   runs/<S>/trainer/    working directory (train_SVDD.py writes data_losses.npy here)
#   runs/<S>/results/    checkpoints and CSV logs (lightning.py expects ../results)
#   runs/<S>/.done       created when training finished successfully
#
# Re-running is safe: a finished structure is skipped (FORCE=1 to retrain) and the
# leftovers of an interrupted run are moved to runs/_previous/ first, because
# lightning.py always loads the pretrained AE from version_0.
#
# Environment variables (all optional):
#   PYTHON_BIN     python interpreter to use (default: first `python` in PATH)
#   CONDA_ENV      if PYTHON_BIN is not set: conda env to activate
#   CONDA_SH       path to conda.sh                 (default: $HOME/anaconda3/etc/profile.d/conda.sh)
#   DEEPSVDD_ROOT  repository root                  (default: parent of scripts/)
#   RUNS_DIR       where runs/ lives                (default: $DEEPSVDD_ROOT/runs)
#   CONFIG_DIR     folder with the YAML configs     (default: $DEEPSVDD_ROOT/configs)
#   EXTRA_ARGS     extra args for train_SVDD.py     (e.g. "--fast-dev-run --no-pretrained")
#   FORCE=1        retrain even if runs/<S>/.done exists

set -u

STRUCT="${1:-}"
case "$STRUCT" in
  THA|PAL|PUT|HIP|AMY) ;;
  *) echo "Usage: $0 {THA|PAL|PUT|HIP|AMY}" >&2; exit 2 ;;
esac

ROOT="${DEEPSVDD_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export DEEPSVDD_ROOT="$ROOT"   # '{root}' in configs/splits resolves to this folder
RUNS_DIR="${RUNS_DIR:-$ROOT/runs}"
CONFIG_DIR="${CONFIG_DIR:-$ROOT/configs}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
# smoke tests (--fast-dev-run) go to their own folder so that they can never
# leave a .done marker or partial results in the real run folder
case " $EXTRA_ARGS " in *" --fast-dev-run "*) RUNS_DIR="$RUNS_DIR/_smoke" ;; esac
RUN_DIR="$RUNS_DIR/$STRUCT"
CFG_AE="$CONFIG_DIR/AE_LeNet_bothHippocampiFlip_lr_001_rd_32_MSE_16filters.yml"
CFG_SVDD="$CONFIG_DIR/AE_LeNet_bothHippocampiFlip_lr_001_rd_32_MSE_16filtersSVDD.yml"

ts() { date '+%F %T'; }

if [ -z "${PYTHON_BIN:-}" ]; then
  if [ -n "${CONDA_ENV:-}" ]; then
    # shellcheck disable=SC1090
    source "${CONDA_SH:-$HOME/anaconda3/etc/profile.d/conda.sh}" && conda activate "$CONDA_ENV" \
      || { echo "ERROR: cannot activate conda env '$CONDA_ENV'" >&2; exit 1; }
  fi
  PYTHON_BIN=python
fi

for f in "$CFG_AE" "$CFG_SVDD"; do
  [ -f "$f" ] || { echo "ERROR: config not found: $f" >&2; exit 1; }
done

if [ -f "$RUN_DIR/.done" ] && [ -z "${FORCE:-}" ]; then
  echo "[$(ts)] $STRUCT: already done ($RUN_DIR/.done), skipping (FORCE=1 to retrain)"
  exit 0
fi
if [ -e "$RUN_DIR" ]; then
  mkdir -p "$RUNS_DIR/_previous"
  mv "$RUN_DIR" "$RUNS_DIR/_previous/${STRUCT}_$(date +%Y%m%d_%H%M%S)"
  echo "[$(ts)] $STRUCT: moved previous run to $RUNS_DIR/_previous/"
fi
mkdir -p "$RUN_DIR/trainer"
cd "$RUN_DIR/trainer" || exit 1

echo "[$(ts)] $STRUCT: starting on $(hostname)"
echo "  root=$ROOT"
echo "  python=$("$PYTHON_BIN" --version 2>&1) ($(command -v "$PYTHON_BIN" || echo "$PYTHON_BIN"))"
echo "  run dir=$RUN_DIR"

start=$(date +%s)
# EXTRA_ARGS is intentionally unquoted so that it splits into several arguments
"$PYTHON_BIN" -u "$ROOT/src/trainer/train_SVDD.py" \
  --configAE "$CFG_AE" --configSVDD "$CFG_SVDD" --structure "$STRUCT" $EXTRA_ARGS
rc=$?
elapsed=$(( $(date +%s) - start ))

if [ "$rc" -eq 0 ]; then
  ts > "$RUN_DIR/.done"
  printf '[%s] %s: finished OK in %dh%02dm\n' "$(ts)" "$STRUCT" $(( elapsed / 3600 )) $(( (elapsed % 3600) / 60 ))
else
  printf '[%s] %s: FAILED with exit code %d after %dh%02dm\n' "$(ts)" "$STRUCT" "$rc" $(( elapsed / 3600 )) $(( (elapsed % 3600) / 60 ))
fi
exit "$rc"
