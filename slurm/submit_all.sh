#!/usr/bin/env bash
# Submit one Slurm job per structure (1 GPU each).
#
#   slurm/submit_all.sh                # THA PAL PUT HIP AMY
#   slurm/submit_all.sh HIP THA        # only those
#   DRY_RUN=1 slurm/submit_all.sh      # show what would be submitted
#
# Structures that already finished (runs/<S>/.done) or that are already queued or
# running are skipped. FORCE=1 retrains finished ones. Extra sbatch options can be
# given in SBATCH_ARGS (e.g. SBATCH_ARGS="--nodelist=thanos").
# The cluster has 2 GPUs per node, so with 2 nodes up the 5th job waits in the queue.

set -eu
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

ALL=(THA PAL PUT HIP AMY)
if [ "$#" -gt 0 ]; then STRUCTS=("$@"); else STRUCTS=("${ALL[@]}"); fi
for s in "${STRUCTS[@]}"; do
  case " ${ALL[*]} " in *" $s "*) ;; *) echo "ERROR: unknown structure '$s' (valid: ${ALL[*]})" >&2; exit 1 ;; esac
done

mkdir -p slurm_logs
for S in "${STRUCTS[@]}"; do
  if [ -f "runs/$S/.done" ] && [ -z "${FORCE:-}" ]; then
    echo "$S: already done, skipping (FORCE=1 to retrain)"; continue
  fi
  if [ -z "${DRY_RUN:-}" ] && squeue -h -u "$USER" -n "dsvdd_$S" 2>/dev/null | grep -q .; then
    echo "$S: already queued or running, skipping"; continue
  fi
  if [ -n "${DRY_RUN:-}" ]; then
    echo "[dry-run] sbatch --job-name=dsvdd_$S ${SBATCH_ARGS:-} slurm/train_structure.sbatch $S"
  else
    # shellcheck disable=SC2086
    jid=$(sbatch --parsable --job-name="dsvdd_$S" ${SBATCH_ARGS:-} slurm/train_structure.sbatch "$S")
    echo "$S: submitted job $jid  (log: slurm_logs/dsvdd_${S}_${jid%%;*}.out)"
  fi
done
