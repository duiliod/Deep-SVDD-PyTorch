#!/bin/bash
#SBATCH --job-name=dsvdd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --time=96:00:00
#SBATCH --output=slurm_logs/%x_%j.out

# Trains Deep SVDD (autoencoder pretraining + SVDD) for ONE structure on 1 GPU.
#
# Submit it from the repository root (the job runs in the submit directory):
#   mkdir -p slurm_logs
#   sbatch --job-name=dsvdd_HIP slurm/train_structure.sbatch HIP
# or submit all of them with slurm/submit_all.sh
#
# Environment variables of the submitting shell are passed to the job
# (PYTHON_BIN, EXTRA_ARGS, FORCE, ... see scripts/train_one.sh).

STRUCT="${1:?usage: sbatch slurm/train_structure.sbatch {THA|PAL|PUT|HIP|AMY}}"

# sbatch runs a copy of this file, so the repository is located via the submit dir
export DEEPSVDD_ROOT="${DEEPSVDD_ROOT:-${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is not set}}"
if [ ! -f "$DEEPSVDD_ROOT/scripts/train_one.sh" ]; then
  echo "ERROR: $DEEPSVDD_ROOT does not look like the repository root; submit from the repository root." >&2
  exit 1
fi
export PYTHON_BIN="${PYTHON_BIN:-$HOME/.conda/envs/deepsvd/bin/python}"

echo "job $SLURM_JOB_ID on $(hostname): structure=$STRUCT cpus=${SLURM_CPUS_PER_TASK:-?} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader 2>&1 || true

exec bash "$DEEPSVDD_ROOT/scripts/train_one.sh" "$STRUCT"
