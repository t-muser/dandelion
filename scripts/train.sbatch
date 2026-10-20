#!/bin/bash
# Train one (model, dataset, lr) cell on a single 4-GPU node, then test it.
# Adjust the #SBATCH lines for your cluster. Example:
#   mkdir -p slurm && DATA=active-turbulence MODEL=dandelion LR=5e-4 BATCH_SIZE=42 sbatch scripts/train.sbatch
#SBATCH --job-name=dandelion
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --time=24:00:00
#SBATCH --output=slurm/%x-%j.out

set -euo pipefail

: "${DATA:?set DATA, e.g. active-turbulence}"
: "${MODEL:?set MODEL, e.g. dandelion}"
LR="${LR:-5e-4}"
BATCH_SIZE="${BATCH_SIZE:?set BATCH_SIZE (per GPU, see README)}"
NGPUS="${NGPUS:-4}"

ARGS=(
    --data "configs/data/${DATA}.yaml"
    --model "configs/models/${MODEL}.yaml"
    --train configs/train.yaml
    optimizer.lr="${LR}"
    data.batch_size="${BATCH_SIZE}"
)

uv run --no-sync torchrun --standalone --nproc_per_node="${NGPUS}" \
    -m dandelion.train "${ARGS[@]}" "$@"
uv run --no-sync torchrun --standalone --nproc_per_node="${NGPUS}" \
    -m dandelion.train "${ARGS[@]}" test_mode=true auto_resume=true "$@"
