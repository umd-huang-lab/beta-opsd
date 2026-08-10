#!/usr/bin/env bash
#SBATCH --job-name=opsd_cmp_base
#SBATCH --account=scavenger
#SBATCH --partition=scavenger
#SBATCH --qos=scavenger
#SBATCH --gres=gpu:rtxa4000:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=slurm_logs/%x-%j.out

set -euo pipefail

module load cuda/12.8.1
module load gcc/11.2.0
source "${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate "${CONDA_ENV:-opsd}"

REPO=${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
RUN=${RUN:-cmp_opsd_baseline_$(date +%Y%m%d_%H%M)}
mkdir -p "$REPO/runs/$RUN/logs"

CACHE_ROOT=${CACHE_ROOT:-$REPO/cache}
mkdir -p "$CACHE_ROOT"/{xdg,wandb,torch_extensions,triton}
export XDG_CACHE_HOME="$CACHE_ROOT/xdg"
export WANDB_DIR="$CACHE_ROOT/wandb"
export WANDB_CACHE_DIR="$CACHE_ROOT/wandb"
export TORCH_EXTENSIONS_DIR="$CACHE_ROOT/torch_extensions"
export TRITON_CACHE_DIR="$CACHE_ROOT/triton"

echo "[sbatch] log: $REPO/runs/$RUN/logs/opsd_cmp_base-${SLURM_JOB_ID}.out"
exec > "$REPO/runs/$RUN/logs/opsd_cmp_base-${SLURM_JOB_ID}.out" 2>&1

RUN="$RUN" bash "$REPO/scripts/run_opsd_1b_compare.sh"
