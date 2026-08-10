#!/usr/bin/env bash
# Generic Slurm template for fixed-teacher OPSD recipes.
#
# Example:
#   sbatch --account=<acct> --partition=<part> --gres=gpu:4 scripts/slurm/submit_opsd.sbatch

#SBATCH --job-name=opsd
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

module load cuda/12.8.1 2>/dev/null || true
module load gcc/11.2.0 2>/dev/null || true

if [[ -f "${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}" ]]; then
    source "${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
    conda activate "${CONDA_ENV:-opsd}"
fi

REPO=${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
cd "$REPO"
bash scripts/train/opsd.sh
