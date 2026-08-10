#!/usr/bin/env bash
#SBATCH --job-name=opsd_4b_train
#SBATCH --account=nexus
#SBATCH --partition=tron
#SBATCH --qos=high
#SBATCH --gres=gpu:9
#SBATCH --cpus-per-task=18
#SBATCH --mem=160G
#SBATCH --time=48:00:00
#SBATCH --output=slurm_logs/%x-%j.out
#
# 8 GPUs for training (Qwen3-4B) + 1 GPU for local vLLM server.
#
# Usage:
#   RUN=myrun sbatch scripts/submit_opsd_4b_localvllm.sbatch

set -euo pipefail

module load cuda/12.8.1
module load gcc/11.2.0
source "${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate "${CONDA_ENV:-opsd}"

REPO=${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
RUN=${RUN:-qwen34b_opsd_localvllm_$(date +%Y%m%d)}
mkdir -p "$REPO/runs/$RUN/logs"

echo "[sbatch] log: $REPO/runs/$RUN/logs/opsd_4b_train-${SLURM_JOB_ID}.out"
exec > "$REPO/runs/$RUN/logs/opsd_4b_train-${SLURM_JOB_ID}.out" 2>&1

NUM_TRAIN_GPUS=8 NUM_VLLM_GPUS=1 MODEL=Qwen/Qwen3-4B RUN="$RUN" \
    bash "$REPO/scripts/run_with_local_vllm.sh" "$REPO/scripts/run_opsd_4b.sh"
