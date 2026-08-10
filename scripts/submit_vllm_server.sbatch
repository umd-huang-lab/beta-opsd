#!/usr/bin/env bash
#SBATCH --job-name=vllm_serve
#SBATCH --account=scavenger
#SBATCH --partition=scavenger
#SBATCH --qos=scavenger
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/%x-%j.out
#
# Usage:
#   MODEL=Qwen/Qwen3-1.7B sbatch scripts/submit_vllm_server.sbatch
#
# After the server starts, note the node hostname from the log and pass it to
# the training job:
#   VLLM_MODE=server VLLM_SERVER_HOST=<node> sbatch scripts/submit_train.sbatch
#
# To use the tron partition for a more stable server:
#   sbatch --account=nexus --partition=tron --qos=high scripts/submit_vllm_server.sbatch

set -euo pipefail

module load cuda/12.8.1
module load gcc/11.2.0

REPO=${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}

MODEL=${MODEL:-Qwen/Qwen3-1.7B}
PORT=${PORT:-8001}
TENSOR_PARALLEL=${TENSOR_PARALLEL:-1}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.9}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-20000}

echo "[vllm-serve] slurm_job=$SLURM_JOB_ID  node=$(hostname)  port=$PORT"
echo "[vllm-serve] Pass to training: VLLM_SERVER_HOST=$(hostname) VLLM_SERVER_PORT=$PORT"

MODEL="$MODEL" PORT="$PORT" TENSOR_PARALLEL="$TENSOR_PARALLEL" \
GPU_MEMORY_UTILIZATION="$GPU_MEMORY_UTILIZATION" MAX_MODEL_LEN="$MAX_MODEL_LEN" \
bash "$REPO/scripts/serve_vllm.sh"
