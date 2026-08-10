#!/usr/bin/env bash
# Submit a prompt-regime analysis for copying vs independent reasoning.
#
# Default behavior:
# - Uses the local Qwen3-1.7B path from existing eval scripts
# - Evaluates student / teacher_independent / teacher_copy on the local refs dataset
# - Writes per-mode JSON plus a summary under analysis/results/copying_eval/$RUN

#SBATCH --job-name=copy_eval
#SBATCH --account=scavenger
#SBATCH --partition=scavenger
#SBATCH --qos=scavenger
#SBATCH --gres=gpu:h200-sxm:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=slurm_logs/%x-%j.out

set -euo pipefail

module load cuda/12.8.1
if ! module load gcc/11.2.0; then
  echo "[copy-eval] gcc/11.2.0 module unavailable on $(hostname); continuing with environment compiler"
fi
source "${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate "${CONDA_ENV:-opsd}"

REPO=${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
cd "$REPO"

export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export TORCHINDUCTOR_CACHE_DIR=${CACHE_ROOT:-$REPO/cache}/inductor
export TRITON_CACHE_DIR=${CACHE_ROOT:-$REPO/cache}/triton
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RUN=${RUN:-copying_eval_$(date +%Y%m%d_%H%M)}
BASE_MODEL=${BASE_MODEL:-/data0/shared/Qwen3-1.7B}
DATASET_PATH=${DATASET_PATH:-openthoughts_refs_1000_qwen3_8b.jsonl}
LIMIT=${LIMIT:-200}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-2048}
TEMPERATURE=${TEMPERATURE:-0.0}
TOP_P=${TOP_P:-1.0}
TOP_K=${TOP_K:--1}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.6}
TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-1}
OUTPUT_DIR=${OUTPUT_DIR:-analysis/results/copying_eval/$RUN}
MODES=${MODES:-student teacher_independent teacher_copy}

mkdir -p "$OUTPUT_DIR"
exec > "$OUTPUT_DIR/run-${SLURM_JOB_ID}.log" 2>&1

cat > "$OUTPUT_DIR/config.json" <<CONFIG
{
  "run": "$RUN",
  "base_model": "$BASE_MODEL",
  "dataset_path": "$DATASET_PATH",
  "limit": $LIMIT,
  "max_new_tokens": $MAX_NEW_TOKENS,
  "temperature": $TEMPERATURE,
  "top_p": $TOP_P,
  "top_k": $TOP_K,
  "modes": "$MODES",
  "gpu_memory_utilization": $GPU_MEMORY_UTILIZATION,
  "tensor_parallel_size": $TENSOR_PARALLEL_SIZE,
  "script": "submit_copying_eval.sbatch",
  "slurm_job_id": "${SLURM_JOB_ID}",
  "launched": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
CONFIG

echo "[copy-eval] run=$RUN"
echo "[copy-eval] base_model=$BASE_MODEL"
echo "[copy-eval] dataset_path=$DATASET_PATH"
echo "[copy-eval] limit=$LIMIT temperature=$TEMPERATURE top_p=$TOP_P top_k=$TOP_K"
echo "[copy-eval] modes=$MODES"
echo "[copy-eval] output_dir=$OUTPUT_DIR"

python analysis/copying_eval.py \
  --base_model "$BASE_MODEL" \
  --dataset_path "$DATASET_PATH" \
  --modes $MODES \
  --limit "$LIMIT" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --temperature "$TEMPERATURE" \
  --top_p "$TOP_P" \
  --top_k "$TOP_K" \
  --gpu_memory_utilization "$GPU_MEMORY_UTILIZATION" \
  --tensor_parallel_size "$TENSOR_PARALLEL_SIZE" \
  --output_dir "$OUTPUT_DIR"
