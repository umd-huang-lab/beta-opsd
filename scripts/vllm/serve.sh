#!/usr/bin/env bash
# Launch TRL's vLLM weight-sync server on a dedicated GPU.

set -euo pipefail

if [[ -f "${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}" ]]; then
    source "${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
    conda activate "${CONDA_ENV:-opsd}"
fi

MODEL=${MODEL:-Qwen/Qwen3-1.7B}
PORT=${PORT:-8001}
TENSOR_PARALLEL=${TENSOR_PARALLEL:-1}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.9}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-20000}
DTYPE=${DTYPE:-bfloat16}

echo "[vllm] model=$MODEL port=$PORT tp=$TENSOR_PARALLEL node=$(hostname)"

trl vllm-serve \
    --model "$MODEL" \
    --port "$PORT" \
    --tensor_parallel_size "$TENSOR_PARALLEL" \
    --gpu_memory_utilization "$GPU_MEMORY_UTILIZATION" \
    --max_model_len "$MAX_MODEL_LEN" \
    --dtype "$DTYPE" \
    --enable_prefix_caching true
