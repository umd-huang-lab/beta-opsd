#!/usr/bin/env bash
# Run a recipe while a local vLLM server uses dedicated GPUs on the same node.

set -euo pipefail

TRAINING_SCRIPT=${1:?Usage: $0 <training_script>}
REPO=${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}

NUM_TRAIN_GPUS=${NUM_TRAIN_GPUS:-4}
NUM_VLLM_GPUS=${NUM_VLLM_GPUS:-1}
PORT=${VLLM_SERVER_PORT:-8001}
MODEL=${MODEL:-Qwen/Qwen3-1.7B}
SERVER_READY_TIMEOUT=${SERVER_READY_TIMEOUT:-300}

TRAIN_GPU_IDS=$(seq -s, 0 $((NUM_TRAIN_GPUS - 1)))
VLLM_GPU_IDS=$(seq -s, $NUM_TRAIN_GPUS $((NUM_TRAIN_GPUS + NUM_VLLM_GPUS - 1)))

echo "[local-vllm] train_gpus=$TRAIN_GPU_IDS vllm_gpus=$VLLM_GPU_IDS port=$PORT"

CUDA_VISIBLE_DEVICES="$VLLM_GPU_IDS" \
    MODEL="$MODEL" PORT="$PORT" TENSOR_PARALLEL="$NUM_VLLM_GPUS" \
    bash "$REPO/scripts/vllm/serve.sh" &
VLLM_PID=$!

trap 'echo "[local-vllm] shutting down server pid=$VLLM_PID"; kill "$VLLM_PID" 2>/dev/null; wait "$VLLM_PID" 2>/dev/null || true' EXIT

elapsed=0
until bash -c "echo >/dev/tcp/localhost/$PORT" 2>/dev/null; do
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "[local-vllm] ERROR: vLLM server exited before it became ready" >&2
        exit 1
    fi
    if (( elapsed >= SERVER_READY_TIMEOUT )); then
        echo "[local-vllm] ERROR: timed out waiting for vLLM server after ${elapsed}s" >&2
        exit 1
    fi
    sleep 5
    elapsed=$((elapsed + 5))
done

CUDA_VISIBLE_DEVICES="$TRAIN_GPU_IDS" \
    VLLM_MODE=server \
    VLLM_SERVER_HOST=localhost \
    VLLM_SERVER_PORT="$PORT" \
    bash "$TRAINING_SCRIPT"
