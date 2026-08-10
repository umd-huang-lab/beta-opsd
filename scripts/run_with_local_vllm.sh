#!/usr/bin/env bash
# Run a training script with a local vLLM server on dedicated GPUs of the same node.
#
# Training gets GPUs 0..NUM_TRAIN_GPUS-1.
# vLLM gets GPUs NUM_TRAIN_GPUS..NUM_TRAIN_GPUS+NUM_VLLM_GPUS-1.
#
# Usage:
#   NUM_TRAIN_GPUS=4 NUM_VLLM_GPUS=1 MODEL=Qwen/Qwen3-1.7B RUN=myrun \
#       bash scripts/run_with_local_vllm.sh scripts/run_opsd_1b.sh
#
# All standard env vars (RUN, VLLM_SERVER_PORT, etc.) are forwarded to the training script.

set -euo pipefail

TRAINING_SCRIPT=${1:?Usage: $0 <training_script>}

REPO=${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}

NUM_TRAIN_GPUS=${NUM_TRAIN_GPUS:-4}
NUM_VLLM_GPUS=${NUM_VLLM_GPUS:-1}
PORT=${VLLM_SERVER_PORT:-8001}
MODEL=${MODEL:-Qwen/Qwen3-1.7B}
SERVER_READY_TIMEOUT=${SERVER_READY_TIMEOUT:-300}  # seconds

# Build comma-separated GPU index lists
TRAIN_GPU_IDS=$(seq -s, 0 $((NUM_TRAIN_GPUS - 1)))
VLLM_GPU_IDS=$(seq -s, $NUM_TRAIN_GPUS $((NUM_TRAIN_GPUS + NUM_VLLM_GPUS - 1)))

echo "[local-vllm] train gpus: $TRAIN_GPU_IDS  vllm gpus: $VLLM_GPU_IDS  port: $PORT"

# ── Start vLLM server in background on its dedicated GPUs ─────────────────────
CUDA_VISIBLE_DEVICES="$VLLM_GPU_IDS" \
    MODEL="$MODEL" PORT="$PORT" TENSOR_PARALLEL="$NUM_VLLM_GPUS" \
    bash "$REPO/scripts/serve_vllm.sh" &
VLLM_PID=$!

# Kill the server when this script exits (success, error, or signal)
trap 'echo "[local-vllm] shutting down server (pid $VLLM_PID)"; kill "$VLLM_PID" 2>/dev/null; wait "$VLLM_PID" 2>/dev/null || true' EXIT

# ── Wait for server to accept connections ─────────────────────────────────────
echo "[local-vllm] waiting for server on port $PORT (timeout: ${SERVER_READY_TIMEOUT}s)..."
elapsed=0
until bash -c "echo >/dev/tcp/localhost/$PORT" 2>/dev/null; do
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "[local-vllm] ERROR: vLLM server process died before becoming ready" >&2
        exit 1
    fi
    if (( elapsed >= SERVER_READY_TIMEOUT )); then
        echo "[local-vllm] ERROR: timed out waiting for vLLM server after ${elapsed}s" >&2
        exit 1
    fi
    sleep 5
    elapsed=$((elapsed + 5))
done
echo "[local-vllm] server ready after ${elapsed}s"

# ── Run training on its dedicated GPUs, pointing at the local server ──────────
CUDA_VISIBLE_DEVICES="$TRAIN_GPU_IDS" \
    VLLM_MODE=server \
    VLLM_SERVER_HOST=localhost \
    VLLM_SERVER_PORT="$PORT" \
    bash "$TRAINING_SCRIPT"
