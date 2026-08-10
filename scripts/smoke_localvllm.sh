#!/bin/bash
#SBATCH --job-name=opsd_smoke_localvllm
#SBATCH --account=scavenger
#SBATCH --partition=scavenger
#SBATCH --qos=scavenger
#SBATCH --gres=gpu:rtxa4000:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:45:00
#SBATCH --output=slurm_logs/%x-%j.out
#
# Smoke test: 3 training steps with local vLLM server mode.
# GPU 0 → training (1 process), GPU 1 → vLLM server.

set -euo pipefail

module load cuda/12.8.1
module load gcc/11.2.0
source "${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate "${CONDA_ENV:-opsd}"

REPO=${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
PORT=8765
MODEL=Qwen/Qwen3-1.7B
SERVER_READY_TIMEOUT=300

cd "$REPO"

echo "[smoke] node=$(hostname)  gpus=$CUDA_VISIBLE_DEVICES"
echo "[smoke] starting vLLM server on GPU 1, port $PORT"

# Start vLLM server on GPU 1
CUDA_VISIBLE_DEVICES=1 \
    MODEL="$MODEL" PORT="$PORT" GPU_MEMORY_UTILIZATION=0.9 MAX_MODEL_LEN=4096 \
    bash scripts/serve_vllm.sh &
VLLM_PID=$!
trap 'echo "[smoke] killing vLLM server (pid $VLLM_PID)"; kill "$VLLM_PID" 2>/dev/null; wait "$VLLM_PID" 2>/dev/null || true' EXIT

# Wait for server
echo "[smoke] waiting for server on port $PORT..."
elapsed=0
until bash -c "echo >/dev/tcp/localhost/$PORT" 2>/dev/null; do
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "[smoke] ERROR: vLLM server died before becoming ready" >&2
        exit 1
    fi
    if (( elapsed >= SERVER_READY_TIMEOUT )); then
        echo "[smoke] ERROR: timed out after ${elapsed}s" >&2
        exit 1
    fi
    sleep 5; elapsed=$((elapsed + 5))
done
echo "[smoke] server ready after ${elapsed}s"

# Run 3-step training on GPU 0
echo "[smoke] starting training on GPU 0"
CUDA_VISIBLE_DEVICES=0 accelerate launch \
    --config_file accelerate_smoke.yaml \
    --num_processes 1 \
    opsd_train.py \
    --model_name_or_path "$MODEL" \
    --learning_rate 5e-6 \
    --max_grad_norm 0.1 \
    --per_device_train_batch_size 1 \
    --gradient_checkpointing \
    --gradient_accumulation_steps 1 \
    --output_dir $REPO/runs/opsd_smoke_localvllm \
    --run_config smoke_localvllm \
    --max_steps 3 \
    --max_completion_length 128 \
    --logging_steps 1 \
    --attn_implementation flash_attention_2 \
    --dtype bfloat16 \
    --max_length 2048 \
    --beta 0 \
    --use_vllm \
    --vllm_mode server \
    --vllm_server_host localhost \
    --vllm_server_port "$PORT" \
    --vllm_server_timeout 120 \
    --use_peft \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
    --temperature 1.1 \
    --top_p 0.95 \
    --top_k 20 \
    --lmbda 1 \
    --fixed_teacher \
    --jsd_token_clip 0.05 \
    --report_to none

echo "[smoke] PASSED"
