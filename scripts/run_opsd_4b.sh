#!/usr/bin/env bash
# Usage: RUN=<run_name> bash scripts/run_opsd_4b.sh
#
# vLLM server mode:
#   VLLM_MODE=server VLLM_SERVER_HOST=<node> VLLM_SERVER_PORT=8001 RUN=... bash scripts/run_opsd_4b.sh

set -euo pipefail

REPO=${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
cd "$REPO"

RUN=${RUN:-qwen34b_opsd_$(date +%Y%m%d)}
RUN_DIR="$REPO/runs/$RUN"
mkdir -p "$RUN_DIR"/{checkpoints,logs,eval}

VLLM_MODE=${VLLM_MODE:-colocate}
VLLM_SERVER_HOST=${VLLM_SERVER_HOST:-localhost}
VLLM_SERVER_PORT=${VLLM_SERVER_PORT:-8001}
VLLM_SERVER_TIMEOUT=${VLLM_SERVER_TIMEOUT:-300}

cat > "$RUN_DIR/config.json" <<CONFIG
{
  "run": "$RUN",
  "model": "Qwen/Qwen3-4B",
  "script": "run_opsd_4b.sh",
  "launched": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "slurm_job_id": "${SLURM_JOB_ID:-local}",
  "vllm_mode": "$VLLM_MODE"
}
CONFIG

echo "[train] run=$RUN  dir=$RUN_DIR  vllm_mode=$VLLM_MODE"

if [[ "$VLLM_MODE" == "server" ]]; then
    echo "[train] vllm server: $VLLM_SERVER_HOST:$VLLM_SERVER_PORT"
    VLLM_ARGS=(
        --vllm_mode server
        --vllm_server_host "$VLLM_SERVER_HOST"
        --vllm_server_port "$VLLM_SERVER_PORT"
        --vllm_server_timeout "$VLLM_SERVER_TIMEOUT"
    )
else
    VLLM_ARGS=(
        --vllm_mode colocate
        --vllm_gpu_memory_utilization 0.6
        --vllm_tensor_parallel_size 1
    )
fi

accelerate launch \
    --config_file accelerate.yaml \
    --num_processes 8 \
    --gradient_accumulation_steps 1 \
    --main_process_port 12949 \
    opsd_train.py \
    --model_name_or_path Qwen/Qwen3-4B \
    --learning_rate 5e-6 \
    --max_grad_norm 0.1 \
    --per_device_train_batch_size 4 \
    --gradient_checkpointing \
    --gradient_accumulation_steps 1 \
    --output_dir "$RUN_DIR/checkpoints" \
    --run_config "$RUN" \
    --num_train_epochs 30 \
    --max_completion_length 1024 \
    --save_steps 25 \
    --logging_steps 2 \
    --attn_implementation flash_attention_2 \
    --dtype bfloat16 \
    --max_length 20000 \
    --beta 0 \
    --use_vllm \
    "${VLLM_ARGS[@]}" \
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
    --wandb_project OPSD
