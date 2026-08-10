#!/usr/bin/env bash
# Usage: RUN=<run_name> bash scripts/run_grpo.sh

set -euo pipefail

REPO=${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
cd "$REPO"

RUN=${RUN:-qwen34b_grpo_$(date +%Y%m%d)}
RUN_DIR="$REPO/runs/$RUN"
mkdir -p "$RUN_DIR"/{checkpoints,logs,eval}

cat > "$RUN_DIR/config.json" <<CONFIG
{
  "run": "$RUN",
  "model": "Qwen/Qwen3-4B",
  "script": "run_grpo.sh",
  "launched": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "slurm_job_id": "${SLURM_JOB_ID:-local}"
}
CONFIG

echo "[train] run=$RUN  dir=$RUN_DIR"

accelerate launch \
    --config_file accelerate.yaml \
    --num_processes 8 \
    --gradient_accumulation_steps 4 \
    --main_process_port 19346 \
    grpo_train.py \
    --model_name_or_path Qwen/Qwen3-4B \
    --learning_rate 5e-6 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --output_dir "$RUN_DIR/checkpoints" \
    --run_config "$RUN" \
    --num_train_epochs 2 \
    --num_iterations 2 \
    --gradient_checkpointing \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
    --max_prompt_length 2048 \
    --max_completion_length 16000 \
    --num_generations 8 \
    --temperature 1.2 \
    --use_vllm \
    --use_peft \
    --vllm_mode colocate \
    --logging_steps 10 \
    --save_steps 20 \
    --beta 0.0 \
    --loss_type grpo \
    --scale_rewards group \
    --wandb_project OPSD
