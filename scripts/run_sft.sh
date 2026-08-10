#!/usr/bin/env bash
# Usage: RUN=<run_name> bash scripts/run_sft.sh

set -euo pipefail

REPO=${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
cd "$REPO"

RUN=${RUN:-qwen34b_sft_$(date +%Y%m%d)}
RUN_DIR="$REPO/runs/$RUN"
mkdir -p "$RUN_DIR"/{checkpoints,logs,eval}

cat > "$RUN_DIR/config.json" <<CONFIG
{
  "run": "$RUN",
  "model": "Qwen/Qwen3-4B",
  "script": "run_sft.sh",
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
    sft_train.py \
    --model_name_or_path Qwen/Qwen3-4B \
    --learning_rate 5e-6 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --output_dir "$RUN_DIR/checkpoints" \
    --run_config "$RUN" \
    --num_train_epochs 4 \
    --gradient_checkpointing \
    --use_peft \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
    --max_length 16000 \
    --logging_steps 5 \
    --save_steps 20 \
    --wandb_project OPSD
