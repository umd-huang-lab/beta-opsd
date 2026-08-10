#!/usr/bin/env bash
# Usage: RUN=<run_name> bash scripts/run_opsd_1b_compare.sh
#
# Pilot baseline OPSD run intended for comparison experiments.
# Defaults use the same local reference dataset and conservative memory settings.

set -euo pipefail

REPO=${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
cd "$REPO"

RUN=${RUN:-cmp_opsd_baseline_$(date +%Y%m%d_%H%M)}
RUN_DIR="$REPO/runs/$RUN"
mkdir -p "$RUN_DIR"/{checkpoints,logs,eval}

MODEL=${MODEL:-Qwen/Qwen3-1.7B}
DATASET_PATH=${DATASET_PATH:-openthoughts_refs_1000_qwen3_8b.jsonl}
MAX_STEPS=${MAX_STEPS:-100}
PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE:-1}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-2}
MAX_COMPLETION_LENGTH=${MAX_COMPLETION_LENGTH:-128}
MAX_LENGTH=${MAX_LENGTH:-2048}
TEMPERATURE=${TEMPERATURE:-1.0}
TOP_P=${TOP_P:-0.95}
TOP_K=${TOP_K:-20}
JSD_TOKEN_CLIP=${JSD_TOKEN_CLIP:-0.05}
VLLM_GPU_MEMORY_UTILIZATION=${VLLM_GPU_MEMORY_UTILIZATION:-0.5}

cat > "$RUN_DIR/config.json" <<CONFIG
{
  "run": "$RUN",
  "model": "$MODEL",
  "dataset_path": "$DATASET_PATH",
  "script": "run_opsd_1b_compare.sh",
  "launched": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "slurm_job_id": "${SLURM_JOB_ID:-local}",
  "variant": "baseline_opsd_compare"
}
CONFIG

echo "[compare-baseline] run=$RUN  dir=$RUN_DIR"

accelerate launch \
    --config_file accelerate.yaml \
    --num_processes 4 \
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
    --main_process_port 12951 \
    opsd_train.py \
    --model_name_or_path "$MODEL" \
    --dataset_path "$DATASET_PATH" \
    --learning_rate 5e-6 \
    --max_grad_norm 0.1 \
    --per_device_train_batch_size "$PER_DEVICE_TRAIN_BATCH_SIZE" \
    --gradient_checkpointing \
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
    --output_dir "$RUN_DIR/checkpoints" \
    --run_config "$RUN" \
    --max_steps "$MAX_STEPS" \
    --max_completion_length "$MAX_COMPLETION_LENGTH" \
    --save_steps "$MAX_STEPS" \
    --logging_steps 5 \
    --attn_implementation flash_attention_2 \
    --dtype bfloat16 \
    --max_length "$MAX_LENGTH" \
    --beta 0 \
    --use_vllm \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization "$VLLM_GPU_MEMORY_UTILIZATION" \
    --vllm_tensor_parallel_size 1 \
    --use_peft \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
    --temperature "$TEMPERATURE" \
    --top_p "$TOP_P" \
    --top_k "$TOP_K" \
    --lmbda 1 \
    --fixed_teacher \
    --jsd_token_clip "$JSD_TOKEN_CLIP" \
    --wandb_project OPSD
