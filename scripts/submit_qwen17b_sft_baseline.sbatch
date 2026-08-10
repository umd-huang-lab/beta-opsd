#!/usr/bin/env bash
#SBATCH --job-name=q17b_sft_base
#SBATCH --account=scavenger
#SBATCH --partition=scavenger
#SBATCH --qos=scavenger
#SBATCH --gres=gpu:rtxa6000:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=slurm_logs/%x-%j.out
#
# Qwen3-1.7B SFT baseline aligned with OPSD/mix training settings.

set -euo pipefail

module load cuda/12.8.1
if ! module load gcc/11.2.0; then
    echo "[sft] gcc/11.2.0 module unavailable on $(hostname); continuing with environment compiler"
fi
source "${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate "${CONDA_ENV:-opsd}"

REPO=${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
cd "$REPO"

export TORCHINDUCTOR_CACHE_DIR=${CACHE_ROOT:-$REPO/cache}/inductor
export TRITON_CACHE_DIR=${CACHE_ROOT:-$REPO/cache}/triton
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RUN=${RUN:-qwen17b_sft_openthoughts_200step_$(date +%Y%m%d_%H%M)}
MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH:-Qwen/Qwen3-1.7B}
MAX_STEPS=${MAX_STEPS:-200}
SAVE_STEPS=${SAVE_STEPS:-25}
PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE:-1}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-8}
NUM_PROCESSES=${NUM_PROCESSES:-4}
MAX_LENGTH=${MAX_LENGTH:-20000}
MAIN_PROCESS_PORT=${MAIN_PROCESS_PORT:-13030}
RUN_DIR="$REPO/runs/$RUN"
mkdir -p "$RUN_DIR"/{checkpoints,logs,eval}
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    scontrol update JobId="$SLURM_JOB_ID" JobName="${SLURM_JOB_NAME_OVERRIDE:-${RUN:0:128}}" || true
fi
exec > "$RUN_DIR/logs/qwen17b_sft-${SLURM_JOB_ID}.out" 2>&1

cat > "$RUN_DIR/config.json" <<CONFIG
{
  "run": "$RUN",
  "model": "$MODEL_NAME_OR_PATH",
  "variant": "sft_baseline",
  "loss": "sft",
  "script": "submit_qwen17b_sft_baseline.sbatch",
  "launched": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "slurm_job_id": "${SLURM_JOB_ID}",
  "max_steps": ${MAX_STEPS},
  "save_steps": ${SAVE_STEPS},
  "per_device_train_batch_size": ${PER_DEVICE_TRAIN_BATCH_SIZE},
  "gradient_accumulation_steps": ${GRADIENT_ACCUMULATION_STEPS},
  "num_processes": ${NUM_PROCESSES},
  "effective_batch_size": $((PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS * NUM_PROCESSES)),
  "max_length": ${MAX_LENGTH},
  "wandb_project": "OPSD_MIX_SAMPLING"
}
CONFIG

echo "[sft] run=$RUN max_steps=$MAX_STEPS save_steps=$SAVE_STEPS"

accelerate launch \
    --config_file accelerate.yaml \
    --num_processes "$NUM_PROCESSES" \
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
    --main_process_port "$MAIN_PROCESS_PORT" \
    sft_train.py \
    --model_name_or_path "$MODEL_NAME_OR_PATH" \
    --learning_rate 5e-6 \
    --max_grad_norm 0.1 \
    --per_device_train_batch_size "$PER_DEVICE_TRAIN_BATCH_SIZE" \
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
    --output_dir "$RUN_DIR/checkpoints" \
    --run_config "$RUN" \
    --max_steps "$MAX_STEPS" \
    --gradient_checkpointing \
    --use_peft \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
    --max_length "$MAX_LENGTH" \
    --attn_implementation flash_attention_2 \
    --dtype bfloat16 \
    --logging_steps 2 \
    --save_steps "$SAVE_STEPS" \
    --wandb_project OPSD_MIX_SAMPLING
