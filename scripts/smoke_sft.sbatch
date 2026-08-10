#!/usr/bin/env bash
#SBATCH --job-name=sft_smoke
#SBATCH --account=scavenger
#SBATCH --partition=scavenger
#SBATCH --qos=scavenger
#SBATCH --gres=gpu:rtxa4000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=slurm_logs/%x-%j.out

set -euo pipefail

module load cuda/12.8.1
module load gcc/11.2.0
source "${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate "${CONDA_ENV:-opsd}"

REPO=${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
RUN=${RUN:-sft_smoke_$(date +%Y%m%d_%H%M)}
RUN_DIR="$REPO/runs/$RUN"
mkdir -p "$RUN_DIR"/{checkpoints,logs,eval}

export WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false

exec > "$RUN_DIR/logs/sft_smoke-${SLURM_JOB_ID}.out" 2>&1
cd "$REPO"

accelerate launch \
    --config_file accelerate_smoke.yaml \
    --num_processes 1 \
    sft_train.py \
    --model_name_or_path Qwen/Qwen3-1.7B \
    --learning_rate 5e-6 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --output_dir "$RUN_DIR/checkpoints" \
    --run_config "$RUN" \
    --max_steps 3 \
    --gradient_checkpointing \
    --use_peft \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
    --max_length 2048 \
    --logging_steps 1 \
    --save_strategy no \
    --eval_strategy no \
    --report_to none
