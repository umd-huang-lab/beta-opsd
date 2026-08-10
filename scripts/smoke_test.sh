#!/bin/bash
#SBATCH --job-name=opsd_smoke
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
cd "$REPO"

accelerate launch \
    --config_file accelerate_smoke.yaml \
    --num_processes 1 \
    opsd_train.py \
    --model_name_or_path Qwen/Qwen3-1.7B \
    --learning_rate 5e-6 \
    --max_grad_norm 0.1 \
    --per_device_train_batch_size 1 \
    --gradient_checkpointing \
    --gradient_accumulation_steps 1 \
    --output_dir $REPO/runs/opsd_smoke \
    --run_config smoke_test \
    --max_steps 3 \
    --max_completion_length 128 \
    --logging_steps 1 \
    --attn_implementation flash_attention_2 \
    --dtype bfloat16 \
    --max_length 2048 \
    --beta 0 \
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
