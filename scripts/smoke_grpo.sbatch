#!/usr/bin/env bash
#SBATCH --job-name=grpo_smoke
#SBATCH --account=scavenger
#SBATCH --partition=scavenger
#SBATCH --qos=scavenger
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=00:45:00
#SBATCH --output=slurm_logs/%x-%j.out

set -euo pipefail

module load cuda/12.8.1
module load gcc/11.2.0
source "${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate "${CONDA_ENV:-opsd}"

REPO=${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
RUN=${RUN:-grpo_smoke_$(date +%Y%m%d_%H%M)}
RUN_DIR="$REPO/runs/$RUN"
mkdir -p "$RUN_DIR"/{checkpoints,logs,eval}

export WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false

exec > "$RUN_DIR/logs/grpo_smoke-${SLURM_JOB_ID}.out" 2>&1
cd "$REPO"

accelerate launch \
    --config_file accelerate_smoke.yaml \
    --num_processes 1 \
    grpo_train.py \
    --model_name_or_path Qwen/Qwen3-1.7B \
    --learning_rate 5e-6 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --output_dir "$RUN_DIR/checkpoints" \
    --run_config "$RUN" \
    --max_steps 2 \
    --num_iterations 1 \
    --gradient_checkpointing \
    --use_peft \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
    --max_prompt_length 512 \
    --max_completion_length 128 \
    --num_generations 2 \
    --temperature 1.0 \
    --use_vllm \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization 0.35 \
    --logging_steps 1 \
    --save_strategy no \
    --eval_strategy no \
    --beta 0.0 \
    --loss_type grpo \
    --scale_rewards group \
    --report_to none
