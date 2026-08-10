#!/usr/bin/env bash
#SBATCH --job-name=smoke_mixture
#SBATCH --account=scavenger
#SBATCH --partition=scavenger
#SBATCH --qos=scavenger
#SBATCH --gres=gpu:rtxa5000:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=slurm_logs/%x-%j.out

set -euo pipefail

module load cuda/12.8.1
module load gcc/11.2.0
source "${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate "${CONDA_ENV:-opsd}"

REPO=${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
RUN=${RUN:-smoke_mixture_$(date +%Y%m%d_%H%M)}
RUN_DIR="$REPO/runs/$RUN"
mkdir -p "$RUN_DIR"/{checkpoints,logs,eval}
mkdir -p "${CACHE_ROOT:-$REPO/cache}"/{xdg,wandb,torch_extensions,triton}

export XDG_CACHE_HOME="${CACHE_ROOT:-$REPO/cache}/xdg"
export WANDB_DIR="${CACHE_ROOT:-$REPO/cache}/wandb"
export WANDB_CACHE_DIR="${CACHE_ROOT:-$REPO/cache}/wandb"
export TORCH_EXTENSIONS_DIR="${CACHE_ROOT:-$REPO/cache}/torch_extensions"
export TRITON_CACHE_DIR="${CACHE_ROOT:-$REPO/cache}/triton"

# Use modified vLLM source (pure Python changes, no recompile needed)
if [ -n "${VLLM_SRC:-}" ]; then export PYTHONPATH="$VLLM_SRC:${PYTHONPATH:-}"; fi

echo "[smoke] run=$RUN  PYTHONPATH=$PYTHONPATH"
echo "[smoke] log: $RUN_DIR/logs/smoke-${SLURM_JOB_ID}.out"
exec > "$RUN_DIR/logs/smoke-${SLURM_JOB_ID}.out" 2>&1

cd "$REPO"

EXTRA_ARGS=()

accelerate launch \
    --config_file accelerate.yaml \
    --num_processes 4 \
    --gradient_accumulation_steps 2 \
    --main_process_port 12949 \
    beta_opsd_train.py \
    --model_name_or_path Qwen/Qwen3-1.7B \
    --learning_rate 5e-6 \
    --max_grad_norm 0.1 \
    --per_device_train_batch_size 4 \
    --gradient_checkpointing \
    --gradient_accumulation_steps 2 \
    --output_dir "$RUN_DIR/checkpoints" \
    --run_config "$RUN" \
    --max_steps 5 \
    --max_completion_length 128 \
    --save_steps 999 \
    --logging_steps 1 \
    --attn_implementation flash_attention_2 \
    --dtype bfloat16 \
    --max_length 4096 \
    --beta 0 \
    --use_vllm \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization 0.5 \
    --vllm_tensor_parallel_size 1 \
    --use_peft \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
    --temperature 1.0 \
    --top_p 0.95 \
    --top_k 20 \
    --lmbda 1 \
    --fixed_teacher \
    --jsd_token_clip 0.05 \
    --use_tinker_loss \
    --use_mixed_sampling \
    --mixed_sampling_teacher_alpha 0.3 \
    "${EXTRA_ARGS[@]}" \
    --wandb_project OPSD
