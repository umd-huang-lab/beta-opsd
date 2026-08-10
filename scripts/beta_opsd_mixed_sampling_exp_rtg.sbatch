#!/usr/bin/env bash
#SBATCH --job-name=cmp_mix_exp_rtg
#SBATCH --account=scavenger
#SBATCH --partition=scavenger
#SBATCH --qos=scavenger
#SBATCH --gres=gpu:rtxa6000:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=slurm_logs/%x-%j.out
#
# Comparison run: mixed-sampling OPSD fork with exponential alpha decay and tinker reward-to-go.

set -euo pipefail

module load cuda/12.8.1
module load gcc/11.2.0
source "${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate "${CONDA_ENV:-opsd}"

REPO=${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
cd "$REPO"

export TORCHINDUCTOR_CACHE_DIR=${CACHE_ROOT:-$REPO/cache}/inductor
export TRITON_CACHE_DIR=${CACHE_ROOT:-$REPO/cache}/triton
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if [ -n "${VLLM_SRC:-}" ]; then export PYTHONPATH="$VLLM_SRC:${PYTHONPATH:-}"; fi

RUN=${RUN:-cmp_mix_sampling_exp_rtg_$(date +%Y%m%d_%H%M)}
MAX_STEPS=${MAX_STEPS:-200}
TINKER_RTG_DISCOUNT=${TINKER_RTG_DISCOUNT:-1.0}
RUN_DIR="$REPO/runs/$RUN"
mkdir -p "$RUN_DIR"/{checkpoints,logs,eval}
exec > "$RUN_DIR/logs/cmp_mix_sampling_exp_rtg-${SLURM_JOB_ID}.out" 2>&1

EXTRA_ARGS=()

cat > "$RUN_DIR/config.json" <<CONFIG
{
  "run": "$RUN",
  "model": "Qwen/Qwen3-1.7B",
  "variant": "mix_sampling_exp_rtg",
  "loss": "tinker_rtg",
  "tinker_reward_to_go_discount": ${TINKER_RTG_DISCOUNT},
  "mixing_space": "pre_softmax",
  "target": "fixed_teacher",
  "alpha_start": 0.8,
  "alpha_end": 0.2,
  "alpha_schedule": "exponential",
  "hardware": "4xrtxa6000",
  "script": "beta_opsd_mixed_sampling_exp_rtg.sbatch",
  "launched": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "slurm_job_id": "${SLURM_JOB_ID}",
  "max_steps": ${MAX_STEPS},
  "effective_batch_size": 32,
  "wandb_project": "OPSD_MIX_SAMPLING"
}
CONFIG

echo "[cmp] run=$RUN  variant=mix_sampling_exp_rtg  alpha=0.8->0.2(exp)  gamma=$TINKER_RTG_DISCOUNT  max_steps=$MAX_STEPS"
echo "[cmp] PYTHONPATH=$PYTHONPATH"

accelerate launch \
    --config_file accelerate.yaml \
    --num_processes 4 \
    --gradient_accumulation_steps 8 \
    --main_process_port 12966 \
    beta_opsd_train.py \
    --model_name_or_path Qwen/Qwen3-1.7B \
    --learning_rate 5e-6 \
    --max_grad_norm 0.1 \
    --per_device_train_batch_size 1 \
    --gradient_checkpointing \
    --gradient_accumulation_steps 8 \
    --output_dir "$RUN_DIR/checkpoints" \
    --run_config "$RUN" \
    --max_steps "$MAX_STEPS" \
    --max_completion_length 1024 \
    --save_steps 50 \
    --logging_steps 2 \
    --attn_implementation flash_attention_2 \
    --dtype bfloat16 \
    --max_length 20000 \
    --beta 0 \
    --use_vllm \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization 0.5 \
    --vllm_tensor_parallel_size 1 \
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
    --use_tinker_loss \
    --tinker_use_reward_to_go \
    --tinker_reward_to_go_discount "$TINKER_RTG_DISCOUNT" \
    --use_mixed_sampling \
    --mixed_sampling_teacher_alpha 0.8 \
    --mixed_sampling_teacher_alpha_exponential_decay \
    --mixed_sampling_teacher_alpha_final 0.2 \
    "${EXTRA_ARGS[@]}" \
    --wandb_project OPSD_MIX_SAMPLING
