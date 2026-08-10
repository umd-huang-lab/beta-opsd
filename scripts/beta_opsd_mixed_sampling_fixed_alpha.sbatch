#!/usr/bin/env bash
#SBATCH --job-name=cmp_mix_fixed_tinker
#SBATCH --account=scavenger
#SBATCH --partition=scavenger
#SBATCH --qos=scavenger
#SBATCH --gres=gpu:rtxa6000:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=slurm_logs/%x-%j.out
#
# Fixed-alpha mixed-sampling OPSD fork with tinker loss on Qwen3-1.7B.

set -euo pipefail

module load cuda/12.8.1
if ! module load gcc/11.2.0; then
    echo "[cmp] gcc/11.2.0 module unavailable on $(hostname); continuing with environment compiler"
fi
source "${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate "${CONDA_ENV:-opsd}"

REPO=${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
cd "$REPO"

export TORCHINDUCTOR_CACHE_DIR=${CACHE_ROOT:-$REPO/cache}/inductor
export TRITON_CACHE_DIR=${CACHE_ROOT:-$REPO/cache}/triton
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if [ -n "${VLLM_SRC:-}" ]; then export PYTHONPATH="$VLLM_SRC:${PYTHONPATH:-}"; fi

RUN=${RUN:-cmp_mix_fixed_alpha05_tinker_$(date +%Y%m%d_%H%M)}
MAX_STEPS=${MAX_STEPS:-200}
MIXED_ALPHA=${MIXED_ALPHA:-0.5}
SAVE_STEPS=${SAVE_STEPS:-50}
MAIN_PROCESS_PORT=${MAIN_PROCESS_PORT:-12967}
USE_PER_TOKEN_IMPORTANCE_SAMPLING=${USE_PER_TOKEN_IMPORTANCE_SAMPLING:-0}
IMPORTANCE_SAMPLING_CLIP_MIN=${IMPORTANCE_SAMPLING_CLIP_MIN:-0}
IMPORTANCE_SAMPLING_CLIP_MAX=${IMPORTANCE_SAMPLING_CLIP_MAX:-0}
RUN_DIR="$REPO/runs/$RUN"
mkdir -p "$RUN_DIR"/{checkpoints,logs,eval}
exec > "$RUN_DIR/logs/cmp_mix_fixed_alpha_tinker-${SLURM_JOB_ID}.out" 2>&1

EXTRA_ARGS=()
if [[ "$USE_PER_TOKEN_IMPORTANCE_SAMPLING" == "1" ]]; then
    EXTRA_ARGS+=(--use_per_token_importance_sampling)
fi
if [[ "$IMPORTANCE_SAMPLING_CLIP_MIN" != "0" ]]; then
    EXTRA_ARGS+=(--importance_sampling_clip_min "$IMPORTANCE_SAMPLING_CLIP_MIN")
fi
if [[ "$IMPORTANCE_SAMPLING_CLIP_MAX" != "0" ]]; then
    EXTRA_ARGS+=(--importance_sampling_clip_max "$IMPORTANCE_SAMPLING_CLIP_MAX")
fi

cat > "$RUN_DIR/config.json" <<CONFIG
{
  "run": "$RUN",
  "model": "Qwen/Qwen3-1.7B",
  "variant": "mix_fixed_alpha_tinker",
  "loss": "tinker",
  "mixing_space": "pre_softmax",
  "target": "fixed_teacher",
  "alpha": ${MIXED_ALPHA},
  "alpha_schedule": "constant",
  "use_per_token_importance_sampling": ${USE_PER_TOKEN_IMPORTANCE_SAMPLING},
  "importance_sampling_clip_min": ${IMPORTANCE_SAMPLING_CLIP_MIN},
  "importance_sampling_clip_max": ${IMPORTANCE_SAMPLING_CLIP_MAX},
  "script": "beta_opsd_mixed_sampling_fixed_alpha.sbatch",
  "launched": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "slurm_job_id": "${SLURM_JOB_ID}",
  "max_steps": ${MAX_STEPS},
  "save_steps": ${SAVE_STEPS},
  "effective_batch_size": 32,
  "wandb_project": "OPSD_MIX_SAMPLING"
}
CONFIG

echo "[cmp] run=$RUN  variant=mix_fixed_alpha_tinker  alpha=$MIXED_ALPHA  max_steps=$MAX_STEPS  save_steps=$SAVE_STEPS  target=fixed_teacher"
echo "[cmp] PYTHONPATH=$PYTHONPATH"

accelerate launch \
    --config_file accelerate.yaml \
    --num_processes 4 \
    --gradient_accumulation_steps 8 \
    --main_process_port "$MAIN_PROCESS_PORT" \
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
    --save_steps "$SAVE_STEPS" \
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
    --use_mixed_sampling \
    --mixed_sampling_teacher_alpha "$MIXED_ALPHA" \
    "${EXTRA_ARGS[@]}" \
    --wandb_project OPSD_MIX_SAMPLING
