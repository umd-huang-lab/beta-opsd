#!/usr/bin/env bash
#SBATCH --job-name=q34b_mix_target
#SBATCH --account=scavenger
#SBATCH --partition=scavenger
#SBATCH --qos=scavenger
#SBATCH --gres=gpu:h200-sxm:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=slurm_logs/%x-%j.out
#
# Qwen3-4B beta-OPSD mix-target Tinker run on one H200.

set -euo pipefail

module load cuda/12.8.1
source "${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate "${CONDA_ENV:-opsd}"

# The gcc/11.2.0 module exports a path that is not mounted on cml36. Force the
# node-local system compiler so FlashInfer's nvcc JIT cannot inherit that path.
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
export CUDAHOSTCXX=/usr/bin/g++
for compiler in "$CC" "$CXX" "$CUDAHOSTCXX"; do
    if [[ ! -x "$compiler" ]]; then
        echo "[cmp] required host compiler is unavailable on $(hostname): $compiler" >&2
        exit 1
    fi
done

REPO=${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
cd "$REPO"

export TORCHINDUCTOR_CACHE_DIR=${CACHE_ROOT:-$REPO/cache}/inductor
export TRITON_CACHE_DIR=${CACHE_ROOT:-$REPO/cache}/triton
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
if [ -n "${VLLM_SRC:-}" ]; then export PYTHONPATH="$VLLM_SRC:${PYTHONPATH:-}"; fi

RUN=${RUN:-qwen34b_mix_target_tinker_$(date +%Y%m%d_%H%M)}
MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH:-Qwen/Qwen3-4B}
MAX_STEPS=${MAX_STEPS:-200}
SAVE_STEPS=${SAVE_STEPS:-25}
MAX_COMPLETION_LENGTH=${MAX_COMPLETION_LENGTH:-1024}
MAX_LENGTH=${MAX_LENGTH:-20000}
TEACHER_WEIGHT_START=${TEACHER_WEIGHT_START:-0.5}
TEACHER_WEIGHT_FINAL=${TEACHER_WEIGHT_FINAL:-$TEACHER_WEIGHT_START}
ALPHA_SCHEDULE=${ALPHA_SCHEDULE:-constant}
TINKER_USE_RTG=${TINKER_USE_RTG:-0}
TINKER_RTG_DISCOUNT=${TINKER_RTG_DISCOUNT:-1.0}
MAIN_PROCESS_PORT=${MAIN_PROCESS_PORT:-13080}

RUN_DIR="$REPO/runs/$RUN"
mkdir -p "$RUN_DIR"/{checkpoints,logs,eval}
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    scontrol update JobId="$SLURM_JOB_ID" JobName="${SLURM_JOB_NAME_OVERRIDE:-${RUN:0:128}}" || true
fi

export FLASHINFER_WORKSPACE_BASE="$RUN_DIR/flashinfer_cache"
mkdir -p "$FLASHINFER_WORKSPACE_BASE"

exec > "$RUN_DIR/logs/qwen34b_mix_target_tinker-${SLURM_JOB_ID}.out" 2>&1

EXTRA_ALPHA_ARGS=()
ALPHA_SCHEDULE_NAME="constant"
if [[ "$ALPHA_SCHEDULE" == "linear" ]]; then
    EXTRA_ALPHA_ARGS+=(
        --mixed_teacher_target_teacher_weight_linear_decay
        --mixed_teacher_target_teacher_weight_final "$TEACHER_WEIGHT_FINAL"
    )
    ALPHA_SCHEDULE_NAME="linear"
fi

EXTRA_RTG_ARGS=()
if [[ "$TINKER_USE_RTG" == "1" ]]; then
    EXTRA_RTG_ARGS+=(
        --tinker_use_reward_to_go
        --tinker_reward_to_go_discount "$TINKER_RTG_DISCOUNT"
    )
fi

cat > "$RUN_DIR/config.json" <<CONFIG
{
  "run": "$RUN",
  "model": "$MODEL_NAME_OR_PATH",
  "variant": "qwen34b_h200_mixed_target_tinker",
  "loss": "tinker",
  "tinker_use_reward_to_go": ${TINKER_USE_RTG},
  "tinker_reward_to_go_discount": ${TINKER_RTG_DISCOUNT},
  "target": "mixed_teacher_target",
  "target_teacher_weight_start": ${TEACHER_WEIGHT_START},
  "target_teacher_weight_end": ${TEACHER_WEIGHT_FINAL},
  "alpha_schedule": "$ALPHA_SCHEDULE_NAME",
  "mixed_teacher_target_reference_model": "current_student",
  "hardware": "1xh200-sxm",
  "script": "beta_opsd_mix_target_qwen34b_h200.sbatch",
  "launched": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "slurm_job_id": "${SLURM_JOB_ID}",
  "max_steps": ${MAX_STEPS},
  "save_steps": ${SAVE_STEPS},
  "per_device_train_batch_size": 1,
  "gradient_accumulation_steps": 32,
  "num_processes": 1,
  "effective_batch_size": 32,
  "max_completion_length": ${MAX_COMPLETION_LENGTH},
  "max_length": ${MAX_LENGTH},
  "wandb_project": "OPSD_MIX_SAMPLING"
}
CONFIG

echo "[cmp] run=$RUN model=$MODEL_NAME_OR_PATH teacher_weight=${TEACHER_WEIGHT_START}->${TEACHER_WEIGHT_FINAL} schedule=$ALPHA_SCHEDULE_NAME rtg=$TINKER_USE_RTG gamma=$TINKER_RTG_DISCOUNT max_steps=$MAX_STEPS"
echo "[cmp] compiler: CC=${CC:-} CXX=${CXX:-} CUDAHOSTCXX=${CUDAHOSTCXX:-}"
echo "[cmp] PYTHONPATH=$PYTHONPATH"

accelerate launch \
    --config_file accelerate.yaml \
    --num_processes 1 \
    --gradient_accumulation_steps 32 \
    --main_process_port "$MAIN_PROCESS_PORT" \
    beta_opsd_train.py \
    --model_name_or_path "$MODEL_NAME_OR_PATH" \
    --learning_rate 5e-6 \
    --max_grad_norm 0.1 \
    --per_device_train_batch_size 1 \
    --gradient_checkpointing \
    --gradient_accumulation_steps 32 \
    --output_dir "$RUN_DIR/checkpoints" \
    --run_config "$RUN" \
    --max_steps "$MAX_STEPS" \
    --max_completion_length "$MAX_COMPLETION_LENGTH" \
    --save_steps "$SAVE_STEPS" \
    --logging_steps 2 \
    --attn_implementation flash_attention_2 \
    --dtype bfloat16 \
    --max_length "$MAX_LENGTH" \
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
    --mixed_teacher_target_teacher_weight "$TEACHER_WEIGHT_START" \
    --mixed_teacher_target_reference_model current_student \
    "${EXTRA_ALPHA_ARGS[@]}" \
    "${EXTRA_RTG_ARGS[@]}" \
    --wandb_project OPSD_MIX_SAMPLING
