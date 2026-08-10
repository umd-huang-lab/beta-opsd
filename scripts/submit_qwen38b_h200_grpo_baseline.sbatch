#!/usr/bin/env bash
#SBATCH --job-name=q38b_grpo_base
#SBATCH --account=cml-furongh
#SBATCH --partition=cml-furongh
#SBATCH --qos=cml-high_long
#SBATCH --nodelist=cml36
#SBATCH --gres=gpu:h200-sxm:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=72:00:00
#SBATCH --output=slurm_logs/%x-%j.out
#
# Qwen3-8B GRPO baseline aligned with the successful 4B 1xH200 baseline.

set -euo pipefail

module load cuda/12.8.1
source "${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate "${CONDA_ENV:-opsd}"

select_gcc_toolchain() {
    local gcc_bin="${CC:-}"
    local gxx_bin="${CXX:-}"

    if [[ -z "$gcc_bin" || -z "$gxx_bin" ]]; then
        if command -v gcc >/dev/null 2>&1 && command -v g++ >/dev/null 2>&1; then
            gcc_bin="$(command -v gcc)"
            gxx_bin="$(command -v g++)"
        fi
    fi

    local major=0
    if [[ -n "$gcc_bin" && -x "$gcc_bin" ]]; then
        major="$("$gcc_bin" -dumpversion | cut -d. -f1)"
    fi

    if [[ "$major" -lt 9 ]] && module load gcc/11.2.0; then
        gcc_bin="$(command -v gcc)"
        gxx_bin="$(command -v g++)"
        major="$("$gcc_bin" -dumpversion | cut -d. -f1)"
    fi

    if [[ -z "$gcc_bin" || -z "$gxx_bin" || ! -x "$gcc_bin" || ! -x "$gxx_bin" || "$major" -lt 9 ]]; then
        echo "[grpo] need GCC >= 9; found gcc=${gcc_bin:-missing} version=$(${gcc_bin:-true} -dumpversion 2>/dev/null || echo missing)" >&2
        exit 1
    fi

    export CC="$gcc_bin"
    export CXX="$gxx_bin"
    export CUDAHOSTCXX="$gxx_bin"
}
select_gcc_toolchain

REPO=${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
cd "$REPO"

export TORCHINDUCTOR_CACHE_DIR=${CACHE_ROOT:-$REPO/cache}/inductor
export TRITON_CACHE_DIR=${CACHE_ROOT:-$REPO/cache}/triton
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
if [ -n "${VLLM_SRC:-}" ]; then export PYTHONPATH="$VLLM_SRC:${PYTHONPATH:-}"; fi

RUN=${RUN:-qwen38b_grpo_openthoughts_200step_16000tok_1xh200_cml36_bs1ga32_$(date +%Y%m%d_%H%M)}
MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH:-Qwen/Qwen3-8B}
MAX_STEPS=${MAX_STEPS:-200}
SAVE_STEPS=${SAVE_STEPS:-25}
MAX_COMPLETION_LENGTH=${MAX_COMPLETION_LENGTH:-16000}
PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE:-1}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-32}
NUM_PROCESSES=${NUM_PROCESSES:-1}
MAIN_PROCESS_PORT=${MAIN_PROCESS_PORT:-13220}
RUN_DIR="$REPO/runs/$RUN"
mkdir -p "$RUN_DIR"/{checkpoints,logs,eval}
export TORCH_EXTENSIONS_DIR="$RUN_DIR/torch_extensions"
export FLASHINFER_WORKSPACE_BASE="$RUN_DIR/flashinfer_cache"
mkdir -p "$TORCH_EXTENSIONS_DIR" "$FLASHINFER_WORKSPACE_BASE"
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    scontrol update JobId="$SLURM_JOB_ID" JobName="${SLURM_JOB_NAME_OVERRIDE:-${RUN:0:128}}" || true
fi
exec > "$RUN_DIR/logs/qwen38b_grpo-${SLURM_JOB_ID}.out" 2>&1

cat > "$RUN_DIR/config.json" <<CONFIG
{
  "run": "$RUN",
  "model": "$MODEL_NAME_OR_PATH",
  "variant": "grpo_baseline",
  "loss": "grpo",
  "script": "submit_qwen38b_h200_grpo_baseline.sbatch",
  "launched": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "slurm_job_id": "${SLURM_JOB_ID}",
  "max_steps": ${MAX_STEPS},
  "save_steps": ${SAVE_STEPS},
  "per_device_train_batch_size": ${PER_DEVICE_TRAIN_BATCH_SIZE},
  "gradient_accumulation_steps": ${GRADIENT_ACCUMULATION_STEPS},
  "num_processes": ${NUM_PROCESSES},
  "effective_batch_size": $((PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS * NUM_PROCESSES)),
  "num_generations": 8,
  "max_prompt_length": null,
  "max_completion_length": ${MAX_COMPLETION_LENGTH},
  "temperature": 1.1,
  "hardware": "1xh200-sxm",
  "wandb_project": "OPSD_MIX_SAMPLING"
}
CONFIG

echo "[grpo] run=$RUN model=$MODEL_NAME_OR_PATH max_steps=$MAX_STEPS save_steps=$SAVE_STEPS max_completion=$MAX_COMPLETION_LENGTH"
echo "[grpo] compiler: CC=${CC:-} CXX=${CXX:-} CUDAHOSTCXX=${CUDAHOSTCXX:-}"

accelerate launch \
    --config_file accelerate.yaml \
    --num_processes "$NUM_PROCESSES" \
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
    --main_process_port "$MAIN_PROCESS_PORT" \
    grpo_train.py \
    --model_name_or_path "$MODEL_NAME_OR_PATH" \
    --learning_rate 5e-6 \
    --max_grad_norm 0.1 \
    --per_device_train_batch_size "$PER_DEVICE_TRAIN_BATCH_SIZE" \
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
    --output_dir "$RUN_DIR/checkpoints" \
    --run_config "$RUN" \
    --max_steps "$MAX_STEPS" \
    --num_iterations 2 \
    --gradient_checkpointing \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
    --max_completion_length "$MAX_COMPLETION_LENGTH" \
    --num_generations 8 \
    --temperature 1.1 \
    --use_vllm \
    --use_peft \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization 0.5 \
    --vllm_tensor_parallel_size 1 \
    --attn_implementation flash_attention_2 \
    --dtype bfloat16 \
    --logging_steps 2 \
    --save_steps "$SAVE_STEPS" \
    --beta 0.0 \
    --loss_type grpo \
    --scale_rewards group \
    --wandb_project OPSD_MIX_SAMPLING
