#!/usr/bin/env bash
#SBATCH --job-name=q34b_vanilla_opsd
#SBATCH --account=scavenger
#SBATCH --partition=scavenger
#SBATCH --qos=scavenger
#SBATCH --gres=gpu:h200-sxm:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=slurm_logs/%x-%j.out
#
# Qwen3-4B vanilla OPSD/Tinker run on one H200. This intentionally does not
# enable mixed sampling, mixed teacher targets, or RTG.

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
        echo "[vanilla] need GCC >= 9 for PyTorch/DeepSpeed extensions; found gcc=${gcc_bin:-missing} version=$(${gcc_bin:-true} -dumpversion 2>/dev/null || echo missing)" >&2
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

RUN=${RUN:-qwen34b_vanilla_opsd_tinker_h200_bs1ga32_$(date +%Y%m%d_%H%M)}
MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH:-Qwen/Qwen3-4B}
MAX_STEPS=${MAX_STEPS:-200}
SAVE_STEPS=${SAVE_STEPS:-25}
MAX_COMPLETION_LENGTH=${MAX_COMPLETION_LENGTH:-1024}
MAX_LENGTH=${MAX_LENGTH:-20000}
MAIN_PROCESS_PORT=${MAIN_PROCESS_PORT:-13110}

RUN_DIR="$REPO/runs/$RUN"
mkdir -p "$RUN_DIR"/{checkpoints,logs,eval}
# Avoid loading a CPUAdam extension built on another host with a newer glibc.
export TORCH_EXTENSIONS_DIR="$RUN_DIR/torch_extensions"
mkdir -p "$TORCH_EXTENSIONS_DIR"
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    scontrol update JobId="$SLURM_JOB_ID" JobName="${SLURM_JOB_NAME_OVERRIDE:-${RUN:0:128}}" || true
fi

export FLASHINFER_WORKSPACE_BASE="$RUN_DIR/flashinfer_cache"
mkdir -p "$FLASHINFER_WORKSPACE_BASE"

exec > "$RUN_DIR/logs/qwen34b_vanilla_opsd-${SLURM_JOB_ID}.out" 2>&1

cat > "$RUN_DIR/config.json" <<CONFIG
{
  "run": "$RUN",
  "model": "$MODEL_NAME_OR_PATH",
  "variant": "qwen34b_h200_vanilla_opsd_tinker",
  "loss": "tinker",
  "tinker_use_reward_to_go": 0,
  "mixing_space": null,
  "use_mixed_sampling": 0,
  "use_mixed_teacher_target": 0,
  "hardware": "1xh200-sxm",
  "script": "beta_opsd_fixed_teacher_qwen34b_h200.sbatch",
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
  "wandb_project": "OPSD"
}
CONFIG

echo "[vanilla] run=$RUN model=$MODEL_NAME_OR_PATH max_steps=$MAX_STEPS save_steps=$SAVE_STEPS"
echo "[vanilla] compiler: CC=${CC:-} CXX=${CXX:-} CUDAHOSTCXX=${CUDAHOSTCXX:-}"
echo "[vanilla] PYTHONPATH=$PYTHONPATH"

accelerate launch \
    --config_file accelerate.yaml \
    --num_processes 1 \
    --gradient_accumulation_steps 32 \
    --main_process_port "$MAIN_PROCESS_PORT" \
    opsd_train.py \
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
    --wandb_project OPSD
