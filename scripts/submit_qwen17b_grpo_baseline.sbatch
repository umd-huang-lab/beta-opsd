#!/usr/bin/env bash
#SBATCH --job-name=q17b_grpo_base
#SBATCH --account=scavenger
#SBATCH --partition=scavenger
#SBATCH --qos=scavenger
#SBATCH --gres=gpu:rtxa6000:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=slurm_logs/%x-%j.out
#
# Qwen3-1.7B GRPO baseline aligned with OPSD/mix training settings.

set -euo pipefail

module load cuda/12.8.1
if ! module load gcc/11.2.0; then
    echo "[grpo] gcc/11.2.0 module unavailable on $(hostname); continuing with environment compiler"
fi
source "${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate "${CONDA_ENV:-opsd}"

# FlashInfer JIT passes CUDAHOSTCXX/CC through to nvcc. Some nodes expose a
# stale gcc/11.2.0 module path, so force these to an executable compiler.
if [[ -n "${CUDAHOSTCXX:-}" && ! -x "$CUDAHOSTCXX" ]]; then
    echo "[grpo] CUDAHOSTCXX=$CUDAHOSTCXX is not executable; resetting to environment gcc"
    unset CUDAHOSTCXX
fi
if [[ -n "${CC:-}" && ! -x "$CC" ]]; then
    echo "[grpo] CC=$CC is not executable; resetting to environment gcc"
    unset CC
fi
if [[ -n "${CXX:-}" && ! -x "$CXX" ]]; then
    echo "[grpo] CXX=$CXX is not executable; resetting to environment g++"
    unset CXX
fi
if [[ -z "${CC:-}" ]]; then
    for candidate in "$(command -v gcc || true)" /usr/bin/gcc /bin/gcc; do
        if [[ -n "$candidate" && -x "$candidate" ]]; then
            export CC="$candidate"
            break
        fi
    done
fi
if [[ -z "${CXX:-}" ]]; then
    for candidate in "$(command -v g++ || true)" /usr/bin/g++ /bin/g++; do
        if [[ -n "$candidate" && -x "$candidate" ]]; then
            export CXX="$candidate"
            break
        fi
    done
fi
if [[ -n "${CC:-}" ]]; then
    export CUDAHOSTCXX="${CUDAHOSTCXX:-$CC}"
else
    unset CUDAHOSTCXX
fi
echo "[grpo] compiler: CC=${CC:-} CXX=${CXX:-} CUDAHOSTCXX=${CUDAHOSTCXX:-}"

REPO=${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
cd "$REPO"

export TORCHINDUCTOR_CACHE_DIR=${CACHE_ROOT:-$REPO/cache}/inductor
export TRITON_CACHE_DIR=${CACHE_ROOT:-$REPO/cache}/triton
export FLASHINFER_WORKSPACE_BASE=${CACHE_ROOT:-$REPO/cache}/flashinfer_grpo
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$FLASHINFER_WORKSPACE_BASE"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
if [ -n "${VLLM_SRC:-}" ]; then export PYTHONPATH="$VLLM_SRC:${PYTHONPATH:-}"; fi

RUN=${RUN:-qwen17b_grpo_openthoughts_200step_$(date +%Y%m%d_%H%M)}
MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH:-Qwen/Qwen3-1.7B}
MAX_STEPS=${MAX_STEPS:-200}
SAVE_STEPS=${SAVE_STEPS:-25}
MAX_COMPLETION_LENGTH=${MAX_COMPLETION_LENGTH:-16000}
PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE:-1}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-8}
NUM_PROCESSES=${NUM_PROCESSES:-4}
MAIN_PROCESS_PORT=${MAIN_PROCESS_PORT:-13031}
RUN_DIR="$REPO/runs/$RUN"
mkdir -p "$RUN_DIR"/{checkpoints,logs,eval}
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    scontrol update JobId="$SLURM_JOB_ID" JobName="${SLURM_JOB_NAME_OVERRIDE:-${RUN:0:128}}" || true
fi

# Do not reuse the shared FlashInfer cache for GRPO jobs. Its build.ninja can
# contain a stale node-specific compiler path, which breaks JIT on other nodes.
export FLASHINFER_WORKSPACE_BASE="$RUN_DIR/flashinfer_cache"
mkdir -p "$FLASHINFER_WORKSPACE_BASE"

exec > "$RUN_DIR/logs/qwen17b_grpo-${SLURM_JOB_ID}.out" 2>&1

cat > "$RUN_DIR/config.json" <<CONFIG
{
  "run": "$RUN",
  "model": "$MODEL_NAME_OR_PATH",
  "variant": "grpo_baseline",
  "loss": "grpo",
  "script": "submit_qwen17b_grpo_baseline.sbatch",
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
  "wandb_project": "OPSD_MIX_SAMPLING"
}
CONFIG

echo "[grpo] run=$RUN max_steps=$MAX_STEPS save_steps=$SAVE_STEPS"

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
