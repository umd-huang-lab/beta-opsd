#!/usr/bin/env bash
#SBATCH --job-name=cmp_revkl
#SBATCH --account=scavenger
#SBATCH --partition=scavenger
#SBATCH --qos=scavenger
#SBATCH --gres=gpu:l40s:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=slurm_logs/%x-%j.out
#
# Comparison run: reverse KL (Thinking-Machines / tinker loss) on Qwen3-1.7B.
# Paired with submit_cmp_cumseq.sbatch — all hyperparams identical except loss.

set -euo pipefail

module load cuda/12.8.1
module load gcc/11.2.0
source "${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate "${CONDA_ENV:-opsd}"

REPO=${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
cd "$REPO"

# Redirect compiler caches to project space — scratch quota is tight
export TORCHINDUCTOR_CACHE_DIR=${CACHE_ROOT:-$REPO/cache}/inductor
export TRITON_CACHE_DIR=${CACHE_ROOT:-$REPO/cache}/triton
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RUN=${RUN:-cmp_revkl_$(date +%Y%m%d)}
MAX_STEPS=${MAX_STEPS:-200}
RESUME_FROM=${RESUME_FROM:-}
RUN_DIR="$REPO/runs/$RUN"
mkdir -p "$RUN_DIR"/{checkpoints,logs,eval}
exec > "$RUN_DIR/logs/cmp_revkl-${SLURM_JOB_ID}.out" 2>&1

cat > "$RUN_DIR/config.json" <<CONFIG
{
  "run": "$RUN",
  "model": "Qwen/Qwen3-1.7B",
  "loss": "reverse_kl",
  "script": "submit_cmp_revkl.sbatch",
  "launched": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "slurm_job_id": "${SLURM_JOB_ID}",
  "max_steps": ${MAX_STEPS},
  "resume_from": "${RESUME_FROM}"
}
CONFIG

echo "[cmp] run=$RUN  loss=reverse_kl  max_steps=$MAX_STEPS  resume_from=${RESUME_FROM:-none}"

RESUME_ARGS=()
[[ -n "$RESUME_FROM" ]] && RESUME_ARGS=(--resume_from_checkpoint "$RESUME_FROM")

accelerate launch \
    --config_file accelerate.yaml \
    --num_processes 4 \
    --gradient_accumulation_steps 2 \
    --main_process_port 12950 \
    opsd_train.py \
    --model_name_or_path Qwen/Qwen3-1.7B \
    --learning_rate 5e-6 \
    --max_grad_norm 0.1 \
    --per_device_train_batch_size 4 \
    --gradient_checkpointing \
    --gradient_accumulation_steps 2 \
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
    --vllm_gpu_memory_utilization 0.6 \
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
    --wandb_project OPSD \
    "${RESUME_ARGS[@]}"
