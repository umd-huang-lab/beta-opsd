#!/usr/bin/env bash
#SBATCH --job-name=opsd_eval
#SBATCH --account=scavenger
#SBATCH --partition=scavenger
#SBATCH --qos=scavenger
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00

# Usage:
#   RUN=<run_name> STEP=<N> sbatch eval/submit.sh
#   RUN=<run_name> CHECKPOINT=<path> sbatch eval/submit.sh
#
# If STEP is set, evaluates runs/<RUN>/checkpoints/checkpoint-<STEP>.
# If CHECKPOINT is set explicitly, uses that path directly.
# If neither is set, evaluates the base model only.

set -euo pipefail

REPO=${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}

module load cuda/12.8.1
if ! module load gcc/11.2.0; then
    echo "[eval] gcc/11.2.0 module unavailable on $(hostname); continuing with environment compiler"
fi
source "${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate "${CONDA_ENV:-opsd}"

cd "$REPO/eval"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

# Keep vLLM/Torch compile artifacts isolated per Slurm job. Shared compile
# caches have produced FXGraphCacheMiss failures when many evals start at once.
CACHE_ROOT=${CACHE_ROOT:-$REPO/cache/eval/${SLURM_JOB_ID:-local}}
mkdir -p "$CACHE_ROOT"/{xdg,torch_extensions,inductor,triton,vllm}
export XDG_CACHE_HOME="$CACHE_ROOT/xdg"
export TORCH_EXTENSIONS_DIR="$CACHE_ROOT/torch_extensions"
export TORCHINDUCTOR_CACHE_DIR="$CACHE_ROOT/inductor"
export TRITON_CACHE_DIR="$CACHE_ROOT/triton"
export VLLM_CACHE_ROOT="$CACHE_ROOT/vllm"

# ── Configuration ─────────────────────────────────────────────────────────────
BASE_MODEL=${BASE_MODEL:-Qwen/Qwen3-1.7B}
RUN=${RUN:-}
STEP=${STEP:-}
CHECKPOINT=${CHECKPOINT:-}
DATA_DIR=${DATA_DIR:-data}
DATASETS=${DATASETS:-}
DATASETS_LIST=${DATASETS_LIST:-}
OUT_SUFFIX=${OUT_SUFFIX:-}
SAMPLES=${SAMPLES:-0}
BATCH_SIZE=${BATCH_SIZE:-32}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-0}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.9}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-}
DECODING=${DECODING:-greedy}
K=${K:-1}
TEMPERATURE=${TEMPERATURE:-0.7}
TOP_P=${TOP_P:-0.95}
TOP_K_SAMPLE=${TOP_K_SAMPLE:-50}
REPEAT=${REPEAT:-1}
THINKING=${THINKING:-0}

if [[ -z "$DATASETS" && -n "$DATASETS_LIST" ]]; then
    DATASETS="${DATASETS_LIST//:/,}"
fi

resolve_checkpoint_path() {
    local run_name="$1"
    local step="$2"
    local direct="$REPO/runs/$run_name/checkpoints/checkpoint-$step"
    local nested="$REPO/runs/$run_name/checkpoints/$run_name/checkpoint-$step"
    if [[ -d "$direct" ]]; then
        printf '%s\n' "$direct"
        return 0
    fi
    if [[ -d "$nested" ]]; then
        printf '%s\n' "$nested"
        return 0
    fi
    printf '%s\n' "$nested"
    return 0
}

# ── Resolve checkpoint and output dir ─────────────────────────────────────────
if [[ -n "$RUN" && -n "$STEP" ]]; then
    CHECKPOINT="$(resolve_checkpoint_path "$RUN" "$STEP")"
    RESULT_DIR="$REPO/runs/$RUN/eval"
    OUT_STEM="step-$STEP"
elif [[ -n "$RUN" && -n "$CHECKPOINT" ]]; then
    RESULT_DIR="$REPO/runs/$RUN/eval"
    OUT_STEM="$(basename "$CHECKPOINT")"
elif [[ -n "$RUN" ]]; then
    # Base model eval stored under the run
    RESULT_DIR="$REPO/runs/$RUN/eval"
    OUT_STEM="base__$(basename "$BASE_MODEL")"
else
    # No RUN — standalone base model eval
    RESULT_DIR="$REPO/eval/results"
    OUT_STEM="base__$(basename "$BASE_MODEL")"
fi

if [[ -n "$OUT_SUFFIX" ]]; then
    OUT_STEM="${OUT_STEM}__${OUT_SUFFIX}"
fi

mkdir -p "$RESULT_DIR"

LOG_DIR="$REPO/runs/${RUN:-base}/logs"
mkdir -p "$LOG_DIR"
LOG_PATH="$LOG_DIR/opsd_eval-${SLURM_JOB_ID:-local}.out"
exec > "$LOG_PATH" 2>&1
echo "[eval] log: $LOG_PATH"
echo "[eval] datasets: ${DATASETS:-all}"

# ── Build args ────────────────────────────────────────────────────────────────
ARGS=(
    --data_dir "$DATA_DIR"
    --samples "$SAMPLES"
    --batch_size "$BATCH_SIZE"
    --gpu_memory_utilization "$GPU_MEMORY_UTILIZATION"
    --decoding "$DECODING"
    --k "$K"
    --temperature "$TEMPERATURE"
    --top_p "$TOP_P"
    --top_k "$TOP_K_SAMPLE"
    --repeat "$REPEAT"
    --output_json  "$RESULT_DIR/$OUT_STEM.json"
    --output_jsonl "$RESULT_DIR/$OUT_STEM.samples.jsonl"
)
[[ -n "$MAX_NEW_TOKENS" ]] && ARGS+=(--max_new_tokens "$MAX_NEW_TOKENS")
[[ -n "$DATASETS" ]] && ARGS+=(--datasets "$DATASETS")
[[ -n "$MAX_MODEL_LEN" ]] && ARGS+=(--max_model_len "$MAX_MODEL_LEN")
[[ "$THINKING" == "1" ]] && ARGS+=(--thinking)

# ── Evaluate ──────────────────────────────────────────────────────────────────
if [[ -z "$CHECKPOINT" ]]; then
    echo "[eval] base model: $BASE_MODEL"
    python eval_math.py --checkpoint "$BASE_MODEL" "${ARGS[@]}"
else
    echo "[eval] checkpoint: $CHECKPOINT"
    echo "[eval] base model: $BASE_MODEL"
    python eval_math.py \
        --checkpoint "$CHECKPOINT" \
        --base_model  "$BASE_MODEL" \
        "${ARGS[@]}"
fi
