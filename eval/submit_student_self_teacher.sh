#!/usr/bin/env bash
#SBATCH --job-name=sst_eval
#SBATCH --account=scavenger
#SBATCH --partition=scavenger
#SBATCH --qos=scavenger
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=12:00:00

set -euo pipefail

REPO=${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}

module load cuda/12.8.1
if ! module load gcc/11.2.0; then
    echo "[sst-eval] gcc/11.2.0 module unavailable on $(hostname); continuing with environment compiler"
fi
source "${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate "${CONDA_ENV:-opsd}"

cd "$REPO"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

CACHE_ROOT=${CACHE_ROOT:-$REPO/cache/student_self_teacher/${SLURM_JOB_ID:-local}}
mkdir -p "$CACHE_ROOT"/{xdg,torch_extensions,inductor,triton,vllm}
export XDG_CACHE_HOME="$CACHE_ROOT/xdg"
export TORCH_EXTENSIONS_DIR="$CACHE_ROOT/torch_extensions"
export TORCHINDUCTOR_CACHE_DIR="$CACHE_ROOT/inductor"
export TRITON_CACHE_DIR="$CACHE_ROOT/triton"
export VLLM_CACHE_ROOT="$CACHE_ROOT/vllm"

RUN=${RUN:-student_self_teacher_$(date +%Y%m%d_%H%M)}
DATASET=${DATASET:?Set DATASET to openthoughts_refs or gsm8k}
MODEL=${MODEL:-Qwen/Qwen3-1.7B}
SEED=${SEED:-0}
K=${K:-12}
TEMPERATURE=${TEMPERATURE:-0.6}
TOP_P=${TOP_P:-0.95}
TOP_K=${TOP_K:-50}
BATCH_SIZE=${BATCH_SIZE:-32}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.9}

case "$DATASET" in
    openthoughts_refs)
        DATA_PATH=${DATA_PATH:-openthoughts_refs_1000_qwen3_8b.jsonl}
        LIMIT=${LIMIT:-100}
        MAX_MODEL_LEN=${MAX_MODEL_LEN:-32768}
        MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-4096}
        ;;
    gsm8k)
        DATA_PATH=${DATA_PATH:-eval/data/gsm8k.jsonl}
        LIMIT=${LIMIT:-200}
        MAX_MODEL_LEN=${MAX_MODEL_LEN:-8192}
        MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-1024}
        ;;
    *)
        echo "Unsupported DATASET=$DATASET" >&2
        exit 2
        ;;
esac

RESULT_DIR="$REPO/runs/$RUN/eval"
LOG_DIR="$REPO/runs/$RUN/logs"
mkdir -p "$RESULT_DIR" "$LOG_DIR"

LOG_PATH="$LOG_DIR/sst_eval-${DATASET}-${SLURM_JOB_ID:-local}.out"
exec > "$LOG_PATH" 2>&1

echo "[sst-eval] run=$RUN"
echo "[sst-eval] dataset=$DATASET"
echo "[sst-eval] data_path=$DATA_PATH"
echo "[sst-eval] limit=$LIMIT k=$K max_model_len=$MAX_MODEL_LEN max_new_tokens=$MAX_NEW_TOKENS"
echo "[sst-eval] output=$RESULT_DIR/${DATASET}.json"

python eval/eval_student_self_teacher.py \
    --dataset "$DATASET" \
    --data_path "$DATA_PATH" \
    --model "$MODEL" \
    --limit "$LIMIT" \
    --seed "$SEED" \
    --k "$K" \
    --temperature "$TEMPERATURE" \
    --top_p "$TOP_P" \
    --top_k "$TOP_K" \
    --batch_size "$BATCH_SIZE" \
    --gpu_memory_utilization "$GPU_MEMORY_UTILIZATION" \
    --max_model_len "$MAX_MODEL_LEN" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --output_json "$RESULT_DIR/${DATASET}.json" \
    --output_jsonl "$RESULT_DIR/${DATASET}.samples.jsonl"
