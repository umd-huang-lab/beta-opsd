#!/usr/bin/env bash
#SBATCH --job-name=debug_h200_srun
#SBATCH --account=cml-furongh
#SBATCH --partition=cml-furongh
#SBATCH --qos=cml-high
#SBATCH --gres=gpu:h200-sxm:1
#SBATCH --nodelist=cml36
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=slurm_logs/%x-%j.out

set -x

source "${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate "${CONDA_ENV:-opsd}"

probe() {
    python - <<'PY'
import os
import torch

print("CUDA_VISIBLE_DEVICES", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("SLURM_STEP_GPUS", os.environ.get("SLURM_STEP_GPUS"))
print("device_count", torch.cuda.device_count())
print("is_available", torch.cuda.is_available())
torch.cuda.init()
print("init ok", torch.cuda.get_device_name(0))
PY
}

echo "===== direct batch ====="
probe || true

echo "===== srun default ====="
srun bash -lc 'source "${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}" && conda activate "${CONDA_ENV:-opsd}" && python - <<'"'"'PY'"'"'
import os
import torch

print("CUDA_VISIBLE_DEVICES", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("SLURM_STEP_GPUS", os.environ.get("SLURM_STEP_GPUS"))
print("device_count", torch.cuda.device_count())
print("is_available", torch.cuda.is_available())
torch.cuda.init()
print("init ok", torch.cuda.get_device_name(0))
PY'

echo "===== srun gpu bind none ====="
srun --gres=gpu:h200-sxm:1 --gpu-bind=none bash -lc 'source "${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}" && conda activate "${CONDA_ENV:-opsd}" && python - <<'"'"'PY'"'"'
import os
import torch

print("CUDA_VISIBLE_DEVICES", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("SLURM_STEP_GPUS", os.environ.get("SLURM_STEP_GPUS"))
print("device_count", torch.cuda.device_count())
print("is_available", torch.cuda.is_available())
torch.cuda.init()
print("init ok", torch.cuda.get_device_name(0))
PY'
