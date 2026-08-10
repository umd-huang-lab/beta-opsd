#!/usr/bin/env bash
# Optional vLLM server template for server-mode generation.
#
# Example:
#   MODEL=Qwen/Qwen3-1.7B sbatch --account=<acct> --partition=<part> --gres=gpu:1 scripts/slurm/submit_vllm_server.sbatch

#SBATCH --job-name=vllm_serve
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

module load cuda/12.8.1 2>/dev/null || true
module load gcc/11.2.0 2>/dev/null || true

REPO=${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
echo "[vllm] node=$(hostname)"
echo "[vllm] pass VLLM_SERVER_HOST=$(hostname) VLLM_SERVER_PORT=${PORT:-8001} to training"

bash "$REPO/scripts/vllm/serve.sh"
