#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
CONFIG=${CONFIG:-recipes/paper/grpo_qwen3_4b.yaml}

cd "$REPO"
python scripts/train/run_recipe.py "$CONFIG" "$@"
