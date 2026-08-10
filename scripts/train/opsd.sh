#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
CONFIG=${CONFIG:-recipes/paper/opsd_fixed_teacher_qwen3_1_7b.yaml}

cd "$REPO"
python scripts/train/run_recipe.py "$CONFIG" "$@"
