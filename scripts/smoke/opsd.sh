#!/usr/bin/env bash
set -euo pipefail

export WANDB_MODE=${WANDB_MODE:-disabled}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export CONFIG=${CONFIG:-recipes/smoke/opsd.yaml}

bash "$(dirname "${BASH_SOURCE[0]}")/../train/opsd.sh" "$@"
