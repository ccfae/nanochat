#!/usr/bin/env bash

# Run on the one-GPU preparation instance. All downloads and generated artifacts
# go to the persistent disk selected by NANOCHAT_BASE_DIR.

set -euo pipefail

: "${NANOCHAT_BASE_DIR:?Set NANOCHAT_BASE_DIR to the persistent data disk, e.g. /data/nanochat}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export WANDB_DIR="${WANDB_DIR:-$NANOCHAT_BASE_DIR/wandb}"
mkdir -p "$NANOCHAT_BASE_DIR" "$WANDB_DIR"

available_kib=$(df -Pk "$NANOCHAT_BASE_DIR" | awk 'NR==2 {print $4}')
required_kib=$((40 * 1024 * 1024))
if (( available_kib < required_kib )); then
    echo "At least 40 GiB free is required for the d12 pipeline."
    exit 1
fi

command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv sync --frozen --extra gpu --group dev

.venv/bin/python - <<'PY'
import torch

assert torch.cuda.is_available(), "CUDA is not available"
major, minor = torch.cuda.get_device_capability(0)
assert (major, minor) >= (8, 0), "A800-class BF16 support requires SM80+"
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"CUDA: {torch.version.cuda}; capability: {major}.{minor}; BF16: {torch.cuda.is_bf16_supported()}")
PY

# 35 training shards plus the pinned validation shard. The first eight are also
# used to train the tokenizer.
.venv/bin/python -m nanochat.dataset -n 35 -w "${DOWNLOAD_WORKERS:-8}"
.venv/bin/python -m scripts.tok_train --max-chars=2000000000 --vocab-size=32768
.venv/bin/python -m scripts.tok_eval

# SFT, ChatCORE and CORE evaluation data.
.venv/bin/python -m scripts.prepare_data

.venv/bin/python -m pytest -q

echo "Preparation complete in $NANOCHAT_BASE_DIR"
du -sh "$NANOCHAT_BASE_DIR"
