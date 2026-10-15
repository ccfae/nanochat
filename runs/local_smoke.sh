#!/bin/bash

# Minimal end-to-end local validation for CPU / Apple Silicon.
# This validates the pipeline; it does not train a useful language model.

set -euo pipefail

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$PWD/.nanochat-data}"

uv sync --frozen --extra cpu --group dev

# One train shard plus the pinned validation shard (~194 MB total).
.venv/bin/python -m nanochat.dataset -n 1 -w 2

# Small tokenizer for a fast local run.
.venv/bin/python -m scripts.tok_train \
    --max-chars=5000000 \
    --doc-cap=4000 \
    --vocab-size=2048
.venv/bin/python -m scripts.tok_eval

# Tiny base model: train, save, then prove optimizer/data state can resume.
BASE_ARGS=(
    --device-type=cpu
    --depth=2
    --aspect-ratio=64
    --head-dim=64
    --max-seq-len=128
    --window-pattern=L
    --device-batch-size=2
    --total-batch-size=512
    --warmup-steps=1
    --eval-every=1
    --eval-tokens=512
    --core-metric-every=-1
    --sample-every=-1
    --save-every=1
    --model-tag=local-smoke
    --run=dummy
)
.venv/bin/python -m scripts.base_train "${BASE_ARGS[@]}" --num-iterations=3
.venv/bin/python -m scripts.base_train "${BASE_ARGS[@]}" --num-iterations=4 --resume-from-step=3

.venv/bin/python -m scripts.base_eval \
    --eval=bpb,sample \
    --model-tag=local-smoke \
    --step=4 \
    --device-batch-size=1 \
    --split-tokens=512 \
    --device-type=cpu

# One real SFT optimizer step. The first run caches the public task datasets.
.venv/bin/python -m scripts.chat_sft \
    --device-type=cpu \
    --model-tag=local-smoke \
    --model-step=4 \
    --load-optimizer=1 \
    --num-iterations=1 \
    --max-seq-len=128 \
    --device-batch-size=1 \
    --total-batch-size=128 \
    --eval-every=1 \
    --eval-tokens=128 \
    --chatcore-every=-1 \
    --mmlu-epochs=0 \
    --gsm8k-epochs=0 \
    --run=dummy

.venv/bin/python -m scripts.chat_cli \
    --source=sft \
    --model-tag=local-smoke \
    --step=1 \
    --prompt="Hello" \
    --max-tokens=16 \
    --temperature=0 \
    --device-type=cpu
