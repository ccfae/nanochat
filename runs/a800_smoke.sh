#!/usr/bin/env bash

# One-GPU, full-d12 pipeline smoke test. It exercises the real architecture,
# checkpoint resume format, SFT, evaluation result files and optional W&B logging.

set -euo pipefail

: "${NANOCHAT_BASE_DIR:?Set NANOCHAT_BASE_DIR to the persistent data disk}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export WANDB_DIR="${WANDB_DIR:-$NANOCHAT_BASE_DIR/wandb}"
export NANOCHAT_DTYPE=bfloat16

run_prefix="${WANDB_RUN_PREFIX:-dummy}"
smoke_batch="${SMOKE_DEVICE_BATCH_SIZE:-2}"
model_tag="d12-a800-smoke"

.venv/bin/python -m scripts.base_train \
    --depth=12 \
    --device-batch-size="$smoke_batch" \
    --total-batch-size=$((smoke_batch * 2048)) \
    --num-iterations=2 \
    --warmup-steps=1 \
    --eval-every=1 \
    --eval-tokens=$((smoke_batch * 2048)) \
    --core-metric-every=-1 \
    --sample-every=-1 \
    --model-tag="$model_tag" \
    --run="$run_prefix"

.venv/bin/python -m scripts.base_eval \
    --eval=bpb,sample \
    --model-tag="$model_tag" \
    --device-batch-size=1 \
    --split-tokens=2048 \
    --run="$run_prefix"

.venv/bin/python -m scripts.chat_sft \
    --model-tag="$model_tag" \
    --load-optimizer=1 \
    --num-iterations=1 \
    --device-batch-size=1 \
    --total-batch-size=2048 \
    --eval-every=1 \
    --eval-tokens=2048 \
    --chatcore-every=-1 \
    --mmlu-epochs=0 \
    --gsm8k-epochs=0 \
    --run="$run_prefix"

.venv/bin/python -m scripts.chat_eval \
    --source=sft \
    --model-tag="$model_tag" \
    --max-problems=2 \
    --max-new-tokens=32 \
    --batch-size=2 \
    --run="$run_prefix"

echo "Full d12 one-GPU smoke test completed."
