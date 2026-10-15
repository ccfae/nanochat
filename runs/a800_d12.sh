#!/usr/bin/env bash

# Formal d12 shared-VE pipeline for one A800 node. Usage:
#   bash runs/a800_d12.sh pretrain
#   bash runs/a800_d12.sh base-eval
#   bash runs/a800_d12.sh sft
#   bash runs/a800_d12.sh chat-eval
#   bash runs/a800_d12.sh all

set -euo pipefail

: "${NANOCHAT_BASE_DIR:?Set NANOCHAT_BASE_DIR to the persistent data disk}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export WANDB_DIR="${WANDB_DIR:-$NANOCHAT_BASE_DIR/wandb}"
export NANOCHAT_DTYPE=bfloat16

stage="${1:-all}"
nproc="${NPROC_PER_NODE:-4}"
device_batch="${DEVICE_BATCH_SIZE:-16}"
model_tag="${MODEL_TAG:-d12}"
run_prefix="${WANDB_RUN_PREFIX:-d12-shared-ve}"
save_every="${SAVE_EVERY:-500}"
resume_from_step="${RESUME_FROM_STEP:--1}"

launch() {
    .venv/bin/torchrun --standalone --nproc_per_node="$nproc" "$@"
}

pretrain() {
    base_args=(
        --depth=12
        --device-batch-size="$device_batch"
        --target-param-data-ratio=12
        --logit-chunk-size=4096
        --save-every="$save_every"
        --model-tag="$model_tag"
        --run="${run_prefix}-base"
    )
    if (( resume_from_step >= 0 )); then
        base_args+=(--resume-from-step="$resume_from_step")
    fi
    launch -m scripts.base_train -- "${base_args[@]}"
}

base_eval() {
    launch -m scripts.base_eval -- \
        --model-tag="$model_tag" \
        --device-batch-size="$device_batch" \
        --run="${run_prefix}-base-eval"
}

sft() {
    launch -m scripts.chat_sft -- \
        --model-tag="$model_tag" \
        --device-batch-size="$device_batch" \
        --run="${run_prefix}-sft"
}

chat_eval() {
    launch -m scripts.chat_eval -- \
        --source=sft \
        --model-tag="$model_tag" \
        --batch-size="$device_batch" \
        --run="${run_prefix}-chat-eval"
}

case "$stage" in
    pretrain) pretrain ;;
    base-eval) base_eval ;;
    sft) sft ;;
    chat-eval) chat_eval ;;
    all)
        pretrain
        base_eval
        sft
        chat_eval
        ;;
    *)
        echo "Unknown stage: $stage (expected pretrain|base-eval|sft|chat-eval|all)"
        exit 2
        ;;
esac
