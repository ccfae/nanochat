#!/usr/bin/env bash

# Four-GPU hardware gate before the formal run. Uses the formal d12 shape and
# batch size, exercises NCCL optimizer sharding, checkpoint resume, SFT and eval.

set -euo pipefail

: "${NANOCHAT_BASE_DIR:?Set NANOCHAT_BASE_DIR to the persistent data disk}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export WANDB_DIR="${WANDB_DIR:-$NANOCHAT_BASE_DIR/wandb}"
export NANOCHAT_DTYPE=bfloat16

nproc="${NPROC_PER_NODE:-4}"
device_batch="${DEVICE_BATCH_SIZE:-16}"
model_tag="d12-a800-${nproc}gpu-smoke"
world_micro_tokens=$((device_batch * 2048 * nproc))
total_batch=$((world_micro_tokens * 2))

launch() {
    .venv/bin/torchrun --standalone --nproc_per_node="$nproc" "$@"
}

base_args=(
    --depth=12
    --device-batch-size="$device_batch"
    --total-batch-size="$total_batch"
    --warmup-steps=1
    --eval-every=1
    --eval-tokens="$world_micro_tokens"
    --core-metric-every=-1
    --sample-every=-1
    --save-every=1
    --model-tag="$model_tag"
    --run=dummy
)

# First run and resume prove that all optimizer rank shards can be restored.
launch -m scripts.base_train -- "${base_args[@]}" --num-iterations=2
launch -m scripts.base_train -- "${base_args[@]}" --num-iterations=3 --resume-from-step=2

launch -m scripts.base_eval -- \
    --eval=bpb,sample \
    --model-tag="$model_tag" \
    --device-batch-size="$device_batch" \
    --split-tokens="$world_micro_tokens" \
    --run=dummy

launch -m scripts.chat_sft -- \
    --model-tag="$model_tag" \
    --model-step=3 \
    --load-optimizer=1 \
    --num-iterations=1 \
    --device-batch-size="$device_batch" \
    --total-batch-size="$world_micro_tokens" \
    --eval-every=1 \
    --eval-tokens="$world_micro_tokens" \
    --chatcore-every=-1 \
    --mmlu-epochs=0 \
    --gsm8k-epochs=0 \
    --run=dummy

launch -m scripts.chat_eval -- \
    --source=sft \
    --model-tag="$model_tag" \
    --task-name=ARC-Easy \
    --max-problems=8 \
    --batch-size="$device_batch" \
    --run=dummy

echo "${nproc}-GPU d12 smoke test, resume, SFT and evaluation completed."
