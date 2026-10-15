"""Pre-download every non-pretraining dataset used by nanochat.

Run this once on a cheap preparation instance with NANOCHAT_BASE_DIR pointing at
the persistent data disk. Pretraining shards and the tokenizer are prepared by
``python -m nanochat.dataset`` and ``python -m scripts.tok_train`` separately.
"""

import gc
import os

from nanochat.common import get_base_dir
from scripts.base_eval import ensure_eval_bundle
from tasks.arc import ARC
from tasks.gsm8k import GSM8K
from tasks.humaneval import HumanEval
from tasks.mmlu import MMLU
from tasks.smoltalk import SmolTalk


def directory_size(path):
    total = 0
    for root, _dirs, files in os.walk(path):
        for filename in files:
            total += os.path.getsize(os.path.join(root, filename))
    return total


def main():
    datasets = [
        ("SmolTalk train", lambda: SmolTalk(split="train")),
        ("SmolTalk test", lambda: SmolTalk(split="test")),
        ("MMLU auxiliary_train", lambda: MMLU(subset="all", split="auxiliary_train")),
        ("MMLU test", lambda: MMLU(subset="all", split="test")),
        ("GSM8K train", lambda: GSM8K(subset="main", split="train")),
        ("GSM8K test", lambda: GSM8K(subset="main", split="test")),
        ("ARC-Easy test", lambda: ARC(subset="ARC-Easy", split="test")),
        ("ARC-Challenge test", lambda: ARC(subset="ARC-Challenge", split="test")),
        ("HumanEval test", HumanEval),
    ]

    for label, factory in datasets:
        dataset = factory()
        print(f"Prepared {label}: {len(dataset):,} examples")
        del dataset
        gc.collect()

    base_dir = get_base_dir()
    eval_bundle_dir = ensure_eval_bundle()
    print(f"Prepared CORE evaluation bundle: {eval_bundle_dir}")
    print(f"Total nanochat data directory size: {directory_size(base_dir) / 1024**3:.2f} GiB")


if __name__ == "__main__":
    main()
