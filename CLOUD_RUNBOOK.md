# d12 shared-VE 云端运行手册

目标配置：12 层、768 隐藏维度、6 个注意力头、单张共享 Value Embedding、
32768 词表、2048 上下文，默认使用 BF16 和 4096-token 分块交叉熵。

## 存储规划

- 如果系统盘和数据盘分开：30 GB 系统盘加 50 GB 数据盘可以运行当前 d12 流程；
  如果希望长期保留所有冒烟和中间 checkpoint，数据盘扩到 80 GB 会更宽裕。
- 如果只有一块盘：建议至少 100 GiB。
- 准备脚本要求数据盘至少还有 40 GiB 空闲空间。

所有数据、tokenizer、checkpoint、评测结果和本地 W&B 日志都必须指向持久盘：

```bash
export NANOCHAT_BASE_DIR=/data/nanochat
export WANDB_DIR=$NANOCHAT_BASE_DIR/wandb
```

## 一张 A800：环境和数据准备

```bash
git clone <YOUR_GITHUB_REPO_URL> nanochat
cd nanochat

export NANOCHAT_BASE_DIR=/data/nanochat
export WANDB_DIR=$NANOCHAT_BASE_DIR/wandb

bash runs/a800_prepare.sh
```

该脚本会安装锁定的 GPU 依赖、验证 CUDA/BF16、下载 35 个预训练 shard
和固定验证 shard、训练正式 tokenizer、下载全部 SFT/ChatEval/CORE 数据并运行测试。

如果要写入 W&B，运行冒烟测试或正式训练前设置密钥：

```bash
export WANDB_API_KEY=<YOUR_KEY>
```

不要把密钥写进脚本、Git commit 或镜像。

## 一张 A800：完整 d12 冒烟测试

```bash
export NANOCHAT_BASE_DIR=/data/nanochat
export WANDB_RUN_PREFIX=d12-smoke  # 使用 dummy 可关闭 W&B
bash runs/a800_smoke.sh
```

该脚本使用正式 d12 架构跑两步预训练、BaseEval、一步 SFT 和小规模 ChatEval。
只有它全部完成后，才启动四卡正式训练。

## 四张 A800：正式流程

同一台四卡机器必须挂载准备阶段的数据盘，并重新设置环境变量：

```bash
export NANOCHAT_BASE_DIR=/data/nanochat
export WANDB_DIR=$NANOCHAT_BASE_DIR/wandb
export WANDB_API_KEY=<YOUR_KEY>
export NPROC_PER_NODE=4
export WANDB_RUN_PREFIX=d12-shared-ve-v1
```

A800 40GB 从单卡 batch 16 开始：

```bash
export DEVICE_BATCH_SIZE=16
```

A800 80GB 可在冒烟测试确认显存后尝试 32：

```bash
export DEVICE_BATCH_SIZE=32
```

先运行四卡硬件冒烟。它会验证 NCCL 参数同步、四份 optimizer shard、断点恢复、
四卡 SFT 和分布式评测：

```bash
bash runs/a800_4gpu_smoke.sh
```

看到脚本完整结束后再进入正式流程。

建议分阶段运行，以便每一步检查产物：

```bash
bash runs/a800_d12.sh pretrain
bash runs/a800_d12.sh base-eval
bash runs/a800_d12.sh sft
bash runs/a800_d12.sh chat-eval
```

也可以连续执行：

```bash
bash runs/a800_d12.sh all
```

预训练默认每 500 步保存一次，最终预计约 2520 步。若从 step 2000 恢复：

```bash
export RESUME_FROM_STEP=2000
export WANDB_RESUME=allow
export WANDB_RUN_ID=<EXISTING_WANDB_RUN_ID>
bash runs/a800_d12.sh pretrain
unset RESUME_FROM_STEP WANDB_RESUME WANDB_RUN_ID
```

恢复时必须保持相同的 GPU 数量、模型配置和 tokenizer，并保留四个 rank 的 optimizer 文件。

## 关键产物

```text
$NANOCHAT_BASE_DIR/
├── tokenizer/
├── base_checkpoints/d12/
├── base_eval/d12/
├── chatsft_checkpoints/d12/
├── chat_eval/sft/d12/
├── task_data/
├── eval_bundle/
└── wandb/
```

释放远程机器前，至少备份：

- `tokenizer/`
- `base_checkpoints/d12/` 最终 checkpoint；需要恢复训练时还要全部 optimizer rank 文件
- `chatsft_checkpoints/d12/` 最终 checkpoint
- `base_eval/d12/` 与 `chat_eval/sft/d12/`
- W&B 在线 run 链接
- 本次训练使用的 Git commit SHA

代码、运行手册和小型 JSON/CSV 结果放 GitHub；大模型权重放 Hugging Face 模型仓库；
完整 optimizer checkpoint 放对象存储或数据盘快照。
