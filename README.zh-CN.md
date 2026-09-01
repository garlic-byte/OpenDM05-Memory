# OpenDM 视觉记忆训练

本仓库基于官方 [Dexmal OpenDM](https://github.com/dexmal/opendm)，增加了一个可迁移的 DM0.5 稀疏视觉记忆训练流程。新增内容仅包含训练必需的数据采样、视频读取、实验配置、启动脚本和测试；数据、权重、日志、W&B 文件、缓存及本地环境均不会提交。

完整参数和数据示例见 [README.md](README.md)。

## 环境搭建

推荐在官方 Docker 环境中运行：

```bash
git clone <your-repository-url> opendm-memory
cd opendm-memory

docker run -it --rm --gpus all --network host \
  --name opendm-memory \
  --shm-size=16g \
  -v "$PWD":/app/opendm \
  -w /app/opendm \
  dexmal/opendm:latest /bin/bash

conda activate opendm
pip install -e .
```

也可以本地安装：

```bash
conda create -n opendm python=3.10 -y
conda activate opendm
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install ninja packaging
MAX_JOBS=2 pip install flash-attn --no-build-isolation
pip install -e .
```

下载基础模型，但不要把权重加入 Git：

```bash
huggingface-cli download Dexmal/DM05 --local-dir ./checkpoints/DM05
```

## 数据要求

每个 JSONL 文件对应一个 episode，每一行对应一个时间步。行内需要包含 `state`、`action`、`task` 和相机字段。相机字段可以指向图片，也可以指向视频帧：

```json
{"type":"video","url":"videos/head.mp4","frame_idx":1}
```

`JSONL_DIR` 指向 episode JSONL 所在目录；`IMAGE_DIR` 是相对 `url` 的媒体根目录。

## 转换 LeRobot 2.1 / 3.0 数据

统一使用 LeRobot 2.1 作为中间格式：

```text
LeRobot 2.1 ───────────────────────→ DM05 episode JSONL
LeRobot 3.0 → LeRobot 2.1 ────────→ DM05 episode JSONL
```

v3.0 转换需要 `ffmpeg` 和包含 v3 数据工具的 LeRobot 环境。原转换实现已在 LeRobot 0.4.0 上验证。建议单独创建转换环境：

```bash
conda create -n lerobot-convert python=3.10 -y
conda activate lerobot-convert
pip install lerobot==0.4.0 jsonlines pyarrow numpy tqdm huggingface-hub
```

如果输入是 v3.0，先非破坏式转换到一个新的 v2.1 目录：

```bash
python script/convert_lerobot_v3_to_v21.py \
  --input-root /path/to/dataset_v30 \
  --output-root /path/to/dataset_v21
```

脚本不会修改源数据，也拒绝覆盖已经存在的输出目录。然后将 v2.1 转成 DM05 JSONL：

```bash
conda activate opendm

python script/convert_lerobot_v21_to_memory_jsonl.py \
  --input-root /path/to/dataset_v21 \
  --output-dir /path/to/dataset_v21/dm05_jsonl \
  --state-key observation.state \
  --action-key action \
  --camera-keys observation.images.head,observation.images.left_wrist,observation.images.right_wrist \
  --output-image-keys images_1,images_2,images_3
```

第二个脚本每个 episode 输出一个 JSONL，并生成 `conversion_manifest.json`。视频不会被重复复制；JSONL 使用相对于 v2.1 数据集根目录的 MP4 路径和 episode 内 `frame_idx`。

训练时使用 manifest 中对应的路径：

```bash
JSONL_DIR=/path/to/dataset_v21/dm05_jsonl \
IMAGE_DIR=/path/to/dataset_v21 \
STATE_DESC=joint,joint,joint,joint,joint,joint,gripper,joint,joint,joint,joint,joint,joint,gripper \
bash script/train_memory_sft.sh
```

省略 `--camera-keys` 时会自动发现视频相机，但建议显式指定，确保相机顺序与 `IMAGE_PROMPTS`、`MEMORY_IMAGE_KEYS` 一致。

## Memory 的使用方式

当前步为 `t` 时，只从同一 episode 的过去采样：

```text
t - MEMORY_FRAMES * MEMORY_STRIDE, ..., t - MEMORY_STRIDE
```

采样结果按从旧到新的顺序排列，绝不会包含当前帧，也不会跨 episode。

- `MEMORY_IMAGE_KEYS`：用作记忆的相机字段，建议先使用一个稳定的全局相机。
- `MEMORY_FRAMES`：历史时间点数量。
- `MEMORY_STRIDE`：相邻历史时间点之间的步数。
- `MAX_MEMORY_IMAGES`：`历史时间点数量 × 历史相机数量` 的上限。
- `LEFT_PAD_MEMORY=True`：episode 开头不足的历史用不可见占位符补齐；设为 `False` 时使用可获得的实际历史帧数。

例如 `MEMORY_FRAMES=5`、`MEMORY_STRIDE=16` 时，时间步 96 使用 `16、32、48、64、80` 作为 memory。

## 启动训练

至少需要设置数据路径和状态维度描述：

```bash
export JSONL_DIR=/path/to/dataset/jsonl
export IMAGE_DIR=/path/to/dataset/media
export STATE_DESC=joint,joint,joint,joint,joint,joint,gripper,joint,joint,joint,joint,joint,joint,gripper

bash script/train_memory_sft.sh
```

常用配置示例：

### 已复现训练默认参数

`train_memory_sft.sh` 默认使用 RoboDojo memory 的 ARX X5 复现配置：

| 参数 | 默认值 | 说明 |
| --- | ---: | --- |
| `ACTION_MODE` | `ABSOLUTE` | 训练绝对关节位置；该复现不能改成 `RELATIVE`。 |
| `CHUNK_SIZE` | `50` | 每个样本预测未来 50 步动作。 |
| `MEMORY_IMAGE_KEYS` | `images_1` | 头部/全局相机作为 memory。 |
| `MEMORY_FRAMES` | `20` | 使用 20 个历史观测。 |
| `MEMORY_STRIDE` | `25` | 历史观测之间相隔 25 个数据步。 |
| `MAX_MEMORY_IMAGES` | `20` | 单 memory 相机的固定容量。 |
| `LEFT_PAD_MEMORY` | `True` | episode 开头用不可见槽位左填充。 |
| `MODEL_MAX_LENGTH` | `1536` | 20 帧 memory 所需 token 上限。 |
| `VIDEO_BACKEND` | `torchcodec` | 稀疏批量读取 MP4。 |
| `AUGMENTATION_PROBABILITY` | `0.0` | 复现配置关闭图像增强。 |
| `NPROC_PER_NODE` | `8` | 8 个训练进程 / GPU。 |
| `PER_DEVICE_BATCH_SIZE` | `4` | 单卡 micro batch。 |
| `GRADIENT_ACCUMULATION_STEPS` | `8` | 全局有效 batch 为 `8 × 4 × 8 = 256`。 |
| `NUM_TRAIN_STEPS` | `10000` | 优化器总步数。 |
| `LEARNING_RATE` | `4e-5` | MuonAdamW 基础学习率。 |
| `WARMUP_STEPS` | `1000` | 学习率 warmup。 |
| `SAVE_STEPS` | `2000` | 每 2,000 步保存。 |
| `USE_LORA` | `False` | 全参数微调。 |
| Attention | `sdpa` | LLM、视觉、动作均使用 SDPA。 |
| `SEED` | `42` | 训练随机种子。 |

启动前检查默认要求 8 张显存不少于 79,000 MiB 的 GPU。只有在明确调整硬件配置时才设置 `VALIDATE_GPU_MEMORY=False`。减少 GPU 数或单卡 batch 时，应相应增加梯度累积，尽量保持全局有效 batch 为 256。

默认状态描述对应 14 维双臂 ARX X5：

```text
joint,joint,joint,joint,joint,joint,gripper,joint,joint,joint,joint,joint,joint,gripper
```

换机器人时必须显式修改 `ROBOT_TYPE`、`STATE_DESC`、`OUTPUT_ACTION_DIM`、`ACTION_MODE`、相机映射，并按需修改 `CHUNK_SIZE`。`STATE_DESC` 的数量和顺序必须同时匹配 state/action。

```bash
DM05_MODEL_PATH=/path/to/DM05 \
JSONL_DIR=/path/to/dataset/jsonl \
IMAGE_DIR=/path/to/dataset/media \
STATE_DESC=joint,joint,joint,joint,joint,joint,gripper,joint,joint,joint,joint,joint,joint,gripper \
IMAGE_KEYS=images_1,images_2,images_3 \
IMAGE_PROMPTS="Head,Left wrist,Right wrist" \
MEMORY_IMAGE_KEYS=images_1 \
MEMORY_FRAMES=20 \
MEMORY_STRIDE=25 \
MAX_MEMORY_IMAGES=20 \
LEFT_PAD_MEMORY=True \
ACTION_MODE=ABSOLUTE \
NPROC_PER_NODE=8 \
PER_DEVICE_BATCH_SIZE=4 \
GRADIENT_ACCUMULATION_STEPS=8 \
NUM_TRAIN_STEPS=10000 \
LEARNING_RATE=4e-5 \
MODEL_MAX_LENGTH=1536 \
VIDEO_BACKEND=torchcodec \
bash script/train_memory_sft.sh
```

默认输出到 `./artifacts/checkpoints/dm05_memory_sft`。可以用 `OUTPUT_DIR` 覆盖。设置 `DRY_RUN=1` 时只校验路径并打印命令，不会启动训练。

## W&B

W&B 默认不启用。需要时在仓库外登录，并通过环境变量传入项目名：

```bash
wandb login
WANDB_PROJECT=dm05-memory bash script/train_memory_sft.sh
```

不要把 `WANDB_API_KEY` 写入代码、配置或脚本；非交互任务应通过作业环境或 secrets manager 注入。

## RoboDojo Memory 复现指标

下表来自 **DM0.5 / OpenDM05** 的 [RoboDojo 官方 rollout 排行榜](https://robodojo-benchmark.com/leaderboard/rollouts/OpenDM05?bench=sim)，核对日期为 2026-09-01。`Avg Score` 和成功率是排行榜中的两个独立指标。

| Memory 任务 | 机器人 | Avg Score | 成功率 |
| --- | --- | ---: | ---: |
| `cover_blocks` | `arx_x5` | 100.00 | 100.00% |
| `press_by_number` | `arx_x5` | 95.33 | 95.00% |
| `match_and_pick_from_conveyor` | `arx_x5` | 70.67 | 71.00% |

这些是模拟器 rollout 指标，不是训练 loss 或验证集 accuracy。复现时需要匹配 RoboDojo 环境、checkpoint、action chunk、历史观测和评测配置。

## 验证

```bash
pytest -q tests/test_memory_dataset.py
pre-commit run --all-files
```

本仓库保留上游 Apache-2.0 许可证，详见 `LICENSE`。
