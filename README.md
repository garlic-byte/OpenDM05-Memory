# OpenDM Visual-Memory Training

This repository is based on the official [Dexmal OpenDM](https://github.com/dexmal/opendm) codebase and adds a portable training path for DM0.5 with sparse visual memory.

The extension keeps the upstream model architecture and training workflow. It adds only the pieces required to sample earlier frames from an episode-oriented JSONL dataset and pass them to DM0.5 as memory images.

## What is added

- `opendm/dataset/memory_dataset.py`: episode-aware memory-frame sampling.
- `opendm/data/video_reader.py`: optional TorchCodec reader for batched sparse MP4 decoding.
- `playground/dm05_memory_sft.py`: generic memory SFT experiment configuration.
- `script/train_memory_sft.sh`: portable training entry point.
- `script/convert_lerobot_v3_to_v21.py`: non-destructive LeRobot v3.0 to v2.1 conversion.
- `script/convert_lerobot_v21_to_memory_jsonl.py`: LeRobot v2.1 to DM05 episode JSONL conversion.
- `tests/test_memory_dataset.py`: unit tests for memory indices, padding, and video batching.

Generated data, checkpoints, logs, W&B files, caches, and local environments are ignored and are not part of the repository.

## Environment setup

### Recommended: Docker

Requirements: Ubuntu 20.04 or 22.04, an NVIDIA GPU and driver, Docker, and NVIDIA Container Toolkit.

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

### Local installation

```bash
conda create -n opendm python=3.10 -y
conda activate opendm

pip install torch torchvision \
  --index-url https://download.pytorch.org/whl/cu128
pip install ninja packaging
MAX_JOBS=2 pip install flash-attn --no-build-isolation
pip install -e .
```

The default `pyav` video backend uses the base dependencies. To use the optional TorchCodec backend, install a TorchCodec build compatible with the active PyTorch and FFmpeg versions.

## Download the base checkpoint

Keep model files outside Git history. The default training command expects the base checkpoint at `./checkpoints/DM05`:

```bash
huggingface-cli download Dexmal/DM05 \
  --local-dir ./checkpoints/DM05
```

You may place it elsewhere and set `DM05_MODEL_PATH` when starting training.

## Dataset format

The trainer expects one or more JSONL files. Every JSONL file represents one episode, and every line represents one step. Earlier lines from the same file are eligible memory frames, so memory never crosses an episode boundary.

A row must provide `state`, `action`, `task`, and one entry per configured image key. Image entries follow the standard OpenDM format and may refer to image files:

```json
{"type": "image", "url": "images/head/000001.jpg"}
```

or frames inside a video:

```json
{"type": "video", "url": "videos/head.mp4", "frame_idx": 1}
```

Example row:

```json
{"state":[0.0,0.1],"action":[0.0,0.2],"task":"place the object","images_1":{"type":"video","url":"videos/head.mp4","frame_idx":1},"images_2":{"type":"video","url":"videos/left.mp4","frame_idx":1},"images_3":{"type":"video","url":"videos/right.mp4","frame_idx":1}}
```

`JSONL_DIR` points to the directory containing episode JSONL files. `IMAGE_DIR` is the root used to resolve each relative `url`.

## Convert LeRobot 2.1 or 3.0 data

The public conversion path uses LeRobot v2.1 as the common intermediate format:

```text
LeRobot v2.1 ───────────────────────→ DM05 episode JSONL
LeRobot v3.0 → LeRobot v2.1 ───────→ DM05 episode JSONL
```

The v3.0 conversion requires `ffmpeg` and a LeRobot environment that provides the v3 dataset utilities used by the converter. The original implementation was tested with LeRobot 0.4.0. Install the conversion dependencies in a separate environment if they conflict with OpenDM:

```bash
conda create -n lerobot-convert python=3.10 -y
conda activate lerobot-convert
pip install lerobot==0.4.0 jsonlines pyarrow numpy tqdm huggingface-hub
```

### Input is LeRobot 3.0

First convert v3.0 to a new v2.1 directory. The source directory is never modified, and the command refuses to overwrite an existing output directory:

```bash
python script/convert_lerobot_v3_to_v21.py \
  --input-root /path/to/dataset_v30 \
  --output-root /path/to/dataset_v21
```

This reconstructs per-episode Parquet and MP4 files, legacy task/episode metadata, and v2.1 path templates.

### Convert LeRobot 2.1 to DM05 JSONL

Run this step directly for an existing v2.1 dataset, or after the v3.0 conversion above:

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

The converter creates one JSONL file per episode and a `conversion_manifest.json`. It does not copy videos: each JSONL row references the original v2.1 episode MP4 with an episode-local `frame_idx`.

Use the paths printed in the manifest for training:

```bash
JSONL_DIR=/path/to/dataset_v21/dm05_jsonl \
IMAGE_DIR=/path/to/dataset_v21 \
STATE_DESC=joint,joint,joint,joint,joint,joint,gripper,joint,joint,joint,joint,joint,joint,gripper \
bash script/train_memory_sft.sh
```

Camera keys are discovered automatically when `--camera-keys` is omitted. Explicit mapping is recommended so the camera order agrees with `IMAGE_PROMPTS` and `MEMORY_IMAGE_KEYS`. The converter validates state/action keys, video existence, task text, camera mapping, and dataset version before writing output.

## How visual memory works

For a current step `t`, memory indices are sampled strictly from the past:

```text
t - MEMORY_FRAMES * MEMORY_STRIDE, ..., t - 2 * MEMORY_STRIDE, t - MEMORY_STRIDE
```

Negative indices are discarded, and returned frames are ordered from oldest to newest.

The key controls are:

- `MEMORY_IMAGE_KEYS`: comma-separated camera keys used as memory. A single stable overview camera is a good starting point.
- `MEMORY_FRAMES`: number of earlier time points to request.
- `MEMORY_STRIDE`: step interval between memory time points.
- `MAX_MEMORY_IMAGES`: safety limit for `MEMORY_FRAMES × number of memory cameras`.
- `LEFT_PAD_MEMORY`: when `True`, unavailable frames at the beginning of an episode become masked slots, keeping a fixed memory layout. When `False`, early steps use fewer memory images.

For example, with `MEMORY_FRAMES=5` and `MEMORY_STRIDE=16`, step 96 uses frames `16, 32, 48, 64, 80`. The current frame is never included in memory.

Memory images are resized and encoded separately from the current views. Their pooled visual tokens replace dedicated memory placeholders in the language-model prefix. Missing left-padded slots use masked placeholders and do not contribute image features or attention.

## Start training

At minimum, set the data paths and state description:

```bash
export JSONL_DIR=/path/to/dataset/jsonl
export IMAGE_DIR=/path/to/dataset/media
export STATE_DESC=joint,joint,joint,joint,joint,joint,gripper,joint,joint,joint,joint,joint,joint,gripper

bash script/train_memory_sft.sh
```

The script uses relative repository defaults for model input and output:

- base checkpoint: `./checkpoints/DM05`
- training output: `./artifacts/checkpoints/dm05_memory_sft`

### Reproduction defaults

`train_memory_sft.sh` defaults to the RoboDojo memory training profile used for the reproduced ARX X5 results:

| Setting | Default | Meaning |
| --- | ---: | --- |
| `ACTION_MODE` | `ABSOLUTE` | Train absolute joint-position targets. Do not use `RELATIVE` for this reproduction. |
| `CHUNK_SIZE` | `50` | Predict 50 future actions per sample. |
| `MEMORY_IMAGE_KEYS` | `images_1` | Use the head/overview camera as memory. |
| `MEMORY_FRAMES` | `20` | Use 20 historical observations. |
| `MEMORY_STRIDE` | `25` | Separate memory observations by 25 dataset steps. |
| `MAX_MEMORY_IMAGES` | `20` | Fixed memory capacity for one camera. |
| `LEFT_PAD_MEMORY` | `True` | Left-pad missing episode-prefix memory with masked slots. |
| `MODEL_MAX_LENGTH` | `1536` | Token budget required by 20 memory frames. |
| `VIDEO_BACKEND` | `torchcodec` | Batched sparse MP4 decoding. |
| `AUGMENTATION_PROBABILITY` | `0.0` | Disable image augmentation for this reproduction. |
| `NPROC_PER_NODE` | `8` | Eight training processes / GPUs. |
| `PER_DEVICE_BATCH_SIZE` | `4` | Per-GPU micro batch size. |
| `GRADIENT_ACCUMULATION_STEPS` | `8` | Effective global batch is `8 × 4 × 8 = 256`. |
| `NUM_TRAIN_STEPS` | `10000` | Total optimizer steps. |
| `LEARNING_RATE` | `4e-5` | MuonAdamW base learning rate. |
| `WARMUP_STEPS` | `1000` | Learning-rate warmup. |
| `SAVE_STEPS` | `2000` | Save every 2,000 steps. |
| `USE_LORA` | `False` | Full-parameter fine-tuning. |
| attention backends | `sdpa` | LLM, vision, and action attention all use SDPA. |
| `SEED` | `42` | Training seed. |

The preflight expects eight GPUs with at least 79,000 MiB each. Set `VALIDATE_GPU_MEMORY=False` only when intentionally adapting the batch configuration to different hardware. Reducing GPU count or memory normally requires lowering `PER_DEVICE_BATCH_SIZE` and increasing gradient accumulation to preserve the effective global batch.

The defaults assume a dual-arm ARX X5 state/action vector with 14 dimensions and the descriptor:

```text
joint,joint,joint,joint,joint,joint,gripper,joint,joint,joint,joint,joint,joint,gripper
```

For another robot, explicitly change `ROBOT_TYPE`, `STATE_DESC`, `OUTPUT_ACTION_DIM`, `ACTION_MODE`, camera mapping, and possibly `CHUNK_SIZE`. The number and order of `STATE_DESC` entries must match both state and action dimensions.

Common overrides:

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

Use `DRY_RUN=1` to validate paths and print the generated command without starting training.

### W&B logging

W&B is optional and disabled unless a project name is provided. Authenticate outside the repository and never commit an API key:

```bash
wandb login
WANDB_PROJECT=dm05-memory bash script/train_memory_sft.sh
```

For non-interactive jobs, pass `WANDB_API_KEY` through the job environment or a secrets manager. Do not store it in source files or shell scripts.

## Reproduced RoboDojo memory results

The following task-level simulation results are reported for **DM0.5 / OpenDM05** on the official [RoboDojo rollout leaderboard](https://robodojo-benchmark.com/leaderboard/rollouts/OpenDM05?bench=sim). Values were checked on 2026-09-01. `Avg Score` and `Success Rate` are separate leaderboard metrics.

| Memory task | Robot | Avg Score | Success Rate |
| --- | --- | ---: | ---: |
| `cover_blocks` | `arx_x5` | 100.00 | 100.00% |
| `press_by_number` | `arx_x5` | 95.33 | 95.00% |
| `match_and_pick_from_conveyor` | `arx_x5` | 70.67 | 71.00% |

These are benchmark rollout results, not training loss or validation-set accuracy. Reproduction requires the matching RoboDojo simulator, policy checkpoint, action chunking, observation history, and evaluation configuration.

## Validation

```bash
pytest -q tests/test_memory_dataset.py
pre-commit run --all-files
```

## License and upstream

The upstream project is maintained at [dexmal/opendm](https://github.com/dexmal/opendm). This repository retains the upstream Apache-2.0 license; see `LICENSE`.
