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

## Train or fine-tune DM05 on LIBERO

The 97.95% result below uses the released `Dexmal/DM05-libero` checkpoint and
does not include a local training run. To train a new LIBERO policy, start from
the base `Dexmal/DM05` checkpoint and use the `libero_pi0_all` training split.
The released `DM05-libero` checkpoint is an evaluation checkpoint, not the base
checkpoint used by this recipe.

The built-in LIBERO configuration uses two images (`Head` and `Left wrist`), an
8-dimensional state, a 7-dimensional absolute action, and an action chunk size
of 10. Training data and base weights are stored outside Git history.

### 1. Prepare data and the base checkpoint

Run from the repository root on the `libero` branch. The same Docker image used
for inference can prepare the files:

```bash
git checkout libero
docker pull dexmal/opendm:latest

docker run --rm \
  --gpus all \
  --network host \
  --ipc=host \
  -v "$PWD":/app/opendm \
  -w /app/opendm \
  dexmal/opendm:latest \
  bash -lc 'source /opt/conda/etc/profile.d/conda.sh && \
    conda activate opendm && \
    pip install -e . && \
    script/libero_runner.sh dataset && \
    script/libero_runner.sh model'
```

This downloads `Dexmal/libero` and organizes its files under `data/libero`,
then downloads `Dexmal/DM05` under `checkpoints/DM05`. If an earlier download
was interrupted, rerun the affected `dataset` or `model` subcommand with
`--force`; Hugging Face resumes files already present in its cache.

Before training, confirm these paths exist:

```text
data/libero/libero_pi0_all/jsonl/
data/libero/libero_pi0_all/image/
checkpoints/DM05/config.json
checkpoints/DM05/model.safetensors
```

### 2. Full-parameter fine-tuning

This is the reference full-training configuration: 8 GPUs, per-GPU batch 4,
global batch 32, 100,000 optimizer steps, learning rate `2e-5`, and checkpoints
every 10,000 steps. The LIBERO entry point enables FSDP for multi-GPU training.

```bash
docker run -d --rm \
  --name dm05-libero-train \
  --gpus all \
  --network host \
  --ipc=host \
  -v "$PWD":/app/opendm \
  -w /app/opendm \
  dexmal/opendm:latest \
  bash -lc 'source /opt/conda/etc/profile.d/conda.sh && \
    conda activate opendm && \
    pip install -e . && \
    script/libero_runner.sh train \
      --nproc-per-node 8 \
      -- \
      --model-config.model-name-or-path ./checkpoints/DM05 \
      --model-config.chunk-size 10 \
      --optimizer-config.base-lr 2e-5 \
      --optimizer-config.warmup-steps 1000 \
      --trainer-config.output-dir ./user_checkpoints/dm05_libero_full \
      --trainer-config.per-device-train-batch-size 4 \
      --trainer-config.gradient-accumulation-steps 1 \
      --trainer-config.num-train-steps 100000 \
      --trainer-config.save-steps 10000'

docker logs -f dm05-libero-train
```

Because the repository is bind-mounted, checkpoints remain under
`user_checkpoints/dm05_libero_full` after the container exits. Starting the
same command again automatically resumes from the latest `checkpoint-*` in
that output directory. Use a new output directory when changing the base
model, dataset, action mode, or chunk size.

GPU memory requirements depend on GPU type and software versions. If batch 4
does not fit, lower `per-device-train-batch-size` and increase
`gradient-accumulation-steps` so their product, multiplied by the GPU count,
remains 32. A short smoke run can use a separate output directory and a small
`num-train-steps`, but its checkpoint is not meaningful for SR comparison.

### 3. LoRA alternative

For lower-memory adaptation, use the dedicated LoRA entry point. Its reference
defaults are rank 32, alpha 16, learning rate `5e-4`, 50,000 steps, and a save
interval of 10,000 steps:

```bash
script/dm05_launcher.sh \
  --exp playground/dm05_libero_lora.py \
  --nproc_per_node 8 \
  --task train \
  --data-config.jsonl-dir ./data/libero/libero_pi0_all \
  --data-config.image-dir ./data/libero/libero_pi0_all/image \
  --model-config.model-name-or-path ./checkpoints/DM05 \
  --trainer-config.output-dir ./user_checkpoints/dm05_libero_lora \
  --trainer-config.num-train-steps 50000 \
  --trainer-config.save-steps 10000
```

Run this command inside the same OpenDM container if training is Docker-based.
For multi-GPU LoRA, the `checkpoint-*` directories are the canonical artifacts
for evaluation. More details are in
[`docs/en/dm05_libero_lora_training.md`](docs/en/dm05_libero_lora_training.md).

### 4. Evaluate a trained checkpoint

Use the LIBERO evaluation procedure below, but replace the released checkpoint
path in the policy-service command:

- Full fine-tuning: keep `--exp playground/dm05_libero.py` and set
  `--model-config.model-name-or-path` to the selected `checkpoint-*` directory.
- LoRA: use `--exp playground/dm05_libero_lora.py` and set the model path to the
  selected LoRA `checkpoint-*` directory.

Keep chunk size 10, output action dimension 7, and the two image prompts
unchanged. Each training checkpoint must retain its matching `norm_stats.json`.

## Reproduced LIBERO evaluation results

The official `Dexmal/DM05-libero` checkpoint was evaluated on 2026-09-06 with the
[Dexbotic benchmark](https://github.com/dexmal/dexbotic-benchmark) at commit
`e399519`. No additional fine-tuning was performed for these results.

The evaluation covers all four standard LIBERO suites, with 10 tasks per suite
and 50 episodes per task (2,000 episodes in total). It uses `seed=7` and
`replan_steps=10`.

| LIBERO suite | Successful episodes | Success rate |
| --- | ---: | ---: |
| Spatial (`libero_spatial`) | 496 / 500 | 99.2% |
| Object (`libero_object`) | 496 / 500 | 99.2% |
| Goal (`libero_goal`) | 493 / 500 | 98.6% |
| Long-horizon (`libero_10`) | 474 / 500 | 94.8% |
| **Overall** | **1,959 / 2,000** | **97.95%** |

All 2,000 rollout videos were generated successfully with no empty files. The
untracked evaluation artifacts on the reproduction machine are stored under
`/data/wudi/code_v6/experiment_runs/opendm-libero-full-sr-20260906`, with the
aggregate machine-readable results in `summary.json`.

### Reproduce the LIBERO evaluation

The evaluator and policy run as separate processes. The policy exposes an HTTP
service on port `7891`; the Dexbotic benchmark runs the LIBERO simulator and
sends the two camera images plus robot state to that service.

#### 1. Download the checkpoint

Run from the OpenDM repository root:

```bash
git checkout libero

hf download Dexmal/DM05-libero \
  --local-dir ./checkpoints/DM05-libero
```

The checkpoint used for the table above had this model-file digest:

```text
model.safetensors SHA-256:
575d0d8e0f75822e95f7adf3a5e62a7c331da0b82e6fc3efeea19ef1b927353f
```

#### 2. Start the DM05 policy service

The following uses the recommended OpenDM image and GPU 0. The repository is
mounted so the `libero` branch code and downloaded checkpoint are visible in
the container.

```bash
docker pull dexmal/opendm:latest

docker run -d --rm \
  --name dm05-libero-server \
  --gpus '"device=0"' \
  --network host \
  --shm-size=16g \
  -v "$PWD":/app/opendm \
  -w /app/opendm \
  dexmal/opendm:latest \
  bash -lc 'source /opt/conda/etc/profile.d/conda.sh && \
    conda activate opendm && \
    pip install -e . && \
    script/dm05_launcher.sh \
      --exp playground/dm05_libero.py \
      --task inference \
      --model-config.model-name-or-path ./checkpoints/DM05-libero \
      --model-config.chunk-size 10 \
      --inference-config.output-action-dim 7 \
      --inference-config.image-prompts "Head" "Left wrist" \
      --inference-config.port 7891'

docker logs -f dm05-libero-server
```

Wait until Flask reports that it is listening on port `7891`. Keep this
container running while evaluating; `Ctrl-C` exits log-following without
stopping the container. This setup uses two images, an
8-dimensional state, a 7-dimensional action, and action chunks of length 10.

#### 3. Prepare the Dexbotic LIBERO benchmark

In a separate directory:

```bash
git clone https://github.com/dexmal/dexbotic-benchmark.git
cd dexbotic-benchmark
git checkout e399519
git submodule update --init --recursive libero
docker pull dexmal/dexbotic_benchmark
```

Set `evaluation/configs/libero/example_dm05_libero.yaml` to:

```yaml
benchmark: libero_spatial
num_trails_per_task: 50
num_steps_wait: 10
seed: 7

base_url: http://127.0.0.1:7891
api_style: v1
replan_step: 10

send_state: true
send_image:
  - image
  - wrist_image
discrete_gripper: false
use_text_template: false

output_dir: results/dm05_libero_spatial
```

`num_trails_per_task` is the spelling used by the benchmark. Run the evaluator
on a different GPU when possible; this example assigns GPU 1 and uses EGL for
headless rendering:

```bash
docker run --rm \
  --gpus '"device=1"' \
  --network host \
  -e EGL_PLATFORM=device \
  -e PYOPENGL_PLATFORM=egl \
  -v "$PWD":/workspace \
  -w /workspace \
  dexmal/dexbotic_benchmark \
  bash /workspace/scripts/env_sh/libero.sh \
    /workspace/evaluation/configs/libero/example_dm05_libero.yaml
```

Repeat the run with `benchmark` and `output_dir` changed for
`libero_object`, `libero_goal`, and `libero_10`. Each suite contains 10 tasks,
so four suites at 50 episodes per task produce 2,000 episodes. Every output
directory contains `results.json`, the resolved `config.yaml`, an evaluation
log, and rollout videos. Suite and overall SR are computed as
`successful_episodes / total_episodes`.

#### Exact setup used for the reported result

The reported run used eight RTX 3090 GPUs and the local
`opendm:libero-cu124-egl` image (CUDA 12.4, PyTorch 2.6.0, plus EGL/Mesa and
FFmpeg). Eight policy services listened on ports `7891` through `7898`. Each of
the four suites was split into task ranges `[0, 5)` and `[5, 10)`, giving eight
evaluator workers and 250 episodes per worker.

Dexbotic commit `e399519` was patched locally to make its task loop honor
`task_start` and `task_end`. This only divided the existing task loop for
parallel execution; it did not change simulator behavior, task horizons,
success conditions, observations, actions, seeds, or episode counts. The stock
single-suite procedure above does not require this patch and evaluates the same
protocol sequentially. The eight shard `results.json` files were merged by
summing successes and episodes, yielding 1,959 / 2,000 = 97.95%.

Stop the policy service after evaluation:

```bash
docker stop dm05-libero-server
```

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
