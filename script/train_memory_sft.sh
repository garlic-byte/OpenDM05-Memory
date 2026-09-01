#!/usr/bin/env bash
# Run portable DM05 visual-memory supervised fine-tuning.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${DM05_PYTHON:-python}"
MODEL_PATH="${DM05_MODEL_PATH:-${REPO_DIR}/checkpoints/DM05}"
JSONL_DIR="${JSONL_DIR:-}"
IMAGE_DIR="${IMAGE_DIR:-}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_DIR}/artifacts/checkpoints/dm05_memory_sft}"

DATASET_NAME="${DATASET_NAME:-memory_sft}"
IMAGE_KEYS="${IMAGE_KEYS:-images_1,images_2,images_3}"
IMAGE_PROMPTS="${IMAGE_PROMPTS:-Head,Left wrist,Right wrist}"
MEMORY_IMAGE_KEYS="${MEMORY_IMAGE_KEYS:-images_1}"
MEMORY_FRAMES="${MEMORY_FRAMES:-20}"
MEMORY_STRIDE="${MEMORY_STRIDE:-25}"
MAX_MEMORY_IMAGES="${MAX_MEMORY_IMAGES:-20}"
LEFT_PAD_MEMORY="${LEFT_PAD_MEMORY:-True}"
ROBOT_TYPE="${ROBOT_TYPE:-Dual ARX5}"
STATE_DESC="${STATE_DESC:-joint,joint,joint,joint,joint,joint,gripper,joint,joint,joint,joint,joint,joint,gripper}"
ACTION_MODE="${ACTION_MODE:-ABSOLUTE}"

NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-4}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-8}"
NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-10000}"
SAVE_STEPS="${SAVE_STEPS:-2000}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-50}"
LEARNING_RATE="${LEARNING_RATE:-4e-5}"
WARMUP_STEPS="${WARMUP_STEPS:-1000}"
CHUNK_SIZE="${CHUNK_SIZE:-50}"
NUM_WORKERS="${NUM_WORKERS:-4}"
MODEL_MAX_LENGTH="${MODEL_MAX_LENGTH:-1536}"
OUTPUT_ACTION_DIM="${OUTPUT_ACTION_DIM:-14}"
VIDEO_BACKEND="${VIDEO_BACKEND:-torchcodec}"
VIDEO_DECODER_CACHE_SIZE="${VIDEO_DECODER_CACHE_SIZE:-32}"
VIDEO_SEEK_MODE="${VIDEO_SEEK_MODE:-exact}"
VIDEO_DECODER_THREADS="${VIDEO_DECODER_THREADS:-1}"
AUGMENTATION_PROBABILITY="${AUGMENTATION_PROBABILITY:-0.0}"
LLM_ATTN_IMPLEMENTATION="${LLM_ATTN_IMPLEMENTATION:-sdpa}"
VISION_ATTN_IMPLEMENTATION="${VISION_ATTN_IMPLEMENTATION:-sdpa}"
ACTION_ATTN_IMPLEMENTATION="${ACTION_ATTN_IMPLEMENTATION:-sdpa}"
LIGER_KERNEL="${LIGER_KERNEL:-False}"
USE_LORA="${USE_LORA:-False}"
SEED="${SEED:-42}"
MASTER_PORT="${MASTER_PORT:-29511}"
MIN_GPU_MEMORY_MIB="${MIN_GPU_MEMORY_MIB:-79000}"
VALIDATE_GPU_MEMORY="${VALIDATE_GPU_MEMORY:-True}"

require_value() {
    local name="$1"
    local value="$2"
    if [[ -z "${value}" ]]; then
        echo "${name} is required. See README.md for an example." >&2
        exit 2
    fi
}

require_path() {
    local label="$1"
    local path="$2"
    if [[ ! -e "${path}" ]]; then
        echo "${label} does not exist: ${path}" >&2
        exit 2
    fi
}

append_bool_switch() {
    local value="$1"
    local enabled_flag="$2"
    local disabled_flag="$3"
    case "${value}" in
        1|true|True|TRUE|yes|Yes|YES|on|On|ON) COMMAND+=("${enabled_flag}") ;;
        0|false|False|FALSE|no|No|NO|off|Off|OFF) COMMAND+=("${disabled_flag}") ;;
        *) echo "Expected a boolean, got: $1" >&2; exit 2 ;;
    esac
}

require_value JSONL_DIR "${JSONL_DIR}"
require_value IMAGE_DIR "${IMAGE_DIR}"
require_value STATE_DESC "${STATE_DESC}"
require_path "Base checkpoint" "${MODEL_PATH}"
require_path "JSONL directory" "${JSONL_DIR}"
require_path "Image directory" "${IMAGE_DIR}"

validate_gpus() {
    if [[ "${DRY_RUN:-0}" == "1" ]]; then
        return
    fi
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "nvidia-smi is required for the GPU preflight." >&2
        exit 2
    fi
    local gpu_count
    gpu_count="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | wc -l | tr -d ' ')"
    if [[ "${gpu_count}" -lt "${NPROC_PER_NODE}" ]]; then
        echo "Expected ${NPROC_PER_NODE} GPUs, found ${gpu_count}." >&2
        exit 2
    fi
    case "${VALIDATE_GPU_MEMORY}" in
        1|true|True|TRUE|yes|Yes|YES|on|On|ON)
            local memory_mib
            while read -r memory_mib; do
                if [[ "${memory_mib}" -lt "${MIN_GPU_MEMORY_MIB}" ]]; then
                    echo "GPU has ${memory_mib} MiB; reproduction default requires at least ${MIN_GPU_MEMORY_MIB} MiB." >&2
                    exit 2
                fi
            done < <(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n "${NPROC_PER_NODE}")
            ;;
    esac
}

validate_gpus

COMMAND=(
    "${REPO_DIR}/script/dm05_launcher.sh"
    --exp "${REPO_DIR}/playground/dm05_memory_sft.py"
    --task train
    --nproc_per_node "${NPROC_PER_NODE}"
    --master_port "${MASTER_PORT}"
    --use-lora "${USE_LORA}"
    --model-config.model-name-or-path "${MODEL_PATH}"
    --model-config.chunk-size "${CHUNK_SIZE}"
    --model-config.llm-attn-implementation "${LLM_ATTN_IMPLEMENTATION}"
    --model-config.vision-attn-implementation "${VISION_ATTN_IMPLEMENTATION}"
    --model-config.action-attn-implementation "${ACTION_ATTN_IMPLEMENTATION}"
    --data-config.dataset-name "${DATASET_NAME}"
    --data-config.jsonl-dir "${JSONL_DIR}"
    --data-config.image-dir "${IMAGE_DIR}"
    --data-config.image-keys "${IMAGE_KEYS}"
    --data-config.image-prompts "${IMAGE_PROMPTS}"
    --data-config.history-image-keys "${MEMORY_IMAGE_KEYS}"
    --data-config.history-frames "${MEMORY_FRAMES}"
    --data-config.history-stride "${MEMORY_STRIDE}"
    --data-config.max-history-images "${MAX_MEMORY_IMAGES}"
    --data-config.robot-type "${ROBOT_TYPE}"
    --data-config.state-desc "${STATE_DESC}"
    --data-config.action-mode "${ACTION_MODE}"
    --data-config.video-backend "${VIDEO_BACKEND}"
    --data-config.video-decoder-cache-size "${VIDEO_DECODER_CACHE_SIZE}"
    --data-config.video-seek-mode "${VIDEO_SEEK_MODE}"
    --data-config.video-decoder-threads "${VIDEO_DECODER_THREADS}"
    --data-config.augmentation-probability "${AUGMENTATION_PROBABILITY}"
    --optimizer-config.optim muon_adamw
    --optimizer-config.base-lr "${LEARNING_RATE}"
    --optimizer-config.warmup-steps "${WARMUP_STEPS}"
    --trainer-config.fsdp1 True
    --trainer-config.per-device-train-batch-size "${PER_DEVICE_BATCH_SIZE}"
    --trainer-config.gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS}"
    --trainer-config.num-train-steps "${NUM_TRAIN_STEPS}"
    --trainer-config.save-steps "${SAVE_STEPS}"
    --trainer-config.save-total-limit "${SAVE_TOTAL_LIMIT}"
    --trainer-config.dataloader-num-workers "${NUM_WORKERS}"
    --trainer-config.model-max-length "${MODEL_MAX_LENGTH}"
    --trainer-config.output-dir "${OUTPUT_DIR}"
    --trainer-config.seed "${SEED}"
    --inference-config.output-action-dim "${OUTPUT_ACTION_DIM}"
)

append_bool_switch "${LEFT_PAD_MEMORY}" \
    --data-config.left-pad-history --data-config.no-left-pad-history
append_bool_switch "${LIGER_KERNEL}" \
    --model-config.liger-kernel --model-config.no-liger-kernel

if [[ -n "${WANDB_PROJECT:-}" ]]; then
    COMMAND+=(--trainer-config.wandb-project "${WANDB_PROJECT}")
fi

printf 'Training command:'
printf ' %q' "${COMMAND[@]}"
printf '\n'
printf 'Effective global batch size: %s\n' \
    "$((NPROC_PER_NODE * PER_DEVICE_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS))"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    exit 0
fi

cd "${REPO_DIR}"
exec "${COMMAND[@]}"
