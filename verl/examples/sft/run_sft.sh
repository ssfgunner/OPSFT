#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <full|opsft_location|opsft|random_location> [Hydra overrides...]" >&2
    exit 1
fi

VARIANT="$1"
shift

SCRIPT_DIR_BC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${ROOT_DIR:-$(cd "${SCRIPT_DIR_BC}/../../.." && pwd)}"
VERL_DIR="${ROOT_DIR}/verl"
MODEL_PATH="${MODEL_PATH:?MODEL_PATH must be explicitly set by the caller}"
MODEL_TAG="${MODEL_TAG:-$(basename "${MODEL_PATH}")}"
TRAIN_FILES="${TRAIN_FILES:?Set TRAIN_FILES to the teacher-generated SFT parquet file}"
VAL_FILES="${VAL_FILES:?Set VAL_FILES to the SFT validation parquet file}"
NUM_GPUS="${NUM_GPUS:-8}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-64}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
MAX_LENGTH="${MAX_LENGTH:-16384}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-3000}"
# Upper bound of the dataloader loop. Explicit total_training_steps controls
# exact early stopping when it is not null.
TOTAL_EPOCHS="${TOTAL_EPOCHS:-5}"
LR="${LR:-2e-6}"
SAVE_FREQ="${SAVE_FREQ:-100}"
MAX_CKPT_TO_KEEP="${MAX_CKPT_TO_KEEP:-8}"
SEED="${SEED:-42}"
PROMPT_KEY="${PROMPT_KEY:-prompt}"
RESPONSE_KEY="${RESPONSE_KEY:-response}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT_DIR}/checkpoints/opsft_sft}"
LOSS_TYPE="${LOSS_TYPE:-ce}"
OPSFT_MASK_PATH="${OPSFT_MASK_PATH:-${ROOT_DIR}/analysis/step_180_top0.05/update_mask.pt}"
OPSFT_DIRECTION_PATH="${OPSFT_DIRECTION_PATH:-${ROOT_DIR}/analysis/step_180_top0.05/parameter_updates.pt}"
RANDOM_MASK_PATH="${RANDOM_MASK_PATH:-${ROOT_DIR}/analysis/random_controls_v2/random_matched_top0.05_seed42/random_mask.pt}"

case "${VARIANT}" in
    full)
        MASK_PATH="null"
        DIRECTION_PATH="null"
        ;;
    opsft_location)
        MASK_PATH="${OPSFT_MASK_PATH}"
        DIRECTION_PATH="null"
        ;;
    opsft)
        MASK_PATH="${OPSFT_MASK_PATH}"
        DIRECTION_PATH="${OPSFT_DIRECTION_PATH}"
        ;;
    random_location)
        MASK_PATH="${RANDOM_MASK_PATH}"
        DIRECTION_PATH="null"
        ;;
    *)
        echo "Unknown variant: ${VARIANT}" >&2
        exit 1
        ;;
esac

OUTPUT_DIR="${OUTPUT_ROOT}/${VARIANT}"

# Fail fast if a wrapper accidentally passes stale model/output metadata.
if [[ -n "${SFT_EXPECTED_MODEL:-}" && "${MODEL_PATH}" != "${SFT_EXPECTED_MODEL}" ]]; then
    echo "[ERROR] MODEL_PATH override mismatch: expected ${SFT_EXPECTED_MODEL}, got ${MODEL_PATH}" >&2
    exit 1
fi
if [[ -n "${SFT_EXPECTED_MODEL_TAG:-}" && "${MODEL_TAG}" != "${SFT_EXPECTED_MODEL_TAG}" ]]; then
    echo "[ERROR] MODEL_TAG override mismatch: expected ${SFT_EXPECTED_MODEL_TAG}, got ${MODEL_TAG}" >&2
    exit 1
fi
if [[ -n "${SFT_EXPECTED_OUTPUT_ROOT:-}" && "${OUTPUT_DIR}" != "${SFT_EXPECTED_OUTPUT_ROOT}" ]]; then
    echo "[ERROR] OUTPUT_ROOT override mismatch: expected ${SFT_EXPECTED_OUTPUT_ROOT}, got ${OUTPUT_DIR}" >&2
    exit 1
fi
[[ -f "${MODEL_PATH}/config.json" ]] || {
    echo "[ERROR] Model config not found: ${MODEL_PATH}" >&2
    exit 1
}

mkdir -p "${OUTPUT_DIR}"
cd "${VERL_DIR}"

cmd=(
    torchrun --standalone --nnodes=1 --nproc_per_node="${NUM_GPUS}"
    -m verl.trainer.fsdp_sft_trainer
    "data.train_files=${TRAIN_FILES}"
    "data.val_files=${VAL_FILES}"
    "data.prompt_key=${PROMPT_KEY}"
    "data.response_key=${RESPONSE_KEY}"
    "data.train_batch_size=${TRAIN_BATCH_SIZE}"
    "data.micro_batch_size_per_gpu=${MICRO_BATCH_SIZE}"
    "data.max_length=${MAX_LENGTH}"
    "data.truncation=right"
    "+data.apply_chat_template_kwargs.enable_thinking=false"
    "model.partial_pretrain=${MODEL_PATH}"
    "model.strategy=fsdp2"
    "model.fsdp_config.model_dtype=${MODEL_DTYPE:-bf16}"
    "model.opsft_mask=${MASK_PATH}"
    "model.opsft_direction=${DIRECTION_PATH}"
    "loss.type=${LOSS_TYPE}"
    "model.enable_gradient_checkpointing=true"
    "optim.lr=${LR}"
    "optim.lr_warmup_steps_ratio=0.0"
    "trainer.default_local_dir=${OUTPUT_DIR}"
    "trainer.project_name=opsft"
    "trainer.experiment_name=${MODEL_TAG}-${VARIANT}"
    "trainer.total_training_steps=${TOTAL_TRAINING_STEPS}"
    "trainer.total_epochs=${TOTAL_EPOCHS}"
    "trainer.save_freq=${SAVE_FREQ}"
    "trainer.test_freq=-1"
    "trainer.seed=${SEED}"
    "trainer.logger=[console]"
    "trainer.checkpoint.save_contents=[model,optimizer,extra,hf_model]"
    "trainer.max_ckpt_to_keep=${MAX_CKPT_TO_KEEP}"
    # "trainer.resume_mode=disable"
)

cmd+=("$@")
printf 'Running:'
printf ' %q' "${cmd[@]}"
printf '\n'
exec "${cmd[@]}"
