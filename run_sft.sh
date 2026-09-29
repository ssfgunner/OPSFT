#!/usr/bin/env bash
# Unified parameterized SFT entry point for OPSFT experiments.
#
#
# Required environment variables:
#   MODEL_PATH    Path to the initial (base or GRPO-checkpoint) Hugging Face model directory.
#   TRAIN_FILES   Path to the teacher-generated SFT training parquet.
#   VAL_FILES     Path to the SFT validation parquet.
#   OUTPUT_ROOT   Root directory for SFT checkpoints.
#
# Common overrides:
#   SFT_VARIANT=full|opsft_location|opsft|random_location (default: full)
#   TOTAL_TRAINING_STEPS=700  LR=1e-7  MODEL_DTYPE=fp32
#   NUM_GPUS=8  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
#   TRAIN_BATCH_SIZE=64  MICRO_BATCH_SIZE=1  MAX_LENGTH=16384
#   SAVE_FREQ=100  MAX_CKPT_TO_KEEP=8  SEED=42
#   LR_SCHEDULER=constant  LOSS_TYPE=ce
#   RESUME_MODE=auto  RESUME_FROM_PATH=/path/to/global_step_N (optional)
#   STAGES=train,eval
#
# Constrained SFT (SFT_VARIANT != full) also requires:
#   SFT_SUBSPACE_DIR  Legacy variable name for the directory containing the
#                     identified update_mask.pt and parameter_updates.pt direction artifacts.
# Or individually:
#   OPSFT_MASK_PATH / OPSFT_DIRECTION_PATH / RANDOM_MASK_PATH
#
# Code evaluation (STAGES contains "eval" and EVAL_DOMAIN=code, default for code SFT):
#   EVAL_SCRIPT  Path to the code evaluation script (default: run_code_eval.sh)
#   EVAL_STEPS   Comma-separated checkpoint steps to evaluate (default: ${TOTAL_TRAINING_STEPS})
#   BENCHMARKS   evalplus,lcb (default) or a subset
#
# Math evaluation (EVAL_DOMAIN=math):
#   Uses math_eval/run_eval_math.sh with DATASETS, GPU_GROUPS, N, etc.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-${SCRIPT_DIR}}"
PYTHON="${PYTHON:-python3}"
CALLER_DIR="$(pwd -P)"

# The inner launcher changes into ${ROOT}/verl before invoking torchrun. Resolve
# caller-provided paths here so Hydra always receives paths independent of that cd.
to_absolute_path() {
    case "$1" in
        /*) printf '%s\n' "$1" ;;
        *) printf '%s/%s\n' "${CALLER_DIR}" "$1" ;;
    esac
}

: "${MODEL_PATH:?Set MODEL_PATH to the initial Hugging Face model directory.}"
: "${TRAIN_FILES:?Set TRAIN_FILES to the SFT training parquet.}"
: "${VAL_FILES:?Set VAL_FILES to the SFT validation parquet.}"
: "${OUTPUT_ROOT:?Set OUTPUT_ROOT for SFT checkpoints.}"

MODEL_PATH="$(to_absolute_path "${MODEL_PATH}")"
TRAIN_FILES="$(to_absolute_path "${TRAIN_FILES}")"
VAL_FILES="$(to_absolute_path "${VAL_FILES}")"
OUTPUT_ROOT="$(to_absolute_path "${OUTPUT_ROOT}")"

SFT_VARIANT="${SFT_VARIANT:-full}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-700}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-999999}"
LR="${LR:-1e-7}"
MODEL_DTYPE="${MODEL_DTYPE:-fp32}"
NUM_GPUS="${NUM_GPUS:-8}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-64}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
MAX_LENGTH="${MAX_LENGTH:-16384}"
SAVE_FREQ="${SAVE_FREQ:-100}"
MAX_CKPT_TO_KEEP="${MAX_CKPT_TO_KEEP:-8}"
SEED="${SEED:-42}"
LR_SCHEDULER="${LR_SCHEDULER:-constant}"
LOSS_TYPE="${LOSS_TYPE:-ce}"
RESUME_MODE="${RESUME_MODE:-auto}"
RESUME_FROM_PATH="${RESUME_FROM_PATH:-null}"
PROMPT_KEY="${PROMPT_KEY:-prompt}"
RESPONSE_KEY="${RESPONSE_KEY:-response}"
STAGES="${STAGES:-train,eval}"
RUN_NAME="${RUN_NAME:-$(basename "${MODEL_PATH}")-sft${TOTAL_TRAINING_STEPS}-${SFT_VARIANT}-${MODEL_DTYPE}-lr${LR}}"

# Resolve mask and direction paths based on SFT_VARIANT.
MASK_PATH="null"
DIRECTION_PATH="null"
RANDOM_MASK_PATH="null"
case "${SFT_VARIANT}" in
    full)
        ;;
    opsft_location)
        : "${SFT_SUBSPACE_DIR:?SFT_SUBSPACE_DIR must be set for opsft_location variant.}"
        MASK_PATH="${SFT_SUBSPACE_DIR}/update_mask.pt"
        ;;
    opsft)
        : "${SFT_SUBSPACE_DIR:?SFT_SUBSPACE_DIR must be set for opsft variant.}"
        MASK_PATH="${SFT_SUBSPACE_DIR}/update_mask.pt"
        DIRECTION_PATH="${SFT_SUBSPACE_DIR}/parameter_updates.pt"
        ;;
    random_location)
        : "${SFT_SUBSPACE_DIR:?SFT_SUBSPACE_DIR must be set to the random-control directory for random_location.}"
        RANDOM_MASK_PATH="${SFT_SUBSPACE_DIR}/random_mask.pt"
        MASK_PATH="${RANDOM_MASK_PATH}"
        ;;
    *)
        echo "[ERROR] SFT_VARIANT must be one of: full, opsft_location, opsft, random_location." >&2
        exit 1
        ;;
esac

# Allow individual overrides for mask/direction paths.
[[ -n "${OPSFT_MASK_PATH:-}" ]] && MASK_PATH="${OPSFT_MASK_PATH}"
[[ -n "${OPSFT_DIRECTION_PATH:-}" ]] && DIRECTION_PATH="${OPSFT_DIRECTION_PATH}"
[[ -n "${RANDOM_MASK_PATH_OVERRIDE:-}" ]] && RANDOM_MASK_PATH="${RANDOM_MASK_PATH_OVERRIDE}"

[[ "${MASK_PATH}" == "null" ]] || MASK_PATH="$(to_absolute_path "${MASK_PATH}")"
[[ "${DIRECTION_PATH}" == "null" ]] || DIRECTION_PATH="$(to_absolute_path "${DIRECTION_PATH}")"
[[ "${RANDOM_MASK_PATH}" == "null" ]] || RANDOM_MASK_PATH="$(to_absolute_path "${RANDOM_MASK_PATH}")"

# Validate required files.
has_hf_weights() {
    local model_dir="$1"
    [[ -d "${model_dir}" ]] || return 1

    # Hugging Face checkpoints may store weights in a single file or in several
    # safetensors shards referenced by model.safetensors.index.json.
    [[ -f "${model_dir}/model.safetensors" || -f "${model_dir}/pytorch_model.bin" || \
       -f "${model_dir}/model.safetensors.index.json" || -f "${model_dir}/pytorch_model.bin.index.json" ]] && return 0

    find "${model_dir}" -maxdepth 1 -type f \( -name '*.safetensors' -o -name '*.bin' \) -print -quit 2>/dev/null | grep -q .
}
require_hf_model() {
    [[ -f "$1/config.json" ]] || { echo "[ERROR] Missing config.json: $1" >&2; exit 1; }
    has_hf_weights "$1" || { echo "[ERROR] Missing HF weights: $1" >&2; exit 1; }
}
require_file() {
    [[ -f "$1" ]] || { echo "[ERROR] Missing required file: $1" >&2; exit 1; }
}

require_hf_model "${MODEL_PATH}"
require_file "${TRAIN_FILES}"
require_file "${VAL_FILES}"
if [[ "${SFT_VARIANT}" == "opsft" || "${SFT_VARIANT}" == "opsft_location" || "${SFT_VARIANT}" == "random_location" ]]; then
    [[ "${MASK_PATH}" != "null" ]] && require_file "${MASK_PATH}"
    [[ "${DIRECTION_PATH}" != "null" ]] && require_file "${DIRECTION_PATH}"
    [[ "${RANDOM_MASK_PATH}" != "null" ]] && require_file "${RANDOM_MASK_PATH}"
fi

case "${RESUME_MODE}" in
    auto|disable|resume_path) ;;
    *) echo "[ERROR] RESUME_MODE must be auto, disable, or resume_path; got ${RESUME_MODE}." >&2; exit 1 ;;
esac
if [[ "${RESUME_MODE}" == "resume_path" ]]; then
    [[ "${RESUME_FROM_PATH}" != "null" ]] || { echo "[ERROR] RESUME_FROM_PATH is required when RESUME_MODE=resume_path." >&2; exit 1; }
    RESUME_FROM_PATH="$(to_absolute_path "${RESUME_FROM_PATH}")"
    [[ -d "${RESUME_FROM_PATH}" ]] || { echo "[ERROR] Missing resume checkpoint: ${RESUME_FROM_PATH}" >&2; exit 1; }
fi

# Validation: GPU count matches.
VISIBLE_GPU_COUNT="$(tr ',' '\n' <<< "${CUDA_VISIBLE_DEVICES}" | sed '/^$/d' | wc -l)"
[[ "${VISIBLE_GPU_COUNT}" -eq "${NUM_GPUS}" ]] || {
    echo "[ERROR] NUM_GPUS=${NUM_GPUS}, but CUDA_VISIBLE_DEVICES exposes ${VISIBLE_GPU_COUNT} GPU(s)." >&2
    exit 1
}

SFT_OUTPUT_DIR="${OUTPUT_ROOT}/${SFT_VARIANT}"
MODEL_TAG="${RUN_NAME}"
EVAL_DOMAIN="${EVAL_DOMAIN:-code}"

# ── Stage: Train ─────────────────────────────────────────────────────
has_stage() {
    local requested=" ${STAGES//,/ } "
    [[ "${requested}" == *" $1 "* ]]
}

checkpoint_dir_for_step() {
    local step="$1"
    local candidate
    candidate="${SFT_OUTPUT_DIR}/global_step_${step}"
    [[ "${step}" =~ ^[1-9][0-9]*$ ]] || return 1
    [[ -d "${candidate}" && -f "${candidate}/huggingface/config.json" ]] || return 1
    printf '%s\n' "${candidate}"
}

merged_model_dir_for_checkpoint() {
    printf '%s/huggingface_merged\n' "$1"
}

fsdp_world_size() {
    local checkpoint_dir="$1"
    "${PYTHON}" - "${checkpoint_dir}/fsdp_config.json" <<'PY'
import json, sys
with open(sys.argv[1], encoding="utf-8") as f:
    ws = json.load(f).get("world_size")
if not isinstance(ws, int) or ws < 1:
    raise ValueError(f"Invalid FSDP world_size in {sys.argv[1]}: {ws!r}")
print(ws)
PY
}

require_complete_fsdp_checkpoint() {
    local checkpoint_dir="$1" world_size rank
    require_file "${checkpoint_dir}/huggingface/config.json"
    require_file "${checkpoint_dir}/fsdp_config.json"
    world_size="$(fsdp_world_size "${checkpoint_dir}")"
    for ((rank = 0; rank < world_size; rank++)); do
        require_file "${checkpoint_dir}/model_world_size_${world_size}_rank_${rank}.pt"
    done
}

merge_checkpoint() {
    local checkpoint_dir="$1" merged_dir temporary_dir
    merged_dir="$(merged_model_dir_for_checkpoint "${checkpoint_dir}")"
    if [[ -f "${merged_dir}/config.json" ]] && has_hf_weights "${merged_dir}"; then
        echo "SKIP FSDP merge; valid merged model already exists: ${merged_dir}"
        return
    fi
    [[ ! -e "${merged_dir}" ]] || {
        echo "[ERROR] Incomplete merged-model directory already exists: ${merged_dir}" >&2
        echo "[ERROR] Refusing to overwrite it. Inspect or remove it, then rerun." >&2
        exit 1
    }
    require_complete_fsdp_checkpoint "${checkpoint_dir}"
    temporary_dir="${merged_dir}.tmp.$$"
    [[ ! -e "${temporary_dir}" ]] || { echo "[ERROR] Temp merge dir exists: ${temporary_dir}" >&2; exit 1; }
    echo "===== Merge FSDP checkpoint: ${checkpoint_dir} ====="
    (
        cd "${ROOT}/verl"
        PYTHONPATH="${ROOT}/verl${PYTHONPATH:+:${PYTHONPATH}}" \
            "${PYTHON}" -m verl.model_merger merge --backend fsdp \
                --local_dir "${checkpoint_dir}" --target_dir "${temporary_dir}" --use_cpu_initialization
    )
    require_hf_model "${temporary_dir}"
    mv "${temporary_dir}" "${merged_dir}"
    require_hf_model "${merged_dir}"
    echo "Merged inference model: ${merged_dir}"
}

train() {
    local completed_checkpoint
    mkdir -p "${OUTPUT_ROOT}/logs"

    if completed_checkpoint="$(checkpoint_dir_for_step "${TOTAL_TRAINING_STEPS}")" && \
        [[ -f "${completed_checkpoint}/fsdp_config.json" ]]; then
        echo "SKIP completed training: ${completed_checkpoint}"
        merge_checkpoint "${completed_checkpoint}"
        return
    fi

    echo "===== Train: ${RUN_NAME} ====="
    echo "Initial model: ${MODEL_PATH}"
    echo "Train parquet: ${TRAIN_FILES}"
    echo "Validation parquet: ${VAL_FILES}"
    echo "Variant: ${SFT_VARIANT}"
    [[ "${SFT_VARIANT}" != "full" ]] && echo "Constraint direction mask: ${MASK_PATH}" && \
        [[ "${DIRECTION_PATH}" != "null" ]] && echo "Constraint direction: ${DIRECTION_PATH}"
    echo "Training: ${MODEL_DTYPE}, ${TOTAL_TRAINING_STEPS} steps, LR=${LR}, scheduler=${LR_SCHEDULER}"
    echo "Resume: ${RESUME_MODE}"
    [[ "${RESUME_FROM_PATH}" == "null" ]] || echo "Resume checkpoint: ${RESUME_FROM_PATH}"
    echo "Output: ${SFT_OUTPUT_DIR}"

    set +e
    ROOT_DIR="${ROOT}" \
    MODEL_PATH="${MODEL_PATH}" MODEL_TAG="${MODEL_TAG}" OUTPUT_ROOT="${OUTPUT_ROOT}" \
    TRAIN_FILES="${TRAIN_FILES}" VAL_FILES="${VAL_FILES}" MODEL_DTYPE="${MODEL_DTYPE}" \
    OPSFT_MASK_PATH="${MASK_PATH}" \
    OPSFT_DIRECTION_PATH="${DIRECTION_PATH}" \
    RANDOM_MASK_PATH="${RANDOM_MASK_PATH}" \
    SFT_EXPECTED_MODEL="${MODEL_PATH}" SFT_EXPECTED_MODEL_TAG="${MODEL_TAG}" \
    SFT_EXPECTED_OUTPUT_ROOT="${SFT_OUTPUT_DIR}" \
    NUM_GPUS="${NUM_GPUS}" CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
    TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE}" MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE}" MAX_LENGTH="${MAX_LENGTH}" \
    TOTAL_EPOCHS="${TOTAL_EPOCHS}" TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS}" LR="${LR}" \
    SAVE_FREQ="${SAVE_FREQ}" MAX_CKPT_TO_KEEP="${MAX_CKPT_TO_KEEP}" SEED="${SEED}" LOSS_TYPE="${LOSS_TYPE}" \
    PROMPT_KEY="${PROMPT_KEY}" RESPONSE_KEY="${RESPONSE_KEY}" \
        bash "${ROOT}/verl/examples/sft/run_sft.sh" "${SFT_VARIANT}" \
            "trainer.resume_mode=${RESUME_MODE}" \
            "trainer.resume_from_path=${RESUME_FROM_PATH}" \
            "trainer.total_epochs=${TOTAL_EPOCHS}" \
            "trainer.total_training_steps=${TOTAL_TRAINING_STEPS}" \
            "optim.lr_scheduler=${LR_SCHEDULER}" \
            2>&1 | tee "${OUTPUT_ROOT}/logs/${SFT_VARIANT}_sft${TOTAL_TRAINING_STEPS}steps_${MODEL_DTYPE}_lr${LR}.log"
    local train_status="${PIPESTATUS[0]}"
    set -e
    (( train_status == 0 )) || { echo "[ERROR] SFT failed; see log." >&2; exit "${train_status}"; }

    completed_checkpoint="$(checkpoint_dir_for_step "${TOTAL_TRAINING_STEPS}")" || {
        echo "[ERROR] No final step-${TOTAL_TRAINING_STEPS} checkpoint was saved under ${SFT_OUTPUT_DIR}." >&2
        echo "[ERROR] If the dataset is smaller than expected, increase TOTAL_EPOCHS before restarting." >&2
        exit 1
    }
    require_complete_fsdp_checkpoint "${completed_checkpoint}"
    merge_checkpoint "${completed_checkpoint}"
}

# ── Stage: Evaluate ──────────────────────────────────────────────────
evaluate() {
    local checkpoint_dir model_dir step run_label model_link

    # Parse EVAL_STEPS (default: just the final step).
    local eval_steps="${EVAL_STEPS:-${TOTAL_TRAINING_STEPS}}"
    local step
    local eval_script

    if [[ "${EVAL_DOMAIN}" == "code" ]]; then
        eval_script="${EVAL_SCRIPT:-${ROOT}/run_code_eval.sh}"
    else
        eval_script="${EVAL_SCRIPT:-${ROOT}/math_eval/run_eval_math.sh}"
    fi
    [[ -f "${eval_script}" ]] || { echo "[ERROR] Missing evaluator: ${eval_script}" >&2; exit 1; }

    while IFS= read -r step; do
        [[ "${step}" =~ ^[1-9][0-9]*$ ]] || { echo "[ERROR] Invalid eval step: ${step}" >&2; exit 1; }
        checkpoint_dir="$(checkpoint_dir_for_step "${step}")" || {
            echo "[ERROR] Requested checkpoint step ${step} is missing or incomplete under ${SFT_OUTPUT_DIR}." >&2
            exit 1
        }
        merge_checkpoint "${checkpoint_dir}"
        model_dir="$(merged_model_dir_for_checkpoint "${checkpoint_dir}")"
        require_hf_model "${model_dir}"
        run_label="${RUN_NAME}_step${step}"

        echo "===== Evaluate ${run_label} ====="

        if [[ "${EVAL_DOMAIN}" == "code" ]]; then
            ROOT="${ROOT}" \
            MODEL_DIR="${model_dir}" \
            MODEL_LINK="${EVAL_MODEL_LINK_PREFIX:-/tmp/opsft_eval}_${step}" \
            RUN_LABEL="${run_label}" \
            BENCHMARKS="${BENCHMARKS:-evalplus,lcb}" \
            EVALPLUS_RESULTS_ROOT="${EVALPLUS_RESULTS_ROOT:-${ROOT}/code_eval/evalplus_results/${run_label}}" \
            LCB_OUTPUT_ROOT="${LCB_OUTPUT_ROOT:-${ROOT}/code_eval/lcb_outputs/${run_label}}" \
                bash "${eval_script}" \
                    2>&1 | tee "${OUTPUT_ROOT}/logs/${SFT_VARIANT}_step${step}_evaluation.log"
        else
            DATASETS="${DATASETS:-aime24 aime25 hmmt25_feb hmmt25_nov}" \
            GPU_GROUPS="${GPU_GROUPS:-0,1 2,3 4,5 6,7}" \
            N="${N:-32}" \
            MAX_TOKENS="${MAX_TOKENS:-16384}" \
            MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}" \
            MAX_NUM_SEQS="${MAX_NUM_SEQS:-256}" \
            TEMPERATURE="${TEMPERATURE:-1.0}" \
            TOP_P="${TOP_P:-1.0}" \
            EVAL_SEED="${EVAL_SEED:-42}" \
            ENABLE_THINKING="${ENABLE_THINKING:-0}" \
            OUTPUT_TAG="${run_label}" \
            STAGGER_SECONDS="${STAGGER_SECONDS:-30}" \
                bash "${eval_script}" "${model_dir}" "${run_label}" \
                    2>&1 | tee "${OUTPUT_ROOT}/logs/${SFT_VARIANT}_step${step}_evaluation.log"
        fi
    done <<< "${eval_steps//,/$'\n'}"
}

has_stage train && train
has_stage eval && evaluate

if ! has_stage train && ! has_stage eval; then
    echo "No stage selected (STAGES=${STAGES}); nothing to do."
fi

echo "Completed requested stages for ${RUN_NAME}."
