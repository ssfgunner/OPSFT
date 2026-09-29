#!/usr/bin/env bash
# Parameterized GRPO trainer. It works for code or math data and any local Hugging Face model.
#
# Required: BASE_MODEL, TRAIN_FILES, VAL_FILES, OUTPUT_ROOT.
# Common overrides: GRPO_STEPS=150, LR=1e-6, NUM_GPUS=8,
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7, ROLLOUT_TP_SIZE=4.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-${SCRIPT_DIR}}"
PYTHON="${PYTHON:-python3}"

: "${BASE_MODEL:?Set BASE_MODEL to a local Hugging Face base model.}"
: "${TRAIN_FILES:?Set TRAIN_FILES to a GRPO training parquet.}"
: "${VAL_FILES:?Set VAL_FILES to a comma-separated list of validation parquets.}"
: "${OUTPUT_ROOT:?Set OUTPUT_ROOT for GRPO checkpoints.}"

GRPO_STEPS="${GRPO_STEPS:-150}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-999999}"
NUM_GPUS="${NUM_GPUS:-8}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-128}"
PPO_MICRO_BATCH_SIZE="${PPO_MICRO_BATCH_SIZE:-1}"
ROLLOUT_TP_SIZE="${ROLLOUT_TP_SIZE:-4}"
ROLLOUT_N="${ROLLOUT_N:-8}"
LR="${LR:-1e-6}"
SAVE_FREQ="${SAVE_FREQ:-50}"
TEST_FREQ="${TEST_FREQ:-999999}"
VAL_MAX_SAMPLES="${VAL_MAX_SAMPLES:-32}"
VAL_ROLLOUT_N="${VAL_ROLLOUT_N:-1}"
SEED="${SEED:-42}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-2048}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-16384}"
PPO_MAX_TOKEN_LEN="${PPO_MAX_TOKEN_LEN:-32768}"
ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.6}"
ENABLE_THINKING="${ENABLE_THINKING:-0}"
RESUME_MODE="${RESUME_MODE:-disable}"
RESUME_FROM_PATH="${RESUME_FROM_PATH:-}"
RUN_NAME="${RUN_NAME:-$(basename "${BASE_MODEL}")-grpo-${GRPO_STEPS}steps-seed${SEED}}"
PROJECT_NAME="${PROJECT_NAME:-opsft-grpo}"

[[ -f "${BASE_MODEL}/config.json" ]] || { echo "[ERROR] Missing base model: ${BASE_MODEL}" >&2; exit 1; }
[[ -f "${TRAIN_FILES}" ]] || { echo "[ERROR] Missing GRPO train parquet: ${TRAIN_FILES}" >&2; exit 1; }
IFS=',' read -r -a VAL_PATHS <<< "${VAL_FILES}"
(( ${#VAL_PATHS[@]} > 0 )) || { echo "[ERROR] VAL_FILES cannot be empty." >&2; exit 1; }
for val_path in "${VAL_PATHS[@]}"; do
    [[ -f "${val_path}" ]] || { echo "[ERROR] Missing GRPO validation parquet: ${val_path}" >&2; exit 1; }
done
[[ "${ENABLE_THINKING}" == "0" || "${ENABLE_THINKING}" == "1" ]] || { echo "[ERROR] ENABLE_THINKING must be 0 or 1." >&2; exit 1; }

VISIBLE_GPU_COUNT="$(tr ',' '\n' <<< "${CUDA_VISIBLE_DEVICES}" | sed '/^$/d' | wc -l)"
[[ "${VISIBLE_GPU_COUNT}" -eq "${NUM_GPUS}" ]] || {
    echo "[ERROR] NUM_GPUS=${NUM_GPUS}, but CUDA_VISIBLE_DEVICES exposes ${VISIBLE_GPU_COUNT} GPU(s)." >&2
    exit 1
}
(( NUM_GPUS % ROLLOUT_TP_SIZE == 0 )) || {
    echo "[ERROR] NUM_GPUS=${NUM_GPUS} must be divisible by ROLLOUT_TP_SIZE=${ROLLOUT_TP_SIZE}." >&2
    exit 1
}

has_hf_weights() {
    [[ -d "$1" ]] && [[ -f "$1/model.safetensors" || -f "$1/pytorch_model.bin" || -n "$(find "$1" -maxdepth 1 -type f \( -name '*.safetensors' -o -name '*.bin' \) -print -quit 2>/dev/null)" ]]
}
require_file() {
    [[ -f "$1" ]] || { echo "[ERROR] Missing required file: $1" >&2; exit 1; }
}
validate_resume_checkpoint() {
    local checkpoint_dir="$1" actor_dir kind count
    [[ -d "${checkpoint_dir}" ]] || { echo "[ERROR] Resume checkpoint does not exist: ${checkpoint_dir}" >&2; exit 1; }
    require_file "${checkpoint_dir}/data.pt"
    actor_dir="${checkpoint_dir}/actor"
    require_file "${actor_dir}/huggingface/config.json"
    require_file "${actor_dir}/fsdp_config.json"
    for kind in model optim extra_state; do
        count="$(find "${actor_dir}" -maxdepth 1 -type f -name "${kind}_world_size_${NUM_GPUS}_rank_*.pt" | wc -l)"
        [[ "${count}" -eq "${NUM_GPUS}" ]] || {
            echo "[ERROR] Resume checkpoint has ${count}/${NUM_GPUS} ${kind} state shards: ${checkpoint_dir}" >&2
            exit 1
        }
    done
}

RESUME_ARGS=()
case "${RESUME_MODE}" in
    disable)
        [[ -z "${RESUME_FROM_PATH}" ]] || { echo "[ERROR] RESUME_FROM_PATH requires auto or resume_path mode." >&2; exit 1; }
        ;;
    auto)
        if [[ -z "${RESUME_FROM_PATH}" ]]; then
            LATEST_CHECKPOINT_FILE="${OUTPUT_ROOT}/latest_checkpointed_iteration.txt"
            require_file "${LATEST_CHECKPOINT_FILE}"
            RESUME_FROM_PATH="${OUTPUT_ROOT}/global_step_$(< "${LATEST_CHECKPOINT_FILE}")"
        fi
        validate_resume_checkpoint "${RESUME_FROM_PATH}"
        RESUME_ARGS=("trainer.resume_from_path=${RESUME_FROM_PATH}")
        ;;
    resume_path)
        [[ -n "${RESUME_FROM_PATH}" ]] || { echo "[ERROR] resume_path mode requires RESUME_FROM_PATH." >&2; exit 1; }
        validate_resume_checkpoint "${RESUME_FROM_PATH}"
        RESUME_ARGS=("trainer.resume_from_path=${RESUME_FROM_PATH}")
        ;;
    *) echo "[ERROR] RESUME_MODE must be disable, auto, or resume_path." >&2; exit 1 ;;
esac

ACTOR_DIR="${OUTPUT_ROOT}/global_step_${GRPO_STEPS}/actor"
MERGED_MODEL_DIR="${ACTOR_DIR}/huggingface_merged"
if has_hf_weights "${MERGED_MODEL_DIR}"; then
    echo "SKIP GRPO: merged final checkpoint already exists: ${MERGED_MODEL_DIR}"
    exit 0
fi

VAL_FILES_HYDRA="$(printf "'%s', " "${VAL_PATHS[@]}")"
VAL_FILES_HYDRA="[${VAL_FILES_HYDRA%, }]"
mkdir -p "${OUTPUT_ROOT}/logs"
LOG_FILE="${LOG_FILE:-${OUTPUT_ROOT}/logs/grpo.log}"
RAY_STORAGE_DIR="${RAY_STORAGE_DIR:-${OUTPUT_ROOT}/ray}"
RAY_SPILL_DIR="${RAY_SPILL_DIR:-${RAY_STORAGE_DIR}/spill}"
RAY_LINK_PARENT="${RAY_LINK_PARENT:-${HOME}}"
RAY_TMP_LINK=""
RAY_SPILL_LINK=""
cleanup_ray_storage_links() {
    for link in "${RAY_TMP_LINK}" "${RAY_SPILL_LINK}"; do
        [[ -n "${link}" && -L "${link}" ]] && rm -f "${link}"
    done
}
setup_ray_storage_links() {
    mkdir -p "${RAY_SPILL_DIR}" "${RAY_LINK_PARENT}"
    RAY_TMP_LINK="$(mktemp -d "${RAY_LINK_PARENT%/}/r.XXXXXX")"
    rmdir "${RAY_TMP_LINK}"
    ln -s "${RAY_STORAGE_DIR}" "${RAY_TMP_LINK}"
    RAY_SPILL_LINK="$(mktemp -d "${RAY_LINK_PARENT%/}/s.XXXXXX")"
    rmdir "${RAY_SPILL_LINK}"
    ln -s "${RAY_SPILL_DIR}" "${RAY_SPILL_LINK}"
}
setup_ray_storage_links
trap cleanup_ray_storage_links EXIT

THINKING_OVERRIDE="+data.apply_chat_template_kwargs.enable_thinking=False"
[[ "${ENABLE_THINKING}" == "1" ]] && THINKING_OVERRIDE="+data.apply_chat_template_kwargs.enable_thinking=True"

echo "===== GRPO ====="
printf '%s\n' \
    "model=${BASE_MODEL}" \
    "train=${TRAIN_FILES}" \
    "validation=${VAL_FILES}" \
    "steps=${GRPO_STEPS}, lr=${LR}, batch=${TRAIN_BATCH_SIZE}, rollout_n=${ROLLOUT_N}" \
    "gpus=${CUDA_VISIBLE_DEVICES}, rollout_tp=${ROLLOUT_TP_SIZE}" \
    "output=${OUTPUT_ROOT}"

set +e
(
    cd "${ROOT}/verl"
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
    RAY_TMPDIR="${RAY_TMP_LINK}" \
    RAY_object_spilling_directory="${RAY_SPILL_LINK}" \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="${ROOT}/verl${PYTHONPATH:+:${PYTHONPATH}}" \
    "${PYTHON}" -m verl.trainer.main_ppo \
        algorithm.adv_estimator=grpo \
        algorithm.rollout_correction.rollout_is=token \
        algorithm.rollout_correction.rollout_is_threshold=5.0 \
        algorithm.rollout_correction.rollout_rs=null \
        algorithm.rollout_correction.bypass_mode=false \
        actor_rollout_ref.rollout.calculate_log_probs=true \
        data.train_files="${TRAIN_FILES}" \
        data.val_files="${VAL_FILES_HYDRA}" \
        data.train_batch_size="${TRAIN_BATCH_SIZE}" \
        data.val_max_samples="${VAL_MAX_SAMPLES}" \
        data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
        data.max_response_length="${MAX_RESPONSE_LENGTH}" \
        data.filter_overlong_prompts=True \
        data.truncation=error \
        data.shuffle=True \
        data.seed="${SEED}" \
        "${THINKING_OVERRIDE}" \
        actor_rollout_ref.model.path="${BASE_MODEL}" \
        actor_rollout_ref.actor.optim.lr="${LR}" \
        actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0 \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.actor.ppo_mini_batch_size="${TRAIN_BATCH_SIZE}" \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE}" \
        actor_rollout_ref.actor.use_kl_loss=True \
        actor_rollout_ref.actor.kl_loss_coef=0.0 \
        actor_rollout_ref.actor.kl_loss_type=low_var_kl \
        actor_rollout_ref.actor.entropy_coeff=0 \
        actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${PPO_MAX_TOKEN_LEN}" \
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        actor_rollout_ref.actor.fsdp_config.param_offload=False \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
        actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TP_SIZE}" \
        actor_rollout_ref.rollout.name=vllm \
        actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEMORY_UTILIZATION}" \
        actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
        actor_rollout_ref.rollout.max_num_batched_tokens="${PPO_MAX_TOKEN_LEN}" \
        actor_rollout_ref.rollout.temperature=1.0 \
        actor_rollout_ref.rollout.top_p=1.0 \
        actor_rollout_ref.rollout.val_kwargs.do_sample=True \
        actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
        actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
        actor_rollout_ref.rollout.val_kwargs.n="${VAL_ROLLOUT_N}" \
        actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
        actor_rollout_ref.ref.fsdp_config.param_offload=True \
        algorithm.use_kl_in_reward=False \
        reward_model.reward_manager=naive \
        trainer.critic_warmup=0 \
        trainer.val_before_train=False \
        trainer.logger='[console]' \
        trainer.project_name="${PROJECT_NAME}" \
        trainer.experiment_name="${RUN_NAME}" \
        trainer.n_gpus_per_node="${NUM_GPUS}" \
        trainer.nnodes=1 \
        trainer.save_freq="${SAVE_FREQ}" \
        trainer.default_local_dir="${OUTPUT_ROOT}" \
        trainer.test_freq="${TEST_FREQ}" \
        trainer.total_training_steps="${GRPO_STEPS}" \
        trainer.total_epochs="${TOTAL_EPOCHS}" \
        trainer.resume_mode="${RESUME_MODE}" \
        "${RESUME_ARGS[@]}" \
        +ray_kwargs.ray_init._temp_dir="${RAY_TMP_LINK}" \
        +ray_kwargs.ray_init.object_spilling_directory="${RAY_SPILL_LINK}"
) 2>&1 | tee -a "${LOG_FILE}"
TRAIN_STATUS="${PIPESTATUS[0]}"
set -e
(( TRAIN_STATUS == 0 )) || { echo "[ERROR] GRPO failed; see ${LOG_FILE}" >&2; exit "${TRAIN_STATUS}"; }

[[ -f "${ACTOR_DIR}/huggingface/config.json" ]] || { echo "[ERROR] Missing final actor checkpoint: ${ACTOR_DIR}" >&2; exit 1; }
if ! has_hf_weights "${MERGED_MODEL_DIR}"; then
    TEMP_MERGED_DIR="${MERGED_MODEL_DIR}.tmp.$$"
    (
        cd "${ROOT}/verl"
        PYTHONPATH="${ROOT}/verl${PYTHONPATH:+:${PYTHONPATH}}" \
        "${PYTHON}" -m verl.model_merger merge --backend fsdp \
            --local_dir "${ACTOR_DIR}" --target_dir "${TEMP_MERGED_DIR}" --use_cpu_initialization
    )
    has_hf_weights "${TEMP_MERGED_DIR}" || { echo "[ERROR] Failed to merge GRPO checkpoint." >&2; exit 1; }
    mv "${TEMP_MERGED_DIR}" "${MERGED_MODEL_DIR}"
fi

echo "Completed GRPO. Merged model: ${MERGED_MODEL_DIR}"
