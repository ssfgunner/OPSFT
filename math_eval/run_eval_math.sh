#!/usr/bin/env bash
set -uo pipefail

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 MODEL_PATH MODEL_NAME" >&2
    echo "Optional env: DATASETS, GPU_GROUPS, N, MAX_TOKENS, MAX_MODEL_LEN, MAX_NUM_SEQS, EVAL_SEED, OUTPUT_TAG" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${ROOT_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
EVAL_DIR="${ROOT_DIR}/math_eval"
MODEL_PATH="$1"
MODEL_NAME="$2"
DATASETS="${DATASETS:-aime24 aime25 hmmt25_feb hmmt25_nov}"
GPU_GROUPS="${GPU_GROUPS:-0,1 2,3 4,5 6,7}"
N="${N:-32}"
MAX_TOKENS="${MAX_TOKENS:-16384}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-256}"
EVAL_SEED="${EVAL_SEED:-42}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-1.0}"
OUTPUT_TAG="${OUTPUT_TAG:-}"
ENABLE_THINKING="${ENABLE_THINKING:-0}"

[[ -f "${MODEL_PATH}/config.json" ]] || {
    echo "[ERROR] Model not found: ${MODEL_PATH}" >&2
    exit 1
}
# Evaluation subsequently switches into ${EVAL_DIR}; resolve caller-provided relative
# paths first so Transformers always receives a valid local model directory.
MODEL_PATH="$(cd "${MODEL_PATH}" && pwd)"

if [[ -n "${OUTPUT_TAG}" ]]; then
    OUTPUT_ROOT="${EVAL_DIR}/eval_outputs/${OUTPUT_TAG}"
else
    OUTPUT_ROOT="${EVAL_DIR}/eval_outputs"
fi
LOG_DIR="${OUTPUT_ROOT}/math_logs"
mkdir -p "${LOG_DIR}"
cd "${EVAL_DIR}" || exit 1

read -ra DATASET_LIST <<< "${DATASETS}"
read -ra GPU_GROUP_LIST <<< "${GPU_GROUPS}"
if (( ${#GPU_GROUP_LIST[@]} < ${#DATASET_LIST[@]} )); then
    echo "[ERROR] Need one GPU group per concurrent dataset: datasets=${#DATASET_LIST[@]}, groups=${#GPU_GROUP_LIST[@]}" >&2
    exit 1
fi

dataset_input() {
    case "$1" in
        aime24) echo "../data/aime24/test.jsonl" ;;
        aime25) echo "../data/aime25/test.jsonl" ;;
        hmmt25_feb) echo "../data/hmmt25_feb/test.jsonl" ;;
        hmmt25_nov) echo "../data/hmmt25_nov/test.jsonl" ;;
        *) return 1 ;;
    esac
}

run_eval() {
    local dataset="$1" gpus="$2" input_file="$3"
    local output_dir="${OUTPUT_ROOT}/${dataset}"
    local output_file="${output_dir}/${MODEL_NAME}.jsonl"
    local metrics_file="${output_dir}/${MODEL_NAME}.metrics.json"
    local log_file="${LOG_DIR}/${dataset}_${MODEL_NAME}.log"
    local cache_root thinking_flag=""

    mkdir -p "${output_dir}"
    if [[ -s "${metrics_file}" ]]; then
        echo "SKIP ${dataset}: ${metrics_file}"
        return 0
    fi
    [[ "${ENABLE_THINKING}" == "1" ]] && thinking_flag="--enable_thinking"
    cache_root=$(mktemp -d "/tmp/sft_math_eval_${dataset}_XXXXXX")
    echo "START ${MODEL_NAME} / ${dataset} on GPU ${gpus}"
    VLLM_CACHE_ROOT="${cache_root}/vllm" \
    TORCHINDUCTOR_CACHE_DIR="${cache_root}/torchinductor" \
    TRITON_CACHE_DIR="${cache_root}/triton" \
    CUDA_VISIBLE_DEVICES="${gpus}" python3 eval_math.py \
        --input_file "${input_file}" \
        --model_path "${MODEL_PATH}" \
        --output_file "${output_file}" \
        --metrics_file "${metrics_file}" \
        --max_tokens "${MAX_TOKENS}" \
        --max_model_len "${MAX_MODEL_LEN}" \
        --temperature "${TEMPERATURE}" \
        --top_p "${TOP_P}" \
        --max_num_seqs "${MAX_NUM_SEQS}" \
        --n "${N}" \
        --begin_idx -1 \
        --end_idx -1 \
        --seed "${EVAL_SEED}" \
        ${thinking_flag} >"${log_file}" 2>&1
    local status=$?
    rm -rf "${cache_root}"
    if (( status != 0 )); then
        echo "FAIL ${dataset}: see ${log_file}" >&2
        return "${status}"
    fi
    echo "DONE ${dataset}: ${metrics_file}"
}

pids=()
labels=()
STAGGER_SECONDS="${STAGGER_SECONDS:-0}"
for index in "${!DATASET_LIST[@]}"; do
    dataset="${DATASET_LIST[$index]}"
    if ! input_file=$(dataset_input "${dataset}"); then
        echo "[ERROR] Unknown dataset: ${dataset}" >&2
        exit 1
    fi
    run_eval "${dataset}" "${GPU_GROUP_LIST[$index]}" "${input_file}" &
    pids+=("$!")
    labels+=("${dataset}")
    # Optionally stagger launches to avoid resource contention during vLLM
    # startup + torch.compile.  Set STAGGER_SECONDS=30 (or similar) to enable.
    if (( STAGGER_SECONDS > 0 )) && (( index + 1 < ${#DATASET_LIST[@]} )); then
        echo "Staggering next launch by ${STAGGER_SECONDS}s..."
        sleep "${STAGGER_SECONDS}"
    fi
done

failures=0
for index in "${!pids[@]}"; do
    if ! wait "${pids[$index]}"; then
        echo "FAILED ${MODEL_NAME} / ${labels[$index]}" >&2
        failures=$((failures + 1))
    fi
done

if (( failures > 0 )); then
    echo "Evaluation completed with ${failures} failures. Logs: ${LOG_DIR}" >&2
    exit 1
fi

echo "All requested math evaluations completed for ${MODEL_NAME}."
for dataset in "${DATASET_LIST[@]}"; do
    metrics_file="${OUTPUT_ROOT}/${dataset}/${MODEL_NAME}.metrics.json"
    if [[ -s "${metrics_file}" ]]; then
        python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print('{}: accuracy={:.2f}, pass@k={:.2f}, avg_tokens={:.1f}'.format(d['dataset'], 100*d['accuracy'], 100*d['pass_at_k'], d['avg_response_tokens']))" "${metrics_file}"
    fi
done
