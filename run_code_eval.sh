#!/usr/bin/env bash
# Unified parameterized code evaluation entry point for OPSFT.
#
# Replaces:
#   run_qwen3_4b_eurus_grpo50init_exact_sft100_code_eval.sh
#   run_qwen3_4b_code_eval_all_ckpts.sh
#
# Evaluates a model on HumanEval+, MBPP+ (EvalPlus) and LiveCodeBench v6.
# Both benchmarks run concurrently: HumanEval+ on GPU 0, MBPP+ on GPU 1,
# LiveCodeBench on GPUs 2,3 (tensor parallelism=2).
#
# Required:
#   MODEL_DIR   Path to the Hugging Face model directory to evaluate.
#
# Common overrides:
#   MODEL_LINK         Symlink path for short model name (default: /tmp/opsft_eval)
#   RUN_LABEL          Label for result directories (default: opsft_eval)
#   BENCHMARKS         evalplus,lcb or a subset (default: evalplus,lcb)
#   EVAL_PROTOCOL      Paper-protocol tag (default: paper_code_n4_t1.0_p1.0_l16384)
#
# EvalPlus parameters:
#   EVALPLUS_N=4  EVALPLUS_TEMPERATURE=1.0  EVALPLUS_TOP_P=1.0
#   EVALPLUS_MAX_NEW_TOKENS=16384  EVALPLUS_TP=1  EVALPLUS_DTYPE=bfloat16
#   HUMANEVAL_GPU=0  MBPP_GPU=1
#
# LiveCodeBench parameters:
#   LCB_CUDA_VISIBLE_DEVICES=2,3  LCB_TP=2  LCB_N=4
#   LCB_TEMPERATURE=1.0  LCB_TOP_P=1.0  LCB_MAX_TOKENS=16384
#   LCB_RELEASE_VERSION=release_v6  LCB_START_DATE=2025-02-01  LCB_END_DATE=2025-05-31

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-${SCRIPT_DIR}}"
PYTHON="${PYTHON:-python3}"

MODEL_DIR="${MODEL_DIR:?Set MODEL_DIR to the Hugging Face model directory to evaluate.}"
MODEL_LINK="${MODEL_LINK:-/tmp/opsft_eval}"
RUN_LABEL="${RUN_LABEL:-opsft_eval}"

BENCHMARKS="${BENCHMARKS:-evalplus,lcb}"
EVAL_PROTOCOL="${EVAL_PROTOCOL:-paper_code_n4_t1.0_p1.0_l16384}"

EVALPLUS_DIR="${EVALPLUS_DIR:-${ROOT}/code_eval/coding/evalplus}"
EVALPLUS_RESULTS_ROOT_BASE="${EVALPLUS_RESULTS_ROOT:-${ROOT}/code_eval/evalplus_results/${RUN_LABEL}}"
EVALPLUS_RESULTS_ROOT="${EVALPLUS_RESULTS_ROOT_BASE}/${EVAL_PROTOCOL}"
EVALPLUS_N="${EVALPLUS_N:-4}"
EVALPLUS_TEMPERATURE="${EVALPLUS_TEMPERATURE:-1.0}"
EVALPLUS_TOP_P="${EVALPLUS_TOP_P:-1.0}"
EVALPLUS_MAX_NEW_TOKENS="${EVALPLUS_MAX_NEW_TOKENS:-16384}"
EVALPLUS_BATCH_SIZE="${EVALPLUS_BATCH_SIZE:-${EVALPLUS_N}}"
EVALPLUS_PARALLEL="${EVALPLUS_PARALLEL:-12}"
EVALPLUS_MIN_TIME_LIMIT="${EVALPLUS_MIN_TIME_LIMIT:-10.0}"
EVALPLUS_GT_TIME_LIMIT_FACTOR="${EVALPLUS_GT_TIME_LIMIT_FACTOR:-8.0}"
HUMANEVAL_GPU="${HUMANEVAL_GPU:-0}"
MBPP_GPU="${MBPP_GPU:-1}"
EVALPLUS_TP="${EVALPLUS_TP:-1}"
EVALPLUS_DTYPE="${EVALPLUS_DTYPE:-bfloat16}"

LCB_DIR="${LCB_DIR:-${ROOT}/code_eval/coding/LiveCodeBench}"
LCB_OUTPUT_ROOT_BASE="${LCB_OUTPUT_ROOT:-${ROOT}/code_eval/lcb_outputs/${RUN_LABEL}}"
LCB_OUTPUT_ROOT="${LCB_OUTPUT_ROOT_BASE}/${EVAL_PROTOCOL}"
LCB_CUDA_VISIBLE_DEVICES="${LCB_CUDA_VISIBLE_DEVICES:-2,3}"
LCB_TP="${LCB_TP:-2}"
LCB_RELEASE_VERSION="${LCB_RELEASE_VERSION:-release_v6}"
LCB_START_DATE="${LCB_START_DATE-2025-02-01}"
LCB_END_DATE="${LCB_END_DATE-2025-05-31}"
LCB_DATA_DIR="${LCB_DATA_DIR:-${ROOT}/code_eval/data/code_generation_lite}"
LCB_N="${LCB_N:-4}"
LCB_TEMPERATURE="${LCB_TEMPERATURE:-1.0}"
LCB_TOP_P="${LCB_TOP_P:-1.0}"
LCB_MAX_TOKENS="${LCB_MAX_TOKENS:-16384}"
LCB_DTYPE="${LCB_DTYPE:-bfloat16}"
LCB_TIMEOUT="${LCB_TIMEOUT:-60}"
LCB_NUM_PROCESS_EVALUATE="${LCB_NUM_PROCESS_EVALUATE:-12}"

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${ROOT}/verl:${EVALPLUS_DIR}:${LCB_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

has_benchmark() {
    local requested=" ${BENCHMARKS//,/ } "
    [[ "${requested}" == *" $1 "* ]]
}

require_file() {
    [[ -f "$1" ]] || { echo "[ERROR] Missing required file: $1" >&2; return 1; }
}
require_dir() {
    [[ -d "$1" ]] || { echo "[ERROR] Missing required directory: $1" >&2; return 1; }
}

find_evalplus_samples() {
    local dataset="$1"
    find "${EVALPLUS_RESULTS_ROOT}/${dataset}" \
        -type f -name "*_vllm_temp_${EVALPLUS_TEMPERATURE}.jsonl" ! -name '*.raw.jsonl' \
        -print -quit 2>/dev/null
}
sample_count() { wc -l < "$1" | tr -d '[:space:]'; }
expected_evalplus_samples() {
    case "$1" in
        humaneval) printf '%s\n' "$((164 * EVALPLUS_N))" ;;
        mbpp) printf '%s\n' "$((378 * EVALPLUS_N))" ;;
        *) printf '0\n' ;;
    esac
}

evalplus_samples_are_exact() {
    "${PYTHON}" - "$1" "$2" "${EVALPLUS_N}" <<'PY'
import json, sys
from collections import Counter
samples_path, dataset, expected_per_task = sys.argv[1:]
expected_per_task = int(expected_per_task)
expected_tasks = {"humaneval": 164, "mbpp": 378}[dataset]
with open(samples_path, encoding="utf-8") as f:
    counts = Counter(json.loads(line)["task_id"] for line in f if line.strip())
if any(c > expected_per_task for c in counts.values()):
    print(f"[ERROR] {dataset} has >{expected_per_task} samples for a task", file=sys.stderr)
    raise SystemExit(2)
if len(counts) != expected_tasks or any(c != expected_per_task for c in counts.values()):
    print(f"[INFO] {dataset} incomplete: {len(counts)}/{expected_tasks} tasks", file=sys.stderr)
    raise SystemExit(1)
PY
}

evalplus_results_are_exact() {
    "${PYTHON}" - "$1" "$2" "${EVALPLUS_N}" <<'PY'
import json, sys
from collections import Counter
results_path, dataset, expected_per_task = sys.argv[1:]
expected_per_task = int(expected_per_task)
expected_tasks = {"humaneval": 164, "mbpp": 378}[dataset]
with open(results_path, encoding="utf-8") as f:
    evaluated = json.load(f).get("eval", {})
counts = {tid: len(c) for tid, c in evaluated.items()}
if any(c > expected_per_task for c in counts.values()):
    print(f"[ERROR] {dataset} has >{expected_per_task} results", file=sys.stderr)
    raise SystemExit(2)
if len(counts) != expected_tasks or any(c != expected_per_task for c in counts.values()):
    print(f"[ERROR] {dataset} not exact-{expected_per_task}-sample", file=sys.stderr)
    raise SystemExit(1)
PY
}

prepare_environment() {
    require_file "${MODEL_DIR}/config.json" || return 1
    require_dir "${EVALPLUS_DIR}" || return 1
    require_dir "${LCB_DIR}" || return 1
    require_file "${ROOT}/code_eval/data/HumanEvalPlus.jsonl" || return 1
    require_file "${ROOT}/code_eval/data/MbppPlus.jsonl" || return 1
    if ! "${PYTHON}" -c 'import tree_sitter_python' >/dev/null 2>&1; then
        echo "Installing tree-sitter-python..."
        PIP_DISABLE_PIP_VERSION_CHECK=1 "${PYTHON}" -m pip install --user tree-sitter-python || true
    fi
    mkdir -p "${EVALPLUS_RESULTS_ROOT}" "${LCB_OUTPUT_ROOT}" "${ROOT}/code_eval/logs/${RUN_LABEL}"
    ln -sfn "${MODEL_DIR}" "${MODEL_LINK}" || { echo "[ERROR] Symlink failed: ${MODEL_LINK}" >&2; return 1; }
    require_file "${MODEL_LINK}/config.json"
}

run_evalplus_dataset() {
    local dataset="$1" gpu="$2" expected samples result_file samples_are_exact=0
    expected="$(expected_evalplus_samples "${dataset}")"
    result_file="${EVALPLUS_RESULTS_ROOT}/${dataset}/${RUN_LABEL}.eval_results.json"
    mkdir -p "${EVALPLUS_RESULTS_ROOT}/${dataset}"

    if [[ -f "${result_file}" ]]; then
        if evalplus_results_are_exact "${result_file}" "${dataset}"; then
            echo "SKIP EvalPlus ${dataset}: ${result_file}"
            return 0
        fi
        echo "[ERROR] Stale EvalPlus ${dataset} result: ${result_file}" >&2
        return 1
    fi

    samples="$(find_evalplus_samples "${dataset}")"
    if [[ -n "${samples}" ]]; then
        if evalplus_samples_are_exact "${samples}" "${dataset}"; then
            samples_are_exact=1
        fi
    fi

    if [[ "${samples_are_exact}" -ne 1 ]]; then
        echo "===== Generate EvalPlus ${dataset} on GPU ${gpu} ====="
        (
            cd "${ROOT}" || return
            HUMANEVAL_OVERRIDE_PATH="${ROOT}/code_eval/data/HumanEvalPlus.jsonl" \
            MBPP_OVERRIDE_PATH="${ROOT}/code_eval/data/MbppPlus.jsonl" \
            CUDA_VISIBLE_DEVICES="${gpu}" \
            "${PYTHON}" -m evalplus.codegen \
                --model "${MODEL_LINK}" --dataset "${dataset}" --root "${EVALPLUS_RESULTS_ROOT}" \
                --backend vllm --tp "${EVALPLUS_TP}" --dtype "${EVALPLUS_DTYPE}" \
                --n-samples "${EVALPLUS_N}" --bs "${EVALPLUS_BATCH_SIZE}" \
                --temperature "${EVALPLUS_TEMPERATURE}" --top-p "${EVALPLUS_TOP_P}" \
                --max-new-tokens "${EVALPLUS_MAX_NEW_TOKENS}" --trust_remote_code
        ) 2>&1 | tee "${ROOT}/code_eval/logs/${RUN_LABEL}/evalplus_${dataset}.log"
        [[ "${PIPESTATUS[0]}" -eq 0 ]] || { echo "[ERROR] EvalPlus ${dataset} gen failed." >&2; return 1; }
        samples="$(find_evalplus_samples "${dataset}")"
    fi

    [[ -n "${samples}" && -f "${samples}" ]] || { echo "[ERROR] No EvalPlus ${dataset} samples." >&2; return 1; }
    evalplus_samples_are_exact "${samples}" "${dataset}" || { echo "[ERROR] EvalPlus ${dataset} sample count mismatch." >&2; return 1; }

    echo "===== Evaluate EvalPlus ${dataset} (${EVAL_PROTOCOL}) ====="
    (
        cd "${ROOT}" || return
        HUMANEVAL_OVERRIDE_PATH="${ROOT}/code_eval/data/HumanEvalPlus.jsonl" \
        MBPP_OVERRIDE_PATH="${ROOT}/code_eval/data/MbppPlus.jsonl" \
        "${PYTHON}" -m evalplus.evaluate --dataset "${dataset}" --samples "${samples}" \
            --output-file "${result_file}" --parallel "${EVALPLUS_PARALLEL}" \
            --min-time-limit "${EVALPLUS_MIN_TIME_LIMIT}" \
            --gt-time-limit-factor "${EVALPLUS_GT_TIME_LIMIT_FACTOR}"
    ) 2>&1 | tee "${ROOT}/code_eval/logs/${RUN_LABEL}/evalplus_${dataset}_evaluate.log"
    [[ "${PIPESTATUS[0]}" -eq 0 ]] || { echo "[ERROR] EvalPlus ${dataset} eval failed." >&2; return 1; }
    [[ -f "${result_file}" ]] && echo "Completed EvalPlus ${dataset}: ${result_file}" && return 0
    echo "[ERROR] EvalPlus ${dataset} did not create ${result_file}." >&2; return 1
}

run_lcb() {
    local eval_file
    local -a LCB_DATE_FILTERS=()
    [[ -n "${LCB_START_DATE}" ]] && LCB_DATE_FILTERS+=(--start_date "${LCB_START_DATE}")
    [[ -n "${LCB_END_DATE}" ]] && LCB_DATE_FILTERS+=(--end_date "${LCB_END_DATE}")

    eval_file="${LCB_OUTPUT_ROOT}/Scenario.codegeneration_${LCB_N}_${LCB_TEMPERATURE}_eval.json"
    [[ -f "${eval_file}" ]] && { echo "SKIP LiveCodeBench: ${eval_file}"; return 0; }

    for f in test.jsonl test2.jsonl test3.jsonl test4.jsonl test5.jsonl test6.jsonl; do
        require_file "${LCB_DATA_DIR}/${f}" || return 1
    done

    echo "===== LiveCodeBench ${LCB_RELEASE_VERSION} on GPUs ${LCB_CUDA_VISIBLE_DEVICES} ====="
    (
        cd "${LCB_DIR}" || return
        CUDA_VISIBLE_DEVICES="${LCB_CUDA_VISIBLE_DEVICES}" \
        LCB_CODE_GENERATION_LITE_DIR="${LCB_DATA_DIR}" \
        LCB_LOCAL_MODEL_PATH="${MODEL_LINK}" \
        "${PYTHON}" -m lcb_runner.runner.main \
            --model "Qwen3-4B-NonThinking" \
            --local_model_path "${MODEL_LINK}" \
            --scenario codegeneration \
            --release_version "${LCB_RELEASE_VERSION}" \
            "${LCB_DATE_FILTERS[@]}" \
            --tensor_parallel_size "${LCB_TP}" \
            --n "${LCB_N}" \
            --temperature "${LCB_TEMPERATURE}" \
            --top_p "${LCB_TOP_P}" \
            --max_tokens "${LCB_MAX_TOKENS}" \
            --dtype "${LCB_DTYPE}" \
            --timeout "${LCB_TIMEOUT}" \
            --num_process_evaluate "${LCB_NUM_PROCESS_EVALUATE}" \
            --custom_output_save_name "${LCB_OUTPUT_ROOT}" \
            --evaluate --continue_existing --continue_existing_with_eval
    ) 2>&1 | tee "${ROOT}/code_eval/logs/${RUN_LABEL}/lcb_release_v6.log"
    [[ "${PIPESTATUS[0]}" -eq 0 ]] || { echo "[ERROR] LiveCodeBench failed." >&2; return 1; }
    [[ -f "${eval_file}" ]] && echo "Completed LiveCodeBench: ${eval_file}" && return 0
    echo "[ERROR] LiveCodeBench did not create ${eval_file}." >&2; return 1
}

if ! prepare_environment; then
    echo "[ERROR] Environment preflight failed; no benchmark was launched." >&2
    exit 1
fi

declare -a job_names=()
declare -a job_pids=()

if has_benchmark evalplus; then
    run_evalplus_dataset humaneval "${HUMANEVAL_GPU}" &
    job_names+=("HumanEval+"); job_pids+=("$!")
    run_evalplus_dataset mbpp "${MBPP_GPU}" &
    job_names+=("MBPP+"); job_pids+=("$!")
fi

if has_benchmark lcb; then
    run_lcb &
    job_names+=("LiveCodeBench v6"); job_pids+=("$!")
fi

failures=0
for index in "${!job_pids[@]}"; do
    if wait "${job_pids[${index}]}"; then
        echo "[DONE] ${job_names[${index}]}"
    else
        echo "[FAILED] ${job_names[${index}]}" >&2
        failures=$((failures + 1))
    fi
done

if [[ "${failures}" -eq 0 ]]; then
    echo "All requested evaluations completed."
else
    echo "${failures} evaluation job(s) failed."
fi
