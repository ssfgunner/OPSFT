#!/usr/bin/env bash
# Generate verified SFT trajectories from a prompt parquet with a vLLM teacher.
# All experiment-specific values are supplied as environment variables.
#
# Required:
#   INPUT, OUTPUT_DIR, TEACHER_MODEL
# Optional:
#   VERIFIER=code|math_verify|math_dapo (default: code)
#   ENABLE_THINKING=0|1 (default: 0), REJECT_THINKING_TAGS=0|1 (default: 1 for code)
#   CUDA_VISIBLE_DEVICES, TEACHER_TP_SIZE, BATCH_SIZE, MAX_NEW_TOKENS,
#   MAX_MODEL_LEN, MAX_NUM_SEQS, TEMPERATURE, TOP_P, SAMPLES_PER_PROMPT,
#   VALIDATION_RATIO, SEED, MAX_SAMPLES, SHARD_COUNT, SHARD_INDEX, RESUME.
#
# A sharded run writes OUTPUT_DIR/shards/shard_XX/verified_trajectories.parquet.
# Use run_merge_trajectory_shards.sh after all shards finish.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-${SCRIPT_DIR}}"
PYTHON="${PYTHON:-python3}"

: "${INPUT:?Set INPUT to an RL prompt parquet.}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR for generated trajectories.}"
: "${TEACHER_MODEL:?Set TEACHER_MODEL to a local Hugging Face teacher model.}"

VERIFIER="${VERIFIER:-code}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
TEACHER_TP_SIZE="${TEACHER_TP_SIZE:-8}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-18432}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-16384}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-256}"
BATCH_SIZE="${BATCH_SIZE:-32}"
TEMPERATURE="${TEMPERATURE:-0.6}"
TOP_P="${TOP_P:-0.95}"
SAMPLES_PER_PROMPT="${SAMPLES_PER_PROMPT:-1}"
MAX_SAMPLES="${MAX_SAMPLES:--1}"
VALIDATION_RATIO="${VALIDATION_RATIO:-0.02}"
SEED="${SEED:-42}"
ENABLE_THINKING="${ENABLE_THINKING:-0}"
STRIP_THINKING_INSTRUCTION="${STRIP_THINKING_INSTRUCTION:-0}"
REJECT_THINKING_TAGS="${REJECT_THINKING_TAGS:-1}"
DEDUPLICATE_PROMPTS="${DEDUPLICATE_PROMPTS:-1}"
SHARD_COUNT="${SHARD_COUNT:-1}"
SHARD_INDEX="${SHARD_INDEX:-0}"
RESUME="${RESUME:-0}"
SKIP_VERIFICATION="${SKIP_VERIFICATION:-0}"

case "${VERIFIER}" in
    code|math_verify|math_dapo) ;;
    *) echo "[ERROR] VERIFIER must be code, math_verify, or math_dapo; got ${VERIFIER}." >&2; exit 1 ;;
esac
for boolean_name in ENABLE_THINKING STRIP_THINKING_INSTRUCTION REJECT_THINKING_TAGS DEDUPLICATE_PROMPTS RESUME SKIP_VERIFICATION; do
    [[ "${!boolean_name}" == "0" || "${!boolean_name}" == "1" ]] || {
        echo "[ERROR] ${boolean_name} must be 0 or 1; got ${!boolean_name}." >&2
        exit 1
    }
done
[[ -f "${INPUT}" ]] || { echo "[ERROR] Missing input parquet: ${INPUT}" >&2; exit 1; }
[[ -f "${TEACHER_MODEL}/config.json" ]] || { echo "[ERROR] Missing teacher model config: ${TEACHER_MODEL}/config.json" >&2; exit 1; }
[[ "${TEACHER_TP_SIZE}" =~ ^[1-9][0-9]*$ ]] || { echo "[ERROR] TEACHER_TP_SIZE must be positive." >&2; exit 1; }
[[ "${BATCH_SIZE}" =~ ^[1-9][0-9]*$ ]] || { echo "[ERROR] BATCH_SIZE must be positive." >&2; exit 1; }
[[ "${MAX_NUM_SEQS}" =~ ^[1-9][0-9]*$ ]] || { echo "[ERROR] MAX_NUM_SEQS must be positive." >&2; exit 1; }
[[ "${SHARD_COUNT}" =~ ^[1-9][0-9]*$ ]] || { echo "[ERROR] SHARD_COUNT must be positive." >&2; exit 1; }
[[ "${SHARD_INDEX}" =~ ^[0-9]+$ ]] || { echo "[ERROR] SHARD_INDEX must be non-negative." >&2; exit 1; }
(( SHARD_INDEX < SHARD_COUNT )) || { echo "[ERROR] SHARD_INDEX must be in [0, SHARD_COUNT)." >&2; exit 1; }
(( BATCH_SIZE <= MAX_NUM_SEQS )) || { echo "[ERROR] BATCH_SIZE cannot exceed MAX_NUM_SEQS." >&2; exit 1; }
if [[ "${VERIFIER}" == "code" && "${SKIP_VERIFICATION}" == "1" ]]; then
    echo "[ERROR] Code trajectories must be execution-verified." >&2
    exit 1
fi

IFS=',' read -r -a GPU_IDS <<< "${CUDA_VISIBLE_DEVICES}"
(( ${#GPU_IDS[@]} == TEACHER_TP_SIZE )) || {
    echo "[ERROR] CUDA_VISIBLE_DEVICES exposes ${#GPU_IDS[@]} GPU(s), but TEACHER_TP_SIZE=${TEACHER_TP_SIZE}." >&2
    exit 1
}

GENERATION_OUTPUT_DIR="${OUTPUT_DIR}"
if (( SHARD_COUNT > 1 )); then
    printf -v SHARD_NAME 'shard_%02d' "${SHARD_INDEX}"
    GENERATION_OUTPUT_DIR="${OUTPUT_DIR}/shards/${SHARD_NAME}"
fi
if [[ -e "${GENERATION_OUTPUT_DIR}/generation_config.json" && "${RESUME}" != "1" ]]; then
    echo "[ERROR] Existing generation state at ${GENERATION_OUTPUT_DIR}; set RESUME=1 or choose another OUTPUT_DIR." >&2
    exit 1
fi

INPUT="${INPUT}" "${PYTHON}" - <<'PY'
import os
import pandas as pd

path = os.environ["INPUT"]
data = pd.read_parquet(path)
required = {"prompt", "reward_model"}
missing = required - set(data.columns)
if missing:
    raise ValueError(f"Input parquet is missing required columns: {sorted(missing)}")
if data.empty:
    raise ValueError("Input parquet is empty")
print(f"Validated trajectory source: rows={len(data)}, columns={sorted(data.columns)}")
PY

command=(
    "${PYTHON}" "${ROOT}/scripts/generate_math_sft_trajectories.py"
    --input "${INPUT}"
    --output-dir "${GENERATION_OUTPUT_DIR}"
    --model "${TEACHER_MODEL}"
    --tensor-parallel-size "${TEACHER_TP_SIZE}"
    --max-model-len "${MAX_MODEL_LEN}"
    --max-new-tokens "${MAX_NEW_TOKENS}"
    --max-num-seqs "${MAX_NUM_SEQS}"
    --batch-size "${BATCH_SIZE}"
    --temperature "${TEMPERATURE}"
    --top-p "${TOP_P}"
    --samples-per-prompt "${SAMPLES_PER_PROMPT}"
    --validation-ratio "${VALIDATION_RATIO}"
    --seed "${SEED}"
    --verifier "${VERIFIER}"
    --shard-count "${SHARD_COUNT}"
    --shard-index "${SHARD_INDEX}"
)
[[ "${ENABLE_THINKING}" == "1" ]] && command+=(--enable-thinking) || command+=(--no-enable-thinking)
[[ "${STRIP_THINKING_INSTRUCTION}" == "1" ]] && command+=(--strip-thinking-instruction)
[[ "${REJECT_THINKING_TAGS}" == "1" ]] && command+=(--reject-thinking-tags)
[[ "${DEDUPLICATE_PROMPTS}" == "1" ]] && command+=(--deduplicate-prompts)
[[ "${SKIP_VERIFICATION}" == "1" ]] && command+=(--skip-verification)
(( MAX_SAMPLES > 0 )) && command+=(--max-samples "${MAX_SAMPLES}")
[[ "${RESUME}" == "1" ]] && command+=(--resume)

printf 'Running:'; printf ' %q' "${command[@]}"; printf '\n'
PYTHONPATH="${ROOT}/verl${PYTHONPATH:+:${PYTHONPATH}}" \
VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}" \
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
"${command[@]}"

GENERATION_OUTPUT_DIR="${GENERATION_OUTPUT_DIR}" \
SHARD_COUNT="${SHARD_COUNT}" \
VERIFIER="${VERIFIER}" \
REJECT_THINKING_TAGS="${REJECT_THINKING_TAGS}" \
"${PYTHON}" - <<'PY'
import os
from pathlib import Path

import pandas as pd

output_dir = Path(os.environ["GENERATION_OUTPUT_DIR"])
shard_count = int(os.environ["SHARD_COUNT"])
verifier = os.environ["VERIFIER"]
reject_thinking_tags = os.environ["REJECT_THINKING_TAGS"] == "1"
paths = [output_dir / "verified_trajectories.parquet"] if shard_count > 1 else [output_dir / "train.parquet", output_dir / "validation.parquet"]
data = pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)
required = {"prompt", "response", "ground_truth", "teacher_verified", "source_index"}
missing = required - set(data.columns)
if missing:
    raise ValueError(f"Output trajectories are missing columns: {sorted(missing)}")
if data.empty:
    raise ValueError("Output trajectories are empty")
if verifier == "code" and not data["teacher_verified"].all():
    raise ValueError("Found code trajectories that are not execution-verified")
if data["source_index"].nunique() != len(data):
    raise ValueError("Found duplicate source indices within one generated shard")
if verifier == "code" and reject_thinking_tags and data["response"].fillna("").str.contains(r"<think|</think", case=False, regex=True).any():
    raise ValueError("Found a thinking tag despite REJECT_THINKING_TAGS=1")
print(f"Validated generated trajectories: total={len(data)}, verifier={verifier}, output={output_dir}")
PY
