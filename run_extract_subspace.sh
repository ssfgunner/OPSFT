#!/usr/bin/env bash
# Unified parameterized direction-identification entry point for OPSFT.
# The legacy filename is retained for compatibility.
#
# Replaces:
#   run_extract_qwen3_multiscale_subspaces.sh
#   run_qwen3_4b_dapo_subspace_extract.sh
#
# Identifies an on-policy parameter-update direction (mask + direction vector)
# from a pair of base and trained (GRPO) Hugging Face checkpoints. Optionally generates
# count-matched random masks for ablation.
#
# Required:
#   BASE_MODEL     Path to the base Hugging Face model directory.
#   TRAINED_MODEL  Path to the GRPO-trained merged Hugging Face model directory.
#   OUTPUT_DIR     Directory for the identified direction artifacts.
#
# Common overrides:
#   SELECTION=exact_support|topk  (default: exact_support)
#   KEEP_RATIO=0.05              (used when SELECTION=topk)
#   GENERATE_RANDOM_CONTROLS=0|1  (default: 0)
#   RANDOM_SEEDS="42 123 2026"   (seeds for random controls)
#   PYTHON=python3

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-${SCRIPT_DIR}}"
PYTHON="${PYTHON:-python3}"

: "${BASE_MODEL:?Set BASE_MODEL to the base Hugging Face model directory.}"
: "${TRAINED_MODEL:?Set TRAINED_MODEL to the GRPO-trained merged Hugging Face model.}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR for the identified direction artifacts.}"

SELECTION="${SELECTION:-exact_support}"
KEEP_RATIO="${KEEP_RATIO:-0.05}"
GENERATE_RANDOM_CONTROLS="${GENERATE_RANDOM_CONTROLS:-0}"
RANDOM_SEEDS="${RANDOM_SEEDS:-42 123 2026}"

has_hf_weights() {
    [[ -d "$1" ]] && [[ -f "$1/model.safetensors" || -f "$1/pytorch_model.bin" || -n "$(find "$1" -maxdepth 1 -type f \( -name '*.safetensors' -o -name '*.bin' \) -print -quit 2>/dev/null)" ]]
}
require_hf_model() {
    [[ -f "$1/config.json" ]] || { echo "[ERROR] Missing config.json: $1" >&2; exit 1; }
    has_hf_weights "$1" || { echo "[ERROR] Missing HF weights: $1" >&2; exit 1; }
}

case "${SELECTION}" in
    exact_support|topk) ;;
    *) echo "[ERROR] SELECTION must be exact_support or topk; got ${SELECTION}." >&2; exit 1 ;;
esac
[[ "${GENERATE_RANDOM_CONTROLS}" == "0" || "${GENERATE_RANDOM_CONTROLS}" == "1" ]] || {
    echo "[ERROR] GENERATE_RANDOM_CONTROLS must be 0 or 1." >&2; exit 1;
}

require_hf_model "${BASE_MODEL}"
require_hf_model "${TRAINED_MODEL}"

if [[ -f "${OUTPUT_DIR}/update_mask.pt" && -f "${OUTPUT_DIR}/parameter_updates.pt" ]]; then
    echo "SKIP direction identification: artifacts already exist at ${OUTPUT_DIR}"
else
    echo "===== Identify on-policy direction ====="
    echo "Base:     ${BASE_MODEL}"
    echo "Trained:  ${TRAINED_MODEL}"
    echo "Output:   ${OUTPUT_DIR}"
    echo "Selection: ${SELECTION}"
    [[ "${SELECTION}" == "topk" ]] && echo "Keep ratio: ${KEEP_RATIO}"
    mkdir -p "${OUTPUT_DIR}"

    local_args=(--base-model "${BASE_MODEL}" --trained-model "${TRAINED_MODEL}" --output-dir "${OUTPUT_DIR}" --selection "${SELECTION}")
    [[ "${SELECTION}" == "topk" ]] && local_args+=(--keep-ratio "${KEEP_RATIO}")

    PYTHONPATH="${ROOT}/verl${PYTHONPATH:+:${PYTHONPATH}}" \
    "${PYTHON}" "${ROOT}/scripts/analyze_update_mask.py" "${local_args[@]}"
fi

echo "Direction artifacts:"
echo "  mask:      ${OUTPUT_DIR}/update_mask.pt"
echo "  direction: ${OUTPUT_DIR}/parameter_updates.pt"

if [[ "${GENERATE_RANDOM_CONTROLS}" == "1" ]]; then
    RANDOM_ROOT="${RANDOM_ROOT:-${OUTPUT_DIR}/../random_controls}"
    echo "===== Generate random-matched controls ====="
    echo "Reference mask: ${OUTPUT_DIR}/update_mask.pt"
    echo "Output root:    ${RANDOM_ROOT}"
    "${PYTHON}" "${ROOT}/scripts/generate_random_matched_controls.py" \
        --base-model "${BASE_MODEL}" \
        --trained-model "${TRAINED_MODEL}" \
        --reference-mask "${OUTPUT_DIR}/update_mask.pt" \
        --output-root "${RANDOM_ROOT}" \
        --seeds ${RANDOM_SEEDS} \
        --name-prefix "random_matched" \
        --torch-dtype float32
    echo "Random controls generated: ${RANDOM_ROOT}"
fi

echo "Direction identification completed."
