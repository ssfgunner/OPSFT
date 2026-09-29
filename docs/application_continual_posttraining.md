# Application 2: Improve Models After Post-Training

This Section 4.2 application reuses the parameter-update direction from a post-trained model's GRPO phase when learning from new verified trajectories. The paper protocol uses 100 GRPO steps, then runs 700-step Vanilla SFT and OPSFT from the same checkpoint.

> **Prerequisites**: Complete [environment setup](environment.md). Use Route A for direct comparison; Route B independently reproduces trajectory generation and GRPO.

## Route A: Released-Asset Reproduction

Download the matching trajectories from [`shufanshen/SFT-Trajectories`](https://huggingface.co/datasets/shufanshen/SFT-Trajectories) and the merged GRPO checkpoint from the [OPSFT collection](https://huggingface.co/collections/shufanshen/opsft-6aa781ffb646c53df3707a3b), then identify its direction locally.

| Setting | Trajectories | Merged GRPO checkpoint |
|---------|--------------|------------------------|
| Math | `application_continual_posttraining/math/deepmath_qwen3_30b_trajectories/{train,validation}.parquet` | [`shufanshen/Qwen3-4B-GRPO-DeepMath-100-steps`](https://huggingface.co/shufanshen/Qwen3-4B-GRPO-DeepMath-100-steps) |
| Code | `application_continual_posttraining/code/eurus_qwen3_30b_trajectories/{train,validation}.parquet` | [`shufanshen/Qwen3-4B-GRPO-CodeEurus-100-steps`](https://huggingface.co/shufanshen/Qwen3-4B-GRPO-CodeEurus-100-steps) |

```bash
export ROOT=/path/to/OPSFT
export BASE_MODEL=/path/to/Qwen3-4B
export TRAJECTORIES_REPO=shufanshen/SFT-Trajectories
cd "${ROOT}"

# Math: download trajectories and the matching 100-step GRPO checkpoint.
huggingface-cli download "${TRAJECTORIES_REPO}" --repo-type dataset --local-dir . \
  --include 'application_continual_posttraining/math/deepmath_qwen3_30b_trajectories/*'
huggingface-cli download shufanshen/Qwen3-4B-GRPO-DeepMath-100-steps \
  --local-dir application_continual_posttraining/math/qwen3_4b_deepmath_grpo100
export POST_TRAINED_MODEL=application_continual_posttraining/math/qwen3_4b_deepmath_grpo100
export SFT_TRAIN=application_continual_posttraining/math/deepmath_qwen3_30b_trajectories/train.parquet
export SFT_VAL=application_continual_posttraining/math/deepmath_qwen3_30b_trajectories/validation.parquet
export SUBSPACE_DIR=analysis/continual_math_grpo100_exact_support
export EVAL_DOMAIN=math

BASE_MODEL="${BASE_MODEL}" TRAINED_MODEL="${POST_TRAINED_MODEL}" \
OUTPUT_DIR="${SUBSPACE_DIR}" SELECTION=exact_support bash run_extract_subspace.sh

# Vanilla SFT
MODEL_PATH="${POST_TRAINED_MODEL}" TRAIN_FILES="${SFT_TRAIN}" VAL_FILES="${SFT_VAL}" \
OUTPUT_ROOT=checkpoints/released_continual_vanilla_sft700 SFT_VARIANT=full \
TOTAL_TRAINING_STEPS=700 LR=1e-7 MODEL_DTYPE=fp32 EVAL_DOMAIN="${EVAL_DOMAIN}" STAGES=train,eval \
  bash run_sft.sh

# OPSFT: same initial checkpoint and trajectories.
MODEL_PATH="${POST_TRAINED_MODEL}" TRAIN_FILES="${SFT_TRAIN}" VAL_FILES="${SFT_VAL}" \
OUTPUT_ROOT=checkpoints/released_continual_opsft_sft700 SFT_VARIANT=opsft \
SFT_SUBSPACE_DIR="${SUBSPACE_DIR}" TOTAL_TRAINING_STEPS=700 LR=1e-7 MODEL_DTYPE=fp32 \
EVAL_DOMAIN="${EVAL_DOMAIN}" STAGES=train,eval bash run_sft.sh
```

For Code, download the trajectory paths in the Code row from `${TRAJECTORIES_REPO}` and download `shufanshen/Qwen3-4B-GRPO-CodeEurus-100-steps` with `--local-dir application_continual_posttraining/code/qwen3_4b_eurus_grpo100`. Then set `POST_TRAINED_MODEL` to that directory, `SUBSPACE_DIR=analysis/continual_code_grpo100_exact_support`, and `EVAL_DOMAIN=code`.

## Route B: Full Reproduction

Generate trajectories following the [Math guide](math.md) or [Code guide](code.md), then run the matching 100-step GRPO phase and local extraction.

```bash
# Math
export BASE_MODEL=/path/to/Qwen3-4B
export RL_TRAIN=data/DeepMath-103K/train_filtered_level6.parquet
export RL_VAL=data/aime24/test.parquet,data/aime25/test.parquet
export SFT_TRAIN=data/deepmath_qwen3_30b_trajectories/train.parquet
export SFT_VAL=data/deepmath_qwen3_30b_trajectories/validation.parquet
BASE_MODEL="${BASE_MODEL}" TRAIN_FILES="${RL_TRAIN}" VAL_FILES="${RL_VAL}" \
OUTPUT_ROOT=checkpoints/continual_math_grpo100 GRPO_STEPS=100 LR=1e-6 ROLLOUT_TP_SIZE=4 ROLLOUT_N=8 \
  bash run_grpo.sh
export POST_TRAINED_MODEL=checkpoints/continual_math_grpo100/global_step_100/actor/huggingface_merged
BASE_MODEL="${BASE_MODEL}" TRAINED_MODEL="${POST_TRAINED_MODEL}" \
OUTPUT_DIR=analysis/continual_math_grpo100_exact_support SELECTION=exact_support bash run_extract_subspace.sh

# Code: change the data paths and output names.
export RL_TRAIN=data/Eurus/code_train.parquet
export RL_VAL=data/Eurus/code_validation.parquet
export SFT_TRAIN=data/eurus_qwen3_30b_trajectories/train.parquet
export SFT_VAL=data/eurus_qwen3_30b_trajectories/validation.parquet
BASE_MODEL="${BASE_MODEL}" TRAIN_FILES="${RL_TRAIN}" VAL_FILES="${RL_VAL}" \
OUTPUT_ROOT=checkpoints/continual_code_grpo100 GRPO_STEPS=100 LR=1e-6 ROLLOUT_TP_SIZE=4 ROLLOUT_N=8 \
  bash run_grpo.sh
```

For either domain, run the two Route A SFT commands with the locally generated trajectory paths, post-trained checkpoint, identified direction, and matching `EVAL_DOMAIN`.

## Comparison Checklist

| Item | Vanilla SFT | OPSFT |
|------|-------------|-------|
| Initial model | Same post-trained GRPO checkpoint | Same post-trained GRPO checkpoint |
| Trajectories | Same verified dataset | Same verified dataset |
| SFT settings | FP32 / 700 / `1e-7` | FP32 / 700 / `1e-7` |
| Difference | `SFT_VARIANT=full` | `SFT_VARIANT=opsft` with the GRPO direction |
