# Application 1: Training-Efficiency Improvement

This application corresponds to Section 4.1 of the paper. It combines a **short GRPO warm-up** for identifying an on-policy parameter-update direction with a **short OPSFT phase**. The paper setting uses 50 GRPO steps to identify the direction, followed by 100 OPSFT steps. The vanilla-SFT reference uses 700 SFT steps.

> **Prerequisites**: Complete [environment setup](environment.md). Start with Route A for direct comparison using our released assets; use Route B only when reproducing trajectory generation and the 50-step GRPO warm-up yourself.

## Route A: Released-Asset Reproduction

This route skips teacher inference and GRPO training, but still runs `run_extract_subspace.sh` locally. Trajectories are in [`shufanshen/SFT-Trajectories`](https://huggingface.co/datasets/shufanshen/SFT-Trajectories), and checkpoints are listed in the [OPSFT collection](https://huggingface.co/collections/shufanshen/opsft-6aa781ffb646c53df3707a3b):

| Setting | Teacher-verified trajectories | Merged GRPO warm-up checkpoint |
|---------|-------------------------------|--------------------------------|
| Math | `application_efficiency/math/deepmath_qwen3_30b_trajectories/{train,validation}.parquet` | [`shufanshen/Qwen3-4B-GRPO-DeepMath-50-steps`](https://huggingface.co/shufanshen/Qwen3-4B-GRPO-DeepMath-50-steps) |
| Code | `application_efficiency/code/eurus_qwen3_30b_trajectories/{train,validation}.parquet` | No 50-step CodeEurus checkpoint is currently published |

```bash
export ROOT=/path/to/OPSFT
export BASE_MODEL=/path/to/Qwen3-4B
export TRAJECTORIES_REPO=shufanshen/SFT-Trajectories
cd "${ROOT}"

# Math: download trajectories and the released 50-step GRPO checkpoint.
huggingface-cli download "${TRAJECTORIES_REPO}" --repo-type dataset --local-dir . \
  --include 'application_efficiency/math/deepmath_qwen3_30b_trajectories/*'
huggingface-cli download shufanshen/Qwen3-4B-GRPO-DeepMath-50-steps \
  --local-dir application_efficiency/math/qwen3_4b_deepmath_grpo50
BASE_MODEL="${BASE_MODEL}" \
TRAINED_MODEL=application_efficiency/math/qwen3_4b_deepmath_grpo50 \
OUTPUT_DIR=analysis/efficiency_math_grpo50_exact_support \
SELECTION=exact_support \
  bash run_extract_subspace.sh

# Code: use Route B to produce the required 50-step checkpoint; only 100/150-step
# CodeEurus checkpoints are currently published.
```

### Math: Run Short OPSFT and the Vanilla Reference

```bash
# Short OPSFT: the GRPO checkpoint supplies only the constraint.
MODEL_PATH="${BASE_MODEL}" \
TRAIN_FILES=application_efficiency/math/deepmath_qwen3_30b_trajectories/train.parquet \
VAL_FILES=application_efficiency/math/deepmath_qwen3_30b_trajectories/validation.parquet \
OUTPUT_ROOT=checkpoints/released_efficiency_math_opsft100 \
SFT_VARIANT=opsft \
SFT_SUBSPACE_DIR=analysis/efficiency_math_grpo50_exact_support \
TOTAL_TRAINING_STEPS=100 LR=1e-6 MODEL_DTYPE=fp32 \
EVAL_DOMAIN=math STAGES=train,eval \
  bash run_sft.sh

# 700-step Vanilla-SFT reference: same trajectories, no constraint.
MODEL_PATH="${BASE_MODEL}" \
TRAIN_FILES=application_efficiency/math/deepmath_qwen3_30b_trajectories/train.parquet \
VAL_FILES=application_efficiency/math/deepmath_qwen3_30b_trajectories/validation.parquet \
OUTPUT_ROOT=checkpoints/released_efficiency_math_vanilla_sft700 \
SFT_VARIANT=full \
TOTAL_TRAINING_STEPS=700 LR=1e-7 MODEL_DTYPE=fp32 \
EVAL_DOMAIN=math STAGES=train,eval \
  bash run_sft.sh
```

### Code: Run Short OPSFT and the Vanilla Reference

```bash
# Short OPSFT: the GRPO checkpoint supplies only the constraint.
MODEL_PATH="${BASE_MODEL}" \
TRAIN_FILES=application_efficiency/code/eurus_qwen3_30b_trajectories/train.parquet \
VAL_FILES=application_efficiency/code/eurus_qwen3_30b_trajectories/validation.parquet \
OUTPUT_ROOT=checkpoints/released_efficiency_code_opsft100 \
SFT_VARIANT=opsft \
SFT_SUBSPACE_DIR=analysis/efficiency_code_grpo50_exact_support \
TOTAL_TRAINING_STEPS=100 LR=1e-6 MODEL_DTYPE=fp32 \
EVAL_DOMAIN=code STAGES=train,eval \
  bash run_sft.sh

# 700-step Vanilla-SFT reference: same trajectories, no constraint.
MODEL_PATH="${BASE_MODEL}" \
TRAIN_FILES=application_efficiency/code/eurus_qwen3_30b_trajectories/train.parquet \
VAL_FILES=application_efficiency/code/eurus_qwen3_30b_trajectories/validation.parquet \
OUTPUT_ROOT=checkpoints/released_efficiency_code_vanilla_sft700 \
SFT_VARIANT=full \
TOTAL_TRAINING_STEPS=700 LR=1e-7 MODEL_DTYPE=fp32 \
EVAL_DOMAIN=code STAGES=train,eval \
  bash run_sft.sh
```

The math evaluation uses AIME 2024/2025 and HMMT 2025 with `N=32` by default; code evaluation uses HumanEval+, MBPP+, and LiveCodeBench v6 with `N=4`. See the task guides for evaluation options.

## Route B: Full Reproduction

Use this route when you want to independently generate trajectories and identify the direction from your own 50-step GRPO warm-up.

### Math Setting: Qwen3-4B on DeepMath

```bash
export ROOT=/path/to/OPSFT
export BASE_MODEL=/path/to/Qwen3-4B
export RL_TRAIN=data/DeepMath-103K/train_filtered_level6.parquet
export RL_VAL=data/aime24/test.parquet,data/aime25/test.parquet
export SFT_TRAIN=data/deepmath_qwen3_30b_trajectories/train.parquet
export SFT_VAL=data/deepmath_qwen3_30b_trajectories/validation.parquet

# First generate SFT_TRAIN/SFT_VAL according to the Math guide, then identify the direction.
BASE_MODEL="${BASE_MODEL}" TRAIN_FILES="${RL_TRAIN}" VAL_FILES="${RL_VAL}" \
OUTPUT_ROOT=checkpoints/efficiency_math_grpo50 GRPO_STEPS=50 LR=1e-6 \
ROLLOUT_TP_SIZE=4 ROLLOUT_N=8 bash run_grpo.sh
BASE_MODEL="${BASE_MODEL}" \
TRAINED_MODEL=checkpoints/efficiency_math_grpo50/global_step_50/actor/huggingface_merged \
OUTPUT_DIR=analysis/efficiency_math_grpo50_exact_support SELECTION=exact_support \
  bash run_extract_subspace.sh

MODEL_PATH="${BASE_MODEL}" TRAIN_FILES="${SFT_TRAIN}" VAL_FILES="${SFT_VAL}" \
OUTPUT_ROOT=checkpoints/efficiency_math_opsft100 SFT_VARIANT=opsft \
SFT_SUBSPACE_DIR=analysis/efficiency_math_grpo50_exact_support \
TOTAL_TRAINING_STEPS=100 LR=1e-6 MODEL_DTYPE=fp32 EVAL_DOMAIN=math STAGES=train,eval \
  bash run_sft.sh
```

### Code Setting: Qwen3-4B on Eurus

```bash
export BASE_MODEL=/path/to/Qwen3-4B
export RL_TRAIN=data/Eurus/code_train.parquet
export RL_VAL=data/Eurus/code_validation.parquet
export SFT_TRAIN=data/eurus_qwen3_30b_trajectories/train.parquet
export SFT_VAL=data/eurus_qwen3_30b_trajectories/validation.parquet

# First generate SFT_TRAIN/SFT_VAL according to the Code guide, then identify the direction.
BASE_MODEL="${BASE_MODEL}" TRAIN_FILES="${RL_TRAIN}" VAL_FILES="${RL_VAL}" \
OUTPUT_ROOT=checkpoints/efficiency_code_grpo50 GRPO_STEPS=50 LR=1e-6 \
ROLLOUT_TP_SIZE=4 ROLLOUT_N=8 bash run_grpo.sh
BASE_MODEL="${BASE_MODEL}" \
TRAINED_MODEL=checkpoints/efficiency_code_grpo50/global_step_50/actor/huggingface_merged \
OUTPUT_DIR=analysis/efficiency_code_grpo50_exact_support SELECTION=exact_support \
  bash run_extract_subspace.sh

MODEL_PATH="${BASE_MODEL}" TRAIN_FILES="${SFT_TRAIN}" VAL_FILES="${SFT_VAL}" \
OUTPUT_ROOT=checkpoints/efficiency_code_opsft100 SFT_VARIANT=opsft \
SFT_SUBSPACE_DIR=analysis/efficiency_code_grpo50_exact_support \
TOTAL_TRAINING_STEPS=100 LR=1e-6 MODEL_DTYPE=fp32 EVAL_DOMAIN=code STAGES=train,eval \
  bash run_sft.sh
```

For both domains, run the same 700-step Vanilla-SFT commands shown in Route A, replacing the `application_efficiency/...` trajectory paths with `${SFT_TRAIN}` and `${SFT_VAL}`.

## Key Settings

| Component | Math / Code application setting |
|-----------|---------------------------------|
| Direction-identification GRPO | 50 steps, LR `1e-6` |
| Direction selection | `exact_support` |
| OPSFT initialization | Base model |
| OPSFT | 100 steps, FP32, LR `1e-6` |
| Vanilla reference | 700 steps, FP32, LR `1e-7` |
| OPSFT constraint | `SFT_VARIANT=opsft` (location + direction) |
