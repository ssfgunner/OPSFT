# Math Task Reproduction Guide

This guide covers OPSFT on the **DeepMath-103K / DAPO math domain** with Qwen3-4B. It provides two equivalent reproduction routes.

> **Prerequisites**: Complete [environment setup](environment.md) first, including model downloads and dataset preparation.

---

## Reproduction Routes

| Route | You provide | Stages to run |
|-------|-------------|---------------|
| **A. Released-asset reproduction** | Base model plus assets downloaded from our Hugging Face dataset repository | Identify direction from the released GRPO checkpoint → SFT |
| **B. Full reproduction** | DeepMath/DAPO RL prompts and the teacher model | Generate trajectories → train GRPO → identify direction → SFT |

## Route A: Released-Asset Reproduction

Download assets, identify the direction locally, and run SFT:

The trajectories are in [`shufanshen/SFT-Trajectories`](https://huggingface.co/datasets/shufanshen/SFT-Trajectories); the matching checkpoint is [`shufanshen/Qwen3-4B-GRPO-DeepMath-150-steps`](https://huggingface.co/shufanshen/Qwen3-4B-GRPO-DeepMath-150-steps). The [OPSFT collection](https://huggingface.co/collections/shufanshen/opsft-6aa781ffb646c53df3707a3b) also provides Qwen3-4B/8B/1.7B DeepMath checkpoints at 50, 100, and 150 steps.

```bash
export ROOT=/path/to/OPSFT
export BASE_MODEL=/path/to/Qwen3-4B
export TRAJECTORIES_REPO=shufanshen/SFT-Trajectories
cd "${ROOT}"

huggingface-cli download "${TRAJECTORIES_REPO}" --repo-type dataset --local-dir . \
  --include 'math/deepmath_qwen3_30b_trajectories/*'
huggingface-cli download shufanshen/Qwen3-4B-GRPO-DeepMath-150-steps \
  --local-dir math/qwen3_4b_deepmath_grpo150

BASE_MODEL="${BASE_MODEL}" \
TRAINED_MODEL=math/qwen3_4b_deepmath_grpo150 \
OUTPUT_DIR=analysis/deepmath_grpo150_exact_support \
SELECTION=exact_support \
  bash run_extract_subspace.sh
```

Run the two SFT arms below after direction identification. Route A intentionally derives the direction locally rather than distributing a precomputed mask/direction.

```bash
# Vanilla SFT: start from the base model with released trajectories.
MODEL_PATH="${BASE_MODEL}" \
TRAIN_FILES=math/deepmath_qwen3_30b_trajectories/train.parquet \
VAL_FILES=math/deepmath_qwen3_30b_trajectories/validation.parquet \
OUTPUT_ROOT=checkpoints/released_math_vanilla_sft700 \
SFT_VARIANT=full TOTAL_TRAINING_STEPS=700 LR=1e-7 MODEL_DTYPE=fp32 \
EVAL_DOMAIN=math STAGES=train,eval \
  bash run_sft.sh

# OPSFT: identical initialization and trajectories, with the extracted constraint.
MODEL_PATH="${BASE_MODEL}" \
TRAIN_FILES=math/deepmath_qwen3_30b_trajectories/train.parquet \
VAL_FILES=math/deepmath_qwen3_30b_trajectories/validation.parquet \
OUTPUT_ROOT=checkpoints/released_math_opsft_sft700 \
SFT_VARIANT=opsft SFT_SUBSPACE_DIR=analysis/deepmath_grpo150_exact_support \
TOTAL_TRAINING_STEPS=700 LR=1e-7 MODEL_DTYPE=fp32 \
EVAL_DOMAIN=math STAGES=train,eval \
  bash run_sft.sh
```

---

## Route B: Full Reproduction

## 1. Data

### RL Prompt Datasets

| Dataset | Path | Format |
|---------|------|--------|
| DeepMath-103K (train) | `data/DeepMath-103K/train_filtered_level6.parquet` | `prompt`, `reward_model` |
| DAPO-math-17k (train) | `data/DAPO-math-17k/train.parquet` | `prompt`, `reward_model` |

### Evaluation Benchmarks

| Benchmark | Path | Problems |
|-----------|------|-----------|
| AIME 2024 | `data/aime24/test.jsonl` | 30 |
| AIME 2025 | `data/aime25/test.jsonl` | 30 |
| HMMT 2025 Feb | `data/hmmt25_feb/test.jsonl` | 30 |
| HMMT 2025 Nov | `data/hmmt25_nov/test.jsonl` | 30 |

---

## 2. Trajectory Generation

Generate math-verified SFT trajectories from the DeepMath prompts using a teacher model.

```bash
INPUT=data/DeepMath-103K/train_filtered_level6.parquet \
OUTPUT_DIR=data/deepmath_qwen3_30b_trajectories \
TEACHER_MODEL=/path/to/Qwen3-30B-A3B-Instruct-2507 \
VERIFIER=math_verify \
ENABLE_THINKING=0 \
    bash run_generate_trajectories.sh
```

> Use `VERIFIER=math_dapo` for DAPO-style math verification.

**Key parameters**:

| Variable | Default | Description |
|----------|---------|-------------|
| `INPUT` | (required) | RL prompt parquet |
| `OUTPUT_DIR` | (required) | Output directory for trajectories |
| `TEACHER_MODEL` | (required) | Hugging Face teacher model path |
| `VERIFIER` | `code` | Set to `math_verify` or `math_dapo` for this task |
| `TEACHER_TP_SIZE` | `8` | Tensor parallelism for vLLM teacher |
| `TEMPERATURE` | `0.6` | Sampling temperature |
| `TOP_P` | `0.95` | Nucleus sampling |
| `SAMPLES_PER_PROMPT` | `1` | Number of samples per prompt |
| `ENABLE_THINKING` | `0` | Disable thinking in teacher generation |
| `SHARD_COUNT` | `1` | Number of shards for parallel generation |

> For large datasets, use sharding (`SHARD_COUNT=4 SHARD_INDEX=0..3`) then merge with `run_merge_trajectory_shards.sh`.

**Output**: `data/deepmath_qwen3_30b_trajectories/train.parquet` and `validation.parquet`

---

## 3. GRPO Training

Train the base model with GRPO on DeepMath math prompts.

```bash
BASE_MODEL=/path/to/Qwen3-4B \
TRAIN_FILES=data/DeepMath-103K/train_filtered_level6.parquet \
VAL_FILES=data/aime24/test.parquet,data/aime25/test.parquet \
OUTPUT_ROOT=checkpoints/qwen3_4b_deepmath_grpo \
GRPO_STEPS=150 \
LR=1e-6 \
ROLLOUT_TP_SIZE=4 \
ROLLOUT_N=8 \
    bash run_grpo.sh
```

**Key parameters**:

| Variable | Default | Description |
|----------|---------|-------------|
| `BASE_MODEL` | (required) | Base Hugging Face model path |
| `TRAIN_FILES` | (required) | GRPO training parquet |
| `VAL_FILES` | (required) | Comma-separated validation parquets |
| `OUTPUT_ROOT` | (required) | GRPO checkpoint output root |
| `GRPO_STEPS` | `150` | Total GRPO training steps |
| `LR` | `1e-6` | Learning rate |
| `NUM_GPUS` | `8` | Number of GPUs |
| `ROLLOUT_TP_SIZE` | `4` | Rollout tensor parallelism |
| `ROLLOUT_N` | `8` | Samples per prompt |
| `TRAIN_BATCH_SIZE` | `128` | PPO mini-batch size |
| `SAVE_FREQ` | `50` | Save checkpoint every N steps |
| `ENABLE_THINKING` | `0` | Disable thinking mode |
| `RESUME_MODE` | `disable` | `disable`, `auto`, or `resume_path` |

**Output**: `checkpoints/qwen3_4b_deepmath_grpo/global_step_150/actor/huggingface_merged`

---

## 4. Direction Identification

Identify the on-policy parameter-update direction (mask + direction vector) from GRPO checkpoints.

```bash
BASE_MODEL=/path/to/Qwen3-4B \
TRAINED_MODEL=checkpoints/qwen3_4b_deepmath_grpo/global_step_150/actor/huggingface_merged \
OUTPUT_DIR=analysis/deepmath_step_150_exact_support \
SELECTION=exact_support \
    bash run_extract_subspace.sh
```

For top-k selection or random controls:

```bash
SELECTION=topk KEEP_RATIO=0.05 bash run_extract_subspace.sh
GENERATE_RANDOM_CONTROLS=1 bash run_extract_subspace.sh
```

**Key parameters**:

| Variable | Default | Description |
|----------|---------|-------------|
| `BASE_MODEL` | (required) | Base Hugging Face model path |
| `TRAINED_MODEL` | (required) | GRPO-trained merged model |
| `OUTPUT_DIR` | (required) | Output directory for artifacts |
| `SELECTION` | `exact_support` | `exact_support` or `topk` |
| `KEEP_RATIO` | `0.05` | Fraction of parameters (topk only) |
| `GENERATE_RANDOM_CONTROLS` | `0` | Generate count-matched random masks |
| `RANDOM_SEEDS` | `42 123 2026` | Seeds for random controls |

**Output**: `analysis/deepmath_step_150_exact_support/update_mask.pt`, `parameter_updates.pt`

---

## 5. SFT

### Vanilla SFT Baseline

```bash
MODEL_PATH=/path/to/Qwen3-4B \
TRAIN_FILES=data/deepmath_qwen3_30b_trajectories/train.parquet \
VAL_FILES=data/deepmath_qwen3_30b_trajectories/validation.parquet \
OUTPUT_ROOT=checkpoints/vanilla_sft_math \
SFT_VARIANT=full \
TOTAL_TRAINING_STEPS=700 \
LR=1e-7 \
MODEL_DTYPE=fp32 \
STAGES=train,eval \
EVAL_DOMAIN=math \
    bash run_sft.sh
```

### OPSFT (Constrained SFT)

```bash
MODEL_PATH=/path/to/Qwen3-4B \
TRAIN_FILES=data/deepmath_qwen3_30b_trajectories/train.parquet \
VAL_FILES=data/deepmath_qwen3_30b_trajectories/validation.parquet \
OUTPUT_ROOT=checkpoints/opsft_sft_math \
SFT_VARIANT=opsft \
SFT_SUBSPACE_DIR=analysis/deepmath_step_150_exact_support \
TOTAL_TRAINING_STEPS=700 \
LR=1e-7 \
MODEL_DTYPE=fp32 \
STAGES=train,eval \
EVAL_DOMAIN=math \
    bash run_sft.sh
```

### SFT Variants

| `SFT_VARIANT` | Mask | Direction | Description |
|---------------|------|-----------|-------------|
| `full` | null | null | Vanilla SFT (no constraint) |
| `opsft_location` | GRPO mask | null | Location-only constraint |
| `opsft` | GRPO mask | GRPO direction | Full OPSFT (location + direction) |
| `random_location` | Random mask | null | Random location control |

### Key SFT Parameters

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | (required) | Initial model (base or GRPO checkpoint) |
| `TRAIN_FILES` | (required) | SFT training parquet |
| `VAL_FILES` | (required) | SFT validation parquet |
| `OUTPUT_ROOT` | (required) | Checkpoint output root |
| `SFT_VARIANT` | `full` | Constraint variant (see table above) |
| `SFT_SUBSPACE_DIR` | (required if constrained; legacy name) | Directory with the identified `update_mask.pt` and `parameter_updates.pt` direction artifacts |
| `TOTAL_TRAINING_STEPS` | `700` | Total training steps |
| `LR` | `1e-7` | Learning rate |
| `MODEL_DTYPE` | `fp32` | Training precision: `fp32` or `bf16` |
| `NUM_GPUS` | `8` | Number of GPUs |
| `TRAIN_BATCH_SIZE` | `64` | Training batch size |
| `MAX_LENGTH` | `16384` | Max sequence length |
| `SAVE_FREQ` | `100` | Save checkpoint every N steps |
| `STAGES` | `train,eval` | Stages to run: `train`, `eval`, or both |
| `EVAL_DOMAIN` | `math` | Set to `math` for this task |
| `EVAL_STEPS` | `${TOTAL_TRAINING_STEPS}` | Comma-separated checkpoint steps to evaluate |

---

## 6. Evaluation

### Math Benchmarks (AIME 2024/2025, HMMT 2025)

Standalone evaluation:

```bash
MODEL_PATH=checkpoints/opsft_sft_math/opsft/global_step_700/huggingface_merged \
MODEL_NAME=opsft_step700 \
DATASETS="aime24 aime25 hmmt25_feb hmmt25_nov" \
GPU_GROUPS="0,1 2,3 4,5 6,7" \
N=32 \
    bash math_eval/run_eval_math.sh \
        checkpoints/opsft_sft_math/opsft/global_step_700/huggingface_merged \
        opsft_step700
```

> If `STAGES` includes `eval` and `EVAL_DOMAIN=math`, `run_sft.sh` will automatically call `math_eval/run_eval_math.sh` after training.

**Paper-aligned protocol**: N=32, temperature=1.0, top-p=1.0, max_tokens=16384.

| Variable | Default | Description |
|----------|---------|-------------|
| `DATASETS` | `aime24 aime25 hmmt25_feb hmmt25_nov` | Datasets to evaluate |
| `GPU_GROUPS` | `0,1 2,3 4,5 6,7` | GPU group per dataset (2 GPUs each) |
| `N` | `32` | Samples per problem |
| `MAX_TOKENS` | `16384` | Max generation tokens |
| `MAX_MODEL_LEN` | `32768` | Max model context length |
| `TEMPERATURE` | `1.0` | Sampling temperature |
| `TOP_P` | `1.0` | Top-p sampling |
| `EVAL_SEED` | `42` | Evaluation seed |
| `ENABLE_THINKING` | `0` | Enable thinking mode |
| `OUTPUT_TAG` | (none) | Subdirectory under `math_eval/eval_outputs/` |
| `STAGGER_SECONDS` | `0` | Delay between dataset launches to reduce startup contention |

> Each of the 4 datasets runs concurrently on a 2-GPU group (8 GPUs total).

