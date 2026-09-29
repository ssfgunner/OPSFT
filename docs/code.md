# Code Task Reproduction Guide

This guide covers OPSFT on the **Eurus code domain** with Qwen3-4B. It provides two equivalent reproduction routes.

> **Prerequisites**: Complete [environment setup](environment.md) first, including model downloads and dataset preparation.

---

## Reproduction Routes

| Route | You provide | Stages to run |
|-------|-------------|---------------|
| **A. Released-asset reproduction** | Base model plus assets downloaded from our Hugging Face dataset repository | Identify direction from the released GRPO checkpoint → SFT |
| **B. Full reproduction** | Eurus RL prompts and the teacher model | Generate trajectories → train GRPO → identify direction → SFT |

## Route A: Released-Asset Reproduction

Download assets, identify the direction locally, and run SFT:

The trajectories are in [`shufanshen/SFT-Trajectories`](https://huggingface.co/datasets/shufanshen/SFT-Trajectories); the matching checkpoint is [`shufanshen/Qwen3-4B-GRPO-CodeEurus-150-steps`](https://huggingface.co/shufanshen/Qwen3-4B-GRPO-CodeEurus-150-steps), also listed in the [OPSFT collection](https://huggingface.co/collections/shufanshen/opsft-6aa781ffb646c53df3707a3b).

```bash
export ROOT=/path/to/OPSFT
export BASE_MODEL=/path/to/Qwen3-4B
export TRAJECTORIES_REPO=shufanshen/SFT-Trajectories
cd "${ROOT}"

huggingface-cli download "${TRAJECTORIES_REPO}" --repo-type dataset --local-dir . \
  --include 'code/eurus_qwen3_30b_trajectories/*'
huggingface-cli download shufanshen/Qwen3-4B-GRPO-CodeEurus-150-steps \
  --local-dir code/qwen3_4b_eurus_grpo150

BASE_MODEL="${BASE_MODEL}" \
TRAINED_MODEL=code/qwen3_4b_eurus_grpo150 \
OUTPUT_DIR=analysis/eurus_grpo150_exact_support \
SELECTION=exact_support \
  bash run_extract_subspace.sh
```

Run the two SFT arms below after direction identification. Route A intentionally derives the direction locally rather than distributing a precomputed mask/direction.

```bash
# Vanilla SFT: start from the base model with released trajectories.
MODEL_PATH="${BASE_MODEL}" \
TRAIN_FILES=code/eurus_qwen3_30b_trajectories/train.parquet \
VAL_FILES=code/eurus_qwen3_30b_trajectories/validation.parquet \
OUTPUT_ROOT=checkpoints/released_code_vanilla_sft700 \
SFT_VARIANT=full TOTAL_TRAINING_STEPS=700 LR=1e-7 MODEL_DTYPE=fp32 \
EVAL_DOMAIN=code STAGES=train,eval \
  bash run_sft.sh

# OPSFT: identical initialization and trajectories, with the extracted constraint.
MODEL_PATH="${BASE_MODEL}" \
TRAIN_FILES=code/eurus_qwen3_30b_trajectories/train.parquet \
VAL_FILES=code/eurus_qwen3_30b_trajectories/validation.parquet \
OUTPUT_ROOT=checkpoints/released_code_opsft_sft700 \
SFT_VARIANT=opsft SFT_SUBSPACE_DIR=analysis/eurus_grpo150_exact_support \
TOTAL_TRAINING_STEPS=700 LR=1e-7 MODEL_DTYPE=fp32 \
EVAL_DOMAIN=code STAGES=train,eval \
  bash run_sft.sh
```

---

## Route B: Full Reproduction

## 1. Data

### RL Prompt Data

| Dataset | Path | Format |
|---------|------|--------|
| Eurus-2 (code train) | `data/Eurus/code_train.parquet` | `prompt`, `reward_model`, `data_source` |
| Eurus-2 (code validation) | `data/Eurus/code_validation.parquet` | same |

### Evaluation Benchmarks

| Benchmark | Path | Problems |
|-----------|------|-----------|
| HumanEval+ | `code_eval/data/HumanEvalPlus.jsonl` | 164 |
| MBPP+ | `code_eval/data/MbppPlus.jsonl` | 378 |
| LiveCodeBench v6 | `code_eval/data/code_generation_lite/` | test–test6 |

---

## 2. Trajectory Generation

Generate execution-verified SFT trajectories from the Eurus code prompts using a teacher model.

```bash
INPUT=data/Eurus/code_train.parquet \
OUTPUT_DIR=data/eurus_qwen3_30b_trajectories \
TEACHER_MODEL=/path/to/Qwen3-30B-A3B-Instruct-2507 \
VERIFIER=code \
ENABLE_THINKING=0 \
REJECT_THINKING_TAGS=1 \
    bash run_generate_trajectories.sh
```

**Key parameters**:

| Variable | Default | Description |
|----------|---------|-------------|
| `INPUT` | (required) | RL prompt parquet |
| `OUTPUT_DIR` | (required) | Output directory for trajectories |
| `TEACHER_MODEL` | (required) | Hugging Face teacher model path |
| `VERIFIER` | `code` | Must be `code` for this task |
| `TEACHER_TP_SIZE` | `8` | Tensor parallelism for vLLM teacher |
| `TEMPERATURE` | `0.6` | Sampling temperature |
| `TOP_P` | `0.95` | Nucleus sampling |
| `SAMPLES_PER_PROMPT` | `1` | Number of samples per prompt |
| `ENABLE_THINKING` | `0` | Disable thinking in teacher generation |
| `REJECT_THINKING_TAGS` | `1` | Reject any response with `<think>` tags |
| `SHARD_COUNT` | `1` | Number of shards for parallel generation |

> For large datasets, use sharding (`SHARD_COUNT=4 SHARD_INDEX=0..3`) then merge with `run_merge_trajectory_shards.sh`.

**Output**: `data/eurus_qwen3_30b_trajectories/train.parquet` and `validation.parquet`

---

## 3. GRPO Training

Train the base model with GRPO on Eurus code prompts.

```bash
BASE_MODEL=/path/to/Qwen3-4B \
TRAIN_FILES=data/Eurus/code_train.parquet \
VAL_FILES=data/Eurus/code_validation.parquet \
OUTPUT_ROOT=checkpoints/qwen3_4b_eurus_grpo \
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

**Output**: `checkpoints/qwen3_4b_eurus_grpo/global_step_150/actor/huggingface_merged`

---

## 4. Direction Identification

Identify the on-policy parameter-update direction (mask + direction vector) from GRPO checkpoints.

```bash
BASE_MODEL=/path/to/Qwen3-4B \
TRAINED_MODEL=checkpoints/qwen3_4b_eurus_grpo/global_step_150/actor/huggingface_merged \
OUTPUT_DIR=analysis/eurus_step_150_exact_support \
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

**Output**: `analysis/eurus_step_150_exact_support/update_mask.pt`, `parameter_updates.pt`

---

## 5. SFT

### Vanilla SFT Baseline

```bash
MODEL_PATH=/path/to/Qwen3-4B \
TRAIN_FILES=data/eurus_qwen3_30b_trajectories/train.parquet \
VAL_FILES=data/eurus_qwen3_30b_trajectories/validation.parquet \
OUTPUT_ROOT=checkpoints/vanilla_sft \
SFT_VARIANT=full \
TOTAL_TRAINING_STEPS=700 \
LR=1e-7 \
MODEL_DTYPE=fp32 \
STAGES=train,eval \
EVAL_DOMAIN=code \
    bash run_sft.sh
```

### OPSFT (Constrained SFT)

```bash
MODEL_PATH=/path/to/Qwen3-4B \
TRAIN_FILES=data/eurus_qwen3_30b_trajectories/train.parquet \
VAL_FILES=data/eurus_qwen3_30b_trajectories/validation.parquet \
OUTPUT_ROOT=checkpoints/opsft_sft \
SFT_VARIANT=opsft \
SFT_SUBSPACE_DIR=analysis/eurus_step_150_exact_support \
TOTAL_TRAINING_STEPS=700 \
LR=1e-7 \
MODEL_DTYPE=fp32 \
STAGES=train,eval \
EVAL_DOMAIN=code \
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
| `EVAL_DOMAIN` | `code` | Evaluation domain (set to `code` for this task) |
| `EVAL_STEPS` | `${TOTAL_TRAINING_STEPS}` | Comma-separated checkpoint steps to evaluate |

---

## 6. Evaluation

### Code Benchmarks (HumanEval+, MBPP+, LiveCodeBench v6)

Standalone evaluation:

```bash
MODEL_DIR=checkpoints/opsft_sft/opsft/global_step_700/huggingface_merged \
RUN_LABEL=opsft_step700 \
BENCHMARKS=evalplus,lcb \
    bash run_code_eval.sh
```

> If `STAGES` includes `eval`, `run_sft.sh` will automatically call `run_code_eval.sh` with the same parameters after training.

**Paper-aligned protocol**: N=4, temperature=1.0, top-p=1.0, max_new_tokens=16384.

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_DIR` | (required) | Hugging Face model directory |
| `BENCHMARKS` | `evalplus,lcb` | Benchmarks to run |
| `EVALPLUS_N` | `4` | Samples per task |
| `EVALPLUS_TEMPERATURE` | `1.0` | Sampling temperature |
| `EVALPLUS_TOP_P` | `1.0` | Top-p sampling |
| `EVALPLUS_MAX_NEW_TOKENS` | `16384` | Max generation tokens |
| `LCB_N` | `4` | LCB samples per task |
| `LCB_RELEASE_VERSION` | `release_v6` | LCB release version |
| `LCB_TEMPERATURE` | `1.0` | LCB sampling temperature |
| `LCB_TOP_P` | `1.0` | LCB top-p |
| `LCB_MAX_TOKENS` | `16384` | LCB max generation tokens |
| `HUMANEVAL_GPU` | `0` | GPU for HumanEval+ |
| `MBPP_GPU` | `1` | GPU for MBPP+ |
| `LCB_CUDA_VISIBLE_DEVICES` | `2,3` | GPUs for LCB (TP=2) |

> HumanEval+ and MBPP+ run concurrently on separate single GPUs; LiveCodeBench runs on 2 GPUs with tensor parallelism=2.
