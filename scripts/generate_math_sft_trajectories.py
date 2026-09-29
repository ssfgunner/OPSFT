#!/usr/bin/env python3
"""Generate verified SFT trajectories from an RL prompt parquet with vLLM."""

import argparse
import ast
import json
import os
import re
import sys
from pathlib import Path

# Prefer the repository's verl source tree over an unrelated installed verl package.
LOCAL_VERL_ROOT = Path(__file__).resolve().parents[1] / "verl"
if LOCAL_VERL_ROOT.is_dir():
    sys.path.insert(0, str(LOCAL_VERL_ROOT))

import numpy as np
import pandas as pd
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


THINKING_INSTRUCTION_RE = re.compile(
    r"\s*You need to think first then write(?: the)?(?: Python)? code\.?\s*$",
    flags=re.IGNORECASE,
)


def decode_structured_value(value):
    """Decode chat/reward columns stored either as objects or Python/JSON strings."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if not isinstance(value, str):
        return value

    candidate = value.strip()
    if not candidate or candidate[0] not in "[{":
        return value

    for decoder in (json.loads, ast.literal_eval):
        try:
            decoded = decoder(candidate)
        except (TypeError, ValueError, SyntaxError, json.JSONDecodeError):
            continue
        if isinstance(decoded, (list, dict)):
            return decoded
    return value


def unwrap_prompt(value):
    """Return the final user message from a chat-formatted or plain prompt."""
    value = decode_structured_value(value)
    if isinstance(value, list):
        for message in reversed(value):
            if isinstance(message, dict) and message.get("role") == "user":
                return str(message.get("content", ""))
        return str(value)
    if isinstance(value, dict):
        return str(value.get("content", value))
    return str(value)


def ground_truth(row):
    reward = decode_structured_value(row.get("reward_model", {}))
    if isinstance(reward, dict):
        return str(reward.get("ground_truth", ""))
    return ""


def normalize_prompt(prompt, strip_thinking_instruction):
    if strip_thinking_instruction:
        return THINKING_INSTRUCTION_RE.sub("", prompt)
    return prompt


def score_to_bool(score):
    if isinstance(score, dict):
        return bool(score.get("acc", False))
    return bool(score)


def has_thinking_tag(text):
    return "<think" in text.lower() or "</think" in text.lower()


def write_json_atomic(path, payload):
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    temporary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary_path, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--model",
        default="/cfs_shtx5_serving_3/mlp/training/docker/user/hadoop-xt-productline-apex/shenshufan/huggingface.co/Qwen/Qwen3-235B-A22B-Instruct-2507",
    )
    parser.add_argument("--tensor-parallel-size", type=int, default=8)
    parser.add_argument("--max-model-len", type=int, default=18432)
    parser.add_argument("--max-new-tokens", type=int, default=16384)
    parser.add_argument("--max-num-seqs", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--samples-per-prompt", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=-1)
    parser.add_argument("--deduplicate-prompts", action="store_true")
    parser.add_argument("--validation-ratio", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-verification", action="store_true")
    parser.add_argument(
        "--verifier",
        choices=("math_verify", "math_dapo", "code"),
        default="math_verify",
        help="Answer verifier matching the source dataset's ground-truth format.",
    )
    parser.add_argument(
        "--enable-thinking",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Render the assistant generation prompt in Qwen thinking mode (default: true).",
    )
    parser.add_argument(
        "--strip-thinking-instruction",
        action="store_true",
        help="Remove a trailing 'You need to think first then write ... code.' instruction from each prompt.",
    )
    parser.add_argument(
        "--reject-thinking-tags",
        action="store_true",
        help="Reject candidates containing <think> or </think>; useful for a non-thinking SFT dataset.",
    )
    parser.add_argument(
        "--shard-count",
        type=int,
        default=1,
        help="Number of deterministic prompt shards. Sharded runs write verified_trajectories.parquet for later merging.",
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="Zero-based shard index; each shard receives prompts whose post-filtered position is congruent to this index.",
    )
    args = parser.parse_args()

    if not 0 < args.batch_size <= args.max_num_seqs:
        raise ValueError("--batch-size must be in (0, --max-num-seqs]")
    if args.samples_per_prompt < 1:
        raise ValueError("--samples-per-prompt must be at least 1")
    if not 0 < args.validation_ratio < 1:
        raise ValueError("--validation-ratio must be in (0, 1)")
    if args.verifier == "code" and args.skip_verification:
        raise ValueError("Code trajectories must be execution-verified; remove --skip-verification.")
    if args.shard_count < 1:
        raise ValueError("--shard-count must be at least 1")
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("--shard-index must be in [0, --shard-count)")

    input_path = Path(args.input).resolve()
    model_path = Path(args.model).resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Input parquet does not exist: {input_path}")
    if not (model_path / "config.json").is_file():
        raise FileNotFoundError(f"Model config does not exist: {model_path / 'config.json'}")

    dataframe = pd.read_parquet(input_path)
    required_columns = {"prompt", "reward_model"}
    missing_columns = required_columns - set(dataframe.columns)
    if missing_columns:
        raise ValueError(f"Input parquet is missing required columns: {sorted(missing_columns)}")
    if dataframe.empty:
        raise ValueError("Input parquet is empty")

    prompts = pd.Series([unwrap_prompt(value) for value in dataframe["prompt"]], index=dataframe.index)
    prompts = prompts.map(lambda prompt: normalize_prompt(prompt, args.strip_thinking_instruction))
    if args.deduplicate_prompts:
        keep = ~prompts.duplicated()
        dataframe = dataframe.loc[keep]
        prompts = prompts.loc[keep]
    if args.max_samples > 0:
        dataframe = dataframe.sample(n=min(args.max_samples, len(dataframe)), random_state=args.seed)
        prompts = prompts.loc[dataframe.index]
    global_prompt_count = len(dataframe)
    if args.shard_count > 1:
        shard_positions = np.arange(global_prompt_count) % args.shard_count == args.shard_index
        dataframe = dataframe.iloc[shard_positions].copy()
        prompts = prompts.iloc[shard_positions]
    source_indices = dataframe.index.to_numpy()
    dataframe = dataframe.reset_index(drop=True)
    prompts = prompts.reset_index(drop=True).tolist()
    if not prompts:
        raise ValueError("This shard has no prompts after filtering")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_jsonl = output_dir / "progress.jsonl"
    legacy_progress_path = output_dir / "progress.parquet"
    state_path = output_dir / "progress.json"
    rejection_path = output_dir / "verification_rejections.jsonl"
    config_path = output_dir / "generation_config.json"

    run_config = {
        "batch_size": args.batch_size,
        "deduplicate_prompts": args.deduplicate_prompts,
        "enable_thinking": args.enable_thinking,
        "input": str(input_path),
        "max_model_len": args.max_model_len,
        "max_new_tokens": args.max_new_tokens,
        "max_num_seqs": args.max_num_seqs,
        "max_samples": args.max_samples,
        "model": str(model_path),
        "reject_thinking_tags": args.reject_thinking_tags,
        "samples_per_prompt": args.samples_per_prompt,
        "seed": args.seed,
        "skip_verification": args.skip_verification,
        "strip_thinking_instruction": args.strip_thinking_instruction,
        "tensor_parallel_size": args.tensor_parallel_size,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "validation_ratio": args.validation_ratio,
        "verifier": args.verifier,
    }
    if args.shard_count > 1:
        run_config["global_prompt_count"] = global_prompt_count
        run_config["shard_count"] = args.shard_count
        run_config["shard_index"] = args.shard_index
    if config_path.exists():
        existing_config = json.loads(config_path.read_text(encoding="utf-8"))
        if existing_config != run_config:
            raise ValueError(
                f"Generation configuration differs from {config_path}. Use a fresh output directory for a new run."
            )
    else:
        write_json_atomic(config_path, run_config)

    records = []
    saved_source_indices = set()
    start_index = 0
    if args.resume and progress_jsonl.exists() and state_path.exists():
        with progress_jsonl.open("r", encoding="utf-8") as progress_file:
            for line in progress_file:
                if line.strip():
                    record = json.loads(line)
                    records.append(record)
                    saved_source_indices.add(record["source_index"])
        state = json.loads(state_path.read_text(encoding="utf-8"))
        start_index = int(state["processed_until"])
        print(f"Resuming from prompt {start_index}/{len(prompts)}: {progress_jsonl}")
    elif args.resume and legacy_progress_path.exists() and state_path.exists():
        progress = pd.read_parquet(legacy_progress_path)
        records = progress.to_dict("records")
        saved_source_indices = {record["source_index"] for record in records}
        state = json.loads(state_path.read_text(encoding="utf-8"))
        start_index = int(state["processed_until"])
        print(f"Resuming legacy progress from prompt {start_index}/{len(prompts)}: {legacy_progress_path}")
    elif args.resume and (progress_jsonl.exists() or legacy_progress_path.exists() or state_path.exists()):
        raise RuntimeError(f"Cannot safely resume without matching progress and state files in {output_dir}")
    elif progress_jsonl.exists() or legacy_progress_path.exists() or state_path.exists():
        raise RuntimeError(f"Partial generation exists in {output_dir}; rerun with --resume.")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    rendered = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=args.enable_thinking,
        )
        for prompt in prompts
    ]
    model = LLM(
        model=str(model_path),
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        trust_remote_code=True,
    )
    sampling = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_new_tokens,
        n=args.samples_per_prompt,
        seed=args.seed,
    )

    verifier = None
    if not args.skip_verification:
        if args.verifier == "math_dapo":
            from verl.utils.reward_score.math_dapo import compute_score
        elif args.verifier == "math_verify":
            from verl.utils.reward_score.math_verify import compute_score
        else:
            from verl.utils.reward_score.prime_code import compute_score
        verifier = compute_score

    accepted_count = len(records)
    rejected_count = 0
    rejected_thinking_count = 0
    with progress_jsonl.open("a", encoding="utf-8") as progress_file, rejection_path.open("a", encoding="utf-8") as rejection_file:
        for batch_start in range(start_index, len(prompts), args.batch_size):
            batch_end = min(batch_start + args.batch_size, len(prompts))
            generations = model.generate(rendered[batch_start:batch_end], sampling)
            for offset, generation in enumerate(generations):
                row_index = batch_start + offset
                source_index = int(source_indices[row_index])
                truth = ground_truth(dataframe.iloc[row_index])
                candidates = [output.text.strip() for output in generation.outputs]
                selected = None
                selected_metadata = None
                rejection_reasons = []

                for candidate_index, candidate in enumerate(candidates):
                    if not candidate:
                        rejection_reasons.append({"candidate_index": candidate_index, "reason": "empty_response"})
                        continue
                    if args.reject_thinking_tags and has_thinking_tag(candidate):
                        rejected_thinking_count += 1
                        rejection_reasons.append({"candidate_index": candidate_index, "reason": "thinking_tag"})
                        continue
                    if verifier is None:
                        selected = candidate
                        break
                    try:
                        score = verifier(candidate, truth)
                        if args.verifier == "code":
                            passed, metadata = score
                        else:
                            passed, metadata = score_to_bool(score), None
                    except Exception as error:
                        rejection_reasons.append(
                            {"candidate_index": candidate_index, "reason": "verifier_error", "error": repr(error)}
                        )
                        continue
                    if passed:
                        selected = candidate
                        selected_metadata = metadata
                        break
                    rejection_reasons.append({"candidate_index": candidate_index, "reason": "verification_failed"})

                if selected is None:
                    rejected_count += 1
                    rejection_file.write(
                        json.dumps(
                            {
                                "source_index": source_index,
                                "candidate_count": len(candidates),
                                "rejections": rejection_reasons,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    continue

                record = {
                    "prompt": prompts[row_index],
                    "response": selected,
                    "ground_truth": truth,
                    "teacher_verified": verifier is not None,
                }
                if args.verifier == "code":
                    record["verification_metadata"] = json.dumps(selected_metadata, ensure_ascii=False, default=str)
                record["source_index"] = source_index
                if source_index not in saved_source_indices:
                    progress_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                    records.append(record)
                    saved_source_indices.add(source_index)
                    accepted_count += 1

            progress_file.flush()
            rejection_file.flush()
            write_json_atomic(
                state_path,
                {
                    "processed_until": batch_end,
                    "total_prompts": len(prompts),
                    "global_prompt_count": global_prompt_count,
                    "shard_count": args.shard_count,
                    "shard_index": args.shard_index,
                    "accepted": accepted_count,
                    "rejected": rejected_count,
                    "rejected_thinking_candidates": rejected_thinking_count,
                },
            )
            print(
                f"Saved progress after prompts {batch_start}:{batch_end}; "
                f"kept {accepted_count}, rejected {rejected_count}."
            )

    result = pd.DataFrame.from_records(records)
    if result.empty:
        raise RuntimeError("No teacher trajectories survived generation and verification")
    if args.shard_count > 1:
        shard_records_path = output_dir / "verified_trajectories.parquet"
        result.sort_values("source_index", kind="stable").reset_index(drop=True).to_parquet(shard_records_path)
        print(
            f"Saved {len(result)} verified trajectories for shard {args.shard_index}/{args.shard_count} "
            f"to {shard_records_path}; merge all shards before creating train/validation splits."
        )
        return

    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(result))
    validation_count = max(1, round(len(result) * args.validation_ratio))
    result.iloc[order[validation_count:]].reset_index(drop=True).to_parquet(output_dir / "train.parquet")
    result.iloc[order[:validation_count]].reset_index(drop=True).to_parquet(output_dir / "validation.parquet")
    print(
        f"Saved {len(result) - validation_count} train and {validation_count} validation trajectories "
        f"to {output_dir}"
    )


if __name__ == "__main__":
    main()
