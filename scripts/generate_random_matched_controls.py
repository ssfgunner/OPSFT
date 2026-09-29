#!/usr/bin/env python3
"""Generate per-tensor count-matched random controls for a Top-K update mask."""

import argparse
import gc
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def random_mask_like(reference_mask, rng):
    """Select exactly the same number of entries as reference_mask."""
    flat_reference = reference_mask.reshape(-1)
    count = int(flat_reference.sum().item())
    mask = torch.zeros(flat_reference.numel(), dtype=torch.bool)
    if count == flat_reference.numel():
        mask.fill_(True)
    elif count > 0:
        indices = rng.choice(flat_reference.numel(), size=count, replace=False, shuffle=False)
        mask[torch.from_numpy(indices.astype(np.int64, copy=False))] = True
    return mask.reshape(reference_mask.shape)


def copy_compatible_assets(tokenizer, model_source, tokenizer_source, output_dir):
    """Preserve model/tokenizer configs from the original Transformers 4.x assets."""
    output_dir = Path(output_dir)
    tokenizer.save_pretrained(output_dir)
    for filename in ("tokenizer_config.json", "tokenizer.json", "vocab.json", "merges.txt", "chat_template.jinja"):
        source = Path(tokenizer_source) / filename
        if source.exists():
            shutil.copy2(source, output_dir / filename)
    for filename in ("config.json", "generation_config.json"):
        source = Path(model_source) / filename
        if source.exists():
            shutil.copy2(source, output_dir / filename)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--trained-model", required=True)
    parser.add_argument("--reference-mask", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 2026])
    parser.add_argument("--name-prefix", default="random_matched_top0.05")
    parser.add_argument("--torch-dtype", choices=["float32", "bfloat16", "float16"], default="float32")
    args = parser.parse_args()

    dtype = getattr(torch, args.torch_dtype)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    print("Loading Base and Full GRPO models...", flush=True)
    base_model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=dtype)
    trained_model = AutoModelForCausalLM.from_pretrained(args.trained_model, torch_dtype=dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    reference_masks = torch.load(args.reference_mask, map_location="cpu", weights_only=True)

    # clone() is essential: base_model is reused as a serialization buffer below.
    # Without cloning, later load_state_dict calls mutate the reference tensors and
    # corrupt all random seeds after the first one.
    base = {name: value.detach().cpu().float().clone() for name, value in base_model.named_parameters() if value.is_floating_point()}
    trained = {name: value.detach().cpu().float().clone() for name, value in trained_model.named_parameters() if value.is_floating_point()}
    if set(base) != set(trained) or set(base) != set(reference_masks):
        raise ValueError("Base, trained, and reference mask parameter names must match")
    del trained_model
    gc.collect()

    updates = {name: trained[name] - base[name] for name in base}
    total_parameters = sum(value.numel() for value in updates.values())
    reference_active = sum(int(mask.sum().item()) for mask in reference_masks.values())
    reference_counts = {name: int(mask.sum().item()) for name, mask in reference_masks.items()}

    all_metrics = []
    for seed in args.seeds:
        print(f"Generating random controls for seed {seed}...", flush=True)
        rng = np.random.default_rng(seed)
        random_masks = {
            name: random_mask_like(reference_masks[name], rng)
            for name in updates
        }
        random_counts = {name: int(mask.sum().item()) for name, mask in random_masks.items()}
        if random_counts != reference_counts:
            raise RuntimeError(f"Per-tensor count mismatch for seed {seed}")

        random_active = sum(random_counts.values())
        selected_l1 = sum(updates[name].abs().masked_select(mask).sum().item() for name, mask in random_masks.items())
        selected_energy = sum(updates[name].square().masked_select(mask).sum().item() for name, mask in random_masks.items())
        total_l1 = sum(value.abs().sum().item() for value in updates.values())
        total_energy = sum(value.square().sum().item() for value in updates.values())

        seed_root = output_root / f"{args.name_prefix}_seed{seed}"
        masked_dir = seed_root / "masked"
        removed_dir = seed_root / "removed"
        seed_root.mkdir(parents=True, exist_ok=True)
        torch.save(random_masks, seed_root / "random_mask.pt")

        state = base_model.state_dict()
        for name, mask in random_masks.items():
            state[name] = (base[name] + updates[name] * mask).to(state[name].dtype)
        base_model.load_state_dict(state, strict=False)
        base_model.save_pretrained(masked_dir, safe_serialization=True)
        copy_compatible_assets(tokenizer, args.base_model, args.base_model, masked_dir)

        state = base_model.state_dict()
        for name, mask in random_masks.items():
            state[name] = (trained[name] - updates[name] * mask).to(state[name].dtype)
        base_model.load_state_dict(state, strict=False)
        base_model.save_pretrained(removed_dir, safe_serialization=True)
        copy_compatible_assets(tokenizer, args.trained_model, args.base_model, removed_dir)

        metrics = {
            "seed": seed,
            "matching": "exact selected count per parameter tensor",
            "reference_mask": str(Path(args.reference_mask).resolve()),
            "parameters": total_parameters,
            "reference_active_parameters": reference_active,
            "random_active_parameters": random_active,
            "active_ratio": random_active / total_parameters,
            "selected_l1_ratio": selected_l1 / max(total_l1, 1e-30),
            "selected_energy_ratio": selected_energy / max(total_energy, 1e-30),
            "masked_model": str(masked_dir.resolve()),
            "removed_model": str(removed_dir.resolve()),
        }
        (seed_root / "metrics.json").write_text(json.dumps(metrics, indent=2))
        all_metrics.append(metrics)
        print(json.dumps(metrics, indent=2), flush=True)
        del random_masks, state
        gc.collect()

    (output_root / f"{args.name_prefix}_summary.json").write_text(json.dumps(all_metrics, indent=2))


if __name__ == "__main__":
    main()
