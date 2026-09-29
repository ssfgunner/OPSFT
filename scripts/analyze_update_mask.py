#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    p = argparse.ArgumentParser(description="Analyze long-horizon parameter updates.")
    p.add_argument("--base-model", required=True)
    p.add_argument("--trained-model", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--keep-ratio", type=float, default=0.05)
    p.add_argument("--threshold", type=float, default=None)
    p.add_argument("--selection", choices=("topk", "threshold", "exact_support"), default=None)
    p.add_argument("--masked-output")
    p.add_argument("--removed-output")
    a = p.parse_args(); out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
    bm = AutoModelForCausalLM.from_pretrained(a.base_model, torch_dtype=torch.float32)
    tm = AutoModelForCausalLM.from_pretrained(a.trained_model, torch_dtype=torch.float32)
    tokenizer = AutoTokenizer.from_pretrained(a.base_model)
    base = {n:x.detach().cpu().float() for n,x in bm.named_parameters() if x.is_floating_point()}
    trained = {n:x.detach().cpu().float() for n,x in tm.named_parameters() if x.is_floating_point()}
    if set(base) != set(trained): raise ValueError("Base and trained models must have identical parameter names")
    updates = {n: trained[n] - base[n] for n in base}
    flat = torch.cat([x.abs().reshape(-1) for x in updates.values()])
    selection = a.selection or ("threshold" if a.threshold is not None else "topk")
    if selection == "exact_support":
        masks = {name: update != 0 for name, update in updates.items()}
        threshold = None
    elif selection == "topk":
        if not 0 < a.keep_ratio <= 1:
            raise ValueError("--keep-ratio must be in (0, 1]")
        total_flat = flat.numel()
        keep_count = max(1, int(total_flat * a.keep_ratio))
        # Explicit global top-k keeps the support size exact despite tied values.
        selected = torch.zeros(total_flat, dtype=torch.bool)
        selected[torch.topk(flat, keep_count, sorted=False).indices] = True
        masks = {}
        offset = 0
        for name, update in updates.items():
            size = update.numel()
            masks[name] = selected[offset : offset + size].reshape(update.shape)
            offset += size
        threshold = flat[selected].min().item()
    elif selection == "threshold":
        if a.threshold is None:
            raise ValueError("--selection threshold requires --threshold")
        if a.threshold < 0:
            raise ValueError("--threshold must be non-negative")
        threshold = a.threshold
        masks = {name: update.abs() >= threshold for name, update in updates.items()}
    else:
        raise ValueError(f"Unknown selection: {selection}")

    total = sum(x.numel() for x in updates.values())
    energy = sum(x.square().sum().item() for x in updates.values())
    l1 = sum(x.abs().sum().item() for x in updates.values())
    active = sum(mask.sum().item() for mask in masks.values())
    ae = sum(updates[name].square().masked_select(mask).sum().item() for name, mask in masks.items())
    al = sum(updates[name].abs().masked_select(mask).sum().item() for name, mask in masks.items())
    metrics = {
        "parameters": total,
        "active_parameters": active,
        "active_ratio": active / max(total, 1),
        "requested_keep_ratio": a.keep_ratio if selection == "topk" else None,
        "update_l1": l1,
        "update_l2_squared": energy,
        "retained_l1_ratio": al / max(l1, 1e-30),
        "retained_energy_ratio": ae / max(energy, 1e-30),
        "threshold": threshold,
        "selection": {
            "topk": "global_exact_topk",
            "threshold": "absolute_threshold",
            "exact_support": "exact_nonzero_support",
        }[selection],
    }
    torch.save(masks, out / "update_mask.pt")
    torch.save(updates, out / "parameter_updates.pt")
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    def save(model, keep, directory):
        directory = Path(directory); directory.mkdir(parents=True, exist_ok=True); state = model.state_dict()
        for n,m in masks.items(): state[n] = ((base[n]+updates[n]*m) if keep else (trained[n]-updates[n]*m)).to(state[n].dtype)
        model.load_state_dict(state, strict=False); model.save_pretrained(directory, safe_serialization=True); tokenizer.save_pretrained(directory)
    if a.masked_output: save(bm, True, a.masked_output)
    if a.removed_output: save(tm, False, a.removed_output)
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__": main()
