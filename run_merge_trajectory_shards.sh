#!/usr/bin/env bash
# Merge deterministic trajectory-generation shards and create one train/validation split.
# Required: TRAJECTORY_DIR (contains shards/shard_*/verified_trajectories.parquet)
# Optional: VALIDATION_RATIO=0.02, SEED=42, OVERWRITE=0.

set -euo pipefail

PYTHON="${PYTHON:-python3}"
: "${TRAJECTORY_DIR:?Set TRAJECTORY_DIR to the parent output directory of a sharded generation run.}"
VALIDATION_RATIO="${VALIDATION_RATIO:-0.02}"
SEED="${SEED:-42}"
OVERWRITE="${OVERWRITE:-0}"

[[ "${OVERWRITE}" == "0" || "${OVERWRITE}" == "1" ]] || { echo "[ERROR] OVERWRITE must be 0 or 1." >&2; exit 1; }
[[ -d "${TRAJECTORY_DIR}/shards" ]] || { echo "[ERROR] Missing shard directory: ${TRAJECTORY_DIR}/shards" >&2; exit 1; }
if [[ "${OVERWRITE}" != "1" && ( -e "${TRAJECTORY_DIR}/train.parquet" || -e "${TRAJECTORY_DIR}/validation.parquet" ) ]]; then
    echo "[ERROR] train.parquet or validation.parquet already exists. Set OVERWRITE=1 to replace them." >&2
    exit 1
fi

TRAJECTORY_DIR="${TRAJECTORY_DIR}" VALIDATION_RATIO="${VALIDATION_RATIO}" SEED="${SEED}" "${PYTHON}" - <<'PY'
import os
from pathlib import Path

import numpy as np
import pandas as pd

output_dir = Path(os.environ["TRAJECTORY_DIR"])
validation_ratio = float(os.environ["VALIDATION_RATIO"])
seed = int(os.environ["SEED"])
if not 0 < validation_ratio < 1:
    raise ValueError("VALIDATION_RATIO must be in (0, 1)")

shard_paths = sorted((output_dir / "shards").glob("shard_*/verified_trajectories.parquet"))
if not shard_paths:
    raise FileNotFoundError(f"No shard parquet files found under {output_dir / 'shards'}")
frames = [pd.read_parquet(path) for path in shard_paths]
data = pd.concat(frames, ignore_index=True)
required = {"prompt", "response", "ground_truth", "teacher_verified", "source_index"}
missing = required - set(data.columns)
if missing:
    raise ValueError(f"Shard output is missing required columns: {sorted(missing)}")
if data.empty:
    raise ValueError("Merged shard data is empty")
if data["source_index"].duplicated().any():
    duplicates = data.loc[data["source_index"].duplicated(), "source_index"].head().tolist()
    raise ValueError(f"Duplicate source indices across shards: {duplicates}")

data = data.sort_values("source_index", kind="stable").reset_index(drop=True)
rng = np.random.default_rng(seed)
order = rng.permutation(len(data))
validation_count = max(1, round(len(data) * validation_ratio))
data.iloc[order[validation_count:]].reset_index(drop=True).to_parquet(output_dir / "train.parquet")
data.iloc[order[:validation_count]].reset_index(drop=True).to_parquet(output_dir / "validation.parquet")
print(
    f"Merged {len(shard_paths)} shards ({len(data)} verified trajectories): "
    f"train={len(data) - validation_count}, validation={validation_count}, output={output_dir}"
)
PY
