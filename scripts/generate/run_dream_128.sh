#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PYTHON:-python}"
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"
DATASET="${1:-commonsenseqa}"
OUT="${2:-outputs/${DATASET}_dream_steps128_400}"
cd "$ROOT"
"$PY" scripts/generate/generate_dream_trajectories.py \
  --model-path "${DREAM_MODEL:-Dream-org/Dream-v0-Instruct-7B}" \
  --dataset "$DATASET" \
  --split "${DATA_SPLIT:-validation}" \
  --subsample "${SUBSAMPLE:-400}" \
  --seed "${SEED:-42}" \
  --output-dir "$OUT" \
  --batch-size "${BATCH_SIZE:-1}" \
  --diffusion-steps "${DIFFUSION_STEPS:-128}" \
  --gen-length "${GEN_LENGTH:-128}" \
  --temperature "${TEMPERATURE:-0.0}" \
  --dtype "${DTYPE:-bfloat16}" \
  --device "${DEVICE:-cuda}"
