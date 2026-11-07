#!/usr/bin/env bash
set -euo pipefail

# Run from the repository root automatically.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

PY=${PY:-python}
export PYTHONNOUSERSITE=1
export PYTHONPATH="$(pwd)/src:$(pwd)/scripts:${PYTHONPATH:-}"

SAMPLES=${SAMPLES:-400}
STEPS=${STEPS:-64}
GEN_LENGTH=${GEN_LENGTH:-64}
BLOCK_LENGTH=${BLOCK_LENGTH:-32}
MODEL_DIR=${MODEL_DIR:-GSAI-ML/LLaDA-8B-Instruct}
OUT=${OUT:-outputs/rcc_mcq_sweep}

mkdir -p "$OUT"

echo "============================================================"
echo "RCC multi-MCQ selective mitigation sweep"
echo "Datasets: ARC-Easy, OpenBookQA, SciQ, HellaSwag"
echo "No LLM judge. Ground-truth multiple-choice evaluation."
echo "Resumable: completed samples are reused."
echo "samples=$SAMPLES steps=$STEPS gen=$GEN_LENGTH block=$BLOCK_LENGTH"
echo "============================================================"

"$PY" scripts/final/run_multi_mcq_rcc_selective.py \
  --datasets arc_easy openbookqa sciq hellaswag \
  --model_dir "$MODEL_DIR" \
  --output_dir "$OUT" \
  --max_samples "$SAMPLES" \
  --seed 42 \
  --steps "$STEPS" \
  --gen_length "$GEN_LENGTH" \
  --block_length "$BLOCK_LENGTH" \
  --temperature 0.0 \
  --train_fraction 0.50

echo
echo "===== Full benchmark summary ====="
cat "$OUT/pilot_summary.csv"
