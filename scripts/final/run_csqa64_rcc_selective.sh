#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PYTHON:-python}"
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
export PYTHONPATH="$ROOT/src:$ROOT/scripts/final:${PYTHONPATH:-}"
cd "$ROOT"
"$PY" scripts/final/analyze_csqa64_rcc_selective.py \
  --root "${RCC_OUTPUT_ROOT:-outputs}" \
  --output_dir "${RCC_CSQA64_OUT:-outputs/rcc_selective_csqa64}" \
  --reference128 "${RCC_CSQA128_RESULTS:-results/main/csqa128_per_sample.csv}"
