#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PYTHON:-python}"
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
export PYTHONPATH="$ROOT/src:$ROOT/scripts/final:${PYTHONPATH:-}"
cd "$ROOT"
"$PY" scripts/final/test_rcc_all_configs.py \
  --auto_root "${RCC_OUTPUT_ROOT:-outputs}" \
  --repo_root "$ROOT" \
  --output_dir "${RCC_ROBUST_OUT:-outputs/rcc_all_configs}"
