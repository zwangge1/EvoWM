#!/usr/bin/env bash
# Classical four-trait BM/OU, CPU only; five folds, no checkpoints required.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
export PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=""
export PYTHONPATH="$ROOT/src:$ROOT/scripts/allen_world${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${THREADS_PER_JOB:-2}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS" OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS" NUMEXPR_NUM_THREADS="$OMP_NUM_THREADS"
exec python scripts/allen_world/phylo_baselines_v1.py --workers "${WORKERS:-5}" \
  --threads "$OMP_NUM_THREADS" "$@"
