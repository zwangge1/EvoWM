#!/usr/bin/env bash
# Frozen CPU inference: no training; no GPUs needed.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
export PYTHONUNBUFFERED=1 EVOWM_ROOT="$ROOT"
export CUDA_VISIBLE_DEVICES=""
export PYTHONPATH="$ROOT/src:$ROOT/hjepa/h_jepa:$ROOT/hjepa${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${THREADS_PER_JOB:-2}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS" OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS" NUMEXPR_NUM_THREADS="$OMP_NUM_THREADS"
exec python scripts/allen_world/ecology_probe_v1.py --workers "${WORKERS:-32}" \
  --threads "$OMP_NUM_THREADS" "$@"
