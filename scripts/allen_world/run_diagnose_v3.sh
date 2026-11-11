#!/usr/bin/env bash
# Reuse completed weights. CPU only; default 32 workers x 2 threads.
# bash scripts/allen_world/run_diagnose_v3.sh --source-run outputs/allen_overnight_v3_TIMESTAMP --smoke
# WORKERS=32 bash scripts/allen_world/run_diagnose_v3.sh --source-run outputs/allen_overnight_v3_TIMESTAMP --full
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
export PYTHONUNBUFFERED=1
export EVOWM_ROOT="$ROOT"
export CUDA_VISIBLE_DEVICES=""
export PYTHONPATH="$ROOT/src:$ROOT/hjepa/h_jepa:$ROOT/hjepa${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${THREADS_PER_JOB:-2}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS" OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS" NUMEXPR_NUM_THREADS="$OMP_NUM_THREADS"
exec python scripts/allen_world/diagnose_v3.py --workers "${WORKERS:-32}" \
  --threads "$OMP_NUM_THREADS" "$@"
