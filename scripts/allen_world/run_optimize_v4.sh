#!/usr/bin/env bash
# Fixed nested-validation grid; retain original model code and checkpoints.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
: "${GPUS:?Set one to four available GPU indices, e.g. GPUS='0 1 2 3'}"
read -r -a GPU_LIST <<< "$GPUS"
export PYTHONUNBUFFERED=1 EVOWM_ROOT="$ROOT"
export PYTHONPATH="$ROOT/src:$ROOT/hjepa/h_jepa:$ROOT/hjepa${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
exec python scripts/allen_world/optimize_v4.py --gpus "${GPU_LIST[@]}" \
  --hours "${HOURS:-8}" --eval-workers "${EVAL_WORKERS:-32}" \
  --loader-workers "${LOADER_WORKERS:-4}" --threads-per-job "${THREADS_PER_JOB:-4}" "$@"
