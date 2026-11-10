#!/usr/bin/env bash
# Default is the short smoke matrix. Select available GPUs explicitly.
# GPUS="4 5" bash scripts/allen_world/run_overnight_v3.sh --smoke
# GPUS="4 5 6 7" HOURS=8 bash scripts/allen_world/run_overnight_v3.sh --full
# GPUS="4 5" bash scripts/allen_world/run_overnight_v3.sh --resume outputs/allen_overnight_v3_TIMESTAMP
# Add --dry-run to snapshot a plan without launching any training.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
: "${GPUS:?Set GPUS to one to four currently available GPU indices, e.g. GPUS='4 5'}"
read -r -a GPU_LIST <<< "$GPUS"
export PYTHONUNBUFFERED=1
export EVOWM_ROOT="$ROOT"
export PYTHONPATH="$ROOT/src:$ROOT/hjepa/h_jepa:$ROOT/hjepa${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
# No automatic table preparation: preserve the exact downloaded frozen folds.
exec python scripts/allen_world/overnight_v3.py --gpus "${GPU_LIST[@]}" \
  --hours "${HOURS:-8}" --eval-workers "${EVAL_WORKERS:-32}" \
  --loader-workers "${LOADER_WORKERS:-8}" --threads-per-job "${THREADS_PER_JOB:-4}" "$@"
