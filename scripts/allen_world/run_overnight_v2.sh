#!/usr/bin/env bash
# In tmux: GPUS="0 1 2 3" HOURS=8 bash scripts/allen_world/run_overnight_v2.sh
# Resume: GPUS="0 1" HOURS=8 bash scripts/allen_world/run_overnight_v2.sh --resume outputs/allen_overnight_v2_TIMESTAMP
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
: "${GPUS:?Set GPUS to one to four currently available GPU indices, e.g. GPUS='0 1'}"
read -r -a GPU_LIST <<< "$GPUS"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT/src:$ROOT/hjepa/h_jepa:$ROOT/hjepa${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
RESUME=0
for arg in "$@"; do
  if [[ "$arg" == "--resume" || "$arg" == --resume=* ]]; then RESUME=1; fi
done
if [[ "$RESUME" == 0 ]]; then
  python scripts/extract_trees.py
  if [[ ! -f outputs/allen_world_v1/input/species_traits.csv ]]; then
    python scripts/allen_world/build_table.py
  fi
  python scripts/allen_world/prepare_v2.py
  python scripts/allen_world/check.py --table outputs/allen_world_v2/input/species_traits.csv \
    --configs allen_v2_flat allen_v2_allocation allen_v2_proximal allen_v2_distal \
    --workers "${CHECK_WORKERS:-32}" --model-smoke --out-dir outputs/allen_world_v2/check
fi
exec python scripts/allen_world/overnight_v2.py --gpus "${GPU_LIST[@]}" --hours "${HOURS:-8}" \
  --eval-workers "${EVAL_WORKERS:-32}" --loader-workers "${LOADER_WORKERS:-8}" \
  --threads-per-job "${THREADS_PER_JOB:-4}" "$@"
