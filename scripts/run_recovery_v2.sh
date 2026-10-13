#!/usr/bin/env bash
# 模拟恢复检验 v2 的训练（预注册 docs/prereg/sim_recovery_v2.md）。
# 判定用：真树训练，效应/零模型 × 种子 3、4；稳健性：种子 3 用 100 棵后验树训练。
# 用法：bash scripts/run_recovery_v2.sh            （GPU，bf16）
#       ACCEL=cpu PRECISION=32 bash scripts/run_recovery_v2.sh   （CPU）
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ACCEL="${ACCEL:-gpu}"
PRECISION="${PRECISION:-bf16}"
WORKERS="${WORKERS:-2}"
LOG="$ROOT/outputs/recovery_v2/logs"
mkdir -p "$LOG"
cd "$ROOT"

run () {  # $1 情景 $2 种子 $3 tree1|post
  local sc=$1 s=$2 tr=$3
  local tree_args=()
  if [ "$tr" = "tree1" ]; then
    tree_args=(data.dataset.tree_file=data/raw/birdtree/hackett_tree1.nwk data.dataset.n_trees=1)
  fi
  echo "start ${sc}_s${s}_${tr} $(date +%T)" >> "$LOG/progress.txt"
  bash scripts/train_hjepa.sh recovery_v2_l1 \
    data.dataset.species_table=data/processed/sim/recovery_v2/${sc}_s${s}/species_traits.csv \
    "${tree_args[@]}" trainer.accelerator=$ACCEL trainer.precision=$PRECISION num_workers=$WORKERS \
    seed=$s subdir=recovery_v2/${sc}_s${s}_${tr} > "$LOG/${sc}_s${s}_${tr}.log" 2>&1
  echo "done ${sc}_s${s}_${tr} $(date +%T)" >> "$LOG/progress.txt"
}

for sc in effect null; do
  for s in 3 4; do run $sc $s tree1; done
done
run effect 3 post
run null 3 post
echo ALLDONE >> "$LOG/progress.txt"
