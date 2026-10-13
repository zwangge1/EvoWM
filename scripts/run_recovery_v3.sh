#!/usr/bin/env bash
# 模拟恢复检验 v3 的训练：按科 5 折交叉验证，效应 / 零模型 × 开发种子 1、2 与检验种子 5、6，共 40 个模型。
# 训练设置同 v2，只改留出方式（configs/hjepa/recovery_v3_l1.yaml）。每个模型单卡、fp32。
# GPUS 里每张卡一个工作进程，按顺序领任务，先跑开发种子，再跑检验种子。已有检查点的任务跳过，可中断后重跑。
# 检验种子（5、6）只训练；它们的评估要等 docs/prereg/sim_recovery_v3.md 写定之后。
# 用法（EvoWM 根目录）：GPUS="0 1 2 3" bash scripts/run_recovery_v3.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
GPUS="${GPUS:-0 1 2 3}"
WORKERS="${WORKERS:-4}"
LOG="$ROOT/outputs/recovery_v3/logs"
mkdir -p "$LOG"
cd "$ROOT"

echo "sim tables $(date +%T)" >> "$LOG/progress.txt"
python scripts/sim_recovery_tables.py --version v3 --seeds 1 2 5 6 > "$LOG/sim_tables.txt" 2>&1

JOBS=()
for s in 1 2 5 6; do
  for sc in effect null; do
    for f in 0 1 2 3 4; do JOBS+=("$sc $s $f"); done
  done
done

run_job () {  # $1 卡号 $2 情景 $3 种子 $4 折
  local g=$1 sc=$2 s=$3 f=$4
  local name="${sc}_s${s}_f${f}"
  if [ -f "outputs/hjepa/ckpts/recovery_v3/$name/evowm_l1_object.ckpt" ]; then
    echo "skip $name" >> "$LOG/progress.txt"
    return 0
  fi
  echo "start $name gpu$g $(date +%T)" >> "$LOG/progress.txt"
  if CUDA_VISIBLE_DEVICES=$g bash scripts/train_hjepa.sh recovery_v3_l1 \
      data.dataset.species_table=data/processed/sim/recovery_v3/${sc}_s${s}/species_traits.csv \
      data.dataset.tree_file=data/raw/birdtree/hackett_tree1.nwk data.dataset.n_trees=1 \
      data.dataset.holdout.fold=$f trainer.accelerator=gpu num_workers=$WORKERS \
      seed=$s subdir=recovery_v3/$name > "$LOG/$name.log" 2>&1; then
    echo "done $name $(date +%T)" >> "$LOG/progress.txt"
  else
    echo "FAIL $name $(date +%T)" >> "$LOG/progress.txt"
  fi
}

worker () {  # $1 卡号 $2 工作进程编号 $3 进程总数：领第 i 个任务，i 除以总数余编号
  local g=$1 w=$2 n=$3 i
  for ((i = w; i < ${#JOBS[@]}; i += n)); do
    # shellcheck disable=SC2086
    run_job "$g" ${JOBS[$i]}
  done
}

read -ra G <<< "$GPUS"
for w in "${!G[@]}"; do
  worker "${G[$w]}" "$w" "${#G[@]}" &
done
wait
echo "ALLDONE $(date +%T)" >> "$LOG/progress.txt"
