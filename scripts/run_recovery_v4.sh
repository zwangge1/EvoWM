#!/usr/bin/env bash
# 模拟恢复检验 v4（预注册 docs/prereg/sim_recovery_v4.md）：生成种子 7、8 的模拟表，训练 20 个模型
# （5 折 × 效应/零模型 × 种子 7、8；训练设置同 v3，配置 recovery_v3_l1），全部训练完后直接按预注册评估。
# GPUS 里每张卡一个工作进程。已有检查点的任务跳过，可中断后重跑。所有输出都在 outputs/ 下。
# 用法（EvoWM 根目录）：GPUS="0 1 2 3" bash scripts/run_recovery_v4.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
GPUS="${GPUS:-0 1 2 3}"
WORKERS="${WORKERS:-4}"
LOG="$ROOT/outputs/recovery_v4/logs"
mkdir -p "$LOG"
cd "$ROOT"
[ -f docs/prereg/sim_recovery_v4.md ] || { echo "缺少预注册 docs/prereg/sim_recovery_v4.md"; exit 1; }

echo "sim tables $(date +%T)" >> "$LOG/progress.txt"
python scripts/sim_recovery_tables.py --version v4 --seeds 7 8 > "$LOG/sim_tables.txt" 2>&1

JOBS=()
for s in 7 8; do
  for sc in effect null; do
    for f in 0 1 2 3 4; do JOBS+=("$sc $s $f"); done
  done
done

run_job () {  # $1 卡号 $2 情景 $3 种子 $4 折
  local g=$1 sc=$2 s=$3 f=$4
  local name="${sc}_s${s}_f${f}"
  if [ -f "outputs/hjepa/ckpts/recovery_v4/$name/evowm_l1_object.ckpt" ]; then
    echo "skip $name" >> "$LOG/progress.txt"
    return 0
  fi
  echo "start $name gpu$g $(date +%T)" >> "$LOG/progress.txt"
  if CUDA_VISIBLE_DEVICES=$g bash scripts/train_hjepa.sh recovery_v3_l1 \
      data.dataset.species_table=data/processed/sim/recovery_v4/${sc}_s${s}/species_traits.csv \
      data.dataset.tree_file=data/raw/birdtree/hackett_tree1.nwk data.dataset.n_trees=1 \
      data.dataset.holdout.fold=$f trainer.accelerator=gpu num_workers=$WORKERS \
      seed=$s subdir=recovery_v4/$name > "$LOG/$name.log" 2>&1; then
    echo "done $name $(date +%T)" >> "$LOG/progress.txt"
  else
    echo "FAIL $name $(date +%T)" >> "$LOG/progress.txt"
  fi
}

worker () {  # $1 卡号 $2 工作进程编号 $3 进程总数
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
echo "trained $(date +%T)" >> "$LOG/progress.txt"

if grep -q FAIL "$LOG/progress.txt"; then
  echo "有训练失败，未评估（见 $LOG/progress.txt）" | tee -a "$LOG/progress.txt"
  exit 1
fi
python scripts/eval_recovery_v4.py 2>&1 | tee "$LOG/eval.txt"
echo "ALLDONE $(date +%T)" >> "$LOG/progress.txt"
