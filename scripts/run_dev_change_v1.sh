#!/usr/bin/env bash
# 开发检查（探索性，不属于预注册）：动作只含习性变化（configs/hjepa/recovery_v5_l1.yaml）。
# 第 0 阶段的模拟表（种子 9、10），第 21 棵树，效应与零模型各 5 折，共 20 个模型；训练完后评估，
# 并与第 0 阶段同一棵树上的完整动作模型并列比较。报告写到 outputs/dev_change_v1/。
# 用法（EvoWM 根目录）：GPUS="0 1 2 3" bash scripts/run_dev_change_v1.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
LOG=outputs/dev_change_v1/logs
mkdir -p "$LOG"
python scripts/extract_trees.py > "$LOG/trees.txt"
if [ ! -f data/processed/sim/recovery_v5/null_s10/species_traits.csv ]; then
  python scripts/sim_recovery_tables.py --version v5 --seeds 9 10 --bg-rate 7.8e-4 --bg-eb -0.025 > "$LOG/sim_tables.txt" 2>&1
fi
md5sum data/processed/sim/recovery_v5/*_s9/species_traits.csv data/processed/sim/recovery_v5/*_s10/species_traits.csv > "$LOG/sim_md5.txt"

LIST="$LOG/jobs.txt"
: > "$LIST"
for s in 9 10; do
  for sc in effect null; do
    for f in 0 1 2 3 4; do
      echo "${sc}_s${s}_t021_f${f} data.dataset.species_table=data/processed/sim/recovery_v5/${sc}_s${s}/species_traits.csv data.dataset.tree_file=data/processed/trees/hackett_t021.nwk data.dataset.n_trees=1 data.dataset.holdout.fold=${f} seed=${s}" >> "$LIST"
    done
  done
done
echo "start $(date +%T)" >> "$LOG/progress.txt"
CONFIG=recovery_v5_l1 bash scripts/gpu_jobs.sh "$LIST" "$LOG" dev_change_v1
echo "trained $(date +%T)" >> "$LOG/progress.txt"

missing=$(while read -r name _; do [ -f "outputs/hjepa/ckpts/dev_change_v1/$name/evowm_l1_object.ckpt" ] || echo "$name"; done < "$LIST")
if [ -n "$missing" ]; then
  echo "以下模型没有训练完成，未评估：$missing" | tee -a "$LOG/progress.txt"
  exit 1
fi
python scripts/exploratory/eval_dev_change_v1.py 2>&1 | tee "$LOG/eval.txt"
echo "ALLDONE $(date +%T)" >> "$LOG/progress.txt"
