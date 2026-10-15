#!/usr/bin/env bash
# 开发检查（探索性，不属于预注册）：按习性尺度分层（设计草案 docs/design/第二层设计草案_2026-10-15.md 第 7 节）。
# 第 0 阶段的模拟表（种子 9、10，带早期爆发背景），第 21 棵树，效应与零模型各 5 折：
#   两层（configs/hjepa/recovery_v5_hab2.yaml）20 个，三层（recovery_v5_hab3.yaml）20 个；
#   只训第 1 层的对照用开发检查 dev_change_v1 的模型（recovery_v5_l1；服务器上已有的直接跳过，缺的补训）。
# 训练完后评估（scripts/exploratory/eval_dev_units_v1.py），报告写到 outputs/dev_units_v1/。
# 用法（EvoWM 根目录）：GPUS="0 1 2 3" bash scripts/run_dev_units_v1.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
LOG=outputs/dev_units_v1/logs
mkdir -p "$LOG"/l1 "$LOG"/hab2 "$LOG"/hab3
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
CONFIG=recovery_v5_l1 CKPT_NAME=evowm_l1 bash scripts/gpu_jobs.sh "$LIST" "$LOG/l1" dev_change_v1
echo "l1 $(date +%T)" >> "$LOG/progress.txt"
CONFIG=recovery_v5_hab2 CKPT_NAME=evowm_hab2 bash scripts/gpu_jobs.sh "$LIST" "$LOG/hab2" dev_units_v1/hab2
echo "hab2 $(date +%T)" >> "$LOG/progress.txt"
CONFIG=recovery_v5_hab3 CKPT_NAME=evowm_hab3 bash scripts/gpu_jobs.sh "$LIST" "$LOG/hab3" dev_units_v1/hab3
echo "trained $(date +%T)" >> "$LOG/progress.txt"

missing=""
for spec in "dev_change_v1 evowm_l1" "dev_units_v1/hab2 evowm_hab2" "dev_units_v1/hab3 evowm_hab3"; do
  read -r sub name <<< "$spec"
  while read -r job _; do
    [ -f "outputs/hjepa/ckpts/$sub/$job/${name}_object.ckpt" ] || missing="$missing $sub/$job"
  done < "$LIST"
done
if [ -n "$missing" ]; then
  echo "以下模型没有训练完成，未评估：$missing" | tee -a "$LOG/progress.txt"
  exit 1
fi
python scripts/exploratory/eval_dev_units_v1.py 2>&1 | tee "$LOG/eval.txt"
echo "ALLDONE $(date +%T)" >> "$LOG/progress.txt"
