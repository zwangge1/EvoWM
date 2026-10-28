#!/usr/bin/env bash
# 热阀反事实 v1 一条命令跑完（设计 docs/design/热阀与落震_反事实闭环_设计_2026-10-28.md 第 4 节）：
# 建带气候的物种表 → 取树 → 训练 50 个模型（Hackett 第 21、41、61、81、100 棵 × 种子 1、2 × 5 折，配置 thermal_v1_l1，
# 与主分析 v2 同样的数量与训练设置）→ 反事实评估（scripts/thermal/eval_thermal_v1.py）。
# 已训练完的模型自动跳过，所以中断后重跑同一条命令即可接着跑。
# 用法（EvoWM 根目录，tmux 里）：GPUS="0 1 2 3" bash scripts/thermal/run_thermal_v1.sh
# 进度看 outputs/thermal_v1/logs/progress.txt（最后一行 ALLDONE 即完成），报告 outputs/thermal_v1/report.md。
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
DESIGN="docs/design/热阀与落震_反事实闭环_设计_2026-10-28.md"
[ -f "$DESIGN" ] || { echo "缺少设计文档 $DESIGN"; exit 1; }
LOG=outputs/thermal_v1/logs
mkdir -p "$LOG"
echo "start $(date +%T)" >> "$LOG/progress.txt"

python scripts/thermal/build_species_table_clim.py | tee "$LOG/table.txt"
md5sum data/processed/species_traits.csv data/processed/species_traits_clim.csv > "$LOG/table_md5.txt"
python scripts/extract_trees.py > "$LOG/trees.txt"

LIST="$LOG/jobs.txt"
: > "$LIST"
for t in 21 41 61 81 100; do
  tt=$(printf %03d "$t")
  for s in 1 2; do
    for f in 0 1 2 3 4; do
      echo "hackett_t${tt}_s${s}_f${f} data.dataset.tree_file=data/processed/trees/hackett_t${tt}.nwk data.dataset.n_trees=1 data.dataset.holdout.fold=${f} seed=${s}" >> "$LIST"
    done
  done
done
CONFIG=thermal_v1_l1 bash scripts/gpu_jobs.sh "$LIST" "$LOG" thermal_v1
echo "trained $(date +%T)" >> "$LOG/progress.txt"

missing=$(while read -r name _; do [ -f "outputs/hjepa/ckpts/thermal_v1/$name/evowm_l1_object.ckpt" ] || echo "$name"; done < "$LIST")
if [ -n "$missing" ]; then
  echo "以下模型没有训练完成，未评估（重跑同一条命令会只补这些）：$missing" | tee -a "$LOG/progress.txt"
  exit 1
fi
python scripts/thermal/eval_thermal_v1.py 2>&1 | tee "$LOG/eval.txt"
echo "ALLDONE $(date +%T)" >> "$LOG/progress.txt"
