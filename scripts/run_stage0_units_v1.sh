#!/usr/bin/env bash
# 按习性分层的第 0 阶段（预注册 docs/prereg/units_main_v1.md 第 4 节）：树不对时的模拟校准。
# 生成新种子 13、14 的模拟表（sim_recovery_tables.py --version v5，带早期爆发布朗背景，与第一层 v2 的第 0 阶段同一设定），
# 训练 100 个三层模型（configs/hjepa/recovery_v5_hab3.yaml；效应/零模型 × 种子 13、14 × 第 21、41、61、81、100 棵树 × 5 折），
# 全部完成后评估（H3 读数为不涉及泛化的第二层步），写出 outputs/stage0_units_v1/decision.json。
# 用法（EvoWM 根目录）：GPUS="0 1 2 3" bash scripts/run_stage0_units_v1.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
[ -f docs/prereg/units_main_v1.md ] || { echo "缺少预注册 docs/prereg/units_main_v1.md"; exit 1; }
LOG=outputs/stage0_units_v1/logs
mkdir -p "$LOG"

python scripts/extract_trees.py > "$LOG/trees.txt"
echo "sim tables $(date +%T)" >> "$LOG/progress.txt"
python scripts/sim_recovery_tables.py --version v5 --seeds 13 14 --bg-rate 7.8e-4 --bg-eb -0.025 > "$LOG/sim_tables.txt" 2>&1
# 必须与预注册第 4 节的 md5 相同，不同就停止
md5sum -c - <<'MD5' | tee -a "$LOG/sim_tables.txt" || { echo "模拟表的 md5 与预注册第 4 节不同，停止" | tee -a "$LOG/progress.txt"; exit 1; }
c03d34812c279fc7e815730dcaec32ec  data/processed/sim/recovery_v5/effect_s13/species_traits.csv
fa8c609ed17ebb17be853cd7a73973ad  data/processed/sim/recovery_v5/null_s13/species_traits.csv
0423f490c2213e0744ec2830895defdf  data/processed/sim/recovery_v5/effect_s14/species_traits.csv
cb3c6b33419ffb604deca8cb9edb45d0  data/processed/sim/recovery_v5/null_s14/species_traits.csv
MD5

LIST="$LOG/jobs.txt"
: > "$LIST"
for s in 13 14; do
  for sc in effect null; do
    for t in 21 41 61 81 100; do
      tt=$(printf %03d "$t")
      for f in 0 1 2 3 4; do
        echo "${sc}_s${s}_t${tt}_f${f} data.dataset.species_table=data/processed/sim/recovery_v5/${sc}_s${s}/species_traits.csv data.dataset.tree_file=data/processed/trees/hackett_t${tt}.nwk data.dataset.n_trees=1 data.dataset.holdout.fold=${f} seed=${s}" >> "$LIST"
      done
    done
  done
done
echo "start $(date +%T)" >> "$LOG/progress.txt"
CONFIG=recovery_v5_hab3 CKPT_NAME=evowm_hab3 bash scripts/gpu_jobs.sh "$LIST" "$LOG" stage0_units_v1
echo "trained $(date +%T)" >> "$LOG/progress.txt"

missing=$(while read -r name _; do [ -f "outputs/hjepa/ckpts/stage0_units_v1/$name/evowm_hab3_object.ckpt" ] || echo "$name"; done < "$LIST")
if [ -n "$missing" ]; then
  echo "以下模型没有训练完成，未评估：$missing" | tee -a "$LOG/progress.txt"
  exit 1
fi
python scripts/eval_stage0_units_v1.py 2>&1 | tee "$LOG/eval.txt"
[ -f outputs/stage0_units_v1/decision.json ] || { echo "第 0 阶段评估没有写出 decision.json" | tee -a "$LOG/progress.txt"; exit 1; }
echo "ALLDONE $(date +%T)" >> "$LOG/progress.txt"
