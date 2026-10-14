#!/usr/bin/env bash
# 真实数据主分析 v2 的第 0 阶段（预注册 docs/prereg/real_main_v2.md 第 4 节）：树不对时的模拟校准，动作只含习性变化
# （配置 recovery_v5_l1）。取出 5 棵训练树，生成新种子 11、12 的模拟表（设定与 v1 的第 0 阶段相同：数据在 Hackett 第 1 棵树上生成，
# 习性的 OU 响应同 v4，另加早期爆发布朗背景，根处速率 7.8e-4、b = −0.025），训练 100 个模型
# （效应/零模型 × 种子 11、12 × 第 21、41、61、81、100 棵树 × 5 折），全部完成后评估并写出 decision.json。
# 用法（EvoWM 根目录）：GPUS="0 1 2 3" bash scripts/run_stage0_v2.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
[ -f docs/prereg/real_main_v2.md ] || { echo "缺少预注册 docs/prereg/real_main_v2.md"; exit 1; }
LOG=outputs/stage0_v2/logs
mkdir -p "$LOG"

python scripts/extract_trees.py > "$LOG/trees.txt"
echo "sim tables $(date +%T)" >> "$LOG/progress.txt"
python scripts/sim_recovery_tables.py --version v5 --seeds 11 12 --bg-rate 7.8e-4 --bg-eb -0.025 > "$LOG/sim_tables.txt" 2>&1
md5sum data/processed/sim/recovery_v5/*_s11/species_traits.csv data/processed/sim/recovery_v5/*_s12/species_traits.csv >> "$LOG/sim_tables.txt"  # 应与预注册第 4 节的 md5 相同

LIST="$LOG/jobs.txt"
: > "$LIST"
for s in 11 12; do
  for sc in effect null; do
    for t in 21 41 61 81 100; do
      tt=$(printf %03d "$t")
      for f in 0 1 2 3 4; do
        echo "${sc}_s${s}_t${tt}_f${f} data.dataset.species_table=data/processed/sim/recovery_v5/${sc}_s${s}/species_traits.csv data.dataset.tree_file=data/processed/trees/hackett_t${tt}.nwk data.dataset.n_trees=1 data.dataset.holdout.fold=${f} seed=${s}" >> "$LIST"
      done
    done
  done
done
CONFIG=recovery_v5_l1 bash scripts/gpu_jobs.sh "$LIST" "$LOG" stage0_v2
echo "trained $(date +%T)" >> "$LOG/progress.txt"

missing=$(while read -r name _; do [ -f "outputs/hjepa/ckpts/stage0_v2/$name/evowm_l1_object.ckpt" ] || echo "$name"; done < "$LIST")
if [ -n "$missing" ]; then
  echo "以下模型没有训练完成，未评估：$missing" | tee -a "$LOG/progress.txt"
  exit 1
fi
python scripts/eval_stage0_v2.py 2>&1 | tee "$LOG/eval.txt"
echo "ALLDONE $(date +%T)" >> "$LOG/progress.txt"
