#!/usr/bin/env bash
# 探索 v2：加"全部习性列"的线性基线后重新评估全部 7 个模型集（scripts/explore_modules_v2.py）。
# 模型都是现成的（主分析、v2.1、第 0 阶段、昨晚 v6 训练的），不训练，只用 CPU，约 15 分钟。
# 用法（EvoWM 根目录）：EVAL_WORKERS=16 bash scripts/run_explore_modules_v2.sh
# 结果：outputs/explore_modules_v2/report.md（另有 traits_gain_v2.csv）；进度在 outputs/explore_modules_v2/progress.txt。
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
OUT=outputs/explore_modules_v2
mkdir -p "$OUT/logs"
P=$OUT/progress.txt
echo "start $(date '+%F %T')" | tee "$P"
python scripts/extract_trees.py > "$OUT/logs/trees.txt" 2>&1 || echo "extract_trees FAILED" | tee -a "$P"
if [ ! -f data/processed/sim/explore_v6/modular_s16/species_traits.csv ]; then
  python scripts/sim_modular_tables.py --seeds 15 16 --bg-rate 7.8e-4 --bg-eb -0.025 > "$OUT/logs/sim_tables.txt" 2>&1
fi
if python scripts/explore_modules_v2.py --sets real_hab3 real_hab3_ericson real_l1 sim_v5_effect sim_v5_null sim_v6_hab3 sim_v6_l1 \
    > "$OUT/logs/eval.txt" 2>&1; then
  echo "evaluated, report written $(date '+%F %T')" | tee -a "$P"
else
  echo "FAILED (see $OUT/logs/eval.txt) $(date '+%F %T')" | tee -a "$P"
  exit 1
fi
echo "ALLDONE $(date '+%F %T')" | tee -a "$P"
