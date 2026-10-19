#!/usr/bin/env bash
# 读数诊断 v1：世界模型的点预测为什么不如照抄，以及变化量读数（scripts/diag_readout_v1.py；设计 docs/design/诊断_读数_2026-10-19.md）。
# 模型都是现成的（与探索 v2 相同的 7 个模型集），不训练，只用 CPU，约 15 分钟。
# 用法（EvoWM 根目录）：EVAL_WORKERS=16 bash scripts/run_diag_readout_v1.sh
# 结果：outputs/diag_readout_v1/report.md（另有 readout_v1.csv）；进度在 outputs/diag_readout_v1/progress.txt。
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
OUT=outputs/diag_readout_v1
mkdir -p "$OUT/logs"
P=$OUT/progress.txt
echo "start $(date '+%F %T')" | tee "$P"
python scripts/extract_trees.py > "$OUT/logs/trees.txt" 2>&1 || echo "extract_trees FAILED" | tee -a "$P"
if [ ! -f data/processed/sim/explore_v6/modular_s16/species_traits.csv ]; then
  python scripts/sim_modular_tables.py --seeds 15 16 --bg-rate 7.8e-4 --bg-eb -0.025 > "$OUT/logs/sim_tables.txt" 2>&1
fi
if python scripts/diag_readout_v1.py --sets real_hab3 real_hab3_ericson real_l1 sim_v5_effect sim_v5_null sim_v6_hab3 sim_v6_l1 \
    > "$OUT/logs/eval.txt" 2>&1; then
  echo "evaluated, report written $(date '+%F %T')" | tee -a "$P"
else
  echo "FAILED (see $OUT/logs/eval.txt) $(date '+%F %T')" | tee -a "$P"
  exit 1
fi
echo "ALLDONE $(date '+%F %T')" | tee -a "$P"
