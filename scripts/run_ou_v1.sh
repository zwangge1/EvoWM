#!/usr/bin/env bash
# OU 式对照（scripts/eval_ou_v1.py；设计 docs/design/OU式对照_2026-10-19.md）。
# 模型都是现成的，不训练，只用 CPU；可以与 GPU 上的训练同时跑。
# 用法（EvoWM 根目录）：
#   EVAL_WORKERS=8 bash scripts/run_ou_v1.sh        原来的 7 个模型集，8 个进程约 1 小时（每个进程约 1.7 GB 内存）
#   EVAL_WORKERS=16 bash scripts/run_ou_v1.sh res   重训后的 7 个模型集（outputs/hjepa/ckpts/residual_v1 跑完以后）
# 结果：outputs/ou_v1/report.md（重训后的模型为 report_res.md）；进度在 outputs/ou_v1/progress.txt。
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
FAMILY="${1:-old}"
OUT=outputs/ou_v1
mkdir -p "$OUT/logs"
P=$OUT/progress.txt
echo "start $FAMILY $(date '+%F %T')" | tee -a "$P"
python scripts/extract_trees.py > "$OUT/logs/trees.txt" 2>&1 || echo "extract_trees FAILED" | tee -a "$P"
if [ ! -f data/processed/sim/explore_v6/modular_s16/species_traits.csv ]; then
  python scripts/sim_modular_tables.py --seeds 15 16 --bg-rate 7.8e-4 --bg-eb -0.025 > "$OUT/logs/sim_tables.txt" 2>&1
fi
SETS="real_hab3 real_hab3_ericson real_l1 sim_v5_effect sim_v5_null sim_v6_hab3 sim_v6_l1"
if [ "$FAMILY" = "res" ]; then
  SETS=$(for s in $SETS; do printf '%s_res ' "$s"; done)
elif [ "$FAMILY" != "old" ]; then
  echo "参数只能是 res 或不写" | tee -a "$P"
  exit 1
fi
if python scripts/eval_ou_v1.py --family "$FAMILY" --sets $SETS > "$OUT/logs/eval_$FAMILY.txt" 2>&1; then
  echo "evaluated $FAMILY, report written $(date '+%F %T')" | tee -a "$P"
else
  echo "FAILED $FAMILY (see $OUT/logs/eval_$FAMILY.txt) $(date '+%F %T')" | tee -a "$P"
  exit 1
fi
echo "ALLDONE $FAMILY $(date '+%F %T')" | tee -a "$P"
