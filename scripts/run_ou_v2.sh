#!/usr/bin/env bash
# 多近亲 OU 对照（scripts/eval_ou_v2.py；设计 docs/design/多近亲OU对照_2026-10-19.md）。只算第一层。
# 模型都是现成的，不训练，只用 CPU；可以与 GPU 上的训练同时跑。需要先有 OU 式对照 v1 的结果（outputs/ou_v1/sets），用来核对。
# 用法（EvoWM 根目录）：
#   EVAL_WORKERS=8 bash scripts/run_ou_v2.sh        原来的模型，判定要的 4 个模型集，8 个进程约 40 分钟
#   EVAL_WORKERS=16 bash scripts/run_ou_v2.sh res   重训后的模型（residual_v1 连同它最后的 OU 式对照跑完以后）
# 结果：outputs/ou_v2/report.md（重训后的模型为 report_res.md）；进度在 outputs/ou_v2/progress.txt。
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
FAMILY="${1:-old}"
OUT=outputs/ou_v2
mkdir -p "$OUT/logs"
P=$OUT/progress.txt
echo "start $FAMILY $(date '+%F %T')" | tee -a "$P"
python scripts/extract_trees.py > "$OUT/logs/trees.txt" 2>&1 || echo "extract_trees FAILED" | tee -a "$P"
if [ ! -f data/processed/sim/explore_v6/modular_s16/species_traits.csv ]; then
  python scripts/sim_modular_tables.py --seeds 15 16 --bg-rate 7.8e-4 --bg-eb -0.025 > "$OUT/logs/sim_tables.txt" 2>&1
fi
SETS="real_hab3 real_hab3_ericson sim_v5_null sim_v6_hab3"
if [ "$FAMILY" = "res" ]; then
  SETS=$(for s in $SETS; do printf '%s_res ' "$s"; done)
elif [ "$FAMILY" != "old" ]; then
  echo "参数只能是 res 或不写" | tee -a "$P"
  exit 1
fi
if python scripts/eval_ou_v2.py --family "$FAMILY" --sets $SETS > "$OUT/logs/eval_$FAMILY.txt" 2>&1; then
  echo "evaluated $FAMILY, report written $(date '+%F %T')" | tee -a "$P"
else
  echo "FAILED $FAMILY (see $OUT/logs/eval_$FAMILY.txt) $(date '+%F %T')" | tee -a "$P"
  exit 1
fi
echo "ALLDONE $FAMILY $(date '+%F %T')" | tee -a "$P"
