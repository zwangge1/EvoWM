#!/usr/bin/env bash
# 岛屿对的外部检验（设计草案 docs/design/外部检验_岛屿_草案_2026-10-18.md），只用 CPU，不训练模型。
#   bash scripts/run_island_v1.sh stage0   第 0 步（预注册之前）：主分析第 0 阶段的 100 个模型与模拟表，只用模拟数据
#   bash scripts/run_island_v1.sh real     真实数据（预注册之后）：主分析的 50 个模型，另 15 个 Ericson 模型作稳健性
# 结果写到 outputs/island_v1/<模式>/（report.md、results.json、predictions.csv.gz），日志 eval.txt。
# 并行进程数用 EVAL_WORKERS 设（默认 8）。在 EvoWM 根目录运行。
set -euo pipefail
MODE=${1:?"用法：bash scripts/run_island_v1.sh stage0|real"}
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
case "$MODE" in stage0|real) ;; *) echo "模式只能是 stage0 或 real"; exit 1 ;; esac
OUT=outputs/island_v1/$MODE
mkdir -p "$OUT"

# 外部数据与登记的 sha256 一致（data/MANIFEST.md）
sha256sum -c - <<'SHA' | tee "$OUT/data_check.txt" || { echo "外部数据与 data/MANIFEST.md 不符，停止"; exit 1; }
79fe71e0f10febabb68ccedd57548f7cfb32096d9c363b340bdd717d74c2d16d  data/raw/ponti2025/morpho_db.csv
13504121c078bb9e68d5e38e3b61754e02b7335b475acdbd8d96e2a6cb6bfe31  data/raw/ponti2025/morpho_db_filtered.csv
4290542ebc0a1570ea62286d6797bb896a8f6294898dab849194c94e976feefa  data/raw/wright2016/skeletal_data.xlsx
SHA
python scripts/extract_trees.py > "$OUT/trees.txt"

if [ "$MODE" = stage0 ]; then
  md5sum -c - <<'MD5' | tee -a "$OUT/data_check.txt" || { echo "模拟表的 md5 与 units_main_v1 第 0 阶段不同，停止"; exit 1; }
c03d34812c279fc7e815730dcaec32ec  data/processed/sim/recovery_v5/effect_s13/species_traits.csv
fa8c609ed17ebb17be853cd7a73973ad  data/processed/sim/recovery_v5/null_s13/species_traits.csv
0423f490c2213e0744ec2830895defdf  data/processed/sim/recovery_v5/effect_s14/species_traits.csv
cb3c6b33419ffb604deca8cb9edb45d0  data/processed/sim/recovery_v5/null_s14/species_traits.csv
MD5
else
  [ -f docs/prereg/island_v1.md ] || { echo "缺少预注册 docs/prereg/island_v1.md；真实数据只在预注册提交之后运行"; exit 1; }
  echo "872bcc14eecae132a83d5f3dec5ea299  data/processed/species_traits.csv" | md5sum -c - | tee -a "$OUT/data_check.txt" \
    || { echo "物种主表的 md5 与主分析不同，停止"; exit 1; }
  python -c "import openpyxl" 2>/dev/null \
    || { echo "缺少 openpyxl（读 Wright 的 xlsx）：pip install --no-cache-dir -i https://pypi.org/simple openpyxl"; exit 1; }
fi

echo "start $(date '+%F %T')" | tee "$OUT/progress.txt"
python scripts/eval_island_v1.py --mode "$MODE" 2>&1 | tee "$OUT/eval.txt"
echo "ALLDONE $(date '+%F %T')" | tee -a "$OUT/progress.txt"
