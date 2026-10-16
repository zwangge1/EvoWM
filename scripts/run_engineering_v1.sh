#!/usr/bin/env bash
# 工程验证 v1（预注册 docs/prereg/engineering_drop_v1.md）：补算推测腿的骨段变化量 Δ，再跑落震。只用 CPU，不训练。
# 需要主分析的 50 个三层模型（outputs/hjepa/ckpts/real_units_v1/）与 outputs/real_units_v1/results.json（自检用）。
# 结果写到 outputs/engineering_v1/。
# 用法（EvoWM 根目录）：EVAL_WORKERS=8 bash scripts/run_engineering_v1.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
[ -f docs/prereg/engineering_drop_v1.md ] || { echo "缺少预注册 docs/prereg/engineering_drop_v1.md"; exit 1; }
LOG=outputs/engineering_v1/logs
mkdir -p "$LOG"
echo "872bcc14eecae132a83d5f3dec5ea299  data/processed/species_traits.csv" | md5sum -c - | tee "$LOG/table_md5.txt" \
  || { echo "物种主表的 md5 与预注册第 2 节不同，停止"; exit 1; }
python -c "import exudyn, sys; print('exudyn', exudyn.__version__); sys.exit(0 if exudyn.__version__ == '1.13.6' else 1)" \
  | tee "$LOG/exudyn.txt" || { echo "Exudyn 版本应为 1.13.6（pip install exudyn==1.13.6）"; exit 1; }
python scripts/extract_trees.py > "$LOG/trees.txt"
echo "delta $(date +%T)" >> "$LOG/progress.txt"
python scripts/eval_engineering_delta_v1.py --workers "${EVAL_WORKERS:-8}" 2>&1 | tee "$LOG/delta.txt"
echo "drop $(date +%T)" >> "$LOG/progress.txt"
python scripts/run_engineering_drop_v1.py --workers "${EVAL_WORKERS:-8}" 2>&1 | tee "$LOG/drop.txt"
echo "ALLDONE $(date +%T)" >> "$LOG/progress.txt"
