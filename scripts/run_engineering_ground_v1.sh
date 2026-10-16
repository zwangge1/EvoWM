#!/usr/bin/env bash
# 工程验证 v2（预注册 docs/prereg/engineering_ground_v1.md）：地面任务筛选与推测腿检验。只用 CPU，不训练。
# 需要 v1 的输出 outputs/engineering_v1/delta.json 与 legs.json（推测的变化量 Δ 直接沿用，不再评估模型）。
# 结果写到 outputs/engineering_ground_v1/。
# 用法（EvoWM 根目录）：EVAL_WORKERS=32 bash scripts/run_engineering_ground_v1.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
[ -f docs/prereg/engineering_ground_v1.md ] || { echo "缺少预注册 docs/prereg/engineering_ground_v1.md"; exit 1; }
LOG=outputs/engineering_ground_v1/logs
mkdir -p "$LOG"
echo "872bcc14eecae132a83d5f3dec5ea299  data/processed/species_traits.csv" | md5sum -c - | tee "$LOG/table_md5.txt" \
  || { echo "物种主表的 md5 与预注册不同，停止"; exit 1; }
python -c "import exudyn, sys; print('exudyn', exudyn.__version__); sys.exit(0 if exudyn.__version__ == '1.13.6' else 1)" \
  | tee "$LOG/exudyn.txt" || { echo "Exudyn 版本应为 1.13.6"; exit 1; }
echo "start $(date +%T)" >> "$LOG/progress.txt"
python scripts/run_engineering_ground_v1.py --workers "${EVAL_WORKERS:-16}" 2>&1 | tee "$LOG/run.txt"
echo "ALLDONE $(date +%T)" >> "$LOG/progress.txt"
