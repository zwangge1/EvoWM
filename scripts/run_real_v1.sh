#!/usr/bin/env bash
# 真实数据主分析 v1（预注册 docs/prereg/real_main_v1.md 第 3、5 节）。只在第 0 阶段通过后运行。
# 主分析：Hackett 第 21、41、61、81、100 棵树 × 训练种子 1、2 × 5 折 = 50 个模型；
# 稳健性：Ericson 第 21、41、61 棵树 × 种子 1 × 5 折 = 15 个模型（缺 Ericson 树文件时跳过）。训练完后评估。
# 用法（EvoWM 根目录）：GPUS="0 1 2 3" bash scripts/run_real_v1.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
[ -f docs/prereg/real_main_v1.md ] || { echo "缺少预注册 docs/prereg/real_main_v1.md"; exit 1; }
python -c "import json,sys; d=json.load(open('outputs/stage0_v1/decision.json')); sys.exit(0 if d['pass'] else 1)" \
  || { echo "第 0 阶段没有通过（或没有运行），不训练真实数据模型"; exit 1; }
LOG=outputs/real_v1/logs
mkdir -p "$LOG"
python scripts/extract_trees.py > "$LOG/trees.txt"
md5sum data/processed/species_traits.csv > "$LOG/table_md5.txt"  # 应为 872bcc14eecae132a83d5f3dec5ea299

LIST="$LOG/jobs.txt"
: > "$LIST"
add () {  # $1 主干 $2 树编号 $3 种子
  local tt
  tt=$(printf %03d "$2")
  for f in 0 1 2 3 4; do
    echo "${1}_t${tt}_s${3}_f${f} data.dataset.species_table=data/processed/species_traits.csv data.dataset.tree_file=data/processed/trees/${1}_t${tt}.nwk data.dataset.n_trees=1 data.dataset.holdout.fold=${f} seed=${3}" >> "$LIST"
  done
}
for t in 21 41 61 81 100; do for s in 1 2; do add hackett "$t" "$s"; done; done
if [ -f data/processed/trees/ericson_t021.nwk ]; then
  for t in 21 41 61; do add ericson "$t" 1; done
fi
echo "start $(date +%T)" >> "$LOG/progress.txt"
bash scripts/gpu_jobs.sh "$LIST" "$LOG" real_v1
echo "trained $(date +%T)" >> "$LOG/progress.txt"

missing=$(grep '^hackett' "$LIST" | while read -r name _; do [ -f "outputs/hjepa/ckpts/real_v1/$name/evowm_l1_object.ckpt" ] || echo "$name"; done)
if [ -n "$missing" ]; then
  echo "以下主分析模型没有训练完成，未评估：$missing" | tee -a "$LOG/progress.txt"
  exit 1
fi
python scripts/eval_real_v1.py 2>&1 | tee "$LOG/eval.txt"
echo "ALLDONE $(date +%T)" >> "$LOG/progress.txt"
