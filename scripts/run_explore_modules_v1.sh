#!/usr/bin/env bash
# 探索：全身形态在各层的习性读数（设计 docs/design/探索_分层模块_2026-10-18.md），可挂过夜。
# 用法（EvoWM 根目录）：GPUS="0 1 2 3" bash scripts/run_explore_modules_v1.sh
# 步骤（某一步失败就记下，能做的继续做）：
#   1. 评估现有模型（只用 CPU）：真实数据三层 50 + Ericson 15、只训第一层 50；第 0 阶段模拟 100，写中间报告；
#   2. 生成全身模块模拟表 v6（种子 15、16）；
#   3. 训练 v6 上的 50 个三层模型与 50 个只训第一层的模型（GPU，约 3 小时）；
#   4. 评估 v6，写最终报告 outputs/explore_modules_v1/report.md（另有 traits_gain.csv、contrasts.csv、summary.json）。
# 进度在 outputs/explore_modules_v1/progress.txt；评估的并行进程数用 EVAL_WORKERS 设（默认 8）。
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
OUT=outputs/explore_modules_v1
LOG=$OUT/logs
mkdir -p "$LOG/hab3" "$LOG/l1"
P=$OUT/progress.txt
step () { echo "$1 $(date '+%F %T')" | tee -a "$P"; }

step "start"
python scripts/extract_trees.py > "$LOG/trees.txt" 2>&1 || step "extract_trees FAILED"

# 1. 现有模型
if python scripts/explore_modules_v1.py --sets real_hab3 real_hab3_ericson real_l1 sim_v5_effect sim_v5_null \
    > "$LOG/eval_existing.txt" 2>&1; then
  step "1 existing models evaluated (interim report written)"
else
  step "1 existing models FAILED (see $LOG/eval_existing.txt)"
fi

# 2. 模拟表
if python scripts/sim_modular_tables.py --seeds 15 16 --bg-rate 7.8e-4 --bg-eb -0.025 > "$LOG/sim_tables.txt" 2>&1; then
  md5sum data/processed/sim/explore_v6/*/species_traits.csv | tee -a "$LOG/sim_tables.txt"
  step "2 v6 tables written"
else
  step "2 v6 tables FAILED; stop"
  exit 1
fi

# 3. 训练（已有检查点的任务会跳过，所以中断后可以重跑）
LIST=$LOG/jobs.txt
: > "$LIST"
for s in 15 16; do
  for t in 21 41 61 81 100; do
    tt=$(printf %03d "$t")
    for f in 0 1 2 3 4; do
      echo "modular_s${s}_t${tt}_f${f} data.dataset.species_table=data/processed/sim/explore_v6/modular_s${s}/species_traits.csv data.dataset.tree_file=data/processed/trees/hackett_t${tt}.nwk data.dataset.n_trees=1 data.dataset.holdout.fold=${f} seed=${s}" >> "$LIST"
    done
  done
done
step "3a train hab3 start"
CONFIG=recovery_v5_hab3 CKPT_NAME=evowm_hab3 bash scripts/gpu_jobs.sh "$LIST" "$LOG/hab3" explore_v6_hab3
step "3b train l1 start"
CONFIG=recovery_v5_l1 CKPT_NAME=evowm_l1 bash scripts/gpu_jobs.sh "$LIST" "$LOG/l1" explore_v6_l1
n_h=$(ls outputs/hjepa/ckpts/explore_v6_hab3/*/evowm_hab3_object.ckpt 2>/dev/null | wc -l)
n_l=$(ls outputs/hjepa/ckpts/explore_v6_l1/*/evowm_l1_object.ckpt 2>/dev/null | wc -l)
step "3 trained: hab3 $n_h/50, l1 $n_l/50"

# 4. 评估 v6 与最终报告
if python scripts/explore_modules_v1.py --sets sim_v6_hab3 sim_v6_l1 > "$LOG/eval_v6.txt" 2>&1; then
  step "4 v6 evaluated, final report written"
else
  step "4 v6 FAILED (see $LOG/eval_v6.txt)"
  python scripts/explore_modules_v1.py --report-only > "$LOG/report_only.txt" 2>&1 || true
fi
step "ALLDONE"
