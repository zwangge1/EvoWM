#!/usr/bin/env bash
# 预测变化量重训（设计 docs/design/预测变化量_重训_2026-10-19.md），可挂过夜，约 10 小时（4 张卡）。
# 用法（EvoWM 根目录）：GPUS="0 1 2 3" EVAL_WORKERS=16 bash scripts/run_residual_v1.sh
# 步骤（某一步失败就记下，能做的继续做；已有检查点的任务跳过，中断时没训完的从最近的存档接着训，可以重跑）：
#   A. 训练真实三层 50（Hackett）、空白模拟三层 50、v6 模拟三层 50，评估，写中间报告（含判据）；
#   B. 训练 v5 效应模拟三层 50、真实三层 15（Ericson）；
#   C. 训练真实与 v6 模拟只训第一层的模型各 50；评估 B、C，写最终报告；
#   D. 用 OU 式对照评估重训后的模型（scripts/run_ou_v1.sh res，只用 CPU，约 30 分钟）。
# 结果：outputs/residual_v1/report.md（另有 residual_v1.csv、sets/）、outputs/ou_v1/report_res.md；
# 进度在 outputs/residual_v1/progress.txt。
# 检查点：outputs/hjepa/ckpts/residual_v1/<模型集>/<运行>_f<折>/。旧模型的结果读 outputs/diag_readout_v1/sets。
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
OUT=outputs/residual_v1
LOG=$OUT/logs
mkdir -p "$LOG"
P=$OUT/progress.txt
CK=residual_v1
step () { echo "$1 $(date '+%F %T')" | tee -a "$P"; }

step "start"
python scripts/extract_trees.py > "$LOG/trees.txt" 2>&1 || step "extract_trees FAILED"
[ -f outputs/diag_readout_v1/sets/real_hab3.npz ] || step "WARNING outputs/diag_readout_v1/sets missing (old-model columns and criterion 3 will be empty)"
if [ ! -f data/processed/sim/explore_v6/modular_s16/species_traits.csv ]; then
  python scripts/sim_modular_tables.py --seeds 15 16 --bg-rate 7.8e-4 --bg-eb -0.025 > "$LOG/sim_tables.txt" 2>&1 || step "v6 tables FAILED"
fi

make_list () {  # $1 清单 $2 名字模板（{s}、{t}） $3 表模板（{s}） $4 树前缀 $5 种子 $6 树
  local list=$1 name=$2 table=$3 tree=$4 seeds=$5 trees=$6 s t tt f n tb
  : > "$list"
  for s in $seeds; do
    tb=${table//\{s\}/$s}
    for t in $trees; do
      tt=$(printf %03d "$t")
      n=${name//\{s\}/$s}
      n=${n//\{t\}/$tt}
      for f in 0 1 2 3 4; do
        echo "${n}_f${f} data.dataset.species_table=${tb} data.dataset.tree_file=data/processed/trees/${tree}_t${tt}.nwk data.dataset.n_trees=1 data.dataset.holdout.fold=${f} seed=${s}" >> "$list"
      done
    done
  done
}
TREES="21 41 61 81 100"
make_list "$LOG/real_hab3.txt" "hackett_t{t}_s{s}" data/processed/species_traits.csv hackett "1 2" "$TREES"
make_list "$LOG/real_ericson.txt" "ericson_t{t}_s{s}" data/processed/species_traits.csv ericson "1" "21 41 61"
make_list "$LOG/sim_null.txt" "null_s{s}_t{t}" "data/processed/sim/recovery_v5/null_s{s}/species_traits.csv" hackett "13 14" "$TREES"
make_list "$LOG/sim_effect.txt" "effect_s{s}_t{t}" "data/processed/sim/recovery_v5/effect_s{s}/species_traits.csv" hackett "13 14" "$TREES"
make_list "$LOG/sim_v6.txt" "modular_s{s}_t{t}" "data/processed/sim/explore_v6/modular_s{s}/species_traits.csv" hackett "15 16" "$TREES"

train () {  # $1 清单 $2 配置 $3 检查点名 $4 模型集子目录
  mkdir -p "$LOG/$4"
  CONFIG=$2 CKPT_NAME=$3 bash scripts/gpu_jobs.sh "$1" "$LOG/$4" "$CK/$4" || step "train $4 ($2) had errors"
  local done_n want
  want=$(wc -l < "$1")
  done_n=$(while read -r name _; do [ -f "outputs/hjepa/ckpts/$CK/$4/$name/${3}_object.ckpt" ] && echo "$name"; done < "$1" | wc -l)
  step "trained $4 ($2): $done_n/$want"
}
evaluate () {  # 模型集名 ...
  if python scripts/eval_residual_v1.py --sets "$@" >> "$LOG/eval.txt" 2>&1; then
    step "evaluated $*"
  else
    step "eval FAILED for $* (see $LOG/eval.txt)"
  fi
}

step "A start"
train "$LOG/real_hab3.txt" recovery_v5_hab3_res evowm_hab3 real_hab3
train "$LOG/sim_null.txt" recovery_v5_hab3_res evowm_hab3 sim_v5
train "$LOG/sim_v6.txt" recovery_v5_hab3_res evowm_hab3 sim_v6_hab3
evaluate real_hab3_res sim_v5_null_res sim_v6_hab3_res
step "B start"
train "$LOG/sim_effect.txt" recovery_v5_hab3_res evowm_hab3 sim_v5
if [ -f data/processed/trees/ericson_t021.nwk ]; then
  train "$LOG/real_ericson.txt" recovery_v5_hab3_res evowm_hab3 real_hab3
else
  step "no Ericson trees, skip"
fi
step "C start"
train "$LOG/real_hab3.txt" recovery_v5_l1_res evowm_l1 real_l1
train "$LOG/sim_v6.txt" recovery_v5_l1_res evowm_l1 sim_v6_l1
evaluate sim_v5_effect_res real_hab3_ericson_res real_l1_res sim_v6_l1_res
step "D start"
if bash scripts/run_ou_v1.sh res > "$LOG/ou_res.txt" 2>&1; then
  step "OU-style evaluation done (outputs/ou_v1/report_res.md)"
else
  step "OU-style evaluation FAILED (see outputs/ou_v1/logs/eval_res.txt)"
fi
step "ALLDONE"
