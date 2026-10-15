#!/usr/bin/env bash
# 按习性分层的主分析 v1，一条命令跑完（预注册 docs/prereg/units_main_v1.md 第 9 节）：
#   1. 第 0 阶段（scripts/run_stage0_units_v1.sh）：100 个三层模型，评估后写 outputs/stage0_units_v1/decision.json；
#   2. G1-L2、G2-L2、G-L1 都通过才接着跑真实数据（scripts/run_real_units_v1.sh）：50 + 15 个三层模型，评估写到 outputs/real_units_v1/。
# 第 0 阶段没通过就停下，不训练任何真实数据模型。总进度在 outputs/units_main_v1.log。
# 用法（EvoWM 根目录）：GPUS="0 1 2 3" bash scripts/run_main_units_v1.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p outputs
LOG=outputs/units_main_v1.log
echo "stage0 start $(date '+%F %T')" | tee -a "$LOG"
bash scripts/run_stage0_units_v1.sh
echo "stage0 done $(date '+%F %T')" | tee -a "$LOG"
if python -c "import json,sys; d=json.load(open('outputs/stage0_units_v1/decision.json', encoding='utf-8')); sys.exit(0 if d.get('pass') else 1)"; then
  echo "stage0 passed; real data start $(date '+%F %T')" | tee -a "$LOG"
  bash scripts/run_real_units_v1.sh
  echo "real data done $(date '+%F %T')" | tee -a "$LOG"
else
  echo "stage0 NOT passed; real data not run $(date '+%F %T')" | tee -a "$LOG"
fi
echo "ALLDONE $(date '+%F %T')" | tee -a "$LOG"
