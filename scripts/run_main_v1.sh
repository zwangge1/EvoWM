#!/usr/bin/env bash
# 真实数据主分析 v1 一条命令跑完（预注册 docs/prereg/real_main_v1.md）：
# 第 0 阶段（模拟校准，100 个模型）→ 通过则训练并评估真实数据（50 + 15 个模型）；不通过就停在第 0 阶段。
# 用法（EvoWM 根目录，tmux 里）：GPUS="0 1 2 3" bash scripts/run_main_v1.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
bash scripts/run_stage0_v1.sh
python -c "import json,sys; d=json.load(open('outputs/stage0_v1/decision.json')); sys.exit(0 if d['pass'] else 1)" \
  || { echo "第 0 阶段未通过，停在这里；报告见 outputs/stage0_v1/report.md"; exit 1; }
bash scripts/run_real_v1.sh
