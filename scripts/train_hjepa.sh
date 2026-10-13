#!/usr/bin/env bash
# 用 H-JEPA 的训练入口训练 EvoWM。
# 用法：bash scripts/train_hjepa.sh evowm_l1 [hydra 覆盖项 ...]
#   例：bash scripts/train_hjepa.sh evowm_l2 seed=1 trainer.max_epochs=50
# 配置在 configs/hjepa/；数据路径相对 EVOWM_ROOT；结果写到 HJEPA_HOME/ckpts/（默认 outputs/hjepa/）。
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CFG="${1:-evowm_l1}"
shift || true
export EVOWM_ROOT="$ROOT"
export HJEPA_HOME="${HJEPA_HOME:-$ROOT/outputs/hjepa}"
export PYTHONPATH="$ROOT/hjepa:$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$ROOT/hjepa/h_jepa"
exec python main_hjepa.py --config-dir "$ROOT/configs/hjepa" --config-name "$CFG" "$@"
