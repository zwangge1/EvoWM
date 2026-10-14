#!/usr/bin/env bash
# 按卡并行训练一批模型（默认配置 recovery_v3_l1，单卡 fp32，与 v3、v4 相同；CONFIG 可换配置，例如 recovery_v5_l1）。
# 用法（EvoWM 根目录）：[CONFIG=recovery_v5_l1] bash scripts/gpu_jobs.sh <任务清单> <日志目录> <检查点子目录>
# 任务清单每行："<名字> <hydra 覆盖项...>"。GPUS 里每张卡一个工作进程，按行号轮流领任务；已有检查点的任务跳过。
set -euo pipefail
LIST=$1
LOG=$2
PREFIX=$3
GPUS="${GPUS:-0 1 2 3}"
WORKERS="${WORKERS:-4}"
CONFIG="${CONFIG:-recovery_v3_l1}"
mapfile -t JOBS < "$LIST"

run_job () {  # $1 卡号 $2 名字 其余为覆盖项
  local g=$1 name=$2
  shift 2
  if [ -f "outputs/hjepa/ckpts/$PREFIX/$name/evowm_l1_object.ckpt" ]; then
    echo "skip $name" >> "$LOG/progress.txt"
    return 0
  fi
  echo "start $name gpu$g $(date +%T)" >> "$LOG/progress.txt"
  if CUDA_VISIBLE_DEVICES=$g bash scripts/train_hjepa.sh "$CONFIG" "$@" \
      trainer.accelerator=gpu num_workers=$WORKERS subdir=$PREFIX/$name > "$LOG/$name.log" 2>&1; then
    echo "done $name $(date +%T)" >> "$LOG/progress.txt"
  else
    echo "FAIL $name $(date +%T)" >> "$LOG/progress.txt"
  fi
}

worker () {  # $1 卡号 $2 工作进程编号 $3 进程总数
  local g=$1 w=$2 n=$3 i
  for ((i = w; i < ${#JOBS[@]}; i += n)); do
    # shellcheck disable=SC2086
    run_job "$g" ${JOBS[$i]}
  done
}

read -ra G <<< "$GPUS"
for w in "${!G[@]}"; do
  worker "${G[$w]}" "$w" "${#G[@]}" &
done
wait
