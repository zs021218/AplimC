#!/usr/bin/env bash

# 简单KD参数扫描脚本（在不同 alpha / temperature / label_smoothing / lr / wd 上网格搜索）
# 用法：bash scripts/sweep_simple_kd.sh data/processed/multimodal_data.h5 /experiments/runs/label_smoothing_resnet18_v2/checkpoints/best_model.pth cuda:2

set -euo pipefail

DATA_PATH=${1:?"need data path"}
TEACHER_PATH=${2:-}
DEVICE=${3:-cuda:2}

ALPHAS=(0.5 0.6 0.7)
TEMPS=(4 5 6)
SMOOTHS=(0.0 0.05)
LRS=(0.001 0.0005)
WDS=(0.0001 0.0005)

for A in "${ALPHAS[@]}"; do
  for T in "${TEMPS[@]}"; do
    for S in "${SMOOTHS[@]}"; do
      for LR in "${LRS[@]}"; do
        for WD in "${WDS[@]}"; do
          NAME="sweep_a${A}_t${T}_ls${S}_lr${LR}_wd${WD}_150ep"
          echo "Running ${NAME}"
          python scripts/train_simple_kd.py \
            --data "${DATA_PATH}" \
            --student-modalities stokes fluorescence \
            --alpha ${A} --temperature ${T} --label-smoothing ${S} \
            --lr ${LR} --weight-decay ${WD} \
            --batch-size 32 --epochs 150 \
            ${TEACHER_PATH:+ --teacher "${TEACHER_PATH}" --freeze-teacher} \
            --device "${DEVICE}" \
            --exp-name "${NAME}" || true
        done
      done
    done
  done
done

echo "Sweep finished."


