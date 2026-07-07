#!/usr/bin/env bash
# ============================================================
# Class-evidence heatmap (CAM류) — 어느 위치를 보고 그 클래스로 판단했는지.
# frozen DINO feature + 선형 분류기(logreg/NCM) 기준.
# ============================================================
set -euo pipefail

DATA_ROOT="${DATA_ROOT:?must set DATA_ROOT=/path/to/class_folders}"   # 분류기 학습용
WEIGHTS="${WEIGHTS:?must set WEIGHTS=/path/to/teacher_checkpoint.pth}"
CONFIG="${CONFIG:-dino_v3/dinov3/configs/train/weaksup/stage2_ssl_weaksup.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-./out/heatmap}"

QUERY_DIR="${QUERY_DIR:-}"          # 비우면 DATA_ROOT 에서 클래스당 PER_CLASS 장 샘플
PER_CLASS="${PER_CLASS:-4}"
IMAGE_SIZE="${IMAGE_SIZE:-448}"
CLF="${CLF:-logreg}"                # logreg | ncm
TARGET="${TARGET:-pred}"            # pred | true
MODE="${MODE:-contrastive}"        # contrastive | raw

EXTRA=()
if [[ -n "${QUERY_DIR}" ]]; then EXTRA+=(--query-dir "${QUERY_DIR}"); fi

python inspection/fewshot_heatmap.py \
    --config-file "${CONFIG}" \
    --pretrained-weights "${WEIGHTS}" \
    --data-root "${DATA_ROOT}" \
    --per-class "${PER_CLASS}" \
    --image-size "${IMAGE_SIZE}" \
    --clf "${CLF}" \
    --target "${TARGET}" \
    --mode "${MODE}" \
    --output-dir "${OUTPUT_DIR}" \
    "${EXTRA[@]}" \
    "$@"
