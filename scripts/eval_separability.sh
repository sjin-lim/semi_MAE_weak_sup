#!/usr/bin/env bash
# ============================================================
# DINO feature 폴더-클래스 분리도 진단 (단일 GPU, 분산 불필요)
#
# 폴더별로 클래스를 모아둔 ImageFolder 레이아웃으로
# 학습된 teacher backbone 의 feature 가 클래스를 분리하는지 검사.
#
#   DATA_ROOT/
#     classA/ *.png
#     classB/ *.png
#     ...
# ============================================================
set -euo pipefail

# ── 사용자 설정 ──────────────────────────────────────────────
DATA_ROOT="${DATA_ROOT:?must set DATA_ROOT=/path/to/class_folders}"
WEIGHTS="${WEIGHTS:?must set WEIGHTS=/path/to/teacher_checkpoint.pth (또는 DCP 디렉토리)}"
CONFIG="${CONFIG:-dino_v3/dinov3/configs/train/weaksup/stage2_ssl_weaksup.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-./out/separability}"

# 진단 파라미터 (override 가능)
IMAGE_SIZE="${IMAGE_SIZE:-448}"     # 학습 global crop 과 동일 권장
FEATURE="${FEATURE:-concat}"        # cls | patchmean | concat
SHOTS="${SHOTS:-}"                  # 비우면 50/50 stratified, 숫자면 K-shot
N_TRIES="${N_TRIES:-5}"
TRAIN_AUG="${TRAIN_AUG:-0}"         # 1이면 support feature augmentation 사용
AUG_VIEWS="${AUG_VIEWS:-4}"        # 이미지당 augmentation view 수
CROP_SCALE="${CROP_SCALE:-0.8}"

EXTRA=()
if [[ -n "${SHOTS}" ]]; then EXTRA+=(--shots "${SHOTS}"); fi
if [[ "${TRAIN_AUG}" == "1" ]]; then EXTRA+=(--train-aug --aug-views "${AUG_VIEWS}" --crop-scale "${CROP_SCALE}"); fi

python dino_v3/dinov3/eval/fewshot_separability.py \
    --config-file "${CONFIG}" \
    --pretrained-weights "${WEIGHTS}" \
    --data-root "${DATA_ROOT}" \
    --image-size "${IMAGE_SIZE}" \
    --feature "${FEATURE}" \
    --n-tries "${N_TRIES}" \
    --output-dir "${OUTPUT_DIR}" \
    "${EXTRA[@]}" \
    "$@"
