#!/usr/bin/env bash
# ============================================================
# Stage 1: Pure SSL Fine-tuning (weak_sup OFF)
#
# 기존 DINOv3 코드와 동일 동작. weak_sup.enabled=false 분기.
# ============================================================
set -euo pipefail

# ── 사용자 설정 ──────────────────────────────────────────────
SHARDS_PATH="${SHARDS_PATH:-/path/to/webdataset/shards}"
OUTPUT_DIR="${OUTPUT_DIR:-./out/stage1}"
PRIOR_TEACHER="${PRIOR_TEACHER:-}"   # 선택: v4_ibot_ft / v7 teacher 에서 출발
NPROC="${NPROC:-4}"                   # GPU 수

# ── Config ──────────────────────────────────────────────────
CONFIG="dino_v3/dinov3/configs/train/weaksup/stage1_ssl_only.yaml"

mkdir -p "${OUTPUT_DIR}"

# ── CLI overrides ───────────────────────────────────────────
CLI_OVERRIDES=(
    "train.dataset_path=WebDataset:path=${SHARDS_PATH}"
    "train.output_dir=${OUTPUT_DIR}"
)
if [[ -n "${PRIOR_TEACHER}" ]]; then
    CLI_OVERRIDES+=("student.resume_from_teacher_chkpt=${PRIOR_TEACHER}")
fi

# ── Launch ──────────────────────────────────────────────────
torchrun --nproc_per_node="${NPROC}" \
    dino_v3/dinov3/train/train.py \
    --config-file "${CONFIG}" \
    --no-resume \
    "${CLI_OVERRIDES[@]}" \
    "$@"
