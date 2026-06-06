#!/usr/bin/env bash
# ============================================================
# Stage 3: High-Resolution Adaptation (weak sup retained, light)
#
# Stage 2 teacher 에서 resume.
# Multi-resolution {512, 768} crop. weak_sup λ 축소 (20 → 5).
# ============================================================
set -euo pipefail

# ── 사용자 설정 ──────────────────────────────────────────────
SHARDS_PATH="${SHARDS_PATH:-/path/to/webdataset/shards}"
LABELED_ROOT="${LABELED_ROOT:-/path/to/labeled_data}"
OUTPUT_DIR="${OUTPUT_DIR:-./out/stage3}"
STAGE2_TEACHER="${STAGE2_TEACHER:?must set STAGE2_TEACHER=/path/to/stage2/eval/.../teacher_checkpoint.pth}"
NPROC="${NPROC:-4}"

# Weak sup light preservation
LABELED_RATIO="${LABELED_RATIO:-0.15}"
LAMBDA_W="${LAMBDA_W:-5.0}"
T_VALUE="${T_VALUE:-8.0}"

# ── Config ──────────────────────────────────────────────────
CONFIG="dino_v3/dinov3/configs/train/weaksup/stage3_hires_adapt.yaml"

mkdir -p "${OUTPUT_DIR}"

# ── CLI overrides ───────────────────────────────────────────
CLI_OVERRIDES=(
    "train.dataset_path=WebDataset:path=${SHARDS_PATH}"
    "train.output_dir=${OUTPUT_DIR}"
    "student.resume_from_teacher_chkpt=${STAGE2_TEACHER}"
    "weak_sup.labeled_root=${LABELED_ROOT}"
    "weak_sup.labeled_ratio=${LABELED_RATIO}"
    "weak_sup.lambda_W=${LAMBDA_W}"
    "weak_sup.T=${T_VALUE}"
)

# ── Launch ──────────────────────────────────────────────────
torchrun --nproc_per_node="${NPROC}" \
    dino_v3/dinov3/train/train.py \
    --config-file "${CONFIG}" \
    --no-resume \
    "${CLI_OVERRIDES[@]}" \
    "$@"
