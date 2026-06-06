#!/usr/bin/env bash
# ============================================================
# Stage 2: SSL + Weak Supervised Per-Patch Pair-wise
#
# Stage 1 teacher 에서 resume.
# Labeled data 의 class 분리 신호 주입.
# ============================================================
set -euo pipefail

# ── 사용자 설정 ──────────────────────────────────────────────
SHARDS_PATH="${SHARDS_PATH:-/path/to/webdataset/shards}"
LABELED_ROOT="${LABELED_ROOT:-/path/to/labeled_data}"
OUTPUT_DIR="${OUTPUT_DIR:-./out/stage2}"
STAGE1_TEACHER="${STAGE1_TEACHER:?must set STAGE1_TEACHER=/path/to/stage1/eval/.../teacher_checkpoint.pth}"
NPROC="${NPROC:-4}"

# Weak sup hyperparameters (override 가능)
LABELED_RATIO="${LABELED_RATIO:-0.25}"
LAMBDA_W="${LAMBDA_W:-20.0}"
T_VALUE="${T_VALUE:-8.0}"

# ── Config ──────────────────────────────────────────────────
CONFIG="dino_v3/dinov3/configs/train/weaksup/stage2_ssl_weaksup.yaml"

mkdir -p "${OUTPUT_DIR}"

# ── CLI overrides ───────────────────────────────────────────
CLI_OVERRIDES=(
    "train.dataset_path=WebDataset:path=${SHARDS_PATH}"
    "train.output_dir=${OUTPUT_DIR}"
    "student.resume_from_teacher_chkpt=${STAGE1_TEACHER}"
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
