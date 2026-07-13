#!/usr/bin/env bash
# ============================================================
# Feature Extraction Service (backbone-as-a-service) 기동
#
# DINO 백본을 1회 로드해 이미지 → embedding 을 주는 REST 서비스.
# GPU 필요. 분류는 클라이언트에서 (service/client_example.py 참고).
# ============================================================
set -euo pipefail

export EM_CONFIG="${EM_CONFIG:-dino_v3/dinov3/configs/train/weaksup/stage2_ssl_weaksup.yaml}"
export EM_CKPT="${EM_CKPT:?must set EM_CKPT=/path/to/teacher_checkpoint.pth}"
export EM_IMAGE_SIZE="${EM_IMAGE_SIZE:-448}"
export EM_FEATURE="${EM_FEATURE:-concat}"     # cls | patchmean | concat
export EM_N_BLOCKS="${EM_N_BLOCKS:-1}"
export EM_PORT="${EM_PORT:-8000}"
export EM_HOST="${EM_HOST:-0.0.0.0}"
export EM_CACHE_DIR="${EM_CACHE_DIR:-./.em_cache}"
export EM_SERVER="${EM_SERVER:-waitress}"     # waitress(프로덕션) | flask(개발)
export EM_THREADS="${EM_THREADS:-4}"          # waitress 스레드 수

python inspection/service/feature_service.py "$@"
