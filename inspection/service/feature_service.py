# Copyright (c) 2026.
#
# Feature Extraction Service (backbone-as-a-service).
#
# DINOv3(weak-sup) teacher 백본을 공통 feature 추출기로 제공하는 stateless REST 서비스.
# 이미지 → embedding 만 반환한다. 분류/이상탐지/분할 등은 이 위에 얹히는 소비자.
#
# 왜 feature 만?
#   무거운 백본(ViT-L, GPU, torch)만 서버에 두고, 우리 분류 헤드/registry 는 torch-free
#   numpy 라 클라이언트에서 embedding 으로 분류 가능(client_example.py 참고).
#
# 기동 (서버, GPU 필요):
#   EM_CONFIG=dino_v3/dinov3/configs/train/weaksup/stage2_ssl_weaksup.yaml \
#   EM_CKPT=/path/to/teacher_checkpoint.pth \
#   python inspection/service/feature_service.py
#   (또는 scripts/serve_features.sh)
#
# WSGI 서버: 기본 waitress(프로덕션·Windows 적합, 순수 파이썬). 미설치 시 Flask 개발 서버 폴백.
#   EM_SERVER=waitress|flask, EM_THREADS=<int>(waitress 스레드 수).
#   단일 GPU라 추론은 내부 _LOCK 으로 직렬화 → 멀티스레드는 큐잉/동시 요청 수신용.
#
# 엔드포인트:
#   GET  /health              → 모델 정보
#   POST /features            → multipart image(1) 또는 images(N) → embedding
#        옵션 쿼리: ?feature_kind=cls|patchmean|concat & n_blocks=<int>

import io
import logging
import os
import sys
import threading
from pathlib import Path

# repo 루트 + dino_v3 를 path 에 (inspection/service/ 위치)
_HERE = Path(__file__).resolve()
_REPO = _HERE.parents[2]
for p in (str(_REPO), str(_REPO / "dino_v3")):
    if p not in sys.path:
        sys.path.insert(0, p)

from flask import Flask, jsonify, request  # noqa: E402
from PIL import Image  # noqa: E402

from inspection.em_classifier import FEATURE_KINDS, EMFeatureExtractor  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("feature_service")

app = Flask(__name__)

# 전역 백본 상태 (1회 로드) + GPU 추론 직렬화 락 + 로드 락
_EXTRACTOR: EMFeatureExtractor | None = None
_LOCK = threading.Lock()
_LOAD_LOCK = threading.Lock()
_CFG: dict = {}


def _ensure_loaded() -> EMFeatureExtractor:
    """백본을 1회만 로드(idempotent). python 실행/gunicorn 양쪽에서 안전."""
    global _EXTRACTOR
    if _EXTRACTOR is None:
        with _LOAD_LOCK:
            if _EXTRACTOR is None:
                _EXTRACTOR = _load_extractor()
    return _EXTRACTOR


def _load_extractor() -> EMFeatureExtractor:
    global _CFG
    config = os.environ.get("EM_CONFIG")
    ckpt = os.environ.get("EM_CKPT")
    if not config or not ckpt:
        raise SystemExit("EM_CONFIG / EM_CKPT 환경변수를 설정하세요.")
    if not os.path.isabs(config):
        config = str(_REPO / config)
    _CFG = {
        "config_file": config,
        "pretrained_weights": ckpt,
        "image_size": int(os.environ.get("EM_IMAGE_SIZE", 448)),
        "feature_kind": os.environ.get("EM_FEATURE", "concat"),
        "n_blocks": int(os.environ.get("EM_N_BLOCKS", 1)),
        "cache_dir": os.environ.get("EM_CACHE_DIR", "./.em_cache"),
    }
    logger.info(f"백본 로드 중... {_CFG}")
    ext = EMFeatureExtractor(
        _CFG["config_file"], _CFG["pretrained_weights"], image_size=_CFG["image_size"],
        n_blocks=_CFG["n_blocks"], feature_kind=_CFG["feature_kind"], cache_dir=_CFG["cache_dir"],
    )
    logger.info(f"로드 완료. embed_dim={ext.embed_dim}, feature={ext.feature_kind}")
    return ext


def _model_info() -> dict:
    return {
        "embed_dim": _EXTRACTOR.embed_dim,
        "image_size": _EXTRACTOR.image_size,
        "feature_kind": _EXTRACTOR.feature_kind,
        "n_blocks": _EXTRACTOR.n_blocks,
        "config_file": _CFG.get("config_file"),
    }


@app.get("/health")
def health():
    _ensure_loaded()
    return jsonify({"status": "ok", "model": _model_info()})


@app.post("/features")
def features():
    _ensure_loaded()

    # 업로드 이미지 수집 (image 단일 또는 images 다중)
    files = request.files.getlist("images") + request.files.getlist("image")
    if not files:
        return jsonify({"error": "이미지 없음 (multipart field 'image' 또는 'images')"}), 400

    # 요청별 override
    feature_kind = request.args.get("feature_kind", _EXTRACTOR.feature_kind)
    if feature_kind not in FEATURE_KINDS:
        return jsonify({"error": f"feature_kind must be one of {FEATURE_KINDS}"}), 400
    try:
        n_blocks = int(request.args.get("n_blocks", _EXTRACTOR.n_blocks))
    except ValueError:
        return jsonify({"error": "n_blocks must be int"}), 400

    imgs, names = [], []
    for f in files:
        try:
            imgs.append(Image.open(io.BytesIO(f.read())).convert("RGB"))
            names.append(f.filename or f"img{len(names)}")
        except Exception as e:  # noqa: BLE001
            return jsonify({"error": f"이미지 디코드 실패({f.filename}): {e}"}), 400

    with _LOCK:  # GPU 추론 직렬화
        feats = _EXTRACTOR.embed(imgs, feature_kind=feature_kind, n_blocks=n_blocks)

    results = [
        {"filename": name, "dim": int(feats.shape[1]), "feature": feats[i].tolist()}
        for i, name in enumerate(names)
    ]
    info = _model_info()
    info["feature_kind"] = feature_kind
    info["n_blocks"] = n_blocks
    return jsonify({"model": info, "count": len(results), "results": results})


def main():
    _ensure_loaded()  # 기동 시 즉시 로드(실패를 바로 표면화)
    port = int(os.environ.get("EM_PORT", 8000))
    host = os.environ.get("EM_HOST", "0.0.0.0")
    threads = int(os.environ.get("EM_THREADS", 4))
    server = os.environ.get("EM_SERVER", "waitress").lower()  # waitress | flask

    # 단일 GPU라 내부 _LOCK 으로 추론이 직렬화됨 → 멀티스레드 WSGI 로 큐잉/동시 요청 처리.
    if server == "waitress":
        try:
            from waitress import serve
        except ImportError:
            logger.warning("waitress 미설치 → Flask 개발 서버로 폴백 (pip install waitress).")
        else:
            logger.info(f"waitress serve {host}:{port} (threads={threads}, GPU는 _LOCK 직렬화)")
            serve(app, host=host, port=port, threads=threads)
            return

    # Flask 내장 서버 (개발/디버그용 — 프로덕션 아님)
    logger.warning("Flask 개발 서버 사용 중 (프로덕션은 EM_SERVER=waitress 권장).")
    app.run(host=host, port=port, threaded=True)


if __name__ == "__main__":
    main()
