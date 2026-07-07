# Feature Extraction Service (backbone-as-a-service)

DINOv3(weak-sup) 백본을 **공통 feature 추출기**로 제공하는 stateless REST 서비스.
이미지 → embedding 만 반환한다. 분류/이상탐지/분할 등은 이 위에 얹히는 소비자.

## 왜 feature 서비스부터?

무거운 백본(ViT-L, GPU, torch)만 서버에 두고, 분류 헤드(`ClassifierHead`)와
`DefectRegistry` 는 **torch-free 순수 numpy** 라 클라이언트 PC에서 embedding 으로 분류한다.
→ 백본은 여러 분석 기능이 공유하는 플랫폼 primitive 가 되고, 분류는 그 소비자 하나.

```
[클라이언트]                          [GPU 서버: feature_service.py]
 image ──POST /features──────────────▶ EMFeatureExtractor (백본 1회 로드)
        ◀──── embedding(JSON) ──────── get_intermediate_layers → pool_tokens
 ClassifierHead(numpy): embedding → class
```

## 기동 (서버, GPU 필요)

```bash
EM_CKPT=/path/to/teacher_checkpoint.pth \
EM_CONFIG=dino_v3/dinov3/configs/train/weaksup/stage2_ssl_weaksup.yaml \
bash scripts/serve_features.sh
```

| env | 기본값 | 설명 |
|---|---|---|
| `EM_CKPT` | (필수) | teacher 체크포인트(.pth 또는 DCP 디렉토리) |
| `EM_CONFIG` | stage2_ssl_weaksup.yaml | 학습에 쓴 config(arch 일치) |
| `EM_IMAGE_SIZE` | 448 | 학습 global crop 과 동일 권장 |
| `EM_FEATURE` | concat | cls / patchmean / concat |
| `EM_N_BLOCKS` | 1 | 사용할 마지막 블록 수 |
| `EM_PORT` / `EM_HOST` | 8000 / 0.0.0.0 | |

의존성: `flask`(추가됨), torch/torchvision/numpy/pillow/scikit-learn(기존).

## 엔드포인트

### `GET /health`
```json
{"status":"ok","model":{"embed_dim":1024,"image_size":448,"feature_kind":"concat","n_blocks":1}}
```

### `POST /features`
- multipart: `image`(1장) 또는 `images`(여러 장)
- 옵션 쿼리: `?feature_kind=cls|patchmean|concat` · `n_blocks=<int>` (기본은 서버 config)
- 응답:
```json
{"model":{...}, "count":1,
 "results":[{"filename":"x.png","dim":2048,"feature":[0.01, ...]}]}
```
- `feature` 는 L2 정규화된 pooled embedding. 길이 = embed_dim×(n_blocks+1) (concat 기준).

```bash
curl http://localhost:8000/health
curl -F image=@sample.png http://localhost:8000/features
curl -F images=@a.png -F images=@b.png "http://localhost:8000/features?feature_kind=cls"
```

## 클라이언트에서 분류/등록 (torch-free)

`inspection/service/client_example.py` — numpy + PIL(+requests/urllib) 만 필요.

```bash
# 1) 클래스별 폴더 → 서버 embedding 수집 → registry/head 구성 (백본 없이 클라에서 완결)
python inspection/service/client_example.py train --server http://SERVER:8000 \
    --data-root /path/to/class_folders --out registry.npz --clf logreg
#   → registry.npz + registry_head.npz 생성

# 2) 새 이미지 분류
python inspection/service/client_example.py classify --server http://SERVER:8000 \
    --input query_or_dir --head registry_head.npz --topk 3
```

feature 규약(정규화·feature_kind·image_size)이 서버에서 오는 embedding 으로 통일되므로
학습/추론이 자동 일치한다.

## 확장 노트

- **서버에서 분류까지 ((2) 구조)**: 이 서비스에 `EMClassifier`(백본+헤드) 를 얹어 `/predict`
  엔드포인트를 추가하면 된다. feature 추출부가 이미 분리돼 있어 얇은 래퍼로 충분.
- **dense 분석**: 현재는 pooled embedding 만. patch 토큰 반환(`include=patch`)은 향후 확장.
- **프로덕션**: GPU 모델 특성상 프로세스 1개가 낫다 → `gunicorn -w 1 --threads N feature_service:app`
  (`service/` 에서 실행). 백본은 첫 요청 시 lazy 로드(`_ensure_loaded`)되어 gunicorn 에서도
  별도 조정 없이 동작.
- enroll/모델관리/UI/인증은 현재 범위 밖.
