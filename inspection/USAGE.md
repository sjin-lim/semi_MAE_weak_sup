# inspection 사용법 (Quickstart)

DINO 백본(frozen)을 소비하는 **불량 분류/분석** 도구 모음. 이 문서가 진입점이다.
(세부는 [README.md](README.md)=분리도/분류 도구, [service/README.md](service/README.md)=feature 서비스,
설계는 [../docs/DESIGN_few_shot_defect_inspection.md](../docs/DESIGN_few_shot_defect_inspection.md).)

## 0. 전제

- **실행 위치: repo 루트** (`semi_MAE_weak_sup/`). 예: `python inspection/em_classifier.py ...`
- **서버(GPU)에서 실행.** 로컬엔 torch 없음(헤드 로직 numpy 단위테스트만 로컬 가능).
- **`<config>`** = 체크포인트를 **학습할 때 쓴 config**(arch 일치). 예:
  `dino_v3/dinov3/configs/train/weaksup/stage2_ssl_weaksup.yaml`
- **`<ckpt>`** = teacher 체크포인트(`.pth` 또는 DCP 디렉토리).

## 1. 구성 요소 한눈에

| 파일 | 역할 | 백본 |
|---|---|---|
| `inspection/fewshot_separability.py` | 폴더-클래스 **분리도 진단**(NCM/kNN/logreg + silhouette) | 직접 로드 |
| `inspection/em_classifier.py` | few-shot **분류 학습/저장/추론** + **증분 registry** | 직접 로드 |
| `inspection/fewshot_heatmap.py` | **어디 보고 판단했나** heatmap(class-evidence) | 직접 로드 |
| `inspection/service/feature_service.py` | 백본을 **feature 서비스**로 서빙(Flask) | 서비스 본체 |
| `inspection/service/client_example.py` | 서비스 소비 **torch-free 클라**(분류/등록) | 서비스 필요 |

## 2. 두 실행 모드

- **direct**(기본): 도구가 백본을 in-process 로드. torch+GPU 필요, 서비스 불필요.
  → separability / em_classifier / heatmap / `em_classification_demo.ipynb`
- **service**: 백본은 서버(feature_service), 분류는 클라(torch-free).
  → `client_example.py` / `em_classification_service_demo.ipynb`

---

## 3. 태스크별 사용법 (repo 루트에서)

### A. 분리도 검사 — **제일 먼저** (DINO feature가 불량을 가르는지 확인)
```bash
DATA_ROOT=/data/classes WEIGHTS=<ckpt> CONFIG=<config> \
bash scripts/eval_separability.sh
```
- 클래스별 폴더(ImageFolder) 필요. 결과: `out/separability/`(정확도·혼동행렬·t-SNE).
- 여기서 갈리면 분류가 바로 동작. 안 갈리면 폴더/해상도/pooling 점검.

### B. few-shot 분류: 학습 → 저장 → 추론
```bash
# 학습 + 아티팩트(.npz) 저장
python inspection/em_classifier.py fit \
    --config-file <config> --pretrained-weights <ckpt> \
    --data-root /data/classes --clf logreg --out out/em_head.npz
# 추론 (백본 경로는 아티팩트에 저장됨)
python inspection/em_classifier.py predict --artifact out/em_head.npz --input /data/query --topk 3
```

### C. 증분 불량 등록/삭제 (백본 재학습 없음)
```bash
python inspection/em_classifier.py enroll --registry out/defects.npz \
    --name scratch --image-dir /data/scratch --config-file <config> --pretrained-weights <ckpt>
python inspection/em_classifier.py enroll --registry out/defects.npz --name dent --image-dir /data/dent
python inspection/em_classifier.py list   --registry out/defects.npz
python inspection/em_classifier.py remove --registry out/defects.npz --name dent
python inspection/em_classifier.py predict --registry out/defects.npz --input /data/query
```

### D. heatmap (판단 근거 시각화)
```bash
DATA_ROOT=/data/classes WEIGHTS=<ckpt> CONFIG=<config> PER_CLASS=4 \
bash scripts/eval_heatmap.sh          # MODE=contrastive(기본)/raw, CLF, TARGET 등 env
```

### E. feature 서비스 + service 기반 분류
```bash
# 1) 서버 기동 (GPU)
EM_CKPT=<ckpt> EM_CONFIG=<config> bash scripts/serve_features.sh   # :8000
# 2) 클라(torch-free): 폴더→registry/head 구성
python inspection/service/client_example.py train --server http://localhost:8000 \
    --data-root /data/classes --out out/registry.npz --clf logreg
# 3) 클라 분류
python inspection/service/client_example.py classify --server http://localhost:8000 \
    --input /data/query --head out/registry_head.npz
```
확인: `curl http://localhost:8000/health`

### F. 노트북 (tests/classification/)
- `em_classification_demo.ipynb` — **direct** 모드(백본 직접). torch+GPU.
- `em_classification_service_demo.ipynb` — **service** 모드(torch-free). 서버 기동 필요.
- 설정 셀 경로만 채우고 위에서부터 실행.

### G. 단위 테스트
```bash
pytest tests/classification            # 헤드/registry numpy 로직(로컬도 가능)
```

---

## 4. 참고
- 정규화는 학습과 동일하게 **PerImageNormalize**(도구 내장) — [../docs/ARCHITECTURE_boundaries.md](../docs/ARCHITECTURE_boundaries.md).
- 백본 해상도/불량 유형별 전략·측정(Stage 2)은 설계 문서 참고.
- 의존 방향: `inspection → dinov3` 단방향(백본은 downstream 미참조).
