# Handoff Context — semi_MAE_weak_sup

> 새 Claude 대화창에서 이 프로젝트를 이어받기 위한 컨텍스트 문서.
> 지금까지 semi_MAE 에서 진행한 SSL 학습 + weak supervised 설계 전체 정리.

---

## 1. 프로젝트 한 줄 요약

**EM (전자현미경) 도메인에 SSL (DINOv3) backbone 을 적응시키는데, "옅은 회색 차이로 구분되는 다른 material" 같은 visually-similar-but-semantically-different 신호를 SSL invariance 가 죽이는 문제를 weak supervision 으로 보완.**

라벨이 일부 (수천 장) 있어서 활용 가능. Pure SSL → SSL + Weak Sup → High-Res adaptation 의 3-stage pipeline.

---

## 2. 두 repo 관계

```
D:\임성진_개발\
├── semi_MAE/              ← 원본 (이전 작업 git history 보존)
│   ├── dino_v3/           DINOv3 코드 + EM augmentation 통합본
│   ├── notebooks/         normalization 검증 등 평가 노트북
│   ├── WEAKLY_SUPERVISED_PIPELINE.md   설계 문서
│   ├── ANAYLYSIS_0527.md  Feature separability 분석
│   └── (MAE legacy 파일들 — 추후 정리 예정)
│
└── semi_MAE_weak_sup/     ← 신규 repo (이 작업의 메인)
    ├── dino_v3/           원본에서 복사 + weak_sup 모듈 추가
    ├── scripts/           train_stage1/2/3.sh
    ├── README.md
    ├── INTEGRATION_NOTES.md
    └── HANDOFF_CONTEXT.md (이 파일)
```

원본 repo 는 절대 건드리지 않고, 신규 repo 에서만 작업 진행.

---

## 3. 핵심 발견 사항 (검증 완료)

### ✅ Pure SSL 변형으론 한계 확인

| 시도 | 결과 |
|---|---|
| v4 (DINOv3 → EM, 200 epoch) | gap 0.025 → 0.753 큰 개선, 하지만 sim[1,2] = 0.916 |
| v4_ibot_ft (iBOT 강화, 50 epoch) | gap 거의 유지, sim[1,2] 변화 미미 |
| v5 high-res adapt | sim[1,2] **오히려 악화** |
| v6 augmentation ablation (CLAHE off 등) | marginal |
| v7 combined (aug + masking 동시) | 큰 변화 없음 |
| Normalization 가설 검증 (notebook) | **fixed norm 도 도움 안 됨** |

→ **결론**: SSL invariance objective 의 본질적 한계. Weak supervision 필요.

### ✅ Inference 단 CLAHE 가설 검증
모델이 CLAHE 적용된 입력에서 오히려 sim[1,2] 가 높아짐 → CLAHE 가 training 시 invariance 학습시킨 결과. **CLAHE 제거가 정당**.

### ✅ Layer 깊이 vs class 분리도
L3 → L11 갈수록 sim[1,2] 악화, intra-class sim 좋아짐. **모든 layer 사용해도 1-2 분리 불가**.

---

## 4. Weak Supervised Loss 설계 (합의 완료)

### 4.1 정식 수식

```
L_total = L_DINO + L_iBOT + λ_K · L_KoLeo                          (모든 batch — SSL 그대로)
        + λ_W · L_weak                                              (labeled image 만)

L_weak = (1/|P|) Σ_{(c_a, c_b) ∈ P} mean(  (exp(T·sim_patch) - 1) / (exp(T) - 1)  )

  P = image 내 non-background class pair 들
  sim_patch = class c_a 의 모든 patch 와 class c_b 의 모든 patch 의 pair-wise cosine
```

### 4.2 핵심 design 결정 (검토 단계마다 합의)

| 결정 | 이유 |
|---|---|
| **Per-patch** (centroid 안 씀) | Centroid 게임 (outlier 만 옮겨도 centroid 변화) 방지 |
| **Pair-wise** (push only) | SSL prototype 구조 보존. "어디에 있어라" 강제 X |
| **Class 0 (bg) 제외** | Background 정의가 dataset 마다 다름 — pair 의미 없음 |
| **Normalized exp** (bounded [0, 1]) | Gradient 발산 방지. λ_W 조정 가능 |
| **Self-curriculum 자연 발생** | exp 함수 특성. Threshold 불필요 |
| **T = 8** | Half-max sim ≈ 0.91. 0.8 이하는 거의 free |
| **λ_W = 20** | SSL gradient 의 ~10-30% 가 target |
| **Class head 없음** | Dataset 마다 class 수 다름. 고정 head 불가능 |
| **iBOT-style 가 아닌 단순 pair-wise 선택** | "Pull to centroid" 가 supervised 에 가까움 → SSL philosophy 보존 위해 pure push |

### 4.3 우려 사항 검토 결과

- **Low sim 에서도 exp 가 loss 만드는가?** → 0.001 수준으로 무시 가능 (수치 검증 완료)
- **Prototype 구조 망가지나?** → backbone level 적용 + 작은 λ → 안전
- **계산 비용?** → centroid 보다 무겁지만 SSL forward 대비 무시 가능
- **너무 많은 hyperparameter?** → T, λ_W 2개로 정리됨

---

## 5. 3-Stage Pipeline

```
Stage 1: Pure SSL
  Config: dino_v3/dinov3/configs/train/weaksup/stage1_ssl_only.yaml
  weak_sup.enabled: false
  목적: EM 도메인 일반 representation 확보
  출발점: 선택적으로 semi_MAE 의 v4_ibot_ft / v7 teacher 사용 가능

Stage 2: SSL + Weak Sup (핵심)
  Config: dino_v3/dinov3/configs/train/weaksup/stage2_ssl_weaksup.yaml
  weak_sup.enabled: true, λ_W=20, T=8
  목적: 다양한 confused pair 분리 신호 주입
  출발점: Stage 1 teacher_checkpoint.pth

Stage 3: High-Res + Weak Sup (light)
  Config: dino_v3/dinov3/configs/train/weaksup/stage3_hires_adapt.yaml
  weak_sup.enabled: true, λ_W=5 (preservation)
  global crops: [512, 768] mixed
  목적: 추론 해상도 확장 + Stage 2 효과 보존
  출발점: Stage 2 teacher_checkpoint.pth
```

---

## 6. 구현 상태

### ✅ 완료된 신규 모듈

```
dino_v3/dinov3/data/labeled/
├── __init__.py
├── dataset.py                   LabeledEMDataset (multi-dataset folder)
├── joint_transforms.py          image+mask 동기 augmentation (tv_tensors)
└── patch_label.py               mask → patch-level majority class

dino_v3/dinov3/train/weaksup/
├── __init__.py
├── losses.py                    per_patch_pairwise_loss, batch_weak_loss,
│                                 compute_pair_sim_stats
├── mixed_loader.py              RatioMixedLoader (labeled/unlabeled ratio mix)
└── integration.py               build_labeled_loader_from_cfg,
                                 labeled_collate_with_weak_sup

dino_v3/dinov3/configs/train/weaksup/
├── stage1_ssl_only.yaml         weak_sup.enabled: false
├── stage2_ssl_weaksup.yaml      weak_sup.enabled: true, λ=20
└── stage3_hires_adapt.yaml      weak_sup.enabled: true, λ=5

scripts/
├── train_stage1.sh
├── train_stage2.sh
└── train_stage3.sh
```

### ✅ 수정된 기존 파일

- `dino_v3/dinov3/configs/ssl_default_config.yaml` — `weak_sup` 섹션 추가
- `dino_v3/dinov3/train/train.py` — `do_train` 에 RatioMixedLoader wrapping +
  `forward_backward` 호출 시 `is_labeled` 전달
- `dino_v3/dinov3/train/ssl_meta_arch.py` — `forward_backward` 에 weak_sup
  분기 추가 (lambda warmup, pair stats logging 포함)

### ⏳ 미완료 / 다음 단계

1. **Toy unit tests**
   - `per_patch_pairwise_loss` 가 random feature 에서 ~0.001, similar feature 에서 ~0.3+ 산출 확인
   - `LabeledEMDataset` 가 toy dataset 폴더 로드 확인
   - `JointGeometricAugmentation` 의 image-mask 동기 확인

2. **Stage 1 sanity test**
   - 신규 repo 에서 `weak_sup.enabled: false` 로 학습 시작
   - Loss 곡선이 semi_MAE 의 v7_combined 와 거의 일치하는지 확인

3. **Stage 2 sanity test**
   - 작은 labeled dataset (예: 10장) 으로 짧은 학습 (200 step)
   - `weak_loss`, `lambda_w`, `pair_sim_max`, `pair_sim_mean` logging 확인
   - iBOT/DINO loss 발산하지 않는지 확인

4. **실제 labeled dataset 구성**
   - 폴더 구조: `labeled_root/dataset_*/images/{stem}.png` + `masks/{stem}.png`
   - 각 dataset 폴더에 `meta.json` (선택)
   - Mask pixel value = class id (0 = background)

5. **본격 학습 — Stage 1 → 2 → 3 순차**

6. **원본 semi_MAE 의 MAE legacy 정리** (별도, 신규 repo 검증 후)

---

## 7. 신규 repo 디렉토리 구조

```
semi_MAE_weak_sup/
├── README.md                              ← 프로젝트 개요
├── INTEGRATION_NOTES.md                   ← 통합 작업 가이드 (대부분 완료됨)
├── HANDOFF_CONTEXT.md                     ← 이 파일
├── LICENSE
├── CODE_OF_CONDUCT.md
├── .gitignore
│
├── dino_v3/                               ← DINO 코드 (semi_MAE 에서 복사)
│   └── dinov3/
│       ├── configs/
│       │   ├── ssl_default_config.yaml    [수정] weak_sup section 추가
│       │   └── train/
│       │       ├── (기존 v4, v6, v7 configs 모두 포함)
│       │       └── weaksup/               [신규]
│       │           ├── stage1_ssl_only.yaml
│       │           ├── stage2_ssl_weaksup.yaml
│       │           └── stage3_hires_adapt.yaml
│       ├── data/
│       │   ├── (기존 데이터 모듈들)
│       │   └── labeled/                   [신규]
│       │       ├── __init__.py
│       │       ├── dataset.py
│       │       ├── joint_transforms.py
│       │       └── patch_label.py
│       └── train/
│           ├── train.py                   [수정] weak_sup 분기 추가
│           ├── ssl_meta_arch.py           [수정] forward_backward 에 weak loss
│           └── weaksup/                   [신규]
│               ├── __init__.py
│               ├── losses.py
│               ├── mixed_loader.py
│               └── integration.py
│
├── util/                                  ← semi_MAE 에서 복사 (PerImageNormalize 등)
│
└── scripts/                               ← 학습 실행 스크립트
    ├── train_stage1.sh
    ├── train_stage2.sh
    └── train_stage3.sh
```

---

## 8. 실행 가이드

### Stage 1
```bash
cd /path/to/semi_MAE_weak_sup
SHARDS_PATH=/path/to/webdataset/shards \
OUTPUT_DIR=./out/stage1 \
PRIOR_TEACHER=/path/to/v4_ibot_ft/.../teacher_checkpoint.pth \
NPROC=4 \
bash scripts/train_stage1.sh
```

### Stage 2 (Stage 1 끝난 후)
```bash
SHARDS_PATH=/path/to/webdataset/shards \
LABELED_ROOT=/path/to/labeled_data \
OUTPUT_DIR=./out/stage2 \
STAGE1_TEACHER=./out/stage1/eval/training_XXXXX/teacher_checkpoint.pth \
NPROC=4 \
bash scripts/train_stage2.sh
```

### Stage 3 (Stage 2 끝난 후)
```bash
SHARDS_PATH=/path/to/webdataset/shards \
LABELED_ROOT=/path/to/labeled_data \
OUTPUT_DIR=./out/stage3 \
STAGE2_TEACHER=./out/stage2/eval/training_XXXXX/teacher_checkpoint.pth \
NPROC=4 \
bash scripts/train_stage3.sh
```

### Hyperparameter Override

```bash
# Stage 2 에서 λ 조정 예시
LAMBDA_W=10 T_VALUE=10 LABELED_RATIO=0.20 \
bash scripts/train_stage2.sh
```

---

## 9. Labeled Dataset 형식 명세

```
labeled_root/
├── dataset_A/
│   ├── images/
│   │   ├── img_001.png
│   │   └── ...
│   ├── masks/
│   │   ├── img_001.png       ← pixel value = class id (0=bg, 1, 2, ...)
│   │   └── ...
│   └── meta.json (선택)
├── dataset_B/
│   ├── images/
│   ├── masks/
│   └── meta.json
└── ...
```

`meta.json` 형식:
```json
{
  "name": "dataset_A",
  "classes": {"0": "background", "1": "phase_alpha", "2": "phase_beta"},
  "background_class": 0,
  "notes": "..."
}
```

- Image stem 과 mask stem 이 매칭됨 (확장자 무관)
- Image: RGB 또는 grayscale → 3채널로 변환됨
- Mask: 2D pixel-level class id (RGB 면 첫 채널 사용)
- 각 dataset 마다 class 수 다를 수 있음 (Per-patch pair-wise loss 가 dynamic 처리)

---

## 10. 학습 중 모니터링 권장

Stage 2/3 에서 다음 metric 추적:

| Metric | 정상 거동 | 이상 신호 |
|---|---|---|
| `weak_loss` | 시간 따라 감소 (self-curriculum) | 0 유지 → 효과 없음 / 폭등 → unstable |
| `lambda_w` | warmup 동안 0 → λ_max 증가 | (자동 schedule) |
| `pair_sim_max` | 점진 감소 | 변동 없음 → 학습 효과 없음 |
| `pair_sim_mean` | 점진 감소 | 0 이하로 over-separation |
| `loss_dino` | 안정 | 발산 → SSL 망가짐 |
| `loss_ibot` | 안정 | 발산 → SSL 망가짐 |
| `*_grad_norm` | clip(3.0) 한계 안 | 자주 clip → λ 줄여야 함 |

---

## 11. 실패 모드 & 대응

| 증상 | 원인 가설 | 대응 |
|---|---|---|
| iBOT loss 급상승 | weak sup 가 backbone 망침 | λ_W 절반 |
| DINO loss 발산 | View consistency 무너짐 | λ_W 절반, warmup 늘리기 |
| `weak_loss = 0` 만 유지 | self-curriculum 도달 X 또는 labeled batch 안 들어옴 | log 확인, labeled_ratio 늘리기 |
| `pair_sim_max < 0` | over-separation | T 낮추기 (8 → 5) |
| 학습 너무 느림 | labeled loader I/O | num_workers_labeled 늘리기 |
| NaN loss | 일반적 SSL 문제 | LR 절반, clip_grad 절반 |

---

## 12. 다음 Claude 세션에서 진행할 만한 작업

(우선순위 순)

1. **Stage 1 sanity test 실행**
   - 신규 repo 에서 `bash scripts/train_stage1.sh` 짧게 (200 step) 돌리기
   - semi_MAE 와 동일 동작 확인 (loss 값 비슷)

2. **`losses.py` toy unit test** — random vs similar features
   - INTEGRATION_NOTES.md 의 "Toy Test" 섹션 참조

3. **`LabeledEMDataset` smoke test**
   - 작은 toy labeled dataset 만들어 `len(ds)`, `ds[0]` 확인
   - mask → patch_label 시각화

4. **Stage 2 sanity test (toy data)**
   - 10장 labeled, 200 step
   - `weak_loss` 가 0 이상 logging 되는지 확인
   - iBOT/DINO loss 안정성 확인

5. **본격 Stage 1 학습**

6. **본격 Stage 2 학습** (Stage 1 완료 후)

7. **본격 Stage 3 학습** (Stage 2 완료 후)

8. **각 Stage 후 평가**
   - sim[1,2] 등 metric (semi_MAE 의 `INSID3` 또는 자체 sim 계산)
   - 다른 class pair sim 분포
   - DINO-UNet downstream
   - PCA 시각화

---

## 13. 보존된 semi_MAE 자료 (참조용)

원본 `D:\임성진_개발\semi_MAE\` 에 다음이 있음 (참조만, 수정 X):

- **`WEAKLY_SUPERVISED_PIPELINE.md`** — 본 설계의 상세 문서 (이 핸드오프 와 부분 중복)
- **`ANAYLYSIS_0527.md`** — Layer 별 feature separability 진단 결과
- **`notebooks/normalization_inference_test.ipynb`** — instance_norm 가설 검증 (negative result)
- **`notebooks/fewshot_*.ipynb`** — fewshot segmentation 평가 (DINO-UNet, INSID3 등)
- **`notebooks/dinov3_pca_eval.ipynb`** — PCA 시각화

학습된 체크포인트 위치 (사용자 환경):
- v4_ibot_ft / v7 teacher checkpoint 는 학습 서버에 있음 (로컬 X)

---

## 14. 핵심 기술 결정 요약

### 왜 이런 loss 가 선택됐는가?

1. **Direct supervised (CE on labels) 안 쓴 이유**:
   - "Patch → 자기 class 로 모이라" 라는 강한 신호
   - SSL prototype 구조 깨질 우려
   - 사실상 supervised classification

2. **Centroid pair-wise 안 쓴 이유**:
   - 5개 outlier 만 push 해도 centroid 평균이 옮겨감 → trivial collapse
   - Per-patch sim 이 robust

3. **Class head (학습 가능) 안 쓴 이유**:
   - Dataset 마다 class 개수 다름 → 고정 N-class head 불가능
   - Background (class 0) 정의 가변

4. **iBOT-style cross-entropy with dynamic centroids 안 쓴 이유**:
   - Pull-to-centroid 가 dominant → "어디에 있어라" 강제
   - SSL philosophy 와 거리감

5. **Normalized exponential 의 self-curriculum 사용 이유**:
   - Threshold 없이 자연스러운 force 곡선
   - Bounded loss [0, 1]
   - Hyperparameter (T, λ) 단순

### 왜 backbone level 에 적용?

Prototype space (DINO/iBOT head 후) 가 아닌 **backbone patch features** 에 적용. 이유:
- Prototype 구조 직접 간섭 X
- Sinkhorn balance 자연 유지
- Backbone 만 class-aware 하게 학습 → head 가 자체적으로 적응

### 왜 Stage 분리?

- Stage 1 first → Stage 2 의 hard negative mining 이 의미 있으려면 baseline 필요
- Stage 3 last → 가장 작은 변화 (해상도 적응), 앞 단계 결과 보존

---

## 15. 한 줄 요약 (TL;DR)

> EM 도메인 DINOv3 fine-tuning 에 **per-patch pair-wise normalized exponential loss** 를 weak sup 으로 추가하여, SSL 이 못 잡는 visually-similar-but-semantically-different class 분리를 보완. **3-Stage** (SSL only → SSL+WeakSup → HiRes). 신규 repo `semi_MAE_weak_sup` 에 코드 통합 완료. Sanity test 부터 시작하면 됨.
